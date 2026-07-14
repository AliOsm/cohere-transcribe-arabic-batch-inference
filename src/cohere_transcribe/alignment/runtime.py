"""MMS CTC emission generation and word-level forced alignment."""

from __future__ import annotations

import gc
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..device import empty_device_cache, is_out_of_memory_error
from ..models import (
    ALIGN_CONTEXT_S,
    ALIGN_MODEL_ID,
    ALIGN_MODEL_REVISION,
    ALIGN_WINDOW_S,
    BAR_FMT,
    INDENT,
    ISO3,
    SR,
    WordTiming,
    info,
)
from ..progress import progress_bar
from ..progress import write as progress_write


def load_aligner(
    device: str,
    dtype: torch.dtype,
    revision: str | None = ALIGN_MODEL_REVISION,
):
    from transformers import AutoTokenizer, Wav2Vec2ForCTC

    tokenizer = AutoTokenizer.from_pretrained(
        ALIGN_MODEL_ID,
        revision=revision,
        word_delimiter_token=None,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        ALIGN_MODEL_ID,
        dtype=dtype,
        attn_implementation="sdpa",
        revision=revision,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def alignment_frame_geometry(model) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    raw_ratio = getattr(config, "inputs_to_logits_ratio", None)
    if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, (int, np.integer)):
        raise RuntimeError(
            "Aligner config must define an integer inputs_to_logits_ratio"
        )
    ratio = int(raw_ratio)
    if ratio <= 0:
        raise RuntimeError("Aligner inputs_to_logits_ratio must be positive")

    window_samples = ALIGN_WINDOW_S * SR
    context_samples = ALIGN_CONTEXT_S * SR
    if window_samples % ratio or context_samples % ratio:
        raise RuntimeError(
            "Alignment window and context must be divisible by the model input stride"
        )
    return ratio, window_samples // ratio, context_samples // ratio


def build_alignment_window_batch(
    audio: np.ndarray,
    window_indices: range,
    window_samples: int,
    context_samples: int,
) -> np.ndarray:
    """Build the evaluated zero-padded 34-second contextual alignment windows."""
    input_samples = window_samples + 2 * context_samples
    batch: np.ndarray = np.zeros((len(window_indices), input_samples), dtype=np.float32)
    audio_samples = len(audio)
    for row, window_index in enumerate(window_indices):
        requested_start = window_index * window_samples - context_samples
        requested_end = requested_start + input_samples
        source_start = max(0, requested_start)
        source_end = min(audio_samples, requested_end)
        if source_end <= source_start:
            continue
        destination_start = source_start - requested_start
        destination_end = destination_start + source_end - source_start
        batch[row, destination_start:destination_end] = audio[source_start:source_end]
    return batch


def _compute_emissions_streaming(audio: np.ndarray, model, batch_size: int, label: str):
    window_samples = ALIGN_WINDOW_S * SR
    context_samples = ALIGN_CONTEXT_S * SR
    ratio, window_frames, context_frames = alignment_frame_geometry(model)
    total_windows = math.ceil(len(audio) / window_samples)
    extension_samples = total_windows * window_samples - len(audio)
    extension_frames = extension_samples // ratio
    emissions: np.ndarray | None = None
    write_offset = 0
    first_window = 0
    learned_batch_size = int(
        getattr(model, "_transcribe_align_batch_size", max(1, batch_size))
    )
    current_batch_size = max(1, min(batch_size, learned_batch_size))

    bar = progress_bar(
        total=total_windows,
        unit="win",
        desc=f"emissions {label}",
        dynamic_ncols=True,
        bar_format=BAR_FMT,
    )
    try:
        while first_window < total_windows:
            window_count = min(current_batch_size, total_windows - first_window)
            indices = range(first_window, first_window + window_count)
            input_batch = None
            values = None
            logits = None
            log_probs_tensor = None
            try:
                input_batch = build_alignment_window_batch(
                    audio, indices, window_samples, context_samples
                )
                values = torch.from_numpy(input_batch).to(
                    model.device, dtype=model.dtype
                )
                with torch.inference_mode():
                    logits = model(values).logits
                    required_frames = context_frames + window_frames
                    if logits.shape[1] < required_frames:
                        raise RuntimeError(
                            "Aligner returned too few frames for the configured window"
                        )
                    logits = logits[
                        :, context_frames : context_frames + window_frames, :
                    ]
                    # Stable in FP32 on the accelerator; avoids exp overflow and
                    # transfers only normalized, cropped frames to host memory.
                    log_probs_tensor = torch.log_softmax(logits.float(), dim=-1)
                batch_log_probs = log_probs_tensor.cpu().numpy()
            except Exception as exc:
                if not is_out_of_memory_error(exc):
                    raise
                input_batch = None
                values = None
                logits = None
                log_probs_tensor = None
                gc.collect()
                empty_device_cache(model.device.type)
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                model._transcribe_align_batch_size = current_batch_size
                info(
                    f"[oom] emissions continuing {label} with batch "
                    f"{current_batch_size} (completed {first_window}/{total_windows} windows)"
                )
                continue
            finally:
                values = None
                logits = None
                log_probs_tensor = None

            input_batch = None
            batch_log_probs = batch_log_probs.reshape(-1, batch_log_probs.shape[-1])
            expected_batch_frames = window_count * window_frames
            if batch_log_probs.shape[0] != expected_batch_frames:
                raise RuntimeError(
                    "Aligner returned an unexpected number of frames for its windows"
                )
            if emissions is None:
                frame_count = total_windows * window_frames - extension_frames
                if frame_count <= 0:
                    raise RuntimeError("Aligner produced no usable CTC frames")
                emissions = np.zeros(
                    (frame_count, batch_log_probs.shape[-1] + 1),
                    dtype=np.float32,
                )
            if first_window + window_count == total_windows and extension_frames:
                batch_log_probs = batch_log_probs[:-extension_frames]

            next_offset = write_offset + len(batch_log_probs)
            if next_offset > len(emissions):
                raise RuntimeError(
                    "Aligner produced more CTC frames than the first window predicted"
                )
            emissions[write_offset:next_offset, :-1] = batch_log_probs
            write_offset = next_offset
            first_window += window_count
            del batch_log_probs
            bar.update(window_count)
    finally:
        bar.close()

    if emissions is None or write_offset != len(emissions):
        raise RuntimeError(
            f"Aligner emission assembly mismatch: wrote {write_offset} frames, "
            f"expected {0 if emissions is None else len(emissions)}"
        )
    stride = ratio * 1000 / SR
    return emissions, stride


def compute_emissions_streaming(
    audio: np.ndarray,
    model,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, float]:
    if len(audio) == 0:
        raise ValueError("Cannot compute CTC emissions for empty audio")
    return _compute_emissions_streaming(audio, model, max(1, batch_size), label)


@dataclass(slots=True)
class AlignmentVocabulary:
    dictionary: dict[str, int]
    index_to_token: dict[int, str]
    blank_id: int


def build_alignment_vocabulary(tokenizer, emission_classes: int) -> AlignmentVocabulary:
    if emission_classes < 2:
        raise ValueError("Aligner must expose at least one token and one <star> column")
    dictionary = {
        key.lower(): int(value) for key, value in tokenizer.get_vocab().items()
    }
    star_id = emission_classes - 1
    if star_id in dictionary.values():
        raise ValueError(
            "The reserved <star> emission column collides with the tokenizer vocabulary"
        )
    dictionary["<star>"] = star_id
    raw_blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)
    if raw_blank_id is None:
        raise ValueError("Aligner tokenizer does not define a blank or pad token")
    blank_id = int(raw_blank_id)
    if not 0 <= blank_id < emission_classes:
        raise ValueError("blank must be within the emissions vocabulary")
    index_to_token = {value: key for key, value in dictionary.items()}
    if blank_id not in index_to_token:
        index_to_token[blank_id] = "<blank>"
    return AlignmentVocabulary(dictionary, index_to_token, blank_id)


def get_alignments_safe(
    emissions: np.ndarray,
    tokens: Sequence[str],
    vocabulary: AlignmentVocabulary,
):
    """Run the maintained TorchAudio CTC op, bypassing the package ctypes extension."""
    from torchaudio.functional import forced_align

    from .alignment_utils import merge_repeats

    token_indices = [
        vocabulary.dictionary[token]
        for token in " ".join(tokens).split(" ")
        if token in vocabulary.dictionary
    ]
    if not token_indices:
        raise ValueError("Transcript produced no aligner vocabulary tokens")
    if vocabulary.blank_id in token_indices:
        raise ValueError(
            f"targets array should not contain blank index ({vocabulary.blank_id})"
        )
    if max(token_indices) >= emissions.shape[-1] or min(token_indices) < 0:
        raise ValueError("targets values must be within the emissions vocabulary")

    targets = torch.tensor([token_indices], dtype=torch.int64)
    log_probs = torch.from_numpy(
        np.ascontiguousarray(emissions[None], dtype=np.float32)
    )
    path, _scores = forced_align(log_probs, targets, blank=vocabulary.blank_id)
    path_values = path.squeeze(0).tolist()
    return (
        merge_repeats(path_values, vocabulary.index_to_token),
        vocabulary.index_to_token[vocabulary.blank_id],
    )


def uniform_word_timings(
    text: str,
    start: float,
    end: float,
    segment_index: int,
    timing_source: str,
) -> list[WordTiming]:
    """Keep every ASR token when precise timing is unavailable."""
    tokens = text.strip().split()
    if not tokens:
        return []
    token_duration = max(0.0, end - start) / len(tokens)
    return [
        {
            "start": start + token_index * token_duration,
            "end": start + (token_index + 1) * token_duration,
            "text": token,
            "segment_index": segment_index,
            "segment_word_index": token_index,
            "timing_source": timing_source,
        }
        for token_index, token in enumerate(tokens)
    ]


def proportional_token_counts(
    token_count: int, spans: Sequence[tuple[float, float]]
) -> list[int]:
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    durations = [max(0.0, end - start) for start, end in spans]
    total = sum(durations)
    if token_count == 0 or total <= 0:
        return [0] * len(spans)
    exact = [token_count * duration / total for duration in durations]
    counts = [int(math.floor(value)) for value in exact]
    remaining = token_count - sum(counts)
    order = sorted(
        range(len(spans)),
        key=lambda index: (exact[index] - counts[index], durations[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def uniform_word_timings_across_spans(
    text: str,
    spans: Sequence[tuple[float, float]],
    segment_index: int,
    timing_source: str,
) -> list[WordTiming]:
    """Distribute words over speech spans without stretching them across silence."""
    tokens = text.strip().split()
    valid_spans = [(start, end) for start, end in spans if end > start]
    if not tokens or not valid_spans:
        return []
    counts = proportional_token_counts(len(tokens), valid_spans)
    words: list[WordTiming] = []
    token_offset = 0
    for (start, end), count in zip(valid_spans, counts, strict=True):
        if count <= 0:
            continue
        token_duration = (end - start) / count
        for local_index in range(count):
            token_index = token_offset + local_index
            words.append(
                {
                    "start": start + local_index * token_duration,
                    "end": start + (local_index + 1) * token_duration,
                    "text": tokens[token_index],
                    "segment_index": segment_index,
                    "segment_word_index": token_index,
                    "timing_source": timing_source,
                }
            )
        token_offset += count
    if token_offset != len(tokens):
        raise RuntimeError("Speech-span token allocation did not preserve every token")
    return words


def speech_spans_within_segment(
    speech_spans: Sequence[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    return [
        (max(start, speech_start), min(end, speech_end))
        for speech_start, speech_end in speech_spans
        if speech_end > start
        and speech_start < end
        and min(end, speech_end) > max(start, speech_start)
    ]


def align_words(
    emissions: np.ndarray,
    stride: float,
    tokenizer,
    segment_times: Sequence[tuple[float, float]],
    segment_texts: Sequence[str],
    language: str,
) -> tuple[list[WordTiming], int]:
    # Construct Uroman only for word alignment. Segment and text-only runs must
    # not pay its startup cost.
    from .alignment_utils import get_spans
    from .text_utils import postprocess_results, preprocess_text

    iso_language = ISO3.get(language, language)
    frame_count = emissions.shape[0]
    words: list[WordTiming] = []
    fallback_count = 0
    pairs = list(zip(segment_times, segment_texts, strict=True))
    vocabulary = build_alignment_vocabulary(tokenizer, emissions.shape[-1])
    bar = progress_bar(
        pairs, unit="seg", desc="aligning", dynamic_ncols=True, bar_format=BAR_FMT
    )
    for segment_index, ((start, end), text) in enumerate(bar):
        text = text.strip()
        if not text:
            continue
        first_frame = max(0, int(round(start * 1000 / stride)))
        last_frame = min(frame_count, int(round(end * 1000 / stride)))
        if last_frame - first_frame < 2:
            fallback_count += 1
            progress_write(
                f"{INDENT}[warn] segment {segment_index} is shorter than two CTC frames; "
                "using uniform word timing"
            )
            words.extend(
                uniform_word_timings(
                    text,
                    start,
                    end,
                    segment_index,
                    "uniform_fallback",
                )
            )
            continue
        try:
            tokens_starred, text_starred = preprocess_text(text, iso_language)
            segments, blank = get_alignments_safe(
                emissions[first_frame:last_frame], tokens_starred, vocabulary
            )
            spans = get_spans(tokens_starred, segments, blank)
            results = postprocess_results(text_starred, spans, stride)
            expected_tokens = text.split()
            aligned_tokens = [word["text"] for word in results]
            if aligned_tokens != expected_tokens:
                raise ValueError(
                    "forced alignment did not preserve the complete ASR transcript "
                    f"({len(aligned_tokens)}/{len(expected_tokens)} words)"
                )
        except Exception as exc:
            fallback_count += 1
            progress_write(
                f"{INDENT}[warn] align failed on segment {segment_index}: {repr(exc)[:100]}"
            )
            words.extend(
                uniform_word_timings(
                    text,
                    start,
                    end,
                    segment_index,
                    "uniform_fallback",
                )
            )
            continue
        for word_index, word in enumerate(results):
            absolute_start = min(end, max(start, start + float(word["start"])))
            absolute_end = min(
                end,
                max(absolute_start, start + float(word["end"])),
            )
            words.append(
                {
                    "start": absolute_start,
                    "end": absolute_end,
                    "text": word["text"],
                    "segment_index": segment_index,
                    "segment_word_index": word_index,
                    "timing_source": "ctc",
                }
            )
    return words, fallback_count
