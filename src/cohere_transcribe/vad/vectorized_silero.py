"""Vectorized Silero VAD v6 inference with Silero-compatible timestamps.

The ONNX sequence export evaluates many 512-sample frames per runtime call. The
timestamp state machine intentionally mirrors silero-vad 6.2.1 so replacing the
frame-by-frame ONNX runner does not alter segment boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np

from ..cancellation import raise_if_cancelled

WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
MAX_SEQUENCE_FRAMES = 256
CANCELLATION_CHECK_FRAMES = 4_096
MODEL_PATH = Path(__file__).with_name("silero_vad_v6.onnx")


class SpeechTimestamp(TypedDict, total=False):
    start: int
    end: int


class SpeechProbabilityModel(Protocol):
    def speech_probabilities(self, audio: np.ndarray) -> np.ndarray: ...


class VectorizedSileroVAD:
    """Thread-confined ONNX session for the Silero v6 sequence export."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )

    def speech_probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Return one speech probability for every zero-padded 512-sample frame."""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Silero VAD expects mono audio, got shape {audio.shape}")
        if not audio.size:
            return np.empty(0, dtype=np.float32)

        hidden = np.zeros((1, 1, 128), dtype=np.float32)
        cell = np.zeros((1, 1, 128), dtype=np.float32)
        previous_context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        frame_count = (audio.size + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
        outputs: list[np.ndarray] = []
        for frame_start in range(0, frame_count, MAX_SEQUENCE_FRAMES):
            chunk_frames = min(MAX_SEQUENCE_FRAMES, frame_count - frame_start)
            sample_start = frame_start * WINDOW_SAMPLES
            sample_end = min(audio.size, (frame_start + chunk_frames) * WINDOW_SAMPLES)
            chunk_audio = audio[sample_start:sample_end]
            if chunk_audio.size == chunk_frames * WINDOW_SAMPLES:
                frames = chunk_audio.reshape(chunk_frames, WINDOW_SAMPLES)
            else:
                frames = np.zeros((chunk_frames, WINDOW_SAMPLES), dtype=np.float32)
                frames.reshape(-1)[: chunk_audio.size] = chunk_audio

            context = np.empty((chunk_frames, CONTEXT_SAMPLES), dtype=np.float32)
            context[0] = previous_context
            if chunk_frames > 1:
                context[1:] = frames[:-1, -CONTEXT_SAMPLES:]
            previous_context = frames[-1, -CONTEXT_SAMPLES:].copy()
            inputs = np.concatenate((context, frames), axis=1)

            probabilities, hidden, cell = self.session.run(
                None,
                {
                    "input": inputs,
                    "h": hidden,
                    "c": cell,
                },
            )
            outputs.append(probabilities)
        return np.concatenate(outputs).astype(np.float32, copy=False)


def get_speech_timestamps(
    audio: np.ndarray,
    model: SpeechProbabilityModel,
    *,
    sampling_rate: int = 16_000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    neg_threshold: float | None = None,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> list[SpeechTimestamp]:
    """Return sample-index timestamps using Silero 6.2.1 segmentation semantics."""
    if sampling_rate != 16_000:
        raise ValueError("The vectorized Silero v6 export requires 16 kHz audio")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Silero VAD expects mono audio, got shape {audio.shape}")

    return get_speech_timestamps_from_probabilities(
        len(audio),
        model.speech_probabilities(audio),
        sampling_rate=sampling_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        max_speech_duration_s=max_speech_duration_s,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        neg_threshold=neg_threshold,
        min_silence_at_max_speech=min_silence_at_max_speech,
        use_max_poss_sil_at_max_speech=use_max_poss_sil_at_max_speech,
    )


def get_speech_timestamps_from_probabilities(
    audio_length_samples: int,
    speech_probabilities: np.ndarray,
    *,
    sampling_rate: int = 16_000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    neg_threshold: float | None = None,
    min_silence_at_max_speech: int = 98,
    use_max_poss_sil_at_max_speech: bool = True,
) -> list[SpeechTimestamp]:
    """Apply Silero 6.2.1 segmentation to precomputed frame probabilities."""
    raise_if_cancelled()
    if sampling_rate != 16_000:
        raise ValueError("The vectorized Silero v6 export requires 16 kHz audio")
    if (
        isinstance(audio_length_samples, bool)
        or not isinstance(audio_length_samples, (int, np.integer))
        or audio_length_samples < 0
    ):
        raise ValueError("audio_length_samples must be a non-negative integer")
    audio_length_samples = int(audio_length_samples)
    speech_probs = np.asarray(speech_probabilities, dtype=np.float32)
    if speech_probs.ndim != 1:
        raise ValueError(
            f"Silero probabilities must be one-dimensional, got {speech_probs.shape}"
        )
    expected_frames = (audio_length_samples + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES
    if len(speech_probs) != expected_frames:
        raise ValueError(
            "Silero probability count does not match the audio length: "
            f"expected {expected_frames}, got {len(speech_probs)}"
        )
    if not np.isfinite(speech_probs).all():
        raise ValueError("Silero probabilities contain non-finite values")
    if np.any((speech_probs < 0) | (speech_probs > 1)):
        raise ValueError("Silero probabilities must be between zero and one")
    raise_if_cancelled()

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s - WINDOW_SAMPLES - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * min_silence_at_max_speech / 1000
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    triggered = False
    speeches: list[SpeechTimestamp] = []
    current_speech: SpeechTimestamp = {}
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends: list[tuple[int, int]] = []
    next_cancellation_check = CANCELLATION_CHECK_FRAMES

    for index, speech_prob in enumerate(speech_probs):
        if index == next_cancellation_check:
            raise_if_cancelled()
            next_cancellation_check += CANCELLATION_CHECK_FRAMES
        current_sample = WINDOW_SAMPLES * index

        if speech_prob >= threshold and temp_end:
            silence_duration = current_sample - temp_end
            if silence_duration > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, silence_duration))
            temp_end = 0
            if next_start < prev_end:
                next_start = current_sample

        if speech_prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = current_sample
            continue

        if triggered and current_sample - current_speech["start"] > max_speech_samples:
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, silence_duration = max(
                    possible_ends, key=lambda candidate: candidate[1]
                )
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + silence_duration
                if next_start < prev_end + current_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            elif prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                current_speech["end"] = current_sample
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

        if speech_prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = current_sample
            current_silence_duration = current_sample - temp_end
            if (
                not use_max_poss_sil_at_max_speech
                and current_silence_duration > min_silence_samples_at_max_speech
            ):
                prev_end = temp_end
            if current_silence_duration < min_silence_samples:
                continue

            current_speech["end"] = temp_end
            if current_speech["end"] - current_speech["start"] > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            possible_ends = []

    raise_if_cancelled()
    if (
        current_speech
        and audio_length_samples - current_speech["start"] > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    next_cancellation_check = CANCELLATION_CHECK_FRAMES
    for index, speech in enumerate(speeches):
        if index == next_cancellation_check:
            raise_if_cancelled()
            next_cancellation_check += CANCELLATION_CHECK_FRAMES
        if index == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if index != len(speeches) - 1:
            silence_duration = speeches[index + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[index + 1]["start"] = int(
                    max(
                        0,
                        speeches[index + 1]["start"] - silence_duration // 2,
                    )
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[index + 1]["start"] = int(
                    max(0, speeches[index + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )
    raise_if_cancelled()
    return speeches
