"""Cohere ASR feature preparation, generation, and output analysis."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..models import (
    REPETITION_CHECK_INTERVAL,
    REPETITION_MAX_PERIOD,
    REPETITION_MIN_GENERATED_TOKENS,
    REPETITION_MIN_PERIOD,
    REPETITION_REPEATS,
    SR,
    SegmentRef,
    TranscriptionConfig,
)
from .model import clear_encoder_projection_cache


def reassemble_chunk_texts(
    chunk_texts: Sequence[str],
    audio_chunk_index: Sequence[tuple[int, int | None]],
    expected_samples: int,
) -> list[str]:
    if len(chunk_texts) != len(audio_chunk_index):
        raise RuntimeError(
            "Processor chunk metadata does not match decoded ASR rows: "
            f"{len(audio_chunk_index)} indices for {len(chunk_texts)} texts"
        )
    if expected_samples < 0:
        raise ValueError("expected_samples must be non-negative")

    outputs: dict[int, str] = {}
    for metadata, text in zip(audio_chunk_index, chunk_texts, strict=True):
        if len(metadata) != 2:
            raise RuntimeError(f"Invalid audio_chunk_index row: {metadata!r}")
        raw_sample_index, raw_chunk_index = metadata
        sample_index = int(raw_sample_index)
        if not 0 <= sample_index < expected_samples:
            raise RuntimeError(
                f"Processor returned sample index {sample_index}; expected 0..{expected_samples - 1}"
            )
        chunk_index = None if raw_chunk_index is None else int(raw_chunk_index)
        if chunk_index not in (None, 0):
            raise RuntimeError(
                f"Processor expanded sample {sample_index} to chunk index {chunk_index}; "
                "expected one model row per sample"
            )
        if sample_index in outputs:
            raise RuntimeError(
                f"Processor returned duplicate rows for sample {sample_index}"
            )
        outputs[sample_index] = text.strip()

    missing = [index for index in range(expected_samples) if index not in outputs]
    if missing:
        raise RuntimeError(
            f"Processor returned no ASR row for sample indices: {missing}"
        )
    return [outputs[index] for index in range(expected_samples)]


@dataclass(slots=True)
class PreparedASRBatch:
    refs: list[SegmentRef]
    model_inputs: dict[str, torch.Tensor]
    chunk_index: list[tuple[int, int | None]]
    prepare_seconds: float
    valid_feature_frames: int
    padded_feature_frames: int
    pin_memory_fallbacks: int = 0


@dataclass(slots=True)
class ASRGenerationResult:
    generated: torch.Tensor
    row_token_counts: list[int]
    truncated_ref_indices: set[int]
    repetition_ref_indices: set[int]
    max_new_tokens: int
    prompt_length: int
    call_wall_seconds: float
    device_generate_seconds: float
    analysis_seconds: float
    h2d_seconds: float = 0.0
    baseline_reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


class RepetitionLoopStoppingCriteria:
    """Stop a row after four repeated 8-32-token blocks in a long decode."""

    def __init__(
        self,
        prompt_length: int,
        min_generated_tokens: int = REPETITION_MIN_GENERATED_TOKENS,
        repeats: int = REPETITION_REPEATS,
        min_period: int = REPETITION_MIN_PERIOD,
        max_period: int = REPETITION_MAX_PERIOD,
        eos_token_ids: Sequence[int] = (),
    ) -> None:
        self.prompt_length = prompt_length
        self.min_generated_tokens = min_generated_tokens
        self.repeats = repeats
        self.min_period = min_period
        self.max_period = max_period
        self.eos_token_ids = tuple(eos_token_ids)
        self._triggered_mask: torch.BoolTensor | None = None
        self._pattern_cache: dict[
            tuple[str, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._eos_cache: dict[str, torch.Tensor] = {}

    @property
    def triggered_rows(self) -> set[int]:
        if self._triggered_mask is None:
            return set()
        rows = self._triggered_mask.nonzero(as_tuple=False).flatten().tolist()
        return set(rows)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _scores: torch.FloatTensor | None,
        **_kwargs,
    ) -> torch.BoolTensor:
        generated = input_ids[:, self.prompt_length :]
        generated_tokens = generated.shape[1]
        if self._triggered_mask is None:
            self._triggered_mask = torch.zeros(
                generated.shape[0], dtype=torch.bool, device=generated.device
            )
        if generated_tokens < self.min_generated_tokens:
            return self._triggered_mask
        if (
            generated_tokens != self.min_generated_tokens
            and (generated_tokens - self.min_generated_tokens)
            % REPETITION_CHECK_INTERVAL
        ):
            return self._triggered_mask

        largest_period = min(self.max_period, generated_tokens // self.repeats)
        if largest_period < self.min_period:
            return self._triggered_mask
        span = self.repeats * largest_period
        cache_key = (str(generated.device), largest_period)
        cached = self._pattern_cache.get(cache_key)
        if cached is None:
            periods = torch.arange(
                self.min_period, largest_period + 1, device=generated.device
            )
            positions = torch.arange(span, device=generated.device)
            base_indices = positions.unsqueeze(0) % periods.unsqueeze(1)
            relevant = positions.unsqueeze(0) < self.repeats * periods.unsqueeze(1)
            cached = (base_indices, relevant)
            self._pattern_cache[cache_key] = cached
        base_indices, relevant = cached

        reverse_tail = generated[:, -span:].flip(dims=(1,))
        expected = reverse_tail[:, base_indices]
        actual = reverse_tail[:, None, :]
        matches = (actual == expected) | ~relevant.unsqueeze(0)
        newly_done = matches.all(dim=2).any(dim=1)
        if self.eos_token_ids:
            device_key = str(generated.device)
            eos_ids = self._eos_cache.get(device_key)
            if eos_ids is None:
                eos_ids = generated.new_tensor(self.eos_token_ids)
                self._eos_cache[device_key] = eos_ids
            eos_seen = (generated.unsqueeze(-1) == eos_ids).any(dim=(1, 2))
            newly_done &= ~eos_seen
        self._triggered_mask |= newly_done
        return self._triggered_mask


def repetition_stopping_criteria(
    decoder_input_ids: torch.Tensor,
    enabled: bool,
    eos_token_ids: Sequence[int] = (),
) -> list[RepetitionLoopStoppingCriteria] | None:
    if not enabled:
        return None
    return [
        RepetitionLoopStoppingCriteria(
            prompt_length=decoder_input_ids.shape[1],
            eos_token_ids=eos_token_ids,
        )
    ]


def maybe_pin_tensor(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def prepare_asr_batch(
    processor,
    refs: Sequence[SegmentRef],
    args: TranscriptionConfig,
) -> PreparedASRBatch:
    started = time.perf_counter()
    waveforms: list[np.ndarray] = []
    for ref in refs:
        if ref.job.audio is None:
            raise RuntimeError(f"Decoded audio was released before ASR: {ref.job.path}")
        first = int(round(ref.start * SR))
        last = int(round(ref.end * SR))
        waveform = ref.job.audio[first:last]
        if waveform.size == 0:
            raise ValueError(
                f"Segment {ref.segment_index} of {ref.job.path} has no audio samples"
            )
        waveforms.append(waveform)

    with torch.inference_mode():
        inputs = processor(
            audio=waveforms,
            sampling_rate=SR,
            return_tensors="pt",
            return_attention_mask=True,
            language=args.language,
        )
    chunk_index = [
        (int(sample_index), None if chunk_index is None else int(chunk_index))
        for sample_index, chunk_index in inputs["audio_chunk_index"]
    ]
    sample_indices = sorted(sample_index for sample_index, _ in chunk_index)
    if len(chunk_index) != len(refs) or sample_indices != list(range(len(refs))):
        raise RuntimeError(
            "The Cohere processor expanded one or more script-controlled segments "
            "into multiple model rows. Lower --max-dur or use a compatible "
            "Transformers release so adaptive batching and OOM recovery remain exact."
        )
    model_inputs = {
        "input_features": inputs["input_features"],
        "decoder_input_ids": inputs["decoder_input_ids"],
    }
    if inputs.get("attention_mask") is not None:
        model_inputs["attention_mask"] = inputs["attention_mask"]

    pin = args.pin_memory and args.device == "cuda"
    pin_memory_fallbacks = 0
    if pin:
        pinned_inputs: dict[str, torch.Tensor] = {}
        for name, tensor in model_inputs.items():
            pinned = maybe_pin_tensor(tensor, True)
            if not pinned.is_pinned():
                pin_memory_fallbacks += 1
            pinned_inputs[name] = pinned
        model_inputs = pinned_inputs

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        rows, frames = model_inputs["input_features"].shape[:2]
        valid_frames = padded_frames = int(rows * frames)
    else:
        valid_frames = int(attention_mask.sum().item())
        padded_frames = int(attention_mask.numel())
    return PreparedASRBatch(
        refs=list(refs),
        model_inputs=model_inputs,
        chunk_index=chunk_index,
        prepare_seconds=time.perf_counter() - started,
        valid_feature_frames=valid_frames,
        padded_feature_frames=padded_frames,
        pin_memory_fallbacks=pin_memory_fallbacks,
    )


def generation_eos_token_ids(model) -> tuple[int, ...]:
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    return tuple(int(token_id) for token_id in eos_token_id)


def analyze_generated_rows(
    generated: torch.Tensor,
    prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    pad_token_id: int | None,
    repetition_rows: set[int],
    chunk_index: Sequence[tuple[int, int | None]],
) -> tuple[list[int], set[int], set[int]]:
    generated_part = generated[:, prompt_length:]
    token_counts: list[int] = []
    truncated_refs: set[int] = set()
    repetition_refs: set[int] = set()
    eos_set = set(eos_token_ids)
    for row_index, row in enumerate(generated_part.tolist()):
        count = len(row)
        saw_eos = False
        for token_index, token_id in enumerate(row):
            if token_id in eos_set:
                count = token_index + 1
                saw_eos = True
                break
            if (
                row_index in repetition_rows
                and pad_token_id is not None
                and token_id == pad_token_id
            ):
                count = token_index
                break
        token_counts.append(count)
        ref_index = int(chunk_index[row_index][0])
        if row_index in repetition_rows:
            repetition_refs.add(ref_index)
        elif not saw_eos and len(row) >= max_new_tokens:
            truncated_refs.add(ref_index)
    return token_counts, truncated_refs, repetition_refs


def generate_asr_batch(
    model,
    prepared: PreparedASRBatch,
    args: TranscriptionConfig,
    max_new_tokens: int,
) -> ASRGenerationResult:
    is_cuda = model.device.type == "cuda"
    non_blocking = bool(is_cuda and args.pin_memory)
    baseline_reserved = 0
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
        baseline_reserved = int(torch.cuda.memory_reserved(model.device))
    call_started = time.perf_counter()

    h2d_start = torch.cuda.Event(enable_timing=True) if is_cuda else None
    h2d_end = torch.cuda.Event(enable_timing=True) if is_cuda else None
    generate_start = torch.cuda.Event(enable_timing=True) if is_cuda else None
    generate_end = torch.cuda.Event(enable_timing=True) if is_cuda else None
    if h2d_start is not None:
        h2d_start.record()

    model_inputs = {
        "input_features": prepared.model_inputs["input_features"].to(
            model.device,
            dtype=model.dtype,
            non_blocking=non_blocking,
        ),
        "decoder_input_ids": prepared.model_inputs["decoder_input_ids"].to(
            model.device, non_blocking=non_blocking
        ),
    }
    if "attention_mask" in prepared.model_inputs:
        model_inputs["attention_mask"] = prepared.model_inputs["attention_mask"].to(
            model.device, non_blocking=non_blocking
        )
    if h2d_end is not None:
        h2d_end.record()

    eos_token_ids = generation_eos_token_ids(model)
    stopping_criteria = repetition_stopping_criteria(
        model_inputs["decoder_input_ids"],
        args.stop_repetition_loops,
        eos_token_ids,
    )
    prompt_length = int(model_inputs["decoder_input_ids"].shape[1])
    clear_encoder_projection_cache(model)
    generate_wall_started = time.perf_counter()
    try:
        if generate_start is not None:
            generate_start.record()
        with torch.inference_mode():
            generated_device = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
        if generate_end is not None:
            generate_end.record()
        # Moving the generated ids to host is the synchronization point, so
        # separate cuda.synchronize() calls would only add launch overhead.
        generated = generated_device.detach().cpu()
        del generated_device
    finally:
        clear_encoder_projection_cache(model)
    call_wall_seconds = time.perf_counter() - call_started
    device_generate_seconds = (
        float(generate_start.elapsed_time(generate_end)) / 1000.0
        if generate_start is not None and generate_end is not None
        else time.perf_counter() - generate_wall_started
    )

    analysis_started = time.perf_counter()
    repetition_rows = (
        stopping_criteria[0].triggered_rows if stopping_criteria else set()
    )
    token_counts, truncated_refs, repetition_refs = analyze_generated_rows(
        generated=generated,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        pad_token_id=model.generation_config.pad_token_id,
        repetition_rows=repetition_rows,
        chunk_index=prepared.chunk_index,
    )
    analysis_seconds = time.perf_counter() - analysis_started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(model.device)) if is_cuda else 0
    )
    peak_reserved = int(torch.cuda.max_memory_reserved(model.device)) if is_cuda else 0
    h2d_seconds = (
        float(h2d_start.elapsed_time(h2d_end)) / 1000.0
        if h2d_start is not None and h2d_end is not None
        else 0.0
    )
    return ASRGenerationResult(
        generated=generated,
        row_token_counts=token_counts,
        truncated_ref_indices=truncated_refs,
        repetition_ref_indices=repetition_refs,
        max_new_tokens=max_new_tokens,
        prompt_length=prompt_length,
        call_wall_seconds=call_wall_seconds,
        device_generate_seconds=device_generate_seconds,
        analysis_seconds=analysis_seconds,
        h2d_seconds=h2d_seconds,
        baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def decode_asr_batch(
    processor,
    generated: torch.Tensor,
    prepared: PreparedASRBatch,
) -> list[str]:
    chunk_texts = processor.batch_decode(generated, skip_special_tokens=True)
    return reassemble_chunk_texts(
        chunk_texts,
        prepared.chunk_index,
        len(prepared.refs),
    )
