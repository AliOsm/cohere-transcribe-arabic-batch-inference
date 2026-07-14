"""Validated, resumable ASR-complete checkpoints."""

from __future__ import annotations

import math
import time
from typing import Any

from ..models import AudioJob
from .io import (
    decode_state,
    ensure_generation_id,
    match_source_and_generation,
    source_payload,
    write_state_atomic,
)


def asr_checkpoint_payload(job: AudioJob) -> dict[str, Any]:
    return {
        "kind": "asr_complete",
        "generation_id": ensure_generation_id(job),
        "asr_contract_key": job.asr_contract_key,
        "source": source_payload(job),
        "updated_unix_seconds": time.time(),
        "checkpoint": {
            "duration": job.duration,
            "segment_times": job.segment_times,
            "speech_spans": job.speech_spans,
            "segment_texts": job.segment_texts,
            "generated_tokens": [
                [index, count] for index, count in sorted(job.generated_tokens.items())
            ],
            "repetition_stopped_segments": sorted(job.repetition_stopped_segments),
            "truncation_retried_segments": sorted(job.truncation_retried_segments),
            "token_limit_segments": sorted(job.token_limit_segments),
            "decode_backend": job.decode_backend,
            "decode_fallback_reason": job.decode_fallback_reason,
            "vad_engine_actual": job.vad_engine_actual,
            "vad_provider": job.vad_provider,
            "vad_provider_options": job.vad_provider_options,
            "vad_fallback_reason": job.vad_fallback_reason,
        },
    }


def _validated_spans(
    value: Any, duration: float, name: str
) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} is not a list")
    spans: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"{name} contains an invalid row")
        start, end = item
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(f"{name} contains a non-numeric row")
        start_float, end_float = float(start), float(end)
        if not (
            math.isfinite(start_float)
            and math.isfinite(end_float)
            and 0 <= start_float <= end_float <= duration + 1e-6
        ):
            raise ValueError(f"{name} contains an out-of-range row")
        if spans and start_float < spans[-1][1]:
            raise ValueError(f"{name} contains overlapping or out-of-order rows")
        spans.append((start_float, min(duration, end_float)))
    return spans


def _restore_payload(job: AudioJob, payload: dict[str, Any]) -> None:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint is not an object")
    duration_raw = checkpoint.get("duration")
    if isinstance(duration_raw, bool) or not isinstance(duration_raw, (int, float)):
        raise ValueError("duration is not numeric")
    duration = float(duration_raw)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("duration is invalid")
    segment_times = _validated_spans(
        checkpoint.get("segment_times"), duration, "segment_times"
    )
    speech_spans = _validated_spans(
        checkpoint.get("speech_spans"), duration, "speech_spans"
    )
    segment_texts = checkpoint.get("segment_texts")
    if not isinstance(segment_texts, list) or not all(
        isinstance(text, str) for text in segment_texts
    ):
        raise ValueError("segment_texts is invalid")
    if len(segment_texts) != len(segment_times):
        raise ValueError("segment text/time counts differ")
    segment_count = len(segment_times)

    def index_set(name: str) -> set[int]:
        raw = checkpoint.get(name, [])
        if not isinstance(raw, list) or not all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < segment_count
            for index in raw
        ):
            raise ValueError(f"{name} is invalid")
        if len(raw) != len(set(raw)):
            raise ValueError(f"{name} contains duplicate indices")
        return set(raw)

    generated_tokens: dict[int, int] = {}
    raw_tokens = checkpoint.get("generated_tokens", [])
    if not isinstance(raw_tokens, list):
        raise ValueError("generated_tokens is invalid")
    for row in raw_tokens:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in row
            )
            or not 0 <= row[0] < segment_count
            or row[1] < 0
        ):
            raise ValueError("generated_tokens contains an invalid row")
        if row[0] in generated_tokens:
            raise ValueError("generated_tokens contains a duplicate segment index")
        generated_tokens[row[0]] = row[1]

    optional_strings = (
        "decode_backend",
        "decode_fallback_reason",
        "vad_engine_actual",
        "vad_provider",
        "vad_fallback_reason",
    )
    if any(
        checkpoint.get(name) is not None and not isinstance(checkpoint.get(name), str)
        for name in optional_strings
    ):
        raise ValueError("checkpoint provenance strings are invalid")
    provider_options = checkpoint.get("vad_provider_options")
    if provider_options is not None and not isinstance(provider_options, dict):
        raise ValueError("vad_provider_options is invalid")

    # Validate and materialize the complete checkpoint before mutating the job.
    # A rejected checkpoint must leave no stale metadata for fresh ASR to inherit.
    restored_segment_texts = list(segment_texts)
    repetition_stopped_segments = index_set("repetition_stopped_segments")
    truncation_retried_segments = index_set("truncation_retried_segments")
    token_limit_segments = index_set("token_limit_segments")
    decode_backend = checkpoint.get("decode_backend")
    decode_fallback_reason = checkpoint.get("decode_fallback_reason")
    vad_engine_actual = checkpoint.get("vad_engine_actual")
    vad_provider = checkpoint.get("vad_provider")
    vad_fallback_reason = checkpoint.get("vad_fallback_reason")

    job.duration = duration
    job.segment_times = segment_times
    job.speech_spans = speech_spans
    job.segment_texts = restored_segment_texts
    job.generated_tokens = generated_tokens
    job.repetition_stopped_segments = repetition_stopped_segments
    job.truncation_retried_segments = truncation_retried_segments
    job.token_limit_segments = token_limit_segments
    job.decode_backend = decode_backend
    job.decode_fallback_reason = decode_fallback_reason
    job.vad_engine_actual = vad_engine_actual
    job.vad_provider = vad_provider
    job.vad_provider_options = provider_options
    job.vad_fallback_reason = vad_fallback_reason


def restore_asr_checkpoint(job: AudioJob) -> tuple[bool, str]:
    if job.checkpoint_path is None:
        return False, "ASR checkpoint path is unavailable"
    payload, reason = decode_state(job.checkpoint_path)
    if payload is None:
        return False, reason
    if payload.get("kind") != "asr_complete":
        return False, f"state is {payload.get('kind')!r}, not an ASR checkpoint"
    if payload.get("asr_contract_key") != job.asr_contract_key:
        return False, "ASR checkpoint contract does not match"
    matched, reason = match_source_and_generation(job, payload)
    if not matched:
        return False, reason
    try:
        _restore_payload(job, payload)
        job.generation_id = str(payload["generation_id"])
        job.asr_checkpoint_loaded = True
        return True, ""
    except (TypeError, ValueError) as exc:
        return False, f"ASR checkpoint is invalid ({exc})"


def write_asr_checkpoint(job: AudioJob) -> None:
    write_state_atomic(job, job.checkpoint_path, asr_checkpoint_payload(job))
