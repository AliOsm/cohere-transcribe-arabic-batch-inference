"""Concurrent decoding, segmentation, and batched VAD preparation."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from ..cancellation import cancellable_executor, raise_if_cancelled
from ..models import (
    SR,
    AudioJob,
    DecodedAudio,
    PreparedAudio,
    PreparedGroup,
    PreparedJobResult,
    TranscriptionConfig,
    VadBatchMetrics,
    info,
)
from ..vad.runtime import (
    SileroBackendUnavailable,
    get_silero_runtime,
    postprocess_silero_probabilities,
    segment_audio_silero,
)
from .decoding import decode_audio_resolved
from .segmentation import (
    segment_audio_auditok,
    segment_audio_fixed,
    validate_segment_times,
)

_preparation_thread_local = threading.local()


def decode_job(job: AudioJob, args: TranscriptionConfig) -> DecodedAudio:
    started = time.perf_counter()
    audio, decode_backend, fallback_reason = decode_audio_resolved(
        job.path,
        args.audio_backend,
        max_decoded_bytes=int(args.audio_memory_gb * 1024**3),
        duration_hint=job.duration_hint,
    )
    return DecodedAudio(
        audio=audio,
        decode_backend=decode_backend,
        decode_seconds=time.perf_counter() - started,
        decode_fallback_reason=fallback_reason,
    )


def prepare_decoded_audio(
    decoded: DecodedAudio, args: TranscriptionConfig
) -> PreparedAudio:
    audio = decoded.audio
    duration = len(audio) / SR

    raise_if_cancelled()
    started = time.perf_counter()
    provider: str | None = None
    provider_options: dict[str, dict[str, str]] | None = None
    fallback_reason: str | None = None
    if args.vad == "silero":
        (
            segment_times,
            speech_spans,
            engine,
            provider,
            provider_options,
            fallback_reason,
        ) = segment_audio_silero(audio, args)
    elif args.vad == "auditok":
        segment_times = segment_audio_auditok(
            audio, args.min_dur, args.max_dur, args.max_silence, args.energy_threshold
        )
        segment_times = validate_segment_times(
            segment_times, duration, max_duration=args.max_dur
        )
        speech_spans = list(segment_times)
        engine = "auditok"
    else:
        segment_times = segment_audio_fixed(audio, args.max_dur)
        segment_times = validate_segment_times(
            segment_times, duration, max_duration=args.max_dur
        )
        speech_spans = list(segment_times)
        engine = "none (fixed windows)"
    raise_if_cancelled()
    vad_seconds = time.perf_counter() - started
    return PreparedAudio(
        audio=audio,
        segment_times=segment_times,
        speech_spans=speech_spans,
        decode_seconds=decoded.decode_seconds,
        vad_seconds=vad_seconds,
        vad_engine=engine,
        decode_backend=decoded.decode_backend,
        decode_fallback_reason=decoded.decode_fallback_reason,
        vad_provider=provider,
        vad_provider_options=provider_options,
        vad_fallback_reason=fallback_reason,
    )


def prepare_audio(job: AudioJob, args: TranscriptionConfig) -> PreparedAudio:
    return prepare_decoded_audio(decode_job(job, args), args)


def prepare_torch_silero_group(
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    decode_workers: int,
) -> PreparedGroup:
    """Decode a group concurrently, then run one bounded packed Torch VAD request."""
    results = {job.index: PreparedJobResult(job=job) for job in jobs}
    decoded: list[tuple[AudioJob, DecodedAudio]] = []
    worker_count = min(max(1, decode_workers), len(jobs))

    with cancellable_executor(
        max_workers=worker_count, thread_name_prefix="audio-decode"
    ) as executor:
        futures = {job.index: executor.submit(decode_job, job, args) for job in jobs}
        # Load the stateless VAD while decoding is in flight. The local loader reads
        # package data directly and does not alter PyTorch's global thread count.
        effective_block = getattr(
            _preparation_thread_local,
            "torch_vad_retry_block",
            args.vad_block_frames,
        )
        runtime_args = (
            args
            if effective_block == args.vad_block_frames
            else replace(args, vad_block_frames=effective_block)
        )
        runtime = get_silero_runtime("torch", runtime_args)
        model_load_seconds = runtime.load_seconds
        runtime.load_seconds = 0.0
        for job in jobs:
            raise_if_cancelled()
            try:
                decoded.append((job, futures[job.index].result()))
            except Exception as exc:
                results[job.index].error = exc

    if not decoded:
        return PreparedGroup(
            results=[results[job.index] for job in jobs],
            vad_metrics=VadBatchMetrics(model_load_seconds=model_load_seconds),
        )

    metrics = VadBatchMetrics(
        model_load_seconds=model_load_seconds,
        prepared_groups=1,
        effective_block_frames=runtime.model.limits.block_frames,
    )
    successful: dict[int, tuple[DecodedAudio, np.ndarray, float]] = {}

    def is_memory_failure(exc: Exception) -> bool:
        message = str(exc).lower()
        return isinstance(exc, MemoryError) or any(
            marker in message
            for marker in (
                "out of memory",
                "cannot allocate memory",
                "defaultcpuallocator",
            )
        )

    def infer(items: Sequence[tuple[AudioJob, DecodedAudio]]) -> None:
        nonlocal runtime
        raise_if_cancelled()
        retry_cap = getattr(_preparation_thread_local, "torch_vad_retry_cap", None)
        if retry_cap is not None and len(items) > retry_cap:
            for offset in range(0, len(items), retry_cap):
                infer(items[offset : offset + retry_cap])
            return

        started = time.perf_counter()
        try:
            batch_probabilities = runtime.model.speech_probabilities_batch(
                [item.audio for _job, item in items]
            )
            raise_if_cancelled()
        except Exception as exc:
            metrics.inference_seconds += time.perf_counter() - started
            if len(items) == 1:
                current_block = runtime.model.limits.block_frames
                if is_memory_failure(exc) and current_block > 1:
                    lower_block = max(1, current_block // 2)
                    _preparation_thread_local.torch_vad_retry_block = lower_block
                    lower_args = replace(args, vad_block_frames=lower_block)
                    runtime = get_silero_runtime("torch", lower_args)
                    metrics.model_load_seconds += runtime.load_seconds
                    runtime.load_seconds = 0.0
                    metrics.effective_block_frames = lower_block
                    info(
                        "[warn] packed Torch VAD ran out of memory for one file; "
                        f"retrying with {lower_block} frames per temporal block"
                    )
                    infer(items)
                    return
                results[items[0][0].index].error = exc
                return
            if is_memory_failure(exc):
                current_cap = retry_cap or args.vad_batch_size
                lower_cap = max(1, min(current_cap // 2, len(items) // 2))
                _preparation_thread_local.torch_vad_retry_cap = lower_cap
                if not getattr(
                    _preparation_thread_local, "torch_vad_retry_reported", False
                ):
                    info(
                        "[warn] packed Torch VAD ran out of memory; retrying with "
                        f"at most {lower_cap} file(s) per pack"
                    )
                    _preparation_thread_local.torch_vad_retry_reported = True
                infer(items)
                return
            midpoint = len(items) // 2
            infer(items[:midpoint])
            infer(items[midpoint:])
            return

        batch_seconds = time.perf_counter() - started
        metrics.inference_seconds += batch_seconds
        execution = runtime.model.last_stats
        metrics.model_calls += execution.model_calls
        metrics.valid_frames += execution.valid_frames
        metrics.padded_frames += execution.padded_frames
        metrics.max_files_per_call = max(
            metrics.max_files_per_call, execution.max_files_per_call
        )
        total_frames = max(1, sum(len(values) for values in batch_probabilities))
        for (job, item), values in zip(items, batch_probabilities, strict=True):
            raise_if_cancelled()
            successful[job.index] = (
                item,
                values,
                batch_seconds * len(values) / total_frames,
            )

    infer(decoded)

    for job, _item in decoded:
        raise_if_cancelled()
        candidate = successful.get(job.index)
        if candidate is None:
            continue
        item, values, inference_share = candidate
        started = time.perf_counter()
        try:
            segments, raw_segments = postprocess_silero_probabilities(
                item.audio, values, args
            )
            raise_if_cancelled()
        except Exception as exc:
            metrics.postprocess_seconds += time.perf_counter() - started
            results[job.index].error = exc
            continue
        item_postprocess_seconds = time.perf_counter() - started
        metrics.postprocess_seconds += item_postprocess_seconds
        results[job.index].prepared = PreparedAudio(
            audio=item.audio,
            segment_times=segments,
            speech_spans=raw_segments,
            decode_seconds=item.decode_seconds,
            vad_seconds=inference_share + item_postprocess_seconds,
            vad_engine="torch" + ("+merge" if args.vad_merge else ""),
            decode_backend=item.decode_backend,
            decode_fallback_reason=item.decode_fallback_reason,
            vad_provider="CPU",
        )

    return PreparedGroup(
        results=[results[job.index] for job in jobs],
        vad_metrics=metrics,
    )


def prepare_source_group(
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    workers: int,
) -> PreparedGroup:
    if args.vad == "silero" and args.vad_engine == "torch":
        try:
            return prepare_torch_silero_group(jobs, args, workers)
        except SileroBackendUnavailable as exc:
            if not all(job.vad_engine_requested == "auto" for job in jobs):
                raise
            info(f"[warn] packed Torch Silero unavailable ({exc}); using sequence ONNX")
            fallback_args = replace(args, vad_engine="auto")
            fallback = prepare_source_group(jobs, fallback_args, workers)
            reason = f"SileroBackendUnavailable: {exc}"
            for result in fallback.results:
                if result.prepared is not None:
                    nested_reason = result.prepared.vad_fallback_reason
                    result.prepared.vad_fallback_reason = (
                        f"{reason}; {nested_reason}" if nested_reason else reason
                    )
            return fallback

    results: list[PreparedJobResult] = []
    worker_count = min(max(1, workers), len(jobs))
    with cancellable_executor(
        max_workers=worker_count, thread_name_prefix="audio-prep"
    ) as executor:
        futures = {job.index: executor.submit(prepare_audio, job, args) for job in jobs}
        for job in jobs:
            raise_if_cancelled()
            try:
                results.append(
                    PreparedJobResult(job=job, prepared=futures[job.index].result())
                )
            except Exception as exc:
                results.append(PreparedJobResult(job=job, error=exc))
    return PreparedGroup(results=results)
