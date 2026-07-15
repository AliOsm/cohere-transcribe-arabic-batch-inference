"""Bounded preparation and ASR orchestration for offline audio batches."""

from __future__ import annotations

import concurrent.futures
import gc
import os
import time
from collections.abc import Sequence

import torch

from ..asr.model import load_asr
from ..asr.orchestration import transcribe_group
from ..audio.preparation import prepare_source_group
from ..audio.segmentation import validate_processor_single_row_window
from ..cancellation import cancellable_executor, raise_if_cancelled
from ..device import empty_device_cache
from ..models import (
    PIPELINE_GROUP_MAX_BYTES,
    PIPELINE_GROUP_MAX_JOBS,
    SR,
    AudioJob,
    PreparedAudio,
    PreparedGroup,
    PreparedJobResult,
    RunStats,
    TranscriptionConfig,
    fmt_dur,
    info,
)
from ..output.pipeline import write_segment_timed_outputs, write_text_only_outputs
from ..runtime.resources import ModelResources
from ..state import write_asr_checkpoint
from .resources import (
    estimated_decoded_bytes,
    partition_audio_jobs,
    release_job_audio,
)


def attach_prepared(job: AudioJob, prepared: PreparedAudio, stats: RunStats) -> None:
    """Attach decoded audio and segmentation metadata to its source job."""
    job.audio = prepared.audio
    job.duration = len(prepared.audio) / SR
    job.segment_times = prepared.segment_times
    job.speech_spans = prepared.speech_spans
    job.segment_texts = [""] * len(prepared.segment_times)
    job.decode_backend = prepared.decode_backend
    job.decode_fallback_reason = prepared.decode_fallback_reason
    job.vad_engine_actual = prepared.vad_engine.removesuffix("+merge")
    job.vad_provider = prepared.vad_provider
    job.vad_provider_options = prepared.vad_provider_options
    job.vad_fallback_reason = prepared.vad_fallback_reason
    stats.decode_seconds += prepared.decode_seconds
    stats.vad_seconds += prepared.vad_seconds
    speech_seconds = sum(end - start for start, end in job.segment_times)
    speech_percent = 0.0 if job.duration == 0 else speech_seconds / job.duration * 100
    coverage_label = (
        "selected audio"
        if prepared.vad_engine.endswith("+merge")
        or prepared.vad_engine.startswith("none")
        else "speech"
    )
    info(
        f"prepared {job.path.name}: {fmt_dur(job.duration)}, {len(job.segment_times)} segments, "
        f"{fmt_dur(speech_seconds)} {coverage_label} ({speech_percent:.0f}%), "
        f"decode={prepared.decode_backend}, VAD={prepared.vad_engine}"
        + (" (TorchCodec fallback)" if prepared.decode_fallback_reason else "")
        + (f" [{prepared.vad_provider}]" if prepared.vad_provider else "")
    )


def finalize_completed_asr_jobs(
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    stats: RunStats,
    *,
    publish_outputs: bool = True,
) -> None:
    """Checkpoint ASR results and progressively publish non-word outputs."""
    checkpoint_started = time.perf_counter()
    for job in jobs:
        if (
            not publish_outputs
            or not job.output_paths
            or job.checkpoint_path is None
            or job.error is not None
            or job.asr_checkpoint_loaded
        ):
            continue
        try:
            write_asr_checkpoint(job)
            stats.asr_checkpoint_written_files += 1
        except OSError as exc:
            info(
                f"[warn] {job.path}: ASR checkpoint could not be stored durably "
                f"({type(exc).__name__}: {exc}); continuing with output publication"
            )
        except Exception as exc:
            job.error = f"ASR checkpoint failed: {type(exc).__name__}: {exc}"
            info(f"[error] {job.path}: {job.error}")
    stats.checkpoint_seconds += time.perf_counter() - checkpoint_started

    output_started = time.perf_counter()
    if args.alignment == "segment":
        write_segment_timed_outputs(jobs, args, publish_outputs=publish_outputs)
    elif args.alignment == "none":
        write_text_only_outputs(jobs, publish_outputs=publish_outputs)
    stats.progressive_output_seconds += time.perf_counter() - output_started


def transcribe_all(
    jobs: list[AudioJob],
    args: TranscriptionConfig,
    device: str,
    dtype: torch.dtype,
    stats: RunStats,
    *,
    publish_outputs: bool = True,
    resources: ModelResources | None = None,
) -> None:
    """Prepare and transcribe jobs using bounded files and targeted audio groups."""
    jobs = [job for job in jobs if not job.skipped]
    resumed_jobs = [job for job in jobs if job.asr_checkpoint_loaded]
    stats.asr_checkpoint_resumed_files += len(resumed_jobs)
    if resumed_jobs and args.alignment != "word":
        finalize_completed_asr_jobs(
            resumed_jobs, args, stats, publish_outputs=publish_outputs
        )
    jobs = [job for job in jobs if not job.asr_checkpoint_loaded]
    if not jobs:
        return

    memory_budget = int(args.audio_memory_gb * 1024**3)
    expected_bytes = [estimated_decoded_bytes(job, memory_budget) for job in jobs]
    retain_audio = args.alignment == "word" and sum(expected_bytes) <= memory_budget
    pipeline_enabled = args.pipeline_preparation and len(jobs) > 1
    group_budget = memory_budget
    group_max_jobs = None
    if pipeline_enabled:
        group_budget = max(1, min(memory_budget // 2, PIPELINE_GROUP_MAX_BYTES))
        group_max_jobs = PIPELINE_GROUP_MAX_JOBS
    source_groups = partition_audio_jobs(jobs, group_budget, group_max_jobs)
    planned_group_bytes = [
        sum(estimated_decoded_bytes(job, memory_budget) for job in group)
        for group in source_groups
    ]
    workers = args.preprocess_workers
    if workers is None:
        workers = (
            1
            if len(jobs) == 1
            else min(2, len(jobs), max(1, (os.cpu_count() or 2) // 2))
        )
    workers = min(workers, len(jobs), max(1, os.cpu_count() or 1))

    cache_strategy = (
        "retain through alignment"
        if retain_audio
        else "bounded groups + re-decode"
        if args.alignment == "word"
        else "release after ASR"
    )
    info(
        f"audio cache: {cache_strategy} "
        f"(group target {group_budget / 1024**3:.2f} GiB); preprocess workers: {workers}; "
        f"next-group preparation: {'on' if pipeline_enabled else 'off'}"
    )

    owned_resources = resources is None
    model_resources = resources or ModelResources()
    processor = None
    model = None
    retained_processed_jobs: list[AudioJob] = []

    def collect_prepared(group: PreparedGroup) -> list[AudioJob]:
        metrics = group.vad_metrics
        stats.vad_model_load_seconds += metrics.model_load_seconds
        stats.vad_inference_seconds += metrics.inference_seconds
        stats.vad_postprocess_seconds += metrics.postprocess_seconds
        stats.vad_prepared_groups += metrics.prepared_groups
        stats.vad_model_calls += metrics.model_calls
        stats.vad_valid_frames += metrics.valid_frames
        stats.vad_padded_frames += metrics.padded_frames
        stats.vad_max_files_per_call = max(
            stats.vad_max_files_per_call, metrics.max_files_per_call
        )
        if metrics.effective_block_frames:
            stats.vad_effective_block_frames = (
                metrics.effective_block_frames
                if stats.vad_effective_block_frames == 0
                else min(
                    stats.vad_effective_block_frames,
                    metrics.effective_block_frames,
                )
            )
        prepared_group: list[AudioJob] = []
        for result in group.results:
            job = result.job
            if result.error is not None:
                job.error = f"audio preparation failed: {result.error}"
                info(f"[error] {job.path}: {job.error}")
                continue
            if result.prepared is None:
                job.error = "audio preparation failed: no result returned"
                info(f"[error] {job.path}: {job.error}")
                continue
            attach_prepared(job, result.prepared, stats)
            prepared_group.append(job)
        return prepared_group

    def enforce_actual_memory_budget() -> None:
        nonlocal retain_audio
        resident_bytes = sum(job.audio_bytes for job in jobs)
        if retain_audio and resident_bytes > memory_budget:
            retain_audio = False
            release_job_audio(retained_processed_jobs)
            retained_processed_jobs.clear()
            info("decoded audio exceeded the estimate; switching to bounded groups")

    def process_prepared_group(prepared_group: Sequence[AudioJob]) -> None:
        if not prepared_group:
            return
        assert processor is not None and model is not None
        actual_groups = partition_audio_jobs(prepared_group, memory_budget)
        if len(actual_groups) > 1 or any(
            job.audio_bytes > memory_budget for job in prepared_group
        ):
            info(
                "decoded audio exceeded its planned group; processing smaller actual-size groups"
            )
        for actual_group in actual_groups:
            try:
                stats.asr_seconds += transcribe_group(
                    processor, model, actual_group, args, stats
                )
            finally:
                if not retain_audio:
                    release_job_audio(actual_group)
            finalize_completed_asr_jobs(
                actual_group,
                args,
                stats,
                publish_outputs=publish_outputs,
            )
        if retain_audio:
            discarded: list[AudioJob] = []
            for job in prepared_group:
                if job.error is None and job.has_text:
                    retained_processed_jobs.append(job)
                else:
                    discarded.append(job)
            release_job_audio(discarded)

    def submit_group(executor, group: Sequence[AudioJob]):
        return executor.submit(prepare_source_group, group, args, workers)

    def resolve_group(
        source_group: Sequence[AudioJob], future: concurrent.futures.Future
    ) -> list[AudioJob]:
        started = time.perf_counter()
        try:
            prepared = future.result()
        except Exception as exc:
            prepared = PreparedGroup(
                results=[PreparedJobResult(job=job, error=exc) for job in source_group]
            )
        stats.preparation_wait_seconds += time.perf_counter() - started
        return collect_prepared(prepared)

    completed = False
    try:
        with cancellable_executor(
            max_workers=1, thread_name_prefix="audio-group"
        ) as executor:
            pending = submit_group(executor, source_groups[0])
            started = time.perf_counter()
            model_revision = args.model_revision
            processor, model, _loaded_now = model_resources.acquire_asr(
                device,
                dtype,
                model_id=args.model,
                model_revision=model_revision,
                model_format=args.model_format or "dense",
                adapter_id=args.adapter,
                adapter_revision=args.adapter_revision,
                loader=lambda load_device, load_dtype: load_asr(
                    load_device,
                    load_dtype,
                    args.model,
                    model_revision,
                    args.model_format or "dense",
                    args.adapter,
                    args.adapter_revision,
                ),
            )
            max_clip = validate_processor_single_row_window(processor, args.max_dur)
            info(f"processor single-row audio limit: {max_clip:g}s")
            stats.asr_load_seconds = time.perf_counter() - started
            info(f"ASR loaded in {fmt_dur(stats.asr_load_seconds)}")
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            for group_index, source_group in enumerate(source_groups):
                raise_if_cancelled()
                prepared_group = resolve_group(source_group, pending)
                enforce_actual_memory_budget()

                next_pending = None
                if pipeline_enabled and group_index + 1 < len(source_groups):
                    resident_bytes = (
                        sum(job.audio_bytes for job in jobs)
                        if retain_audio
                        else sum(job.audio_bytes for job in prepared_group)
                    )
                    can_overlap = (
                        resident_bytes + planned_group_bytes[group_index + 1]
                        <= memory_budget
                    )
                    if can_overlap:
                        next_pending = submit_group(
                            executor, source_groups[group_index + 1]
                        )

                process_prepared_group(prepared_group)

                if group_index + 1 < len(source_groups):
                    pending = next_pending or submit_group(
                        executor, source_groups[group_index + 1]
                    )
        completed = True
    finally:
        if device == "cuda" and torch.cuda.is_available():
            stats.peak_cuda_gib = max(
                stats.peak_cuda_gib,
                torch.cuda.max_memory_allocated() / 1024**3,
            )
            stats.peak_cuda_reserved_gib = max(
                stats.peak_cuda_reserved_gib,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
        circuit_broken = model_resources.asr_circuit_broken
        model = None
        processor = None
        evict_model = owned_resources or not completed or circuit_broken
        if evict_model:
            model_resources.evict_asr()
        if not completed:
            release_job_audio(jobs)
        if not evict_model:
            gc.collect()
            empty_device_cache(device)
