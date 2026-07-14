"""Detailed command-line presentation for the shared runtime."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..api.types import TranscriptionRun
from ..asr.batching import default_asr_batch_size
from ..models import (
    INDENT,
    OUTPUT_PATH_DISPLAY_LIMIT,
    AudioJob,
    RunStats,
    TranscriptionConfig,
    fmt_dur,
)


def print_header(
    jobs: Sequence[AudioJob], args: TranscriptionConfig, device: str, dtype: torch.dtype
) -> None:
    """Print resolved device, batching, segmentation, and output mode."""
    default_initial = default_asr_batch_size(device)
    initial_batch = args.batch_size or (
        min(default_initial, args.batch_max_size)
        if args.batch_max_size is not None
        else default_initial
    )
    if not args.adaptive_batch:
        batch_label = f"{initial_batch} fixed"
    elif args.batch_size is not None and args.batch_max_size is None:
        batch_label = f"{initial_batch} cap + OOM learning"
    else:
        cap_label = (
            str(args.batch_max_size)
            if args.batch_max_size is not None
            else "VRAM-derived"
        )
        batch_label = f"adaptive {initial_batch}->{cap_label}"
    total_hint = sum(job.duration_hint or 0.0 for job in jobs)
    vad_label = {
        "silero": (
            f"Silero {args.vad_engine}{' + merge' if args.vad_merge else ''}"
            + (
                f" (files {args.vad_batch_size}, block {args.vad_block_frames})"
                if args.vad_engine == "torch"
                else ""
            )
        ),
        "auditok": "auditok",
        "none": f"no VAD ({args.max_dur:g}s fixed windows)",
    }[args.vad]
    print(
        f"{INDENT}{len(jobs)} file(s), {fmt_dur(total_hint)} probed audio | "
        f"{device} / {dtype} | ASR batch {batch_label} (length sorted) | "
        f"{vad_label} | "
        f"{'text only' if args.alignment == 'none' else args.alignment + ' timing'}",
        flush=True,
    )


def print_summary(
    run: TranscriptionRun,
    jobs: Sequence[AudioJob],
    stats: RunStats,
    args: TranscriptionConfig,
) -> None:
    """Print execution, memory, batching, quality-guard, and output telemetry."""
    successful = [job for job in jobs if job.error is None]
    prepared_duration = sum(job.duration for job in jobs if job.duration > 0)
    fallback_count = sum(job.fallback_alignments for job in successful)
    repetition_stop_count = sum(
        len(job.repetition_stopped_segments) for job in successful
    )
    token_limit_count = sum(len(job.token_limit_segments) for job in successful)
    memory_label = (
        f", CUDA peak {stats.peak_cuda_gib:.2f} GiB allocated / "
        f"{stats.peak_cuda_reserved_gib:.2f} GiB reserved"
        if stats.cuda_total_gib
        else ""
    )
    attempted = len([job for job in jobs if not job.skipped])
    print(
        f"{INDENT}{len(run.successful)}/{attempted} files finished in "
        f"{fmt_dur(run.statistics.elapsed_seconds)} "
        f"(RTFx {run.statistics.real_time_factor_x:.1f}{memory_label})",
        flush=True,
    )
    print(
        f"{INDENT}ASR load {fmt_dur(stats.asr_load_seconds)} | "
        f"ASR wall {fmt_dur(stats.asr_seconds)} | "
        f"feature worker {stats.asr_feature_seconds:.3f}s | "
        f"feature wait {stats.asr_feature_wait_seconds:.3f}s | "
        f"H2D {stats.asr_h2d_seconds:.3f}s | "
        f"generation call {stats.asr_generation_call_seconds:.3f}s "
        f"(includes transfer; device {stats.asr_generate_device_seconds:.3f}s) | "
        f"token analysis {stats.asr_generation_analysis_seconds:.3f}s | "
        f"decode text {stats.asr_decode_seconds:.3f}s",
        flush=True,
    )
    print(
        f"{INDENT}runtime import {stats.runtime_import_seconds:.3f}s | "
        f"serialization wait {stats.serialization_wait_seconds:.3f}s | "
        f"input/probe {stats.input_validation_seconds:.3f}s | "
        f"exposed preparation wait {stats.preparation_wait_seconds:.3f}s | "
        f"checkpoints {stats.checkpoint_seconds:.3f}s | "
        f"progressive output {stats.progressive_output_seconds:.3f}s | "
        f"post-ASR stage {stats.post_asr_seconds:.3f}s",
        flush=True,
    )
    if stats.vad_prepared_groups:
        vad_padding = (
            0.0
            if stats.vad_padded_frames == 0
            else 1.0 - stats.vad_valid_frames / stats.vad_padded_frames
        )
        print(
            f"{INDENT}packed VAD {stats.vad_inference_seconds:.3f}s inference + "
            f"{stats.vad_postprocess_seconds:.3f}s timestamps | "
            f"{stats.vad_prepared_groups} group(s), "
            f"{stats.vad_model_calls} model call(s), "
            f"max {stats.vad_max_files_per_call} files/call, "
            f"padding {vad_padding:.1%}",
            flush=True,
        )
    print(
        f"{INDENT}aligner load {fmt_dur(stats.align_load_seconds)} | "
        f"emissions {fmt_dur(stats.emissions_seconds)} | "
        f"Viterbi {fmt_dur(stats.viterbi_seconds)}",
        flush=True,
    )
    padding_ratio = (
        0.0
        if stats.asr_padded_feature_frames == 0
        else 1.0 - stats.asr_valid_feature_frames / stats.asr_padded_feature_frames
    )
    if stats.asr_batches:
        print(
            f"{INDENT}{stats.asr_batches} generation batch(es), "
            f"{stats.asr_processor_rows} processor row(s), "
            f"batch range {stats.effective_batch_min}-{stats.effective_batch_max}, "
            f"feature padding {padding_ratio:.1%}, "
            f"{stats.asr_generated_tokens} generated token(s)",
            flush=True,
        )
    if stats.asr_oom_retries or stats.asr_truncation_retries:
        print(
            f"{INDENT}adaptive retries: {stats.asr_oom_retries} OOM, "
            f"{stats.asr_truncation_retries} token-limit segment(s)",
            flush=True,
        )
    if fallback_count:
        print(
            f"{INDENT}warning: {fallback_count} segment(s) used approximate word timing",
            flush=True,
        )
    if repetition_stop_count:
        triggered = [
            f"{job.relative_path}:{segment_index}"
            for job in successful
            for segment_index in sorted(job.repetition_stopped_segments)
        ]
        provenance_hint = _provenance_hint(triggered, args.formats)
        print(
            f"{INDENT}warning: repetition guard stopped "
            f"{repetition_stop_count} segment(s); {provenance_hint}",
            flush=True,
        )
    if token_limit_count:
        limited = [
            f"{job.relative_path}:{segment_index}"
            for job in successful
            for segment_index in sorted(job.token_limit_segments)
        ]
        provenance_hint = _provenance_hint(limited, args.formats)
        print(
            f"{INDENT}warning: {token_limit_count} segment(s) reached the final "
            f"decoder token limit without EOS; {provenance_hint}",
            flush=True,
        )
    written_paths = [path for job in jobs for path in job.written]
    for path in written_paths[:OUTPUT_PATH_DISPLAY_LIMIT]:
        print(f"{INDENT}{path}", flush=True)
    if len(written_paths) > OUTPUT_PATH_DISPLAY_LIMIT:
        print(
            f"{INDENT}... and "
            f"{len(written_paths) - OUTPUT_PATH_DISPLAY_LIMIT} more output files",
            flush=True,
        )
    if run.failed and prepared_duration > run.statistics.successful_audio_seconds:
        attempted_rtfx = (
            prepared_duration / run.statistics.elapsed_seconds
            if run.statistics.elapsed_seconds > 0
            else 0.0
        )
        print(
            f"{INDENT}attempted {fmt_dur(prepared_duration)} of audio "
            f"(RTFx {attempted_rtfx:.1f})",
            flush=True,
        )
    if run.failed:
        print(f"{INDENT}Failures:", flush=True)
        for result in run.failed:
            print(f"{INDENT}- {result.path}: {result.error}", flush=True)


def _provenance_hint(segment_labels: Sequence[str], formats: Sequence[str]) -> str:
    if "json" in formats:
        return "see JSON provenance for segment indices"
    displayed = ", ".join(segment_labels[:10])
    remainder = len(segment_labels) - 10
    return "segments " + displayed + (f" (+{remainder} more)" if remainder > 0 else "")


__all__ = ["print_header", "print_summary"]
