"""Performance profile construction and atomic publication."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import stat
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models import (
    ALIGN_MODEL_ID,
    ALIGN_MODEL_REVISION,
    ALIGN_PACKAGE_REPOSITORY,
    ALIGN_PACKAGE_REVISION,
    ASR_MODEL_REVISION,
    MODEL_ID,
    PROFILE_SCHEMA_VERSION,
    SILERO_VERSION,
    UROMAN_VERSION,
    AudioJob,
    RunStats,
    TranscriptionConfig,
    default_output_mode,
    package_version,
    runtime_implementation,
)
from .output.publication import apply_file_mode, fsync_directories


def _duration_quantiles(durations: Sequence[float]) -> dict[str, float] | None:
    values = sorted(durations)
    if not values:
        return None
    return {
        "min": values[0],
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": values[-1],
    }


def validate_profile_output_path(
    path_text: str | None, jobs: Sequence[AudioJob]
) -> Path | None:
    if path_text is None:
        return None
    supplied_path = Path(path_text).expanduser()
    if supplied_path.is_symlink():
        raise SystemExit(f"Profile path must not be a symlink: {supplied_path}")
    path = supplied_path.resolve(strict=False)
    source_paths = {job.path.resolve() for job in jobs}
    output_paths = {
        output.resolve(strict=False)
        for job in jobs
        for output in job.output_paths.values()
    }
    if path in source_paths:
        raise SystemExit(f"Profile path collides with an input audio file: {path}")
    if path in output_paths:
        raise SystemExit(f"Profile path collides with a transcript output: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"Cannot create profile directory {path.parent}: {exc}"
        ) from exc
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SystemExit(f"Profile path is not a regular file: {path}")
    if not os.access(path.parent, os.W_OK):
        raise SystemExit(f"Profile directory is not writable: {path.parent}")
    return path


def runtime_environment(device: str, dtype: torch.dtype) -> dict[str, Any]:
    package_names = (
        "torch",
        "torchaudio",
        "transformers",
        "numpy",
        "tqdm",
        "onnxruntime",
        "cohere-transcribe-arabic",
        "uroman",
        "torchcodec",
        "librosa",
        "auditok",
    )
    environment: dict[str, Any] = {
        "python": sys.version.splitlines()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "dtype": str(dtype),
        "packages": {
            package: version
            for package in package_names
            if (version := package_version(package)) is not None
        },
        "torch_cuda_build": torch.version.cuda,
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "pytorch_effective_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        or os.environ.get("PYTORCH_ALLOC_CONF"),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        allocator_backend = None
        get_allocator_backend = getattr(torch.cuda, "get_allocator_backend", None)
        if callable(get_allocator_backend):
            with contextlib.suppress(Exception):
                allocator_backend = get_allocator_backend()
        environment["cuda"] = {
            "device_index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_gib": total_bytes / 1024**3,
            "free_memory_at_profile_gib": free_bytes / 1024**3,
            "driver_visible_device_count": torch.cuda.device_count(),
            "cudnn_version": torch.backends.cudnn.version(),
            "allocator_backend": allocator_backend,
        }
    return environment


def build_profile_payload(
    args: TranscriptionConfig,
    requested_configuration: dict[str, Any],
    stats: RunStats,
    jobs: Sequence[AudioJob],
    elapsed: float,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    successful = [job for job in jobs if job.error is None]
    successful_duration = sum(job.duration for job in successful)
    padded_frames = stats.asr_padded_feature_frames
    padding_ratio = (
        0.0
        if padded_frames == 0
        else 1.0 - stats.asr_valid_feature_frames / padded_frames
    )
    all_segment_durations = [
        end - start for job in jobs for start, end in job.segment_times
    ]
    inferred_segment_durations = [
        job.segment_times[index][1] - job.segment_times[index][0]
        for job in jobs
        if not job.asr_checkpoint_loaded
        for index in sorted(job.generated_tokens)
        if 0 <= index < len(job.segment_times)
    ]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "implementation": runtime_implementation(),
        "models": {
            "asr": {"id": MODEL_ID, "revision": ASR_MODEL_REVISION},
            "vad": (
                {
                    "source": "silero-vad",
                    "source_version": SILERO_VERSION,
                    "distribution": "cohere-transcribe-arabic",
                    "version": package_version("cohere-transcribe-arabic"),
                    "torch_weight_asset": "cohere_transcribe/vad/silero_vad.jit",
                    "onnx_weight_asset": "cohere_transcribe/vad/silero_vad_v6.onnx",
                    "packed_torch_implementation": "packed-sequence-v1",
                }
                if args.vad == "silero"
                else None
            ),
            "aligner": (
                {
                    "id": ALIGN_MODEL_ID,
                    "revision": ALIGN_MODEL_REVISION,
                    "kernel": {
                        "distribution": "torchaudio",
                        "operation": "torchaudio.functional.forced_align",
                        "version": package_version("torchaudio"),
                    },
                    "utility_package": {
                        "distribution": "cohere-transcribe-arabic",
                        "location": "cohere_transcribe.alignment",
                        "repository": ALIGN_PACKAGE_REPOSITORY,
                        "revision": ALIGN_PACKAGE_REVISION,
                    },
                    "romanizer": {
                        "distribution": "uroman",
                        "version": UROMAN_VERSION,
                    },
                }
                if args.alignment == "word"
                else None
            ),
        },
        "environment": runtime_environment(device, dtype),
        "configuration": requested_configuration,
        "resolved_configuration": asdict(args),
        "run": {
            "elapsed_seconds": elapsed,
            "successful_files": len(successful),
            "failed_files": len(jobs) - len(successful),
            "successful_audio_seconds": successful_duration,
            "real_time_factor_x": (
                successful_duration / elapsed if elapsed > 0 else 0.0
            ),
        },
        "timings": {
            "runtime_import_seconds": stats.runtime_import_seconds,
            "serialization_wait_seconds": stats.serialization_wait_seconds,
            "input_validation_seconds": stats.input_validation_seconds,
            "decode_worker_seconds": stats.decode_seconds,
            "vad_worker_seconds": stats.vad_seconds,
            "vad_model_load_seconds": stats.vad_model_load_seconds,
            "vad_inference_seconds": stats.vad_inference_seconds,
            "vad_postprocess_seconds": stats.vad_postprocess_seconds,
            "preparation_wait_seconds": stats.preparation_wait_seconds,
            "asr_load_seconds": stats.asr_load_seconds,
            "asr_wall_seconds": stats.asr_seconds,
            "asr_feature_worker_seconds": stats.asr_feature_seconds,
            "asr_discarded_feature_seconds": stats.asr_discarded_feature_seconds,
            "asr_feature_wait_seconds": stats.asr_feature_wait_seconds,
            "asr_h2d_seconds": stats.asr_h2d_seconds,
            "asr_generation_call_wall_seconds": stats.asr_generation_call_seconds,
            "asr_generate_device_seconds": stats.asr_generate_device_seconds,
            "asr_generation_analysis_seconds": stats.asr_generation_analysis_seconds,
            "asr_decode_seconds": stats.asr_decode_seconds,
            "aligner_load_seconds": stats.align_load_seconds,
            "emissions_seconds": stats.emissions_seconds,
            "viterbi_seconds": stats.viterbi_seconds,
            "post_asr_seconds": stats.post_asr_seconds,
            "checkpoint_seconds": stats.checkpoint_seconds,
            "progressive_output_seconds": stats.progressive_output_seconds,
        },
        "vad": {
            "requested_engines": sorted(
                {
                    job.vad_engine_requested
                    for job in jobs
                    if job.vad_engine_requested is not None
                }
            ),
            "actual_engines": sorted(
                {
                    job.vad_engine_actual
                    for job in jobs
                    if job.vad_engine_actual is not None
                }
            ),
            "torch_device": "cpu" if stats.vad_prepared_groups else None,
            "torch_intraop_threads": (
                torch.get_num_threads() if stats.vad_prepared_groups else None
            ),
            "configured_file_batch_size": args.vad_batch_size,
            "configured_block_frames": args.vad_block_frames,
            "effective_block_frames": stats.vad_effective_block_frames or None,
            "prepared_groups": stats.vad_prepared_groups,
            "model_calls": stats.vad_model_calls,
            "valid_frames": stats.vad_valid_frames,
            "padded_frames": stats.vad_padded_frames,
            "padding_ratio": (
                0.0
                if stats.vad_padded_frames == 0
                else 1.0 - stats.vad_valid_frames / stats.vad_padded_frames
            ),
            "max_files_per_call": stats.vad_max_files_per_call,
        },
        "asr": {
            "batches": stats.asr_batches,
            "processor_rows": stats.asr_processor_rows,
            "generated_tokens": stats.asr_generated_tokens,
            "valid_feature_frames": stats.asr_valid_feature_frames,
            "padded_feature_frames": stats.asr_padded_feature_frames,
            "discarded_processor_rows": stats.asr_discarded_processor_rows,
            "discarded_valid_feature_frames": stats.asr_discarded_valid_feature_frames,
            "discarded_padded_feature_frames": stats.asr_discarded_padded_feature_frames,
            "padding_ratio": padding_ratio,
            "effective_batch_min": stats.effective_batch_min,
            "effective_batch_max": stats.effective_batch_max,
            "final_batch_size": stats.final_batch_size,
            "final_batch_cap": stats.final_batch_cap,
            "oom_retries": stats.asr_oom_retries,
            "truncation_retries": stats.asr_truncation_retries,
            "discarded_feature_batches": stats.asr_discarded_feature_batches,
            "pin_memory_fallbacks": stats.pin_memory_fallbacks,
            "all_segment_duration_seconds": _duration_quantiles(all_segment_durations),
            "inferred_segment_duration_seconds": _duration_quantiles(
                inferred_segment_durations
            ),
            "batch_history": stats.batch_history,
            "checkpoint_resumed_files": stats.asr_checkpoint_resumed_files,
            "checkpoint_written_files": stats.asr_checkpoint_written_files,
        },
        "cuda_memory": {
            "total_gib": stats.cuda_total_gib,
            "free_start_gib": stats.cuda_free_start_gib,
            "free_end_gib": stats.cuda_free_end_gib,
            "peak_allocated_gib": stats.peak_cuda_gib,
            "peak_reserved_gib": stats.peak_cuda_reserved_gib,
        },
        "files": [
            {
                "path": os.fspath(job.path),
                "relative_path": os.fspath(job.relative_path),
                "duration_seconds": job.duration,
                "segment_count": len(job.segment_times),
                "raw_speech_span_count": len(job.speech_spans),
                "raw_speech_seconds": sum(
                    end - start for start, end in job.speech_spans
                ),
                "selected_audio_seconds": sum(
                    end - start for start, end in job.segment_times
                ),
                "decode_backend": job.decode_backend,
                "decode_fallback_reason": job.decode_fallback_reason,
                "vad_engine": job.vad_engine_actual,
                "vad_provider": job.vad_provider,
                "vad_provider_options": job.vad_provider_options,
                "vad_fallback_reason": job.vad_fallback_reason,
                "generated_tokens": sum(job.generated_tokens.values()),
                "repetition_stopped_segments": sorted(job.repetition_stopped_segments),
                "truncation_retried_segments": sorted(job.truncation_retried_segments),
                "token_limit_segments": sorted(job.token_limit_segments),
                "fallback_alignment_segments": job.fallback_alignments,
                "outputs": [os.fspath(path) for path in job.written],
                "resumed_from_asr_checkpoint": job.asr_checkpoint_loaded,
                "published": job.published,
                "error": job.error,
            }
            for job in jobs
        ],
    }


def write_profile_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        output_mode = (
            stat.S_IMODE(path.stat().st_mode)
            if path.exists()
            else default_output_mode()
        )
        apply_file_mode(descriptor, temporary_path, output_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directories(iter((path.parent,)))
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)
