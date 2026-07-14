from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from cohere_transcribe import (
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionRun,
    TranscriptionStatistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def statistics() -> TranscriptionStatistics:
    return TranscriptionStatistics(
        elapsed_seconds=0.0,
        successful_audio_seconds=0.0,
        real_time_factor_x=0.0,
        runtime_import_seconds=0.0,
        serialization_wait_seconds=0.0,
        input_validation_seconds=0.0,
        decode_seconds=0.0,
        vad_seconds=0.0,
        asr_load_seconds=0.0,
        asr_seconds=0.0,
        aligner_load_seconds=0.0,
        emissions_seconds=0.0,
        viterbi_seconds=0.0,
        peak_cuda_allocated_gib=0.0,
        peak_cuda_reserved_gib=0.0,
        asr_batches=0,
        asr_processor_rows=0,
        generated_tokens=0,
        oom_retries=0,
        truncation_retries=0,
    )


def result(
    name: str,
    *,
    status: str = "completed",
    text: str | None = "transcript",
    error: str | None = None,
) -> TranscriptionResult:
    return TranscriptionResult(
        path=Path(name),
        relative_path=Path(name),
        status=cast(Any, status),
        text=text,
        duration=1.0,
        error=error,
    )


def run_for(
    options: TranscriptionOptions,
    *results: TranscriptionResult,
    errors: tuple[str, ...] = (),
) -> TranscriptionRun:
    return TranscriptionRun(
        results=tuple(results),
        requested_options=options,
        resolved_options=options,
        statistics=statistics(),
        errors=errors,
    )


def patch_execute(monkeypatch: pytest.MonkeyPatch, implementation: Any) -> None:
    import cohere_transcribe.runtime.engine as runtime

    monkeypatch.setattr(runtime, "execute", implementation)


def patch_cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    import torch

    import cohere_transcribe.runtime.engine as runtime

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(runtime, "preflight_runtime", lambda _args: None)
    return runtime
