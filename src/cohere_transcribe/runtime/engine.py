"""Shared execution engine for the CLI and the public Python API."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import NoReturn, TypeVar

import torch

from .. import device as device_module
from .. import inputs as inputs_module
from ..api.input import normalize_audio_input
from ..api.types import (
    AudioInput,
    BatchTranscriptionError,
    ProgressCallback,
    ProgressCallbackError,
    ProgressEvent,
    TranscriberBusyError,
    TranscriberClosedError,
    TranscriptionConfigurationError,
    TranscriptionError,
    TranscriptionInputError,
    TranscriptionOptions,
    TranscriptionRun,
    TranscriptionRuntimeError,
)
from ..cancellation import TerminationRequested, reset_cancellation
from ..config import config_from_options, validate_args
from ..models import (
    INDENT,
    RunStats,
    TranscriptionConfig,
    default_output_mode,
    fmt_dur,
    info,
)
from ..output import pipeline as output_pipeline
from ..pipeline import transcription as transcription_pipeline
from ..preflight import preflight_runtime
from ..profiling import (
    build_profile_payload,
    validate_profile_output_path,
    write_profile_json,
)
from ..progress import _ProgressCallbackAbort, reporting
from ..state import release_all_output_locks
from .console import print_header, print_summary
from .resources import ModelResources
from .results import build_run

_PROCESS_RUNTIME_GATE = threading.RLock()
_PROCESS_RUNTIME_STATE = threading.local()
E = TypeVar("E", bound=TranscriptionError)


def _raise_translated(error_type: type[E], exc: SystemExit) -> NoReturn:
    message = str(exc)
    raise error_type(message or "Transcription setup failed") from exc


def _resolve_precision(
    args: TranscriptionConfig,
) -> tuple[str, str, torch.dtype, torch.dtype, str, str]:
    """Resolve device, ASR dtype, alignment dtype, and automatic VAD engine."""
    try:
        device = device_module.pick_device(args.device)
    except SystemExit as exc:
        _raise_translated(TranscriptionRuntimeError, exc)

    requested_dtype = args.dtype
    resolved_dtype = requested_dtype
    if requested_dtype == "auto":
        if device == "cuda":
            resolved_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        elif device == "mps":
            resolved_dtype = "fp16"
        else:
            resolved_dtype = "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        resolved_dtype
    ]
    if device == "cpu":
        resolved_dtype = "fp32"
        dtype = torch.float32
    if (
        device == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise TranscriptionRuntimeError(
            "This CUDA device does not support BF16; use --dtype fp16"
        )
    if device == "mps" and dtype == torch.bfloat16:
        try:
            probe = torch.zeros(1, device="mps", dtype=torch.bfloat16)
            del probe
        except (RuntimeError, TypeError) as exc:
            raise TranscriptionRuntimeError(
                "This MPS device/runtime does not support BF16; use --dtype fp16"
            ) from exc
    if args.alignment == "word" and args.align_dtype == "fp16" and device != "cuda":
        raise TranscriptionRuntimeError(
            "--align-dtype fp16 is supported only with CUDA"
        )
    align_dtype = torch.float16 if args.align_dtype == "fp16" else torch.float32
    requested_vad_engine = args.vad_engine
    return (
        device,
        resolved_dtype,
        dtype,
        align_dtype,
        requested_dtype,
        requested_vad_engine,
    )


def execute(
    args: TranscriptionConfig,
    requested_options: TranscriptionOptions,
    *,
    requested_configuration: dict[str, object],
    resources: ModelResources | None,
    publication_enabled: bool,
    console: bool,
    runtime_import_seconds: float = 0.0,
    serialization_wait_seconds: float = 0.0,
    started: float | None = None,
    preflight: Callable[[TranscriptionConfig], None] | None = None,
) -> TranscriptionRun:
    """Execute one validated configuration through the shared offline pipeline."""
    started = time.perf_counter() if started is None else started
    if console:
        print("\n[1/4] Validating inputs and outputs", flush=True)
    (
        device,
        resolved_dtype,
        dtype,
        align_dtype,
        requested_dtype,
        requested_vad_engine,
    ) = _resolve_precision(args)
    contract_args = replace(args, device=device, dtype=resolved_dtype)
    try:
        if publication_enabled:
            if console:
                jobs = inputs_module.build_jobs(args, contract_args=contract_args)
            else:
                jobs = inputs_module.build_jobs(
                    args,
                    contract_args=contract_args,
                    capture_results=True,
                    retain_skipped=True,
                )
        else:
            jobs = inputs_module.build_jobs(
                args,
                contract_args=contract_args,
                publication_enabled=False,
                capture_results=True,
                retain_skipped=True,
            )
    except SystemExit as exc:
        _raise_translated(TranscriptionInputError, exc)

    args.device = device
    args.dtype = resolved_dtype
    args.vad_engine = (
        "torch"
        if args.vad == "silero" and requested_vad_engine == "auto"
        else requested_vad_engine
    )
    runnable_jobs = [job for job in jobs if not job.skipped]
    stats = RunStats(
        runtime_import_seconds=runtime_import_seconds,
        serialization_wait_seconds=serialization_wait_seconds,
        input_validation_seconds=max(
            0.0,
            time.perf_counter()
            - started
            - runtime_import_seconds
            - serialization_wait_seconds,
        ),
    )
    if not runnable_jobs:
        if console:
            print(f"{INDENT}All inputs were skipped; no model was loaded.", flush=True)
        return build_run(
            jobs,
            requested_options,
            args,
            stats,
            time.perf_counter() - started,
        )

    if device == "cpu" and requested_dtype not in {"auto", "fp32"}:
        info(
            f"CPU inference uses FP32; ignoring requested {requested_dtype.upper()} precision"
        )
    if requested_vad_engine == "auto" and args.vad == "silero":
        info("Silero auto selected packed CPU Torch from the batch benchmark")
    if args.vad_threads is not None:
        torch.set_num_threads(args.vad_threads)

    try:
        profile_path = validate_profile_output_path(args.profile_json, jobs)
    except SystemExit as exc:
        _raise_translated(TranscriptionInputError, exc)
    try:
        (preflight or preflight_runtime)(args)
    except SystemExit as exc:
        _raise_translated(TranscriptionRuntimeError, exc)
    stats.input_validation_seconds = max(
        0.0,
        time.perf_counter()
        - started
        - runtime_import_seconds
        - serialization_wait_seconds,
    )
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        stats.cuda_total_gib = total_bytes / 1024**3
        stats.cuda_free_start_gib = free_bytes / 1024**3

    if console:
        print_header(runnable_jobs, args, device, dtype)
        print("\n[2/4] Loading ASR + preparing audio", flush=True)
    try:
        if resources is None:
            transcription_pipeline.transcribe_all(
                runnable_jobs, args, device, dtype, stats
            )
        else:
            transcription_pipeline.transcribe_all(
                runnable_jobs,
                args,
                device,
                dtype,
                stats,
                publish_outputs=publication_enabled,
                resources=resources,
            )
    except SystemExit as exc:
        _raise_translated(TranscriptionRuntimeError, exc)

    if console:
        segment_count = sum(
            len(job.segment_times) for job in runnable_jobs if job.error is None
        )
        print(
            f"{INDENT}ASR done: {segment_count} segments in "
            f"{fmt_dur(stats.asr_seconds)} (decode {fmt_dur(stats.decode_seconds)}, "
            f"VAD {fmt_dur(stats.vad_seconds)} compute)",
            flush=True,
        )
    post_asr_started = time.perf_counter()
    if args.alignment == "word":
        if console:
            print("\n[3/4] Forced alignment + transactional outputs", flush=True)
        output_pipeline.align_and_write_all(
            runnable_jobs,
            args,
            device,
            align_dtype,
            stats,
            publish_outputs=publication_enabled,
        )
    elif console and args.alignment == "segment":
        print("\n[3/4] Segment-timed transactional outputs", flush=True)
    elif console:
        print("\n[3/4] Text-only transactional outputs", flush=True)
    stats.post_asr_seconds = time.perf_counter() - post_asr_started

    if device == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info()
        stats.cuda_free_end_gib = free_bytes / 1024**3
    elapsed = time.perf_counter() - started
    profile_error: str | None = None
    if profile_path is not None:
        try:
            payload = build_profile_payload(
                args,
                requested_configuration,
                stats,
                runnable_jobs,
                elapsed,
                device,
                dtype,
            )
            write_profile_json(profile_path, payload)
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"

    run = build_run(
        jobs,
        requested_options,
        args,
        stats,
        elapsed,
        (() if profile_error is None else (f"profile output failed: {profile_error}",)),
    )
    if console:
        print("\n[4/4] Summary", flush=True)
        print_summary(run, runnable_jobs, stats, args)
        if profile_path is not None and profile_error is None:
            print(f"{INDENT}{profile_path}", flush=True)
        if profile_error is not None:
            print(
                f"{INDENT}[error] writing performance profile failed: {profile_error}",
                flush=True,
            )
    return run


class _TranscriberSession:
    """Heavy, private implementation behind the dependency-light public façade."""

    __slots__ = ("_active", "_closed", "_gate", "_resources", "options", "progress")

    def __init__(
        self,
        options: TranscriptionOptions,
        progress: ProgressCallback | None,
    ) -> None:
        self.options = options
        self.progress = progress
        self._resources = ModelResources()
        self._gate = threading.RLock()
        self._active = False
        self._closed = False

    def _report_progress(self, event: ProgressEvent) -> None:
        """Mark user callback execution so cross-thread reentry cannot block."""
        callback = self.progress
        if callback is None:  # pragma: no cover - reporting omits this wrapper
            return
        previous = getattr(_PROCESS_RUNTIME_STATE, "progress_callback", False)
        _PROCESS_RUNTIME_STATE.progress_callback = True
        try:
            callback(event)
        finally:
            _PROCESS_RUNTIME_STATE.progress_callback = previous

    def transcribe(
        self,
        audio: AudioInput,
        *,
        raise_on_error: bool = False,
        _started: float | None = None,
        _runtime_import_seconds: float = 0.0,
        _serialization_wait_seconds: float = 0.0,
    ) -> TranscriptionRun:
        if getattr(_PROCESS_RUNTIME_STATE, "progress_callback", False):
            raise TranscriberBusyError(
                "Reentrant transcription from a progress callback is not supported"
            )
        if getattr(_PROCESS_RUNTIME_STATE, "active", False):
            raise TranscriberBusyError(
                "Reentrant transcription in one process is not supported"
            )
        session_wait_started = time.perf_counter()
        with self._gate:
            serialization_wait_seconds = (
                _serialization_wait_seconds + time.perf_counter() - session_wait_started
            )
            if self._closed:
                raise TranscriberClosedError("This Transcriber has been closed")
            if self._active:
                raise TranscriberBusyError(
                    "Reentrant transcription with one Transcriber is not supported"
                )
            self._active = True
            try:
                process_wait_started = time.perf_counter()
                with _PROCESS_RUNTIME_GATE:
                    serialization_wait_seconds += (
                        time.perf_counter() - process_wait_started
                    )
                    if getattr(_PROCESS_RUNTIME_STATE, "active", False):
                        raise TranscriberBusyError(
                            "Reentrant transcription in one process is not supported"
                        )
                    _PROCESS_RUNTIME_STATE.active = True
                    previous_threads: int | None = None
                    try:
                        previous_threads = torch.get_num_threads()
                        reset_cancellation()
                        paths = normalize_audio_input(audio)
                        try:
                            args = config_from_options(paths, self.options)
                            validate_args(args)
                        except SystemExit as exc:
                            _raise_translated(TranscriptionConfigurationError, exc)
                        except (AttributeError, TypeError, ValueError) as exc:
                            raise TranscriptionConfigurationError(str(exc)) from exc
                        if self.options.publication is not None:
                            default_output_mode()
                        callback = (
                            self._report_progress if self.progress is not None else None
                        )
                        with reporting(callback=callback, quiet=True):
                            run = execute(
                                args,
                                self.options,
                                requested_configuration=asdict(args),
                                resources=self._resources,
                                publication_enabled=(
                                    self.options.publication is not None
                                ),
                                console=False,
                                runtime_import_seconds=_runtime_import_seconds,
                                serialization_wait_seconds=serialization_wait_seconds,
                                started=_started,
                            )
                        if raise_on_error and not run.ok:
                            raise BatchTranscriptionError(run)
                        return run
                    except (
                        BatchTranscriptionError,
                        TranscriberBusyError,
                        TranscriptionConfigurationError,
                        TranscriptionInputError,
                    ):
                        raise
                    except TranscriptionRuntimeError:
                        self._resources.evict_asr()
                        raise
                    except _ProgressCallbackAbort as exc:
                        self._resources.evict_asr()
                        raise ProgressCallbackError(exc.original) from exc.original
                    except SystemExit as exc:
                        self._resources.evict_asr()
                        _raise_translated(TranscriptionRuntimeError, exc)
                    except (KeyboardInterrupt, TerminationRequested):
                        self._resources.evict_asr()
                        raise
                    except Exception as exc:
                        self._resources.evict_asr()
                        raise TranscriptionRuntimeError(
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    finally:
                        try:
                            if (
                                previous_threads is not None
                                and torch.get_num_threads() != previous_threads
                            ):
                                torch.set_num_threads(previous_threads)
                        finally:
                            try:
                                release_all_output_locks()
                            finally:
                                _PROCESS_RUNTIME_STATE.active = False
            finally:
                self._active = False

    def close(self) -> None:
        if getattr(_PROCESS_RUNTIME_STATE, "progress_callback", False) or getattr(
            _PROCESS_RUNTIME_STATE, "active", False
        ):
            raise TranscriberBusyError(
                "Cannot close a Transcriber while transcription is active"
            )
        with self._gate:
            if self._closed:
                return
            if self._active:
                raise TranscriberBusyError(
                    "Cannot close a Transcriber while its transcription is active"
                )
            with _PROCESS_RUNTIME_GATE:
                self._resources.close()
                self._closed = True
