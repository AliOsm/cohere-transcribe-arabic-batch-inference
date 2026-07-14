"""Silero runtime construction and backend diagnostics."""

from __future__ import annotations

import importlib
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..audio.segmentation import (
    merge_speech_segments,
    sample_timestamps_to_seconds,
    validate_segment_times,
)
from ..models import SR, TranscriptionConfig, info


@dataclass(slots=True)
class SileroRuntime:
    model: Any
    engine: str
    runner: Callable[[np.ndarray, Any, TranscriptionConfig], Sequence[dict[str, Any]]]
    provider: str | None = None
    provider_options: dict[str, dict[str, str]] | None = None
    load_seconds: float = 0.0


class SileroBackendUnavailable(RuntimeError):
    """A Silero execution backend is unavailable in the current environment."""


_vad_thread_local = threading.local()


def onnx_provider_details(
    model: Any,
) -> tuple[str | None, dict[str, dict[str, str]] | None]:
    for candidate in (
        model,
        getattr(model, "session", None),
        getattr(model, "ort_session", None),
        getattr(model, "_session", None),
    ):
        if candidate is None or not hasattr(candidate, "get_providers"):
            continue
        try:
            providers = candidate.get_providers()
        except Exception:
            continue
        if providers:
            raw_options = None
            if hasattr(candidate, "get_provider_options"):
                try:
                    raw_options = candidate.get_provider_options()
                except Exception:
                    raw_options = None
            options = (
                {
                    str(provider): {
                        str(key): str(value) for key, value in values.items()
                    }
                    for provider, values in raw_options.items()
                }
                if isinstance(raw_options, dict)
                else None
            )
            return ",".join(map(str, providers)), options
    return None, None


def is_onnxruntime_failure(exc: BaseException) -> bool:
    """Recognize ONNX Runtime failures without importing it for JIT-only runs."""
    return exc.__class__.__module__.startswith("onnxruntime.")


def packaged_silero_jit_path() -> Path:
    path = Path(__file__).with_name("silero_vad.jit")
    if not path.is_file():
        raise SileroBackendUnavailable(
            f"packaged Silero TorchScript asset is missing: {path}"
        )
    return path


def build_silero_jit_runtime() -> SileroRuntime:
    timestamps_fn = getattr(
        importlib.import_module("cohere_transcribe.vad.vectorized_silero"),
        "get_speech_timestamps",
    )

    class SafeTorchScriptSilero:
        def __init__(self) -> None:
            self.model = torch.jit.load(
                str(packaged_silero_jit_path()), map_location="cpu"
            ).eval()

        def speech_probabilities(self, audio: np.ndarray) -> np.ndarray:
            audio = np.ascontiguousarray(audio, dtype=np.float32)
            if audio.ndim != 1:
                raise ValueError(
                    f"Silero VAD expects mono audio, got shape {audio.shape}"
                )
            if not audio.size:
                return np.empty(0, dtype=np.float32)
            values = torch.from_numpy(audio).unsqueeze(0)
            if values.shape[1] < 512:
                values = torch.nn.functional.pad(values, (0, 512 - values.shape[1]))
            self.model.reset_states()
            with torch.inference_mode():
                probabilities = self.model.audio_forward(values, SR)
            expected_frames = (audio.size + 511) // 512
            return (
                probabilities.reshape(-1)[:expected_frames]
                .to(dtype=torch.float32)
                .numpy()
            )

    model = SafeTorchScriptSilero()

    def run(
        audio: np.ndarray, active_model: Any, args: TranscriptionConfig
    ) -> Sequence[dict[str, Any]]:
        return timestamps_fn(
            audio,
            active_model,
            sampling_rate=SR,
            threshold=args.vad_threshold,
            min_speech_duration_ms=int(round(args.min_dur * 1000)),
            max_speech_duration_s=args.max_dur,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
        )

    return SileroRuntime(model=model, engine="jit", runner=run)


def build_silero_onnx_runtime() -> SileroRuntime:
    try:
        module = importlib.import_module("cohere_transcribe.vad.vectorized_silero")
    except ImportError as exc:
        raise SileroBackendUnavailable(str(exc)) from exc
    model_class = getattr(module, "VectorizedSileroVAD")
    timestamps_fn = getattr(module, "get_speech_timestamps")
    try:
        model = model_class()
    except Exception as exc:
        if isinstance(exc, (ImportError, OSError)) or is_onnxruntime_failure(exc):
            raise SileroBackendUnavailable(str(exc)) from exc
        raise

    def run(
        audio: np.ndarray, active_model: Any, args: TranscriptionConfig
    ) -> Sequence[dict[str, Any]]:
        try:
            return timestamps_fn(
                audio,
                active_model,
                sampling_rate=SR,
                threshold=args.vad_threshold,
                min_speech_duration_ms=int(round(args.min_dur * 1000)),
                max_speech_duration_s=args.max_dur,
                min_silence_duration_ms=args.min_silence_ms,
                speech_pad_ms=args.speech_pad_ms,
            )
        except Exception as exc:
            if isinstance(exc, (ImportError, OSError)) or is_onnxruntime_failure(exc):
                raise SileroBackendUnavailable(str(exc)) from exc
            raise

    provider, provider_options = onnx_provider_details(model)

    return SileroRuntime(
        model=model,
        engine="onnx",
        runner=run,
        provider=provider,
        provider_options=provider_options,
    )


def build_silero_torch_runtime(args: TranscriptionConfig) -> SileroRuntime:
    started = time.perf_counter()
    try:
        module = importlib.import_module("cohere_transcribe.vad.torch_silero")
        limits_class = getattr(module, "BatchLimits")
        model_class = getattr(module, "TorchSileroSequenceVAD")
        timestamps_fn = getattr(
            importlib.import_module("cohere_transcribe.vad.vectorized_silero"),
            "get_speech_timestamps_from_probabilities",
        )
        limits = limits_class(
            block_frames=args.vad_block_frames,
            max_files=args.vad_batch_size,
            max_valid_frames=args.vad_block_frames * args.vad_batch_size,
            max_padded_frames=args.vad_block_frames * args.vad_batch_size,
            # max_valid_frames is already the exact integer work bound. Avoid a
            # frames -> float seconds -> frames round-trip, which can round legal
            # configurations such as 2,001 frames down to 2,000.
            max_audio_seconds=None,
        )
        model = model_class(sampling_rate=SR, limits=limits)
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SileroBackendUnavailable(str(exc)) from exc
    load_seconds = time.perf_counter() - started

    def run(
        audio: np.ndarray, active_model: Any, config: TranscriptionConfig
    ) -> Sequence[dict[str, Any]]:
        probabilities = active_model.speech_probabilities(audio)
        return timestamps_fn(
            len(audio),
            probabilities,
            sampling_rate=SR,
            threshold=config.vad_threshold,
            min_speech_duration_ms=int(round(config.min_dur * 1000)),
            max_speech_duration_s=config.max_dur,
            min_silence_duration_ms=config.min_silence_ms,
            speech_pad_ms=config.speech_pad_ms,
        )

    return SileroRuntime(
        model=model,
        engine="torch",
        runner=run,
        provider="CPU",
        load_seconds=load_seconds,
    )


def get_silero_runtime(
    requested_engine: str, args: TranscriptionConfig
) -> SileroRuntime:
    cache = getattr(_vad_thread_local, "runtimes", None)
    if cache is None:
        cache = {}
        _vad_thread_local.runtimes = cache
    cache_key = (
        requested_engine,
        args.vad_batch_size,
        args.vad_block_frames,
    )
    if cache_key in cache:
        return cache[cache_key]

    if requested_engine == "torch":
        runtime = build_silero_torch_runtime(args)
        cache[cache_key] = runtime
        return runtime

    if requested_engine in {"auto", "onnx"}:
        try:
            runtime = build_silero_onnx_runtime()
        except SileroBackendUnavailable as exc:
            if requested_engine == "onnx":
                raise
            _vad_thread_local.onnx_fallback_error = f"{type(exc).__name__}: {exc}"
        else:
            cache[cache_key] = runtime
            return runtime

    runtime = build_silero_jit_runtime()
    cache[cache_key] = runtime
    return runtime


def segment_audio_silero(
    audio: np.ndarray, args: TranscriptionConfig
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    str,
    str | None,
    dict[str, dict[str, str]] | None,
    str | None,
]:
    runtime = get_silero_runtime(args.vad_engine, args)
    fallback_reason = getattr(_vad_thread_local, "onnx_fallback_error", None)
    if (
        args.vad_engine == "auto"
        and runtime.engine == "jit"
        and fallback_reason
        and not getattr(_vad_thread_local, "onnx_fallback_reported", False)
    ):
        info(f"[warn] ONNX Silero unavailable ({fallback_reason}); using TorchScript")
        _vad_thread_local.onnx_fallback_reported = True
    try:
        timestamps = runtime.runner(audio, runtime.model, args)
    except SileroBackendUnavailable as exc:
        if args.vad_engine != "auto" or runtime.engine != "onnx":
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        _vad_thread_local.onnx_fallback_error = fallback_reason
        _vad_thread_local.onnx_fallback_reported = True
        info(
            f"[warn] ONNX Silero failed ({type(exc).__name__}: {exc}); "
            "falling back to TorchScript"
        )
        runtime = build_silero_jit_runtime()
        cache = getattr(_vad_thread_local, "runtimes", None)
        if cache is None:
            cache = {}
            _vad_thread_local.runtimes = cache
        cache[("auto", args.vad_batch_size, args.vad_block_frames)] = runtime
        timestamps = runtime.runner(audio, runtime.model, args)

    raw_segments = sample_timestamps_to_seconds(timestamps, len(audio))
    segments = (
        merge_speech_segments(raw_segments, args.max_dur)
        if args.vad_merge
        else list(raw_segments)
    )
    segments = validate_segment_times(
        segments, len(audio) / SR, max_duration=args.max_dur
    )
    engine = runtime.engine + ("+merge" if args.vad_merge else "")
    return (
        segments,
        raw_segments,
        engine,
        runtime.provider,
        runtime.provider_options,
        fallback_reason,
    )


def postprocess_silero_probabilities(
    audio: np.ndarray,
    probabilities: np.ndarray,
    args: TranscriptionConfig,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    module = importlib.import_module("cohere_transcribe.vad.vectorized_silero")
    timestamps_fn = getattr(module, "get_speech_timestamps_from_probabilities")
    timestamps = timestamps_fn(
        len(audio),
        probabilities,
        sampling_rate=SR,
        threshold=args.vad_threshold,
        min_speech_duration_ms=int(round(args.min_dur * 1000)),
        max_speech_duration_s=args.max_dur,
        min_silence_duration_ms=args.min_silence_ms,
        speech_pad_ms=args.speech_pad_ms,
    )
    raw_segments = sample_timestamps_to_seconds(timestamps, len(audio))
    segments = (
        merge_speech_segments(raw_segments, args.max_dur)
        if args.vad_merge
        else list(raw_segments)
    )
    return (
        validate_segment_times(segments, len(audio) / SR, max_duration=args.max_dur),
        raw_segments,
    )
