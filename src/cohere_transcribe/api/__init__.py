"""Lightweight public façade for programmatic transcription."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import TracebackType
from typing import Protocol, cast

from .input import normalize_audio_input
from .types import (
    AudioInput,
    BatchTranscriptionError,
    ProgressCallback,
    TranscriberBusyError,
    TranscriberClosedError,
    TranscriptionOptions,
    TranscriptionRun,
    TranscriptionRuntimeError,
)


class _Session(Protocol):
    def transcribe(
        self,
        audio: AudioInput,
        *,
        raise_on_error: bool = False,
        _started: float | None = None,
        _runtime_import_seconds: float = 0.0,
        _serialization_wait_seconds: float = 0.0,
    ) -> TranscriptionRun: ...

    def close(self) -> None: ...


class Transcriber:
    """Reusable, serialized transcription session with lazy model loading."""

    __slots__ = (
        "_closed",
        "_closing",
        "_implementation",
        "_lock",
        "_options",
        "_progress",
    )

    def __init__(
        self,
        options: TranscriptionOptions | None = None,
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        resolved_options = options if options is not None else TranscriptionOptions()
        if not isinstance(resolved_options, TranscriptionOptions):
            raise TypeError("options must be a TranscriptionOptions instance")
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        self._options = resolved_options
        self._progress = progress
        self._implementation: _Session | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._closing = False

    def _get_implementation(self) -> tuple[_Session, float, float]:
        lock_started = time.perf_counter()
        with self._lock:
            serialization_wait_seconds = time.perf_counter() - lock_started
            if self._closed:
                raise TranscriberClosedError("This Transcriber has been closed")
            if self._closing:
                raise TranscriberBusyError("This Transcriber is being closed")
            runtime_import_seconds = 0.0
            if self._implementation is None:
                import_started = time.perf_counter()
                try:
                    from .._environment import configure_runtime_environment

                    configure_runtime_environment()
                    from ..runtime.engine import _TranscriberSession

                    self._implementation = cast(
                        _Session, _TranscriberSession(self._options, self._progress)
                    )
                except SystemExit as exc:
                    raise TranscriptionRuntimeError(
                        str(exc) or "Transcription runtime initialization failed"
                    ) from exc
                except Exception as exc:
                    raise TranscriptionRuntimeError(
                        f"Cannot initialize the transcription runtime: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                runtime_import_seconds = time.perf_counter() - import_started
            return (
                self._implementation,
                runtime_import_seconds,
                serialization_wait_seconds,
            )

    @property
    def options(self) -> TranscriptionOptions:
        """Immutable options fixed for this reusable session."""
        return self._options

    @property
    def progress(self) -> ProgressCallback | None:
        """Progress callback fixed for this reusable session."""
        return self._progress

    def transcribe(
        self,
        audio: AudioInput,
        *,
        raise_on_error: bool = False,
    ) -> TranscriptionRun:
        """Transcribe one path or an ordered sequence of files and directories."""
        started = time.perf_counter()
        normalized_audio = normalize_audio_input(audio)
        (
            implementation,
            runtime_import_seconds,
            serialization_wait_seconds,
        ) = self._get_implementation()
        try:
            run = implementation.transcribe(
                normalized_audio,
                raise_on_error=raise_on_error,
                _started=started,
                _runtime_import_seconds=runtime_import_seconds,
                _serialization_wait_seconds=serialization_wait_seconds,
            )
        except BatchTranscriptionError as exc:
            adjusted = _with_elapsed(exc.run, time.perf_counter() - started)
            raise BatchTranscriptionError(adjusted) from exc
        return _with_elapsed(run, time.perf_counter() - started)

    def close(self) -> None:
        """Release retained models and make the session unusable."""
        with self._lock:
            if self._closed:
                return
            if self._closing:
                raise TranscriberBusyError("This Transcriber is already being closed")
            self._closing = True
            implementation = self._implementation
        try:
            if implementation is not None:
                implementation.close()
        except BaseException:
            with self._lock:
                self._closing = False
            raise
        with self._lock:
            self._closed = True
            self._closing = False
            if self._implementation is implementation:
                self._implementation = None

    def __enter__(self) -> Transcriber:
        with self._lock:
            if self._closed:
                raise TranscriberClosedError("This Transcriber has been closed")
            if self._closing:
                raise TranscriberBusyError("This Transcriber is being closed")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def transcribe(
    audio: AudioInput,
    *,
    options: TranscriptionOptions | None = None,
    progress: ProgressCallback | None = None,
    raise_on_error: bool = False,
) -> TranscriptionRun:
    """Transcribe audio in a one-shot session that always releases its models."""
    started = time.perf_counter()
    transcriber = Transcriber(options, progress=progress)
    batch_error: BatchTranscriptionError | None = None
    run: TranscriptionRun | None = None
    try:
        try:
            run = transcriber.transcribe(audio, raise_on_error=raise_on_error)
        except BatchTranscriptionError as exc:
            batch_error = exc
    finally:
        transcriber.close()
    elapsed = time.perf_counter() - started
    if batch_error is not None:
        adjusted = _with_elapsed(batch_error.run, elapsed)
        raise BatchTranscriptionError(adjusted) from batch_error
    assert run is not None
    return _with_elapsed(run, elapsed)


def _with_elapsed(run: TranscriptionRun, elapsed: float) -> TranscriptionRun:
    statistics = replace(
        run.statistics,
        elapsed_seconds=elapsed,
        real_time_factor_x=(
            run.statistics.successful_audio_seconds / elapsed if elapsed > 0 else 0.0
        ),
    )
    return replace(run, statistics=statistics)


__all__ = ["Transcriber", "transcribe"]
