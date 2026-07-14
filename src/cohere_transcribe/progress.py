"""Per-run console suppression and serialized Python progress callbacks."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from tqdm import tqdm

from .api.types import ProgressCallback, ProgressEvent

T = TypeVar("T")


class _ProgressCallbackAbort(BaseException):
    """Escape broad per-file handlers without misclassifying callback failures."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))


@dataclass(slots=True)
class _ProgressState:
    callback: ProgressCallback | None
    quiet: bool
    callback_lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: ProgressEvent) -> None:
        if self.callback is None:
            return
        with self.callback_lock:
            try:
                self.callback(event)
            except Exception as exc:
                raise _ProgressCallbackAbort(exc) from exc


_STATE_LOCK = threading.RLock()
_CURRENT: _ProgressState | None = None


def _current_state() -> _ProgressState | None:
    with _STATE_LOCK:
        return _CURRENT


@contextlib.contextmanager
def reporting(
    *, callback: ProgressCallback | None = None, quiet: bool = False
) -> Iterator[None]:
    """Apply reporting behavior to one serialized transcription run."""
    global _CURRENT
    transformers_progress_enabled: bool | None = None
    hub_progress_disabled: bool | None = None
    transformers_logging = None
    hub_utils = None
    if quiet:
        try:
            from transformers.utils import logging as transformers_logging

            transformers_progress_enabled = (
                transformers_logging.is_progress_bar_enabled()
            )
            transformers_logging.disable_progress_bar()
        except (ImportError, AttributeError):
            transformers_logging = None
        try:
            from huggingface_hub import utils as hub_utils

            hub_progress_disabled = hub_utils.are_progress_bars_disabled()
            hub_utils.disable_progress_bars()
        except (ImportError, AttributeError):
            hub_utils = None
    with _STATE_LOCK:
        previous = _CURRENT
        _CURRENT = _ProgressState(callback=callback, quiet=quiet)
    try:
        yield
    finally:
        with _STATE_LOCK:
            _CURRENT = previous
        if transformers_progress_enabled and transformers_logging is not None:
            transformers_logging.enable_progress_bar()
        if hub_progress_disabled is False and hub_utils is not None:
            hub_utils.enable_progress_bars()


def write(message: str) -> None:
    """Write a console-safe message or deliver it to the active callback."""
    state = _current_state()
    if state is None:
        tqdm.write(message)
    elif state.callback is not None:
        state.emit(ProgressEvent(stage="message", message=message))
    elif not state.quiet:
        tqdm.write(message)


class ProgressBar(Generic[T]):
    """Small tqdm-compatible surface used by ASR and alignment loops."""

    __slots__ = ("_bar", "_iterable", "_stage", "_state", "_total", "n")

    def __init__(
        self,
        iterable: Iterable[T] | None = None,
        *,
        total: int | None = None,
        desc: str = "progress",
        **tqdm_kwargs: Any,
    ) -> None:
        self._state = _current_state()
        self._stage = desc
        self._iterable = iterable
        self._total = total
        if total is None and iterable is not None:
            with contextlib.suppress(TypeError):
                self._total = len(iterable)  # type: ignore[arg-type]
        self.n = 0
        self._bar = (
            tqdm(iterable, total=total, desc=desc, **tqdm_kwargs)
            if self._state is None or not self._state.quiet
            else None
        )
        if self._state is not None and self._state.callback is not None:
            self._state.emit(
                ProgressEvent(stage=self._stage, current=0, total=self._total)
            )

    def update(self, count: int = 1) -> None:
        self.n += count
        if self._bar is not None:
            self._bar.update(count)
        if self._state is not None and self._state.callback is not None:
            self._state.emit(
                ProgressEvent(
                    stage=self._stage,
                    current=self.n,
                    total=self._total,
                )
            )

    def __iter__(self) -> Iterator[T]:
        if self._bar is not None:
            yield from self._bar
            return
        if self._iterable is None:
            return
        for item in self._iterable:
            yield item
            self.update()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str = "progress",
    **kwargs: Any,
) -> ProgressBar[T]:
    return ProgressBar(iterable, total=total, desc=desc, **kwargs)
