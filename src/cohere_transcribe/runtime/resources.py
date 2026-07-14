"""Private ownership of reusable heavyweight inference resources."""

from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import Callable
from typing import Any

import torch

from ..device import empty_device_cache

AsrLoader = Callable[[str, torch.dtype], tuple[Any, Any]]

_ASR_OWNER_GUARD = threading.RLock()
_ASR_OWNER: weakref.ReferenceType[ModelResources] | None = None


def evict_current_asr_owner() -> None:
    """Evict whichever reusable ASR model currently owns the process lease."""
    with _ASR_OWNER_GUARD:
        owner = _ASR_OWNER() if _ASR_OWNER is not None else None
        if owner is not None:
            owner.evict_asr()


class ModelResources:
    """Own a lazy ASR model while keeping the word aligner one-shot."""

    __slots__ = (
        "__weakref__",
        "_asr_key",
        "_asr_model",
        "_asr_processor",
        "_closed",
    )

    def __init__(self) -> None:
        self._asr_key: tuple[str, torch.dtype] | None = None
        self._asr_processor: Any | None = None
        self._asr_model: Any | None = None
        self._closed = False

    def acquire_asr(
        self,
        device: str,
        dtype: torch.dtype,
        *,
        loader: AsrLoader,
    ) -> tuple[Any, Any, bool]:
        """Return a compatible ASR pair, loading it only when necessary."""
        global _ASR_OWNER
        with _ASR_OWNER_GUARD:
            if self._closed:
                raise RuntimeError("Model resources have been closed")
            key = (device, dtype)
            if self._asr_model is not None and self._asr_key != key:
                self.evict_asr()

            owner = _ASR_OWNER() if _ASR_OWNER is not None else None
            if owner is not self:
                if owner is not None:
                    owner.evict_asr()
                _ASR_OWNER = weakref.ref(self)

            if self._asr_model is None:
                processor, model = loader(device, dtype)
                self._asr_processor = processor
                self._asr_model = model
                self._asr_key = key
                return processor, model, True
            return self._asr_processor, self._asr_model, False

    @property
    def asr_circuit_broken(self) -> bool:
        """Return whether handled fatal generation errors poisoned the cached model."""
        if self._asr_model is None:
            return False
        controller = getattr(self._asr_model, "_transcribe_batch_controller", None)
        return bool(getattr(controller, "circuit_breaker_error", None))

    @property
    def has_asr(self) -> bool:
        return self._asr_model is not None

    def evict_asr(self) -> None:
        """Drop cached ASR state and release the relevant device allocator cache."""
        global _ASR_OWNER
        with _ASR_OWNER_GUARD:
            owner = _ASR_OWNER() if _ASR_OWNER is not None else None
            if owner is self:
                _ASR_OWNER = None
            if self._asr_model is None and self._asr_processor is None:
                self._asr_key = None
                return
            device = self._asr_key[0] if self._asr_key is not None else None
            self._asr_model = None
            self._asr_processor = None
            self._asr_key = None
        gc.collect()
        if device is not None:
            empty_device_cache(device)

    def close(self) -> None:
        """Release all resources permanently. This operation is idempotent."""
        if self._closed:
            return
        self.evict_asr()
        self._closed = True
