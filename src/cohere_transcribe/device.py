"""Compute-device selection and allocator cleanup."""

from __future__ import annotations

import torch

_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "cannot allocate memory",
    "cublas_status_alloc_failed",
    "cuda error: memory allocation",
    "cuda_error_memory_allocation",
    "cudaerrormemoryallocation",
    "cudnn_status_alloc_failed",
    "defaultcpuallocator",
)


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but CUDA is not available to PyTorch"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "--device mps was requested, but MPS is not available to PyTorch"
        )
    return requested


def empty_device_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def is_out_of_memory_error(exc: BaseException) -> bool:
    """Recognize allocator failures reported through Python or backend errors."""
    return isinstance(exc, (torch.OutOfMemoryError, MemoryError)) or any(
        marker in str(exc).lower() for marker in _OUT_OF_MEMORY_MARKERS
    )
