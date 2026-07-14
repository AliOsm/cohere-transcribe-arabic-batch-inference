"""Process environment defaults established immediately before runtime imports."""

from __future__ import annotations

import os


def configure_runtime_environment() -> None:
    """Set conservative defaults without replacing caller-provided configuration."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not {
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    }.intersection(os.environ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
