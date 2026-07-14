"""Dependency-light input normalization for the public Python API."""

from __future__ import annotations

import os
from collections.abc import Sequence

from .types import AudioInput, TranscriptionInputError


def normalize_audio_input(audio: AudioInput) -> list[str]:
    """Normalize one path or an ordered path sequence without expanding directories."""

    def normalize_one(value: object, index: int | None = None) -> str:
        label = "audio" if index is None else f"audio[{index}]"
        if isinstance(value, bytes):
            raise TranscriptionInputError(f"{label} must not be bytes")
        if not isinstance(value, (str, os.PathLike)):
            raise TranscriptionInputError(
                f"{label} must be a string or path-like object"
            )
        path_text = os.fspath(value)
        if isinstance(path_text, bytes):
            raise TranscriptionInputError(
                f"{label} must resolve to a text path, not bytes"
            )
        if not path_text.strip():
            raise TranscriptionInputError(f"{label} must not be empty")
        return path_text

    if isinstance(audio, bytes):
        raise TranscriptionInputError("audio must not be bytes")
    if isinstance(audio, (str, os.PathLike)):
        return [normalize_one(audio)]
    if not isinstance(audio, Sequence):
        raise TranscriptionInputError(
            "audio must be one path or an ordered sequence of paths"
        )
    if not audio:
        raise TranscriptionInputError("audio must contain at least one path")
    return [normalize_one(value, index) for index, value in enumerate(audio)]
