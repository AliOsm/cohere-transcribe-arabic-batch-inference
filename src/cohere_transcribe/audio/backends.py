"""Lightweight audio-backend capability detection and selection."""

from __future__ import annotations

import functools
import importlib
import shutil
from importlib import metadata as importlib_metadata
from typing import NamedTuple

from packaging.version import Version

MIN_TORCHCODEC_VERSION = "0.14.0"


class TorchCodecStatus(NamedTuple):
    usable: bool
    version: str | None
    detail: str | None


def _concise_native_error(error: BaseException) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    for marker in ("libnvrtc", "libcudart", "libavcodec", "libavutil"):
        if line := next((line for line in lines if marker in line.lower()), None):
            return f"{type(error).__name__}: {line}"
    detail = lines[0] if lines else str(error)
    return f"{type(error).__name__}: {detail}"


@functools.cache
def probe_torchcodec() -> TorchCodecStatus:
    """Return TorchCodec version and a concise native-loading diagnostic."""
    try:
        version = importlib_metadata.version("torchcodec")
    except importlib_metadata.PackageNotFoundError:
        return TorchCodecStatus(False, None, "package is not installed")
    try:
        if Version(version) < Version(MIN_TORCHCODEC_VERSION):
            return TorchCodecStatus(
                False,
                version,
                f"version is older than {MIN_TORCHCODEC_VERSION}",
            )
        importlib.import_module("torchcodec")
    except Exception as exc:
        return TorchCodecStatus(False, version, _concise_native_error(exc))
    return TorchCodecStatus(True, version, None)


def torchcodec_is_usable() -> bool:
    """Return whether the installed TorchCodec runtime can actually initialize."""
    return probe_torchcodec().usable


def resolve_audio_backend(backend: str) -> str:
    """Resolve automatic decoding to TorchCodec or the FFmpeg fallback."""
    if backend != "auto":
        return backend
    if torchcodec_is_usable():
        return "torchcodec"
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    raise RuntimeError(
        "Automatic audio decoding requires either a working TorchCodec installation "
        "or the ffmpeg executable on PATH"
    )
