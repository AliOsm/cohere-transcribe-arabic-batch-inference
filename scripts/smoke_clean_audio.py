#!/usr/bin/env python3
"""Verify that a clean core wheel decodes audio through TorchCodec."""

from __future__ import annotations

import importlib.util
import struct
import tempfile
import wave
from pathlib import Path

import numpy as np

from cohere_transcribe.audio.decoding import decode_audio_resolved

SAMPLE_RATE = 16_000
SAMPLE_COUNT = 1_600
SAMPLE_PATTERN = (-32_768, -16_384, -1, 0, 1, 16_384, 32_767)


def main() -> int:
    for optional_module in (
        "auditok",
        "onnxruntime",
        "silero_vad",
        "torchaudio",
        "uroman",
    ):
        if importlib.util.find_spec(optional_module) is not None:
            raise RuntimeError(
                f"clean core-wheel environment unexpectedly contains {optional_module}"
            )
    if importlib.util.find_spec("torchcodec") is None:
        raise RuntimeError("clean core-wheel environment is missing TorchCodec")
    if importlib.util.find_spec("librosa") is None:
        raise RuntimeError("clean core-wheel environment is missing Librosa")

    samples = tuple(
        SAMPLE_PATTERN[index % len(SAMPLE_PATTERN)] for index in range(SAMPLE_COUNT)
    )
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "decoder-smoke.wav"
        with wave.open(str(fixture), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        decoded, backend, fallback_reason = decode_audio_resolved(fixture, "auto")

    if backend != "torchcodec" or fallback_reason is not None:
        raise RuntimeError(
            "core-wheel automatic decoding did not resolve directly to TorchCodec: "
            f"backend={backend!r}, fallback_reason={fallback_reason!r}"
        )
    expected = np.asarray(samples, dtype=np.float32) / np.float32(32_768)
    np.testing.assert_array_equal(decoded, expected)
    if decoded.dtype != np.float32 or not decoded.flags.c_contiguous:
        raise RuntimeError(
            f"decoded audio must be contiguous float32, got {decoded.dtype}"
        )

    print(
        f"Validated core-wheel auto decoding: {backend}, {decoded.size} float32 samples"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
