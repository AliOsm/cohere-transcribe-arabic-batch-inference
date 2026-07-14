"""Audio decoder selection and mono PCM loading."""

from __future__ import annotations

import contextlib
import io
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

from ..cancellation import raise_if_cancelled, registered_process, terminate_process
from ..models import FFMPEG_DECODE_TIMEOUT_S, SR
from .backends import resolve_audio_backend


class DecodedAudioLimitError(RuntimeError):
    """Raised before a decoder may retain PCM beyond the configured limit."""


class UnboundedAudioError(RuntimeError):
    """Raised when a decoder cannot establish a safe pre-decode duration."""


def decoded_bytes_for_duration(duration: float) -> int:
    return int(math.ceil(max(0.0, duration) * SR)) * np.dtype(np.float32).itemsize


def enforce_duration_limit(
    path: Path, duration: float | None, max_decoded_bytes: int | None
) -> None:
    if duration is None or max_decoded_bytes is None:
        return
    required = decoded_bytes_for_duration(duration)
    if required > max_decoded_bytes:
        raise DecodedAudioLimitError(
            f"Decoded audio for {path} would require at least "
            f"{required / 1024**3:.2f} GiB, exceeding the "
            f"{max_decoded_bytes / 1024**3:.2f} GiB per-file limit"
        )


def enforce_array_limit(
    path: Path, audio: np.ndarray, max_decoded_bytes: int | None
) -> None:
    if max_decoded_bytes is not None and audio.nbytes > max_decoded_bytes:
        raise DecodedAudioLimitError(
            f"Decoded audio for {path} is {audio.nbytes / 1024**3:.2f} GiB, "
            f"exceeding the {max_decoded_bytes / 1024**3:.2f} GiB per-file limit"
        )


def load_audio_ffmpeg(path: Path, max_decoded_bytes: int | None = None) -> np.ndarray:
    """Decode to mono float32 PCM without retaining a second full-size bytes copy."""
    raise_if_cancelled()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is not installed; install it or select an installed optional "
            "decoder explicitly"
        )
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "0",
        "-i",
        os.fspath(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    output = io.BytesIO()
    timed_out = threading.Event()
    with tempfile.TemporaryFile() as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not launch FFmpeg for {path}: {exc}") from exc
        assert process.stdout is not None

        def terminate_on_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                with contextlib.suppress(OSError):
                    process.kill()

        with registered_process(process):
            timer = threading.Timer(FFMPEG_DECODE_TIMEOUT_S, terminate_on_timeout)
            timer.daemon = True
            timer.start()
            try:
                while True:
                    raise_if_cancelled()
                    chunk = process.stdout.read(1024 * 1024)
                    if not chunk:
                        break
                    if (
                        max_decoded_bytes is not None
                        and output.tell() + len(chunk) > max_decoded_bytes
                    ):
                        raise DecodedAudioLimitError(
                            f"FFmpeg output for {path} exceeded the "
                            f"{max_decoded_bytes / 1024**3:.2f} GiB per-file limit"
                        )
                    output.write(chunk)
                returncode = process.wait()
                raise_if_cancelled()
            except BaseException:
                terminate_process(process)
                raise
            finally:
                timer.cancel()
                process.stdout.close()

        if timed_out.is_set():
            raise RuntimeError(
                f"FFmpeg exceeded the {FFMPEG_DECODE_TIMEOUT_S}s decode timeout for {path}"
            )
        if returncode:
            stderr_handle.seek(0)
            message = " ".join(
                stderr_handle.read().decode("utf-8", errors="replace").split()
            )
            raise RuntimeError(f"FFmpeg failed for {path}: {message[-500:]}")

    view = output.getbuffer()
    if len(view) % np.dtype("<f4").itemsize:
        raise RuntimeError(
            f"FFmpeg returned an incomplete float32 sample for {path}: {len(view)} bytes"
        )
    # The ndarray retains the memoryview/BytesIO owner, so this is zero-copy.
    return np.frombuffer(view, dtype="<f4")


def load_audio_torchcodec(
    path: Path, max_decoded_bytes: int | None = None
) -> np.ndarray:
    """Decode through TorchCodec after checking container duration metadata."""
    from torchcodec.decoders import AudioDecoder

    raise_if_cancelled()
    decoder = AudioDecoder(path, sample_rate=SR, num_channels=1)
    metadata = decoder.metadata

    def finite_seconds(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    begin_seconds = finite_seconds(getattr(metadata, "begin_stream_seconds", None))
    if begin_seconds is None:
        begin_seconds = 0.0
    header_duration = finite_seconds(
        getattr(metadata, "duration_seconds_from_header", None)
    )
    duration = finite_seconds(getattr(metadata, "duration_seconds", None))
    if header_duration is not None and header_duration >= 0:
        playable_duration = header_duration
    elif duration is not None and duration >= 0:
        playable_duration = (
            duration - begin_seconds
            if begin_seconds > 0 and duration > begin_seconds
            else duration
        )
    else:
        playable_duration = None

    if playable_duration is None:
        if max_decoded_bytes is not None:
            raise UnboundedAudioError(
                f"TorchCodec could not determine the duration of {path} before "
                "decoding; use automatic or FFmpeg decoding to enforce the per-file limit"
            )
    else:
        enforce_duration_limit(path, playable_duration, max_decoded_bytes)
    if max_decoded_bytes is None:
        samples = decoder.get_all_samples().data
    else:
        max_samples = max_decoded_bytes // np.dtype(np.float32).itemsize
        # Request one sample beyond the retained limit. This bounds the decoded
        # output even when container duration metadata under-reports the stream,
        # while still allowing the exact-size postcondition below to detect it.
        stop_seconds = begin_seconds + (max_samples + 1) / SR
        samples = decoder.get_samples_played_in_range(
            start_seconds=begin_seconds,
            stop_seconds=stop_seconds,
        ).data
    raise_if_cancelled()
    if samples.ndim != 2 or samples.shape[0] != 1:
        raise ValueError(
            f"Expected one TorchCodec audio channel, got shape {tuple(samples.shape)}"
        )
    audio = samples.squeeze(0).numpy()
    enforce_array_limit(path, audio, max_decoded_bytes)
    return audio


def decode_audio_resolved(
    path: Path,
    backend: str,
    *,
    max_decoded_bytes: int | None = None,
    duration_hint: float | None = None,
) -> tuple[np.ndarray, str, str | None]:
    """Decode audio and return the concrete backend used for this file."""
    enforce_duration_limit(path, duration_hint, max_decoded_bytes)

    def decode_ffmpeg() -> np.ndarray:
        if max_decoded_bytes is None:
            return load_audio_ffmpeg(path)
        return load_audio_ffmpeg(path, max_decoded_bytes)

    def decode_with(concrete_backend: str) -> np.ndarray:
        if concrete_backend == "ffmpeg":
            return decode_ffmpeg()
        if concrete_backend == "torchcodec":
            if max_decoded_bytes is None:
                return load_audio_torchcodec(path)
            return load_audio_torchcodec(path, max_decoded_bytes)
        from transformers.audio_utils import load_audio

        return load_audio(os.fspath(path), sampling_rate=SR, backend=concrete_backend)

    concrete_backend = resolve_audio_backend(backend)
    fallback_reason: str | None = None
    try:
        audio = decode_with(concrete_backend)
    except DecodedAudioLimitError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as backend_exc:
        if (
            backend != "auto"
            or concrete_backend == "ffmpeg"
            or not shutil.which("ffmpeg")
        ):
            raise
        try:
            audio = decode_ffmpeg()
        except DecodedAudioLimitError:
            raise
        except (OSError, RuntimeError, ValueError) as ffmpeg_exc:
            backend_label = {
                "torchcodec": "TorchCodec",
                "librosa": "Librosa",
            }.get(concrete_backend, concrete_backend)
            raise RuntimeError(
                f"{backend_label} decoding failed "
                f"({type(backend_exc).__name__}: {backend_exc}); FFmpeg fallback "
                f"also failed ({type(ffmpeg_exc).__name__}: {ffmpeg_exc})"
            ) from ffmpeg_exc
        resolved_backend = "ffmpeg"
        fallback_reason = f"{type(backend_exc).__name__}: {backend_exc}"
    else:
        resolved_backend = concrete_backend
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio after decoding, got shape {audio.shape}")
    enforce_array_limit(path, audio, max_decoded_bytes)
    # Bound the validation temporary for multi-hour recordings. Packed Silero
    # repeats this check because its public runtime also accepts external arrays.
    for start in range(0, audio.size, 1_048_576):
        raise_if_cancelled()
        if not np.isfinite(audio[start : start + 1_048_576]).all():
            raise ValueError("Decoded audio contains NaN or infinite samples")
    return np.ascontiguousarray(audio), resolved_backend, fallback_reason


def decode_audio(
    path: Path,
    backend: str,
    *,
    max_decoded_bytes: int | None = None,
    duration_hint: float | None = None,
) -> np.ndarray:
    audio, _, _ = decode_audio_resolved(
        path,
        backend,
        max_decoded_bytes=max_decoded_bytes,
        duration_hint=duration_hint,
    )
    return audio
