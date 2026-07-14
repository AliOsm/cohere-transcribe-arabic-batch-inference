"""Dependency and runtime compatibility checks for selected features."""

from __future__ import annotations

import importlib
import shutil

from .models import (
    SILERO_VERSION,
    TRANSFORMERS_VERSION,
    TranscriptionConfig,
    package_version,
    release_pair,
)


def preflight_forced_align() -> None:
    """Verify that TorchAudio's maintained CPU CTC operation is callable."""
    try:
        import torch
        from torchaudio.functional import forced_align

        emissions = torch.log_softmax(
            torch.tensor([[[4.0, 0.0], [0.0, 4.0]]], dtype=torch.float32), dim=-1
        )
        targets = torch.tensor([[1]], dtype=torch.int64)
        path, scores = forced_align(emissions, targets, blank=0)
        if path.shape != (1, 2) or scores.shape != (1, 2):
            raise RuntimeError(
                f"unexpected forced_align output shapes {path.shape}, {scores.shape}"
            )
    except Exception as exc:
        raise SystemExit(
            "TorchAudio forced alignment is unavailable or incompatible with this "
            f"PyTorch build ({type(exc).__name__}: {exc}). Install matching torch "
            "and torchaudio releases."
        ) from exc


def preflight_runtime(args: TranscriptionConfig) -> None:
    """Import only dependencies required by the selected execution path."""
    import torch

    required: list[tuple[str, str, str]] = [
        (
            "transformers",
            "Cohere ASR",
            f"transformers=={TRANSFORMERS_VERSION}",
        ),
    ]
    if args.vad == "silero":
        if args.vad_engine == "torch":
            required.append(
                (
                    "cohere_transcribe.vad.torch_silero",
                    "packed Torch Silero VAD",
                    "the installed cohere_transcribe.vad package",
                )
            )
        required.append(
            (
                "cohere_transcribe.vad.vectorized_silero",
                "Silero timestamp runtime",
                "the installed cohere_transcribe.vad package",
            )
        )
        if args.vad_engine == "onnx":
            required.append(
                (
                    "onnxruntime",
                    "ONNX Silero VAD",
                    "cohere-transcribe-arabic[onnx]",
                )
            )
    elif args.vad == "auditok":
        required.append(
            (
                "auditok.core",
                "Auditok VAD",
                "cohere-transcribe-arabic[auditok]",
            )
        )
    if args.alignment == "word":
        required.extend(
            [
                (
                    "torchaudio",
                    "word alignment",
                    "cohere-transcribe-arabic[word]",
                ),
                (
                    "uroman",
                    "word-alignment romanization",
                    "cohere-transcribe-arabic[word]",
                ),
                (
                    "cohere_transcribe.alignment.alignment_utils",
                    "word alignment span utilities",
                    "cohere-transcribe-arabic[word]",
                ),
                (
                    "cohere_transcribe.alignment.text_utils",
                    "word alignment text utilities",
                    "cohere-transcribe-arabic[word]",
                ),
            ]
        )
    if args.audio_backend == "torchcodec":
        required.append(
            (
                "torchcodec",
                "TorchCodec audio decoding",
                "cohere-transcribe-arabic",
            )
        )
    elif args.audio_backend == "librosa":
        required.append(
            (
                "librosa",
                "Librosa audio decoding",
                "cohere-transcribe-arabic",
            )
        )

    for module_name, feature, package in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise SystemExit(
                f"Cannot initialize {feature}: import {module_name!r} failed ({exc}).\n"
                f"  Install a compatible build with: pip install {package}"
            ) from exc

    if args.vad == "silero" and args.vad_engine in {"torch", "jit"}:
        from .vad.runtime import SileroBackendUnavailable, packaged_silero_jit_path

        try:
            packaged_silero_jit_path()
        except SileroBackendUnavailable as exc:
            raise SystemExit(
                f"Silero {SILERO_VERSION} package data is unavailable ({exc}). "
                "Reinstall cohere-transcribe-arabic."
            ) from exc

    from packaging.version import Version

    transformers_version = package_version("transformers")
    if transformers_version is None or Version(transformers_version) != Version(
        TRANSFORMERS_VERSION
    ):
        raise SystemExit(
            "The optimized Cohere hot path is validated only with "
            f"transformers=={TRANSFORMERS_VERSION}; "
            f"found {transformers_version or 'unknown'}"
        )

    if args.alignment == "word":
        torch_pair = release_pair(torch.__version__)
        torchaudio_version = package_version("torchaudio")
        audio_pair = release_pair(torchaudio_version or "")
        if (
            torch_pair is not None
            and audio_pair is not None
            and torch_pair != audio_pair
        ):
            raise SystemExit(
                "PyTorch and TorchAudio must use matching major/minor releases for "
                f"forced alignment; found torch {torch.__version__} and "
                f"torchaudio {torchaudio_version}"
            )
        preflight_forced_align()
    if args.audio_backend in {"auto", "torchcodec"}:
        from .audio.backends import resolve_audio_backend, torchcodec_is_usable

        if args.audio_backend == "auto":
            try:
                resolve_audio_backend("auto")
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
        elif not torchcodec_is_usable():
            raise SystemExit(
                "--audio-backend torchcodec requires a working TorchCodec >= 0.14 "
                "installation and compatible system FFmpeg libraries"
            )
    elif args.audio_backend == "ffmpeg" and not shutil.which("ffmpeg"):
        raise SystemExit(
            "--audio-backend ffmpeg requires the ffmpeg executable on PATH"
        )
