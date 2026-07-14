from __future__ import annotations

from unittest import mock

import pytest
import torch

from cohere_transcribe import device, preflight
from cohere_transcribe.audio import backends as audio_backends
from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.models import TRANSFORMERS_VERSION, TranscriptionConfig
from cohere_transcribe.vad import runtime as vad_runtime


def runtime_args(
    *,
    vad: str = "none",
    vad_engine: str = "auto",
    alignment: str = "none",
    audio_backend: str = "ffmpeg",
) -> TranscriptionConfig:
    args = parse_args(
        [
            "audio.wav",
            "--vad",
            vad,
            "--vad-engine",
            vad_engine,
            "--alignment",
            alignment,
            "--audio-backend",
            audio_backend,
        ]
    )
    validate_args(args)
    return args


def allow_required_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    torchaudio_version: str | None = None,
) -> list[str]:
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return object()

    def package_version(name: str) -> str | None:
        if name == "transformers":
            return TRANSFORMERS_VERSION
        if name == "torchaudio":
            return torchaudio_version
        return None

    monkeypatch.setattr(preflight.importlib, "import_module", import_module)
    monkeypatch.setattr(preflight, "package_version", package_version)
    return imported


def test_missing_required_import_has_actionable_install_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_transformers(name: str) -> object:
        if name == "transformers":
            raise ImportError("broken wheel")
        return object()

    monkeypatch.setattr(preflight.importlib, "import_module", fail_transformers)

    with pytest.raises(SystemExit) as raised:
        preflight.preflight_runtime(runtime_args())

    message = str(raised.value)
    assert "Cannot initialize Cohere ASR" in message
    assert "import 'transformers' failed (broken wheel)" in message
    assert f"pip install transformers=={TRANSFORMERS_VERSION}" in message


def test_transformers_version_must_match_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(preflight.importlib, "import_module", lambda _name: object())
    monkeypatch.setattr(preflight, "package_version", lambda _name: "5.12.0")

    with pytest.raises(SystemExit) as raised:
        preflight.preflight_runtime(runtime_args())

    message = str(raised.value)
    assert f"transformers=={TRANSFORMERS_VERSION}" in message
    assert "found 5.12.0" in message


def test_missing_packaged_jit_asset_has_reinstall_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_required_imports(monkeypatch)

    def missing_asset() -> None:
        raise vad_runtime.SileroBackendUnavailable("asset is missing")

    monkeypatch.setattr(vad_runtime, "packaged_silero_jit_path", missing_asset)

    with pytest.raises(SystemExit) as raised:
        preflight.preflight_runtime(runtime_args(vad="silero", vad_engine="jit"))

    message = str(raised.value)
    assert "Silero 6.2.1 package data is unavailable (asset is missing)" in message
    assert "Reinstall cohere-transcribe-arabic" in message


def test_torch_and_torchaudio_release_mismatch_fails_before_alignment_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_required_imports(monkeypatch, torchaudio_version="2.10.0")
    forced_align = mock.Mock()
    monkeypatch.setattr(preflight, "preflight_forced_align", forced_align)
    monkeypatch.setattr(torch, "__version__", "2.11.0")

    with pytest.raises(SystemExit) as raised:
        preflight.preflight_runtime(runtime_args(alignment="word"))

    message = str(raised.value)
    assert "PyTorch and TorchAudio must use matching major/minor releases" in message
    assert "torch 2.11.0 and torchaudio 2.10.0" in message
    forced_align.assert_not_called()


def test_automatic_decoder_absence_is_reported_by_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_required_imports(monkeypatch)
    monkeypatch.setattr(
        audio_backends,
        "resolve_audio_backend",
        mock.Mock(side_effect=RuntimeError("no automatic decoder is available")),
    )

    with pytest.raises(SystemExit, match="no automatic decoder is available"):
        preflight.preflight_runtime(runtime_args(audio_backend="auto"))


def test_explicit_broken_torchcodec_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_required_imports(monkeypatch)
    monkeypatch.setattr(audio_backends, "torchcodec_is_usable", lambda: False)

    with pytest.raises(SystemExit) as raised:
        preflight.preflight_runtime(runtime_args(audio_backend="torchcodec"))

    message = str(raised.value)
    assert "--audio-backend torchcodec requires a working TorchCodec >= 0.14" in message
    assert "compatible system FFmpeg libraries" in message


def test_explicit_ffmpeg_requires_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allow_required_imports(monkeypatch)
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: None)

    with pytest.raises(SystemExit, match="ffmpeg executable on PATH"):
        preflight.preflight_runtime(runtime_args(audio_backend="ffmpeg"))


def test_librosa_backend_imports_its_core_decoder_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = allow_required_imports(monkeypatch)

    preflight.preflight_runtime(runtime_args(audio_backend="librosa"))

    assert imported == ["transformers", "librosa"]


def test_auditok_vad_imports_only_its_vad_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = allow_required_imports(monkeypatch)
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    preflight.preflight_runtime(runtime_args(vad="auditok"))

    assert imported == ["transformers", "auditok.core"]


def test_successful_word_preflight_checks_all_dependencies_and_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = allow_required_imports(
        monkeypatch,
        torchaudio_version=torch.__version__,
    )
    forced_align = mock.Mock()
    monkeypatch.setattr(preflight, "preflight_forced_align", forced_align)
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: "/usr/bin/ffmpeg")

    preflight.preflight_runtime(runtime_args(alignment="word"))

    assert imported == [
        "transformers",
        "torchaudio",
        "uroman",
        "cohere_transcribe.alignment.alignment_utils",
        "cohere_transcribe.alignment.text_utils",
    ]
    forced_align.assert_called_once_with()


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_pick_device_auto_priority(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(
        device.torch.backends.mps, "is_available", lambda: mps_available
    )

    assert device.pick_device("auto") == expected


@pytest.mark.parametrize("requested", ["cpu", "cuda", "mps"])
def test_pick_device_accepts_available_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device.torch.backends.mps, "is_available", lambda: True)

    assert device.pick_device(requested) == requested


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ("cuda", "CUDA is not available to PyTorch"),
        ("mps", "MPS is not available to PyTorch"),
    ],
)
def test_pick_device_rejects_unavailable_explicit_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    message: str,
) -> None:
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(SystemExit, match=message):
        device.pick_device(requested)
