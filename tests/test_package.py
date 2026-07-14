from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from cohere_transcribe import __version__, doctor
from cohere_transcribe.alignment.text_utils import preprocess_text
from cohere_transcribe.audio.backends import TorchCodecStatus
from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.inputs import build_jobs
from cohere_transcribe.models import (
    ALIGN_PACKAGE_REVISION,
    is_model_access_error,
    model_access_message,
)

ROOT = Path(__file__).resolve().parents[1]


def run_python(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_pyproject_and_runtime_versions_match() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["version"] == __version__


def test_retained_arabic_uroman_behavior() -> None:
    assert ALIGN_PACKAGE_REVISION == "11855d1de76af2b490dd2e8e2db2661805ae90a0"
    tokens, text = preprocess_text("مرحبا بكم في العالم", "ara")
    assert text[-1] == "العالم"
    assert tokens[-1] == "a l ' a l m"


def test_packaged_silero_assets_are_the_validated_models() -> None:
    onnx_asset = ROOT / "src/cohere_transcribe/vad/silero_vad_v6.onnx"
    assert onnx_asset.stat().st_size == 1_249_744
    assert hashlib.sha256(onnx_asset.read_bytes()).hexdigest() == (
        "914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9"
    )
    jit_asset = ROOT / "src/cohere_transcribe/vad/silero_vad.jit"
    assert jit_asset.stat().st_size == 2_272_526
    assert hashlib.sha256(jit_asset.read_bytes()).hexdigest() == (
        "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
    )


def test_output_names_do_not_use_the_legacy_aligned_suffix(tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"not decoded during input discovery")
    args = parse_args([str(source), "--alignment", "segment"])
    validate_args(args)
    jobs = build_jobs(args)
    assert len(jobs) == 1
    assert {path.name for path in jobs[0].output_paths.values()} == {
        "sample.txt",
        "sample.srt",
        "sample.vtt",
    }


def test_segment_timestamps_are_the_default_output_mode(tmp_path: Path) -> None:
    source = tmp_path / "sample.wav"
    source.write_bytes(b"input discovery does not decode this fixture")
    args = parse_args([str(source)])
    validate_args(args)

    assert args.alignment == "segment"
    assert args.formats == ["txt", "srt", "vtt"]
    assert doctor.parse_args([]).mode == "segment"


def test_model_access_error_has_actionable_guidance() -> None:
    error = RuntimeError("Cannot access gated repo for this model")

    assert is_model_access_error(error)
    message = model_access_message(error)
    assert "https://huggingface.co/CohereLabs/" in message
    assert "hf auth login" in message
    assert "HF_TOKEN" in message


def test_model_access_error_recognizes_wrapped_http_authorization() -> None:
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    response = httpx.Response(
        403,
        request=httpx.Request("GET", "https://huggingface.co/gated-model"),
    )
    denied = HfHubHTTPError("forbidden", response=response)
    try:
        raise RuntimeError("Transformers wrapped the download failure") from denied
    except RuntimeError as wrapped:
        assert is_model_access_error(wrapped)

    assert not is_model_access_error(RuntimeError("connection timed out"))


def test_module_launcher_initializes() -> None:
    completed = run_python("-m", "cohere_transcribe", "--help")
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout


def test_module_launcher_reports_version() -> None:
    completed = run_python("-m", "cohere_transcribe", "--version")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"cohere-transcribe {__version__}"


def test_package_import_has_no_cli_environment_side_effects() -> None:
    variables = (
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    )
    script = (
        "import os; "
        + "; ".join(f"os.environ.pop({name!r}, None)" for name in variables)
        + "; import cohere_transcribe; "
        + f"assert not any(name in os.environ for name in {variables!r})"
    )
    completed = run_python("-c", script)
    assert completed.returncode == 0, completed.stderr


def test_alignment_import_does_not_construct_uroman() -> None:
    script = (
        "import sys; import cohere_transcribe.alignment; "
        "assert 'cohere_transcribe.alignment.text_utils' "
        "not in sys.modules"
    )
    completed = run_python("-c", script)
    assert completed.returncode == 0, completed.stderr


def test_default_segment_imports_without_word_extra() -> None:
    script = (
        "import sys; "
        "sys.modules['torchaudio'] = None; sys.modules['uroman'] = None; "
        "from cohere_transcribe.config import parse_args, validate_args; "
        "args = parse_args(['input.wav']); validate_args(args); "
        "assert args.alignment == 'segment'"
    )
    completed = run_python("-c", script)
    assert completed.returncode == 0, completed.stderr


def test_doctor_does_not_retry_an_unimportable_onnx_runtime() -> None:
    results = doctor.Results()
    with (
        mock.patch.object(doctor.importlib.util, "find_spec", return_value=object()),
        mock.patch.object(doctor, "import_required", return_value=None) as importer,
        mock.patch(
            "cohere_transcribe.vad.vectorized_silero.VectorizedSileroVAD"
        ) as vectorized,
    ):
        doctor.validate_silero(results, torch=None)

    importer.assert_called_once_with(
        results, "onnxruntime", "optional ONNX Silero runtime"
    )
    vectorized.assert_not_called()


def test_doctor_does_not_report_a_broken_torchcodec_as_available() -> None:
    results = doctor.Results()
    with (
        mock.patch.object(
            doctor,
            "probe_torchcodec",
            return_value=TorchCodecStatus(False, "0.2.1", "version is too old"),
        ),
        mock.patch.object(doctor, "package_version", return_value=None),
        mock.patch.object(doctor.importlib.util, "find_spec", return_value=None),
        mock.patch.object(doctor.shutil, "which", return_value="/usr/bin/ffmpeg"),
    ):
        doctor.report_optional_runtime(results)

    assert results.failures == 0
    assert results.warnings == 2


def test_doctor_auto_fails_without_any_decoder() -> None:
    results = doctor.Results()
    with (
        mock.patch.object(
            doctor,
            "probe_torchcodec",
            return_value=TorchCodecStatus(False, None, "package is not installed"),
        ),
        mock.patch.object(doctor, "package_version", return_value=None),
        mock.patch.object(doctor.importlib.util, "find_spec", return_value=None),
        mock.patch.object(doctor.shutil, "which", return_value=None),
    ):
        doctor.report_optional_runtime(results, "auto")

    assert results.failures == 1
    assert results.warnings == 2


def test_doctor_explicit_torchcodec_requires_a_usable_version() -> None:
    results = doctor.Results()
    with (
        mock.patch.object(
            doctor,
            "probe_torchcodec",
            return_value=TorchCodecStatus(False, "0.2.1", "version is too old"),
        ),
        mock.patch.object(doctor, "package_version", return_value=None),
        mock.patch.object(doctor.importlib.util, "find_spec", return_value=None),
        mock.patch.object(doctor.shutil, "which") as which,
    ):
        doctor.report_optional_runtime(results, "torchcodec")

    assert results.failures == 1
    assert results.warnings == 1
    which.assert_not_called()


def test_doctor_explicit_librosa_does_not_require_ffmpeg() -> None:
    results = doctor.Results()

    def version(name: str) -> str | None:
        return "0.11.0" if name == "librosa" else None

    with (
        mock.patch.object(doctor, "package_version", side_effect=version),
        mock.patch.object(doctor, "import_required", return_value=object()) as importer,
        mock.patch.object(doctor.importlib.util, "find_spec", return_value=None),
        mock.patch.object(doctor.shutil, "which") as which,
        mock.patch.object(doctor, "probe_torchcodec") as torchcodec_probe,
    ):
        doctor.report_optional_runtime(results, "librosa")

    assert results.failures == 0
    assert results.warnings == 1
    importer.assert_called_once_with(results, "librosa", "Librosa audio decoder")
    which.assert_not_called()
    torchcodec_probe.assert_not_called()
