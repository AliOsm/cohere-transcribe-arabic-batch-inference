#!/usr/bin/env python3
"""Validate the local transcription runtime without loading the 2B model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "transcribe.py"
ASSET = ROOT / "transcribe_assets" / "silero_vad_v6.onnx"
EXPECTED_SCRIPT_SHA256 = (
    "fe781182519f9afa2e60b76afc5f82409fd6a98b799989109f3a38f696f4bd54"
)
EXPECTED_ASSET_SHA256 = (
    "914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9"
)
ASR_MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
ASR_REVISION = "0a8193caa4f3f92131471ab08824e488141cb392"
ALIGN_MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"
ALIGN_REVISION = "49402e9577b1158620820667c218cd494cc44486"
ALIGN_PACKAGE_REPOSITORY = "https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
ALIGN_PACKAGE_REVISION = "c344f5bc900323aa434a7cb200b7c629d463bd02"
ALIGN_PACKAGE_VERSION = "0.3.0"
UROMAN_VERSION = "1.3.1.1"
ALIGN_VOCABULARY = (
    "<blank>",
    "<pad>",
    "</s>",
    "<unk>",
    "a",
    "i",
    "e",
    "n",
    "o",
    "u",
    "t",
    "s",
    "r",
    "m",
    "k",
    "l",
    "d",
    "g",
    "h",
    "y",
    "b",
    "p",
    "w",
    "c",
    "v",
    "j",
    "z",
    "f",
    "'",
    "q",
    "x",
)


class Results:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_alignment_provenance(results: Results) -> None:
    try:
        distribution = importlib.metadata.distribution("ctc-forced-aligner")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError) as exc:
        results.fail(f"alignment package provenance: {type(exc).__name__}: {exc}")
        return

    if distribution.version != ALIGN_PACKAGE_VERSION:
        results.fail(
            f"alignment package version: expected {ALIGN_PACKAGE_VERSION}, "
            f"found {distribution.version}"
        )
        return
    if not isinstance(direct_url, dict):
        results.fail("alignment package has no PEP 610 direct Git provenance")
        return
    vcs_info = direct_url.get("vcs_info")
    repository = str(direct_url.get("url", "")).rstrip("/")
    if (
        repository != ALIGN_PACKAGE_REPOSITORY.rstrip("/")
        or not isinstance(vcs_info, dict)
        or vcs_info.get("vcs") != "git"
        or vcs_info.get("commit_id") != ALIGN_PACKAGE_REVISION
    ):
        results.fail(
            "alignment package provenance does not match the evaluated official "
            f"revision {ALIGN_PACKAGE_REVISION}"
        )
        return
    results.ok(
        f"official alignment package {distribution.version} at {ALIGN_PACKAGE_REVISION}"
    )


def import_required(results: Results, module: str, feature: str):
    try:
        imported = importlib.import_module(module)
    except Exception as exc:
        results.fail(
            f"{feature}: cannot import {module!r}: {type(exc).__name__}: {exc}"
        )
        return None
    results.ok(f"{feature}: {module}")
    return imported


def release_pair(version: str) -> tuple[int, int] | None:
    text = version.split("+", 1)[0].split(".")
    if len(text) < 2:
        return None
    try:
        return int(text[0]), int(text[1])
    except ValueError:
        return None


def validate_files(results: Results) -> None:
    if not SCRIPT.is_file():
        results.fail(f"missing script: {SCRIPT}")
    else:
        digest = sha256(SCRIPT)
        if digest == EXPECTED_SCRIPT_SHA256:
            results.ok(f"script integrity: {digest}")
        else:
            results.warn(
                "script differs from the production snapshot: "
                f"expected {EXPECTED_SCRIPT_SHA256}, found {digest}"
            )

    if not ASSET.is_file():
        results.fail(f"missing Silero asset: {ASSET}")
    else:
        digest = sha256(ASSET)
        if digest == EXPECTED_ASSET_SHA256:
            results.ok(f"Silero asset integrity: {digest}")
        else:
            results.fail(
                f"Silero asset checksum mismatch: expected {EXPECTED_ASSET_SHA256}, "
                f"found {digest}"
            )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        results.ok("script imports and CLI parser initializes")
    else:
        results.fail(f"script --help failed: {completed.stderr.strip()}")


def validate_common_runtime(results: Results):
    if sys.version_info < (3, 10):
        results.fail(
            f"Python 3.10 or newer is required; found {sys.version.split()[0]}"
        )
    else:
        results.ok(f"Python {sys.version.split()[0]}")

    torch = import_required(results, "torch", "PyTorch")
    import_required(results, "transformers", "Cohere ASR runtime")
    import_required(results, "accelerate", "model loading")
    import_required(results, "sentencepiece", "model tokenizer")
    import_required(results, "google.protobuf", "processor serialization")
    import_required(results, "packaging", "version validation")
    import_required(results, "numpy", "numeric runtime")
    import_required(results, "soundfile", "audio I/O")
    import_required(results, "librosa", "portable audio decoder")
    import_required(results, "tqdm", "progress display")

    version = distribution_version("transformers")
    if version is not None:
        from packaging.version import Version

        if Version("5.13") <= Version(version) < Version("5.14"):
            results.ok(f"Transformers compatibility range: {version}")
        else:
            results.fail(
                f"Transformers {version} is outside the validated >=5.13,<5.14 range"
            )

    if torch is None:
        return None
    if torch.cuda.is_available():
        results.ok(
            f"accelerator: CUDA device {torch.cuda.current_device()} - "
            f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        results.ok("accelerator: Apple MPS")
    else:
        results.warn("accelerator: CPU only; the 2B model will be substantially slower")
    return torch


def validate_silero(results: Results) -> None:
    import_required(results, "silero_vad", "Silero TorchScript fallback")
    import_required(results, "onnxruntime", "vectorized Silero runtime")
    try:
        import numpy as np

        sys.path.insert(0, str(ROOT))
        from transcribe_assets.vectorized_silero import VectorizedSileroVAD

        probabilities = VectorizedSileroVAD().speech_probabilities(
            np.zeros(1024, dtype=np.float32)
        )
        if probabilities.shape != (2,) or not all(
            math.isfinite(float(value)) for value in probabilities
        ):
            raise RuntimeError(f"unexpected probability output {probabilities!r}")
    except Exception as exc:
        results.fail(f"bundled Silero ONNX smoke test: {type(exc).__name__}: {exc}")
    else:
        results.ok("bundled Silero ONNX model executes on CPU")


def validate_word_alignment(results: Results, torch) -> None:
    torchaudio = import_required(results, "torchaudio", "word alignment")
    aligner = import_required(results, "ctc_forced_aligner", "alignment utilities")
    import_required(results, "uroman", "Arabic alignment romanization")
    validate_alignment_provenance(results)
    if distribution_version("uroman") != UROMAN_VERSION:
        results.fail(
            f"Uroman version: expected {UROMAN_VERSION}, "
            f"found {distribution_version('uroman') or 'missing'}"
        )
    else:
        results.ok(f"Uroman version: {UROMAN_VERSION}")
    if aligner is not None:
        required_symbols = (
            "merge_repeats",
            "get_spans",
            "preprocess_text",
            "postprocess_results",
        )
        missing = [name for name in required_symbols if not hasattr(aligner, name)]
        if missing:
            results.fail(f"alignment utility exports are missing: {missing}")
        else:
            try:
                tokens, _ = aligner.preprocess_text(
                    "مرحبا بكم في العالم", romanize=True, language="ara"
                )
                if tokens[-1] != "a l ' a l m":
                    raise RuntimeError(f"unexpected Uroman tokens: {tokens!r}")
            except Exception as exc:
                results.fail(
                    f"official Arabic romanization smoke test: {type(exc).__name__}: {exc}"
                )
            else:
                results.ok("official Arabic Uroman path executes")
    if torch is None or torchaudio is None:
        return

    torch_pair = release_pair(torch.__version__)
    audio_pair = release_pair(torchaudio.__version__)
    if torch_pair != audio_pair:
        results.fail(
            f"torch {torch.__version__} and torchaudio {torchaudio.__version__} "
            "must have matching major/minor releases"
        )
        return
    results.ok(
        f"matched torch/torchaudio releases: {torch.__version__} / "
        f"{torchaudio.__version__}"
    )
    try:
        from torchaudio.functional import forced_align

        emissions = torch.log_softmax(
            torch.tensor([[[4.0, 0.0], [0.0, 4.0]]], dtype=torch.float32), dim=-1
        )
        path, scores = forced_align(
            emissions, torch.tensor([[1]], dtype=torch.int64), blank=0
        )
        if path.shape != (1, 2) or scores.shape != (1, 2):
            raise RuntimeError(f"unexpected output shapes {path.shape}, {scores.shape}")
    except Exception as exc:
        results.fail(f"TorchAudio forced-align smoke test: {type(exc).__name__}: {exc}")
    else:
        results.ok("TorchAudio forced-align operation executes")


def report_optional_runtime(results: Results) -> None:
    if importlib.util.find_spec("torchcodec") is None:
        results.warn("TorchCodec is not installed; auto decoding uses librosa/FFmpeg")
    else:
        results.ok(
            f"optional TorchCodec {distribution_version('torchcodec') or 'installed'}"
        )
    if importlib.util.find_spec("auditok") is None:
        results.warn("Auditok is not installed; --vad auditok is unavailable")
    else:
        results.ok(f"optional Auditok {distribution_version('auditok') or 'installed'}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        results.warn(
            "FFmpeg is not on PATH; librosa/TorchCodec must decode every input"
        )
    else:
        results.ok(f"FFmpeg executable: {ffmpeg}")


def validate_model_access(results: Results, include_aligner: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        processor = AutoProcessor.from_pretrained(ASR_MODEL_ID, revision=ASR_REVISION)
        maximum = getattr(processor.feature_extractor, "max_audio_clip_s", None)
        if maximum is None:
            raise RuntimeError("processor does not expose max_audio_clip_s")
        hf_hub_download(ASR_MODEL_ID, "config.json", revision=ASR_REVISION)
        if include_aligner:
            aligner_config = AutoConfig.from_pretrained(
                ALIGN_MODEL_ID, revision=ALIGN_REVISION
            )
            aligner_tokenizer = AutoTokenizer.from_pretrained(
                ALIGN_MODEL_ID,
                revision=ALIGN_REVISION,
                word_delimiter_token=None,
            )
            expected_vocabulary = {
                token: index for index, token in enumerate(ALIGN_VOCABULARY)
            }
            if aligner_tokenizer.get_vocab() != expected_vocabulary:
                raise RuntimeError("pinned aligner tokenizer vocabulary changed")
            if aligner_tokenizer.pad_token_id != 1:
                raise RuntimeError("pinned aligner tokenizer pad ID changed")
            if getattr(aligner_config, "inputs_to_logits_ratio", None) != 320:
                raise RuntimeError("pinned aligner input stride changed")
    except Exception as exc:
        results.fail(f"pinned model access: {type(exc).__name__}: {exc}")
    else:
        suffix = " and aligner" if include_aligner else ""
        results.ok(
            f"pinned ASR processor{suffix} accessible; one-row limit is {maximum}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the production transcription bundle without loading model weights."
    )
    parser.add_argument(
        "--mode",
        choices=("word", "segment", "text"),
        default="word",
        help="Validate dependencies for this output mode (default: word).",
    )
    parser.add_argument(
        "--model-access",
        action="store_true",
        help="Also contact Hugging Face and download the small pinned processor/config files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = Results()
    validate_files(results)
    torch = validate_common_runtime(results)
    validate_silero(results)
    if args.mode == "word":
        validate_word_alignment(results, torch)
    report_optional_runtime(results)
    if args.model_access:
        validate_model_access(results, include_aligner=args.mode == "word")

    print()
    if results.failures:
        print(
            f"Validation failed: {results.failures} failure(s), "
            f"{results.warnings} warning(s)."
        )
        return 1
    print(f"Validation passed for {args.mode} mode with {results.warnings} warning(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
