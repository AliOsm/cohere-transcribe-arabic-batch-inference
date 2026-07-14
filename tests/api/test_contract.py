from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

import cohere_transcribe
from cohere_transcribe import (
    PublicationOptions,
    Transcriber,
    TranscriptionInputError,
    TranscriptionOptions,
    transcribe,
)
from cohere_transcribe.api.input import normalize_audio_input
from cohere_transcribe.config import (
    config_from_options,
    options_from_config,
    parse_args,
    validate_args,
)
from cohere_transcribe.inputs import expand_inputs
from cohere_transcribe.models import TranscriptionConfig

from ._support import PROJECT_ROOT, patch_execute, result, run_for


def test_root_import_is_dependency_light_and_exports_the_documented_api() -> None:
    script = """
import sys
import cohere_transcribe as package
heavy = {'torch', 'numpy', 'transformers', 'librosa', 'torchcodec', 'torchaudio'}
assert not heavy.intersection(sys.modules), sorted(heavy.intersection(sys.modules))
for name in (
    'transcribe', 'Transcriber', 'TranscriptionOptions', 'PublicationOptions',
    'TranscriptionRun', 'TranscriptionResult', 'TranscriptionError',
):
    assert hasattr(package, name), name
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    # Isolated mode ignores PYTHONPATH, so retry with an explicit source insertion.
    if completed.returncode:
        script = f"sys.path.insert(0, {os.fspath(PROJECT_ROOT / 'src')!r});\n" + script
        completed = subprocess.run(
            [sys.executable, "-I", "-c", "import sys\n" + script],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    assert completed.returncode == 0, completed.stderr


def test_public_exports_are_explicit_and_resolve_to_the_public_types() -> None:
    assert cohere_transcribe.transcribe is transcribe
    assert cohere_transcribe.Transcriber is Transcriber
    assert set(cohere_transcribe.__all__) >= {
        "transcribe",
        "Transcriber",
        "TranscriptionOptions",
        "PublicationOptions",
        "TranscriptionRun",
        "TranscriptionResult",
    }


def test_api_options_cover_every_non_positional_cli_configuration_field() -> None:
    api_fields = {item.name for item in fields(TranscriptionOptions)} - {"publication"}
    internal_fields = {item.name for item in fields(TranscriptionConfig)} - {
        "audio",
        "formats",
        "output_dir",
        "existing",
        "profile_json",
    }
    assert api_fields == internal_fields


def test_default_api_and_cli_configuration_are_behaviorally_identical() -> None:
    cli_config = parse_args(["input.wav"])
    validate_args(cli_config)
    api_config = config_from_options(
        ["input.wav"],
        TranscriptionOptions(publication=PublicationOptions()),
    )
    validate_args(api_config)
    assert asdict(api_config) == asdict(cli_config)


def test_options_round_trip_all_publication_and_runtime_values(tmp_path: Path) -> None:
    options = TranscriptionOptions(
        language="en",
        recursive=False,
        device="cpu",
        dtype="fp32",
        audio_backend="librosa",
        audio_memory_gb=2.5,
        preprocess_workers=3,
        pipeline_preparation=False,
        vad="none",
        vad_engine="jit",
        vad_batch_size=7,
        vad_block_frames=64,
        vad_threads=None,
        min_dur=0.2,
        max_dur=20.0,
        batch_size=3,
        batch_audio_seconds=50.0,
        batch_vram_target=0.75,
        pin_memory=True,
        max_new_tokens=300,
        max_retry_tokens=500,
        truncation_policy="warn",
        stop_repetition_loops=False,
        alignment="segment",
        align_batch_size=2,
        max_chars=60,
        max_cue_dur=4.0,
        max_gap=0.4,
        publication=PublicationOptions(
            formats=("json", "txt", "json"),
            output_dir=tmp_path / "out",
            existing="overwrite",
            profile_json=tmp_path / "profile.json",
        ),
    )
    config = config_from_options(["input.wav"], options)
    validate_args(config)
    resolved = options_from_config(config, publication_enabled=True)

    assert resolved == replace(
        options,
        publication=PublicationOptions(
            formats=("json", "txt"),
            output_dir=tmp_path / "out",
            existing="overwrite",
            profile_json=tmp_path / "profile.json",
        ),
    )


@pytest.mark.parametrize(
    ("audio", "expected"),
    [
        ("first.wav", ["first.wav"]),
        (Path("first.wav"), ["first.wav"]),
        (["first.wav", Path("second.mp3")], ["first.wav", "second.mp3"]),
        ((Path("first.wav"), "second.mp3"), ["first.wav", "second.mp3"]),
    ],
)
def test_audio_input_normalization_preserves_order(audio, expected) -> None:
    assert normalize_audio_input(audio) == expected


@pytest.mark.parametrize(
    "audio",
    [
        "",
        "   ",
        [],
        (),
        b"audio.wav",
        ["audio.wav", ""],
        ["audio.wav", b"other.wav"],
        ["audio.wav", object()],
        42,
        {"audio.wav"},
    ],
)
def test_audio_input_normalization_rejects_ambiguous_or_invalid_values(audio) -> None:
    with pytest.raises(TranscriptionInputError):
        normalize_audio_input(cast(Any, audio))


def test_public_api_passes_single_and_multiple_paths_to_one_shared_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_execute(args, requested_options, **_kwargs):
        seen.append(args.audio)
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    with Transcriber() as session:
        session.transcribe("one.wav")
        session.transcribe(["two.wav", Path("three.wav")])

    assert seen == [["one.wav"], ["two.wav", "three.wav"]]


def test_public_directory_input_expands_recursively_and_in_stable_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audio"
    (root / "nested").mkdir(parents=True)
    (root / "z.wav").write_bytes(b"z")
    (root / "nested" / "a.mp3").write_bytes(b"a")
    (root / "nested" / "ignored.txt").write_text("no", encoding="utf-8")

    def fake_execute(args, requested_options, **_kwargs):
        expanded = expand_inputs(args.audio, args.recursive)
        results = tuple(
            result(os.fspath(path), text=os.fspath(relative))
            for path, relative in expanded
        )
        return run_for(requested_options, *results)

    patch_execute(monkeypatch, fake_execute)
    run = transcribe(root)

    assert [item.text for item in run] == ["nested/a.mp3", "z.wav"]
    with pytest.raises(ValueError, match="exactly one"):
        _ = run.single


def test_non_recursive_directory_input_excludes_nested_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "audio"
    (root / "nested").mkdir(parents=True)
    (root / "top.wav").write_bytes(b"top")
    (root / "nested" / "inside.wav").write_bytes(b"inside")

    def fake_execute(args, requested_options, **_kwargs):
        expanded = expand_inputs(args.audio, args.recursive)
        return run_for(
            requested_options,
            *(result(os.fspath(path)) for path, _relative in expanded),
        )

    patch_execute(monkeypatch, fake_execute)
    run = transcribe(root, options=TranscriptionOptions(recursive=False))
    assert [item.path.name for item in run] == ["top.wav"]
