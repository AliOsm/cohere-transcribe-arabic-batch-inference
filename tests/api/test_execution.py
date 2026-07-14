from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from cohere_transcribe import (
    PublicationOptions,
    TranscriptionConfigurationError,
    TranscriptionOptions,
    TranscriptionRun,
    TranscriptionRuntimeError,
    transcribe,
)

from ._support import patch_execute, run_for


def test_publication_none_is_in_memory_and_publication_options_are_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[bool, list[str], str | None, str, str | None]] = []

    def fake_execute(args, requested_options, *, publication_enabled, **_kwargs):
        seen.append(
            (
                publication_enabled,
                args.formats,
                args.output_dir,
                args.existing,
                args.profile_json,
            )
        )
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    transcribe("memory.wav")
    transcribe(
        "published.wav",
        options=TranscriptionOptions(
            publication=PublicationOptions(
                formats=("txt", "json"),
                output_dir=tmp_path / "out",
                existing="overwrite",
                profile_json=tmp_path / "profile.json",
            )
        ),
    )

    assert seen == [
        (False, ["txt", "srt", "vtt"], None, "error", None),
        (
            True,
            ["txt", "json"],
            os.fspath(tmp_path / "out"),
            "overwrite",
            os.fspath(tmp_path / "profile.json"),
        ),
    ]


def test_public_in_memory_execution_returns_text_without_creating_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.runtime.engine as runtime
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")

    def fake_transcribe(jobs, args, *_args, publish_outputs, **_kwargs):
        assert not publish_outputs
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["captured text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=False)

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )

    run = transcribe(
        source,
        options=TranscriptionOptions(
            vad="none",
            audio_backend="librosa",
            alignment="segment",
        ),
    )

    assert run.single.status == "completed"
    assert run.single.text == "captured text"
    assert run.single.outputs == ()
    assert not run.single.provenance.published
    assert {path.name for path in tmp_path.iterdir()} == {"clip.wav"}


def test_publication_writes_outputs_and_verified_skip_does_not_run_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.runtime.engine as runtime
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    output_dir = tmp_path / "out"
    calls = 0

    def fake_transcribe(jobs, args, *_args, publish_outputs, **_kwargs):
        nonlocal calls
        calls += 1
        assert publish_outputs
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["published text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=True)

    monkeypatch.setattr(
        runtime,
        "_resolve_precision",
        lambda args: (
            "cpu",
            "fp32",
            torch.float32,
            torch.float32,
            args.dtype,
            args.vad_engine,
        ),
    )
    monkeypatch.setattr(runtime.inputs_module, "probe_duration", lambda _path: 1.0)
    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )
    base_publication = PublicationOptions(
        formats=("txt", "json"),
        output_dir=output_dir,
        existing="overwrite",
    )
    first = transcribe(
        source,
        options=TranscriptionOptions(
            vad="none",
            audio_backend="librosa",
            publication=base_publication,
        ),
    )
    second = transcribe(
        source,
        options=TranscriptionOptions(
            vad="none",
            audio_backend="librosa",
            publication=replace(base_publication, existing="skip"),
        ),
    )

    assert calls == 1
    assert first.single.status == "completed"
    assert first.single.provenance.published
    assert {path.name for path in first.single.outputs} == {"clip.txt", "clip.json"}
    assert second.single.status == "skipped"
    assert second.single.provenance.published
    assert {path.name for path in second.single.outputs} == {"clip.txt", "clip.json"}


def test_configuration_system_exit_is_exposed_as_a_typed_api_error() -> None:
    invalid = TranscriptionOptions(language=cast(Any, "fr"))
    with pytest.raises(TranscriptionConfigurationError, match="language"):
        transcribe("input.wav", options=invalid)


def test_runtime_system_exit_is_exposed_as_a_typed_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_execute(*_args, **_kwargs):
        raise SystemExit("backend initialization failed")

    patch_execute(monkeypatch, fake_execute)
    with pytest.raises(
        TranscriptionRuntimeError, match="backend initialization failed"
    ):
        transcribe("input.wav")


def test_invalid_publication_object_is_exposed_as_a_configuration_error() -> None:
    options = TranscriptionOptions(publication=cast(Any, object()))
    with pytest.raises(TranscriptionConfigurationError):
        transcribe("input.wav", options=options)


def test_in_memory_api_does_not_initialize_filesystem_output_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    def fail_output_mode() -> int:
        raise AssertionError("in-memory API must not initialize publication state")

    def fake_execute(
        _args, requested_options, *, publication_enabled: bool, **_kwargs
    ) -> TranscriptionRun:
        assert not publication_enabled
        return run_for(requested_options)

    monkeypatch.setattr(runtime, "default_output_mode", fail_output_mode)
    monkeypatch.setattr(runtime, "execute", fake_execute)

    assert transcribe("memory.wav").ok
