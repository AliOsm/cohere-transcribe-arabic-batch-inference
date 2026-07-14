from dataclasses import FrozenInstanceError

import pytest

from cohere_transcribe import (
    BatchTranscriptionError,
    TranscriptionOptions,
    transcribe,
)

from ._support import patch_execute, result, run_for


def test_transcription_run_is_sequence_like_deeply_immutable_and_classified() -> None:
    completed = result("done.wav")
    failed = result("failed.wav", status="failed", text="", error="decode failed")
    skipped = result("skipped.wav", status="skipped", text=None)
    options = TranscriptionOptions()
    run = run_for(options, completed, failed, skipped)

    assert len(run) == 3
    assert list(run) == [completed, failed, skipped]
    assert run[0] is completed
    assert run[1:] == (failed, skipped)
    assert run.successful == (completed,)
    assert run.failed == (failed,)
    assert run.skipped == (skipped,)
    assert not run.ok
    with pytest.raises(FrozenInstanceError):
        run.errors = ("mutated",)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        completed.text = "mutated"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        options.language = "en"  # type: ignore[misc]


def test_single_returns_exactly_one_expanded_result() -> None:
    only = result("only.wav")
    assert run_for(TranscriptionOptions(), only).single is only
    with pytest.raises(ValueError, match="found 0"):
        _ = run_for(TranscriptionOptions()).single
    with pytest.raises(ValueError, match="found 2"):
        _ = run_for(TranscriptionOptions(), only, result("second.wav")).single


def test_result_statuses_and_raise_on_error_preserve_the_partial_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = result("done.wav")
    failed = result("failed.wav", status="failed", error="bad input")
    skipped = result("skipped.wav", status="skipped", text=None)

    def fake_execute(_args, requested_options, **_kwargs):
        return run_for(requested_options, done, failed, skipped)

    patch_execute(monkeypatch, fake_execute)
    partial = transcribe(["done.wav", "failed.wav", "skipped.wav"])
    assert partial.successful == (done,)
    assert partial.failed == (failed,)
    assert partial.skipped == (skipped,)

    with pytest.raises(BatchTranscriptionError) as raised:
        transcribe(["done.wav", "failed.wav", "skipped.wav"], raise_on_error=True)
    assert raised.value.run.results == (done, failed, skipped)


def test_batch_error_reports_run_level_errors_when_no_file_failed() -> None:
    run = run_for(
        TranscriptionOptions(),
        result("completed.wav"),
        errors=("profile output failed: disk full",),
    )

    error = BatchTranscriptionError(run)

    assert error.run is run
    assert "run error" in str(error).lower()
    assert "0 transcription file" not in str(error)
