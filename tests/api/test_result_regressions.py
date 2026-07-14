from dataclasses import fields
from pathlib import Path

import pytest

from cohere_transcribe import TranscriptionOptions, TranscriptionStatistics
from cohere_transcribe.models import AudioJob, RunStats, SourceSnapshot
from cohere_transcribe.runtime.results import _statistics


def test_public_statistics_name_and_value_are_explicitly_rtfx(tmp_path: Path) -> None:
    public_fields = {field.name for field in fields(TranscriptionStatistics)}
    assert "real_time_factor_x" in public_fields
    assert "real_time_factor" not in public_fields

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=10.0,
        language="ar",
        vad_mode="none",
        alignment_mode="none",
        duration=10.0,
    )

    statistics = _statistics(RunStats(), elapsed=2.0, jobs=[job])
    assert statistics.real_time_factor_x == pytest.approx(5.0)


def test_failed_unprepared_result_is_distinct_from_completed_empty_transcript(
    tmp_path: Path,
) -> None:
    from cohere_transcribe.config import config_from_options, validate_args
    from cohere_transcribe.runtime.results import build_run

    failed_path = tmp_path / "failed.wav"
    empty_path = tmp_path / "empty.wav"
    failed_path.write_bytes(b"failed")
    empty_path.write_bytes(b"empty")
    failed = AudioJob(
        index=0,
        path=failed_path,
        relative_path=Path(failed_path.name),
        snapshot=SourceSnapshot.capture(failed_path),
        duration_hint=3.25,
        language="ar",
        vad_mode="none",
        alignment_mode="none",
        error="audio preparation failed",
    )
    completed_empty = AudioJob(
        index=1,
        path=empty_path,
        relative_path=Path(empty_path.name),
        snapshot=SourceSnapshot.capture(empty_path),
        duration_hint=2.0,
        language="ar",
        vad_mode="none",
        alignment_mode="none",
        duration=2.0,
        result_completed=True,
        result_payload={"transcript": [], "segments": [], "words": [], "cues": []},
    )
    options = TranscriptionOptions(vad="none", alignment="none")
    args = config_from_options(
        [str(failed_path), str(empty_path)],
        options,
    )
    validate_args(args)

    run = build_run(
        [failed, completed_empty],
        options,
        args,
        RunStats(),
        elapsed=1.0,
    )

    assert run[0].status == "failed"
    assert run[0].text is None
    assert run[0].duration == pytest.approx(3.25)
    assert run[1].status == "completed"
    assert run[1].text == ""
    assert run[1].duration == pytest.approx(2.0)
