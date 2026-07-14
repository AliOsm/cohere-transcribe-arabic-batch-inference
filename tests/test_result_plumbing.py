from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.inputs import build_jobs
from cohere_transcribe.models import AudioJob, RunStats, SourceSnapshot
from cohere_transcribe.output.pipeline import write_segment_timed_outputs
from cohere_transcribe.output.publication import complete_job_result
from cohere_transcribe.pipeline.transcription import finalize_completed_asr_jobs
from cohere_transcribe.state import release_output_locks


def make_config(
    source: Path,
    *,
    output_dir: Path | None = None,
    existing: str = "overwrite",
):
    argv = [
        os.fspath(source),
        "--vad",
        "none",
        "--alignment",
        "segment",
        "--formats",
        "txt",
        "json",
        "--existing",
        existing,
    ]
    if output_dir is not None:
        argv.extend(("--output-dir", os.fspath(output_dir)))
    args = parse_args(argv)
    validate_args(args)
    return args


def populate_result(job: AudioJob) -> None:
    job.duration = 1.0
    job.segment_times = [(0.0, 1.0)]
    job.speech_spans = [(0.0, 1.0)]
    job.segment_texts = ["test transcript"]
    job.decode_backend = "test"
    job.vad_engine_actual = "none"


def test_in_memory_jobs_create_no_publication_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    output_dir = tmp_path / "must-not-exist"
    args = make_config(source, output_dir=output_dir)

    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        jobs = build_jobs(
            args,
            publication_enabled=False,
            capture_results=True,
            retain_skipped=True,
        )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.capture_result
    assert job.output_paths == {}
    assert job.state_path is None
    assert job.checkpoint_path is None
    assert job.output_lock is None
    assert job.duration_hint == 1.0
    assert not output_dir.exists()
    assert list(tmp_path.glob(".*cohere-transcribe*")) == []


def test_in_memory_completion_captures_a_detached_payload(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    args = make_config(source)
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="segment",
        capture_result=True,
    )
    populate_result(job)

    write_segment_timed_outputs([job], args, publish_outputs=False)

    assert job.result_completed
    assert not job.published
    assert job.result_payload is not None
    assert job.result_payload["transcript"] == ["test transcript"]
    assert job.result_payload["segments"] == [
        {
            "segment_index": 0,
            "start": 0.0,
            "end": 1.0,
            "text": "test transcript",
        }
    ]
    job.segment_texts[0] = "mutated"
    assert job.result_payload["transcript"] == ["test transcript"]


def test_in_memory_completion_rejects_a_changed_source(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"before")
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="none",
        capture_result=True,
    )
    source.write_bytes(b"after source mutation")

    with pytest.raises(RuntimeError, match="Source changed while processing"):
        complete_job_result(job, [], publish_outputs=False)
    assert not job.result_completed
    assert job.result_payload is None


def test_no_publication_skips_checkpoint_but_completes_result(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    args = make_config(source)
    checkpoint = tmp_path / "must-not-exist.checkpoint.json"
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="segment",
        checkpoint_path=checkpoint,
        capture_result=True,
    )
    populate_result(job)
    stats = RunStats()

    finalize_completed_asr_jobs([job], args, stats, publish_outputs=False)

    assert not checkpoint.exists()
    assert stats.asr_checkpoint_written_files == 0
    assert job.result_completed
    assert job.result_payload is not None


def test_verified_skip_can_be_retained_as_an_ordered_result(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    output_dir = tmp_path / "out"
    overwrite = make_config(source, output_dir=output_dir)
    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        original = build_jobs(overwrite)[0]
    try:
        populate_result(original)
        complete_job_result(original, [], transcript_lines=original.segment_texts)
    finally:
        release_output_locks([original])

    skip = make_config(source, output_dir=output_dir, existing="skip")
    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        retained = build_jobs(skip, capture_results=True, retain_skipped=True)

    assert len(retained) == 1
    skipped = retained[0]
    assert skipped.index == 0
    assert skipped.skipped
    assert skipped.published
    assert skipped.result_completed
    assert skipped.output_lock is None
    assert skipped.written == list(skipped.output_paths.values())
    assert json.loads(skipped.output_paths["json"].read_text(encoding="utf-8"))[
        "transcript"
    ] == ["test transcript"]


def test_publication_and_capture_share_one_equivalent_result(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    args = make_config(source, output_dir=tmp_path / "out")
    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        job = build_jobs(args, capture_results=True)[0]
    try:
        populate_result(job)
        complete_job_result(job, [], transcript_lines=job.segment_texts)
        assert job.result_payload is not None
        on_disk = json.loads(job.output_paths["json"].read_text(encoding="utf-8"))
        assert job.result_payload == {
            key: on_disk[key] for key in ("transcript", "segments", "words", "cues")
        }

        job.segment_texts[0] = "later mutation"
        assert job.result_payload["transcript"] == ["test transcript"]
    finally:
        release_output_locks([job])
