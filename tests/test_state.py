from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

import pytest

from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.inputs import build_jobs
from cohere_transcribe.models import (
    AudioJob,
    RunStats,
    SourceSnapshot,
    TranscriptionConfig,
)
from cohere_transcribe.output.publication import atomic_write_outputs
from cohere_transcribe.pipeline.transcription import finalize_completed_asr_jobs
from cohere_transcribe.state import (
    OutputLockTarget,
    OutputSetLock,
    asr_checkpoint_payload,
    asr_contract_key,
    create_state_temporary,
    lock_target_for_outputs,
    release_output_locks,
    render_contract_key,
    restore_asr_checkpoint,
    write_asr_checkpoint,
)


def make_config(
    source: Path,
    output_dir: Path,
    *,
    alignment: str = "segment",
    formats: tuple[str, ...] = ("txt", "srt"),
    existing: str = "overwrite",
    language: str = "ar",
) -> TranscriptionConfig:
    args = parse_args(
        [
            os.fspath(source),
            "--output-dir",
            os.fspath(output_dir),
            "--alignment",
            alignment,
            "--formats",
            *formats,
            "--existing",
            existing,
            "--device",
            "cpu",
            "--dtype",
            "fp32",
            "--vad",
            "none",
            "--audio-backend",
            "ffmpeg",
            "--language",
            language,
        ]
    )
    validate_args(args)
    return args


def build_one(args: TranscriptionConfig) -> AudioJob:
    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        jobs = build_jobs(args)
    assert len(jobs) == 1
    return jobs[0]


def populate_asr(job: AudioJob, text: str = "transcript") -> None:
    job.duration = 1.0
    job.segment_times = [(0.0, 1.0)]
    job.speech_spans = [(0.0, 1.0)]
    job.segment_texts = [text]
    job.generated_tokens = {0: 3}
    job.decode_backend = "ffmpeg"
    job.vad_engine_actual = "none"


def publish(job: AudioJob) -> None:
    populate_asr(job)
    write_asr_checkpoint(job)
    atomic_write_outputs(
        job,
        [{"start": 0.0, "end": 1.0, "text": job.segment_texts[0]}],
    )


def test_asr_and_render_contracts_have_separate_semantics(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    base = make_config(source, tmp_path / "out")
    render_change = replace(
        base,
        alignment="word",
        formats=["txt", "json"],
        max_chars=37,
        max_gap=0.2,
    )

    assert asr_contract_key(base) == asr_contract_key(render_change)
    assert render_contract_key(base) != render_contract_key(render_change)
    assert asr_contract_key(base) != asr_contract_key(replace(base, dtype="fp16"))
    assert asr_contract_key(base) != asr_contract_key(
        replace(base, audio_memory_gb=base.audio_memory_gb / 2)
    )
    assert asr_contract_key(base) != asr_contract_key(
        replace(base, pipeline_preparation=not base.pipeline_preparation)
    )
    assert asr_contract_key(base) != asr_contract_key(
        replace(base, vad="silero", vad_engine="torch")
    )
    auto_silero = replace(base, vad="silero", vad_engine="auto")
    assert asr_contract_key(auto_silero) != asr_contract_key(
        replace(auto_silero, vad_engine="torch")
    )
    original_implementation_key = asr_contract_key(base)
    with mock.patch(
        "cohere_transcribe.state.contracts._implementation_fingerprint",
        return_value="changed-implementation",
    ):
        assert original_implementation_key != asr_contract_key(replace(base))


def test_checkpoint_binds_canonical_source_path(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = build_one(make_config(source, tmp_path / "out"))
    try:
        populate_asr(job)
        write_asr_checkpoint(job)
        assert job.checkpoint_path is not None
        envelope = json.loads(job.checkpoint_path.read_text(encoding="utf-8"))
        assert envelope["payload"]["source"] == {
            "canonical_path": os.fspath(source.resolve()),
            "snapshot": {
                "ctime_ns": job.snapshot.ctime_ns,
                "device": job.snapshot.device,
                "inode": job.snapshot.inode,
                "mtime_ns": job.snapshot.mtime_ns,
                "size": job.snapshot.size,
            },
        }

        payload = envelope["payload"]
        payload["source"]["canonical_path"] = os.fspath(tmp_path / "other.wav")
        temporary = create_state_temporary(job.checkpoint_path, payload)
        os.replace(temporary, job.checkpoint_path)
        restored, reason = restore_asr_checkpoint(job)
        assert not restored
        assert "source snapshot" in reason
    finally:
        release_output_locks([job])


def test_rejected_checkpoint_does_not_mutate_any_job_field(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = build_one(make_config(source, tmp_path / "out"))
    try:
        populate_asr(job, "checkpoint text")
        payload = asr_checkpoint_payload(job)
        checkpoint = payload["checkpoint"]
        assert isinstance(checkpoint, dict)
        checkpoint["token_limit_segments"] = [len(job.segment_times)]
        assert job.checkpoint_path is not None
        temporary = create_state_temporary(job.checkpoint_path, payload)
        os.replace(temporary, job.checkpoint_path)

        job.generation_id = ""
        job.duration = 0.25
        job.segment_times = [(0.0, 0.25)]
        job.speech_spans = []
        job.segment_texts = ["fresh state"]
        job.generated_tokens = {0: 7}
        job.repetition_stopped_segments = set()
        job.truncation_retried_segments = {0}
        job.token_limit_segments = set()
        job.decode_backend = "fresh-backend"
        job.decode_fallback_reason = "fresh-decode-reason"
        job.vad_engine_actual = "fresh-vad"
        job.vad_provider = "fresh-provider"
        job.vad_provider_options = {"fresh": {"option": "value"}}
        job.vad_fallback_reason = "fresh-vad-reason"

        before: dict[str, object] = {}
        for field in fields(AudioJob):
            value = getattr(job, field.name)
            before[field.name] = (
                value if field.name == "output_lock" else copy.deepcopy(value)
            )

        restored, reason = restore_asr_checkpoint(job)

        assert not restored
        assert "invalid" in reason
        for field in fields(AudioJob):
            actual = getattr(job, field.name)
            expected = before[field.name]
            if field.name == "output_lock":
                assert actual is expected
            else:
                assert actual == expected, field.name
    finally:
        release_output_locks([job])


def test_skip_rebuilds_after_same_size_mtime_preserving_source_rewrite(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"first")
    output_dir = tmp_path / "out"
    original = build_one(make_config(source, output_dir))
    before = original.snapshot
    original_stat = source.stat()
    try:
        publish(original)
    finally:
        release_output_locks([original])

    source.write_bytes(b"other")
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    after = SourceSnapshot.capture(source)
    assert after.device == before.device
    assert after.inode == before.inode
    assert after.size == before.size
    assert after.mtime_ns == before.mtime_ns
    assert after.ctime_ns != before.ctime_ns
    assert after != before

    rebuilt = build_one(make_config(source, output_dir, existing="skip"))
    try:
        assert not rebuilt.asr_checkpoint_loaded
    finally:
        release_output_locks([rebuilt])


@pytest.mark.parametrize("corruption", ["overlap", "duplicate_tokens"])
def test_checkpoint_rejects_ambiguous_segment_metadata(
    tmp_path: Path, corruption: str
) -> None:
    source = tmp_path / f"{corruption}.wav"
    source.write_bytes(b"audio")
    job = build_one(make_config(source, tmp_path / "out"))
    try:
        populate_asr(job)
        payload = asr_checkpoint_payload(job)
        checkpoint = payload["checkpoint"]
        assert isinstance(checkpoint, dict)
        if corruption == "overlap":
            checkpoint["segment_times"] = [[0.0, 0.7], [0.5, 1.0]]
            checkpoint["segment_texts"] = ["first", "second"]
        else:
            checkpoint["generated_tokens"] = [[0, 3], [0, 4]]
        assert job.checkpoint_path is not None
        temporary = create_state_temporary(job.checkpoint_path, payload)
        os.replace(temporary, job.checkpoint_path)

        restored, reason = restore_asr_checkpoint(job)
        assert not restored
        assert "invalid" in reason
    finally:
        release_output_locks([job])


def test_verified_skip_rejects_tampered_output_and_reuses_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    output_dir = tmp_path / "out"
    job = build_one(make_config(source, output_dir))
    try:
        publish(job)
        assert job.state_path is not None and job.state_path.is_file()
        assert job.checkpoint_path is not None and job.checkpoint_path.is_file()
    finally:
        release_output_locks([job])

    skip_args = make_config(source, output_dir, existing="skip")
    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        assert build_jobs(skip_args) == []

    (output_dir / "clip.txt").write_text("tampered\n", encoding="utf-8")
    rebuilt = build_one(skip_args)
    try:
        assert rebuilt.asr_checkpoint_loaded
        assert rebuilt.segment_texts == ["transcript"]
    finally:
        release_output_locks([rebuilt])


def test_render_changes_resume_asr_but_asr_changes_do_not(tmp_path: Path) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    output_dir = tmp_path / "out"
    original = build_one(make_config(source, output_dir))
    try:
        publish(original)
        original_asr_key = original.asr_contract_key
        original_render_key = original.render_contract_key
    finally:
        release_output_locks([original])

    word_args = make_config(
        source,
        output_dir,
        alignment="word",
        formats=("txt", "json"),
        existing="overwrite",
    )
    resumed = build_one(word_args)
    try:
        assert resumed.asr_checkpoint_loaded
        assert resumed.asr_contract_key == original_asr_key
        assert resumed.render_contract_key != original_render_key
        assert resumed.segment_times == [(0.0, 1.0)]
    finally:
        release_output_locks([resumed])

    changed = build_one(
        make_config(source, output_dir, existing="overwrite", language="en")
    )
    try:
        assert not changed.asr_checkpoint_loaded
        assert changed.asr_contract_key != original_asr_key
    finally:
        release_output_locks([changed])


def test_progressive_finalizer_publishes_and_retains_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    args = make_config(source, tmp_path / "out")
    job = build_one(args)
    stats = RunStats()
    try:
        populate_asr(job)
        finalize_completed_asr_jobs([job], args, stats)

        assert job.published
        assert (tmp_path / "out" / "clip.txt").read_text(encoding="utf-8") == (
            "transcript\n"
        )
        assert job.state_path is not None and job.state_path.is_file()
        assert job.checkpoint_path is not None and job.checkpoint_path.is_file()
        assert stats.asr_checkpoint_written_files == 1
    finally:
        release_output_locks([job])


def test_manifest_is_committed_last_and_failure_restores_generation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = build_one(make_config(source, tmp_path / "out"))
    try:
        publish(job)
        assert job.state_path is not None
        assert job.checkpoint_path is not None
        old_files = {
            path: path.read_bytes()
            for path in (*job.output_paths.values(), job.state_path)
        }
        checkpoint_bytes = job.checkpoint_path.read_bytes()
        populate_asr(job, "replacement")
        real_replace = os.replace
        destinations: list[Path] = []

        def fail_manifest(source_path, destination_path):
            destination = Path(destination_path)
            destinations.append(destination)
            if destination == job.state_path and Path(source_path).suffix == ".tmp":
                raise OSError("simulated manifest commit failure")
            return real_replace(source_path, destination_path)

        with (
            mock.patch(
                "cohere_transcribe.output.publication.os.replace",
                side_effect=fail_manifest,
            ),
            pytest.raises(OSError, match="manifest commit failure"),
        ):
            atomic_write_outputs(
                job,
                [{"start": 0.0, "end": 1.0, "text": "replacement"}],
            )

        assert destinations[:2] == list(job.output_paths.values())
        assert destinations[2] == job.state_path
        for path, content in old_files.items():
            assert path.read_bytes() == content
        assert job.checkpoint_path.read_bytes() == checkpoint_bytes
    finally:
        release_output_locks([job])


def _run_lock_child(target: OutputLockTarget) -> subprocess.CompletedProcess[str]:
    script = """
import sys
from pathlib import Path
from cohere_transcribe.state import OutputLockTarget, OutputSetLock

target = OutputLockTarget(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
try:
    lock = OutputSetLock.acquire(target)
except RuntimeError:
    raise SystemExit(23)
lock.release()
"""
    environment = os.environ.copy()
    source_root = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (os.fspath(source_root), environment.get("PYTHONPATH", "")))
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            os.fspath(target.path),
            str(target.offset),
            target.identity,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=10,
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX byte-range lock semantics")
def test_output_locks_contend_per_stem_across_processes(tmp_path: Path) -> None:
    first_target = lock_target_for_outputs({"txt": tmp_path / "first.txt"})
    second_target = lock_target_for_outputs(
        {"txt": tmp_path / "another-directory" / "second.txt"}
    )
    assert first_target.path == second_target.path
    assert first_target.offset != second_target.offset
    first_lock = OutputSetLock.acquire(first_target)
    try:
        assert _run_lock_child(first_target).returncode == 23
        assert _run_lock_child(second_target).returncode == 0
    finally:
        first_lock.release()

    assert first_target.path.is_file()
    reacquired = OutputSetLock.acquire(first_target)
    reacquired.release()


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux FD accounting")
def test_many_directory_locks_share_one_registry_fd(tmp_path: Path) -> None:
    targets = [
        lock_target_for_outputs({"txt": tmp_path / f"directory-{index}" / "clip.txt"})
        for index in range(1500)
    ]
    assert len({target.path for target in targets}) == 1
    before = len(list(Path("/proc/self/fd").iterdir()))
    locks = [
        OutputSetLock.acquire(target)
        for target in sorted(targets, key=lambda t: t.sort_key)
    ]
    try:
        during = len(list(Path("/proc/self/fd").iterdir()))
        assert during - before <= 2
    finally:
        for lock in reversed(locks):
            lock.release()
    after = len(list(Path("/proc/self/fd").iterdir()))
    assert after - before <= 1


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_lock_open_rejects_symlink_without_following_it(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    target = OutputLockTarget(registry / "outputs.lock", 1, "test output")
    victim = tmp_path / "victim"
    victim.write_text("do not change", encoding="utf-8")
    target.path.symlink_to(victim)

    with pytest.raises(RuntimeError, match="not a regular file"):
        OutputSetLock.acquire(target)
    assert victim.read_text(encoding="utf-8") == "do not change"


def test_same_process_cannot_acquire_one_output_twice(tmp_path: Path) -> None:
    target = lock_target_for_outputs({"txt": tmp_path / "clip.txt"})
    lock = OutputSetLock.acquire(target)
    try:
        with pytest.raises(RuntimeError, match="owns output set"):
            OutputSetLock.acquire(target)
    finally:
        lock.release()

    reacquired = OutputSetLock.acquire(target)
    reacquired.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_lock_registry_requires_private_directory_permissions(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o755)
    registry.chmod(0o755)
    target = OutputLockTarget(registry / "outputs.lock", 1, "test output")

    with pytest.raises(RuntimeError, match=r"permissions.*0700"):
        OutputSetLock.acquire(target)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_lock_registry_requires_private_file_permissions(tmp_path: Path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir(mode=0o700)
    path = registry / "outputs.lock"
    path.touch(mode=0o644)
    path.chmod(0o644)
    target = OutputLockTarget(path, 1, "test output")

    with pytest.raises(RuntimeError, match=r"permissions.*0600"):
        OutputSetLock.acquire(target)


@pytest.mark.skipif(os.name == "nt", reason="POSIX byte-range lock semantics")
def test_releasing_one_registry_range_keeps_other_ranges_locked(
    tmp_path: Path,
) -> None:
    first_target = lock_target_for_outputs({"txt": tmp_path / "first" / "clip.txt"})
    second_target = lock_target_for_outputs({"txt": tmp_path / "second" / "clip.txt"})
    first_lock = OutputSetLock.acquire(first_target)
    second_lock = OutputSetLock.acquire(second_target)
    try:
        first_lock.release()
        assert _run_lock_child(first_target).returncode == 0
        assert _run_lock_child(second_target).returncode == 23
    finally:
        first_lock.release()
        second_lock.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX byte-range lock semantics")
def test_planner_contention_releases_ranges_acquired_earlier(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "first.wav").write_bytes(b"")
    (source_root / "second.wav").write_bytes(b"")
    output_root = tmp_path / "output"
    output_root.mkdir()
    targets = sorted(
        (
            lock_target_for_outputs({"txt": output_root / "first.txt"}),
            lock_target_for_outputs({"txt": output_root / "second.txt"}),
        ),
        key=lambda target: target.sort_key,
    )
    blocker = OutputSetLock.acquire(targets[1])
    try:
        args = make_config(source_root, output_root, formats=("txt",))
        with (
            mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0),
            pytest.raises(SystemExit, match="owns output set"),
        ):
            build_jobs(args)
        assert _run_lock_child(targets[0]).returncode == 0
    finally:
        blocker.release()

    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=1.0):
        jobs = build_jobs(args)
    try:
        assert len(jobs) == 2
    finally:
        release_output_locks(jobs)


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX resource limits")
def test_nested_batch_planning_stays_within_a_low_fd_limit(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    for index in range(200):
        source = source_root / f"directory-{index:03d}" / "clip.wav"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"")

    script = """
import os
import resource
import sys
from cohere_transcribe import inputs
from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.state import release_output_locks

soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(64, hard), hard))
args = parse_args([
    sys.argv[1], "--output-dir", sys.argv[2], "--alignment", "none",
    "--formats", "txt", "--existing", "overwrite",
])
validate_args(args)
inputs._probe_duration_hints = lambda jobs: None
jobs = inputs.build_jobs(args)
try:
    assert len(jobs) == 200
finally:
    release_output_locks(jobs)
"""
    environment = os.environ.copy()
    source_path = Path(__file__).parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (os.fspath(source_path), environment.get("PYTHONPATH", "")))
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            os.fspath(source_root),
            os.fspath(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
