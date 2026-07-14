from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.inputs import build_jobs, expand_inputs
from cohere_transcribe.models import AUDIO_EXTENSIONS, AudioJob, TranscriptionConfig
from cohere_transcribe.profiling import validate_profile_output_path
from cohere_transcribe.state import release_output_locks

EXPECTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
    ".webm",
    ".wma",
}


def make_config(
    *sources: Path,
    output_dir: Path | None = None,
    alignment: str = "none",
    formats: tuple[str, ...] | None = None,
    profile_json: Path | None = None,
) -> TranscriptionConfig:
    argv = [*(os.fspath(source) for source in sources), "--alignment", alignment]
    argv.extend(("--existing", "overwrite"))
    if formats is not None:
        argv.extend(("--formats", *formats))
    if output_dir is not None:
        argv.extend(("--output-dir", os.fspath(output_dir)))
    if profile_json is not None:
        argv.extend(("--profile-json", os.fspath(profile_json)))
    args = parse_args(argv)
    validate_args(args)
    return args


def build_without_probe(args: TranscriptionConfig) -> list[AudioJob]:
    with mock.patch("cohere_transcribe.inputs.probe_duration", return_value=None):
        return build_jobs(args)


@contextlib.contextmanager
def planned_jobs(args: TranscriptionConfig) -> Iterator[list[AudioJob]]:
    jobs = build_without_probe(args)
    try:
        yield jobs
    finally:
        release_output_locks(jobs)


def create_special_path(path: Path, kind: str) -> None:
    if kind == "symlink":
        target = path.with_name(f"{path.name}.target")
        target.write_bytes(b"target")
        path.symlink_to(target.name)
    elif kind == "self-symlink":
        path.symlink_to(path.name)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation is unavailable on this platform")
        os.mkfifo(path)
    else:  # pragma: no cover - test helper invariant
        raise AssertionError(f"unknown special path kind: {kind}")


def test_directory_discovery_recurses_only_when_requested(tmp_path: Path) -> None:
    direct = tmp_path / "direct.wav"
    nested = tmp_path / "season" / "episode.mp3"
    ignored = tmp_path / "season" / "notes.txt"
    nested.parent.mkdir()
    direct.write_bytes(b"")
    nested.write_bytes(b"")
    ignored.write_text("not audio", encoding="utf-8")

    assert expand_inputs([os.fspath(tmp_path)], recursive=False) == [
        (direct.resolve(), Path("direct.wav"))
    ]
    assert expand_inputs([os.fspath(tmp_path)], recursive=True) == [
        (direct.resolve(), Path("direct.wav")),
        (nested.resolve(), Path("season/episode.mp3")),
    ]


def test_directory_discovery_accepts_every_supported_suffix_case_insensitively(
    tmp_path: Path,
) -> None:
    assert AUDIO_EXTENSIONS == EXPECTED_AUDIO_EXTENSIONS
    for index, suffix in enumerate(sorted(EXPECTED_AUDIO_EXTENSIONS)):
        (tmp_path / f"audio-{index:02d}{suffix.upper()}").write_bytes(b"")
    (tmp_path / "audio.raw").write_bytes(b"")

    entries = expand_inputs([os.fspath(tmp_path)], recursive=True)

    assert len(entries) == len(EXPECTED_AUDIO_EXTENSIONS)
    assert {path.suffix.lower() for path, _relative in entries} == (
        EXPECTED_AUDIO_EXTENSIONS
    )


def test_unsupported_files_are_ignored_in_directories_but_allowed_explicitly(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "clip.wav"
    metadata = tmp_path / "clip.transcript"
    audio.write_bytes(b"")
    metadata.write_text("input is validated by the decoder later", encoding="utf-8")

    assert expand_inputs([os.fspath(tmp_path)], recursive=True) == [
        (audio.resolve(), Path("clip.wav"))
    ]
    assert expand_inputs([os.fspath(metadata)], recursive=True) == [
        (metadata.resolve(), Path("clip.transcript"))
    ]


def test_discovery_deduplicates_symlinks_by_canonical_path(tmp_path: Path) -> None:
    source = tmp_path / "real.wav"
    alias = tmp_path / "00 alias.wav"
    source.write_bytes(b"")
    alias.symlink_to(source.name)

    assert expand_inputs([os.fspath(tmp_path)], recursive=True) == [
        (source.resolve(), Path("00 alias.wav"))
    ]
    assert expand_inputs(
        [os.fspath(source), os.fspath(alias), os.fspath(tmp_path)], recursive=True
    ) == [(source.resolve(), Path("real.wav"))]


def test_discovery_preserves_input_group_order_and_sorts_directory_entries(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "outside.opus"
    directory = tmp_path / "collection"
    explicit.write_bytes(b"")
    directory.mkdir()
    names = (
        "z last.wav",
        "A first.mp3",
        "archive.tar.WAV",
        "مرحبا.WEBM",
        ".hidden.ogg",
    )
    for name in names:
        (directory / name).write_bytes(b"")

    entries = expand_inputs([os.fspath(explicit), os.fspath(directory)], recursive=True)

    assert [relative.as_posix() for _path, relative in entries] == [
        "outside.opus",
        ".hidden.ogg",
        "A first.mp3",
        "archive.tar.WAV",
        "z last.wav",
        "مرحبا.WEBM",
    ]


def test_missing_broken_and_empty_directory_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.wav"
    with pytest.raises(SystemExit, match="Input does not exist"):
        expand_inputs([os.fspath(missing)], recursive=True)

    broken = tmp_path / "broken.wav"
    broken.symlink_to("absent.wav")
    with pytest.raises(SystemExit, match="Input does not exist"):
        expand_inputs([os.fspath(broken)], recursive=True)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No audio files found"):
        expand_inputs([os.fspath(empty)], recursive=True)


def test_empty_regular_audio_is_valid_during_planning(tmp_path: Path) -> None:
    source = tmp_path / "empty.wav"
    source.write_bytes(b"")

    with planned_jobs(make_config(source)) as jobs:
        assert len(jobs) == 1
        assert jobs[0].snapshot.size == 0
        assert jobs[0].duration_hint is None


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX FIFO semantics")
def test_explicit_fifo_is_rejected_as_a_nonregular_input(tmp_path: Path) -> None:
    fifo = tmp_path / "audio.wav"
    os.mkfifo(fifo)

    with pytest.raises(SystemExit, match="not a regular file or directory"):
        expand_inputs([os.fspath(fifo)], recursive=True)


def test_output_root_preserves_nested_relative_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "course" / "day 01" / "مقدمة.wav"
    output_root = tmp_path / "transcripts"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")

    with planned_jobs(make_config(source_root, output_dir=output_root)) as jobs:
        assert len(jobs) == 1
        job = jobs[0]
        expected_parent = output_root / "course" / "day 01"
        assert job.relative_path == Path("course/day 01/مقدمة.wav")
        assert job.output_paths == {"txt": expected_parent / "مقدمة.txt"}
        assert job.state_path == (
            expected_parent / ".مقدمة.cohere-transcribe.manifest.json"
        )
        assert job.checkpoint_path == (
            expected_parent / ".مقدمة.cohere-transcribe.asr.json"
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("audio", "Invalid input path"),
        ("output_dir", "Invalid output directory path"),
        ("profile_json", "Invalid profile path"),
    ],
)
@pytest.mark.parametrize(
    "invalid_path",
    ["invalid\0path", "~__cohere_transcribe_missing_user_81f9__/path"],
)
def test_invalid_user_paths_are_concise_planner_errors(
    tmp_path: Path, field: str, message: str, invalid_path: str
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"")
    args = make_config(source, output_dir=tmp_path / "out")
    if field == "audio":
        args.audio = [invalid_path]
    else:
        setattr(args, field, invalid_path)

    with pytest.raises(SystemExit, match=message) as captured:
        build_without_probe(args)

    assert "\0" not in str(captured.value)


def test_nested_output_symlink_cannot_escape_output_root(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "redirect" / "clip.wav"
    output_root = tmp_path / "out"
    outside = tmp_path / "outside"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    output_root.mkdir()
    outside.mkdir()
    (output_root / "redirect").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SystemExit, match="escapes output root"):
        build_without_probe(make_config(source_root, output_dir=output_root))

    assert list(outside.iterdir()) == []


def test_same_stem_inputs_in_one_directory_are_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "clip.wav").write_bytes(b"")
    (source_root / "clip.MP3").write_bytes(b"")

    with pytest.raises(SystemExit, match="collision"):
        build_without_probe(make_config(source_root, output_dir=tmp_path / "out"))


def test_matching_relative_stems_from_separate_roots_are_rejected(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = first_root / "nested" / "clip.wav"
    second = second_root / "nested" / "clip.opus"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"")
    second.write_bytes(b"")

    with pytest.raises(SystemExit, match="collision"):
        build_without_probe(
            make_config(first_root, second_root, output_dir=tmp_path / "out")
        )


def test_transcript_output_cannot_overwrite_an_explicit_input(tmp_path: Path) -> None:
    source = tmp_path / "transcript.txt"
    source.write_text("explicit non-audio input", encoding="utf-8")

    with pytest.raises(SystemExit, match="Output path collides with an input"):
        build_without_probe(make_config(source))


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
@pytest.mark.parametrize("label", ["input", "output directory", "profile"])
def test_planner_reports_symlink_loops_consistently(tmp_path: Path, label: str) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"")
    loop = tmp_path / "loop"
    loop.symlink_to(loop.name)
    args = {
        "input": make_config(loop),
        "output directory": make_config(source, output_dir=loop),
        "profile": make_config(source, profile_json=loop),
    }[label]

    with pytest.raises(SystemExit, match=rf"Invalid {label} path .*Symlink loop"):
        build_without_probe(args)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges")
def test_nested_output_directory_loop_is_rejected_before_creation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source = source_root / "nested" / "clip.wav"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"")
    output_root = tmp_path / "out"
    output_root.mkdir()
    (output_root / "nested").symlink_to("nested")

    with pytest.raises(
        SystemExit, match=r"Cannot resolve output directory .*Symlink loop"
    ):
        build_without_probe(make_config(source_root, output_dir=output_root))


@pytest.mark.parametrize(
    ("reserved_name", "label"),
    [
        (".clip.cohere-transcribe.manifest.json", "state marker"),
        (".clip.cohere-transcribe.asr.json", "ASR checkpoint"),
    ],
)
def test_reserved_paths_cannot_also_be_explicit_inputs(
    tmp_path: Path, reserved_name: str, label: str
) -> None:
    source = tmp_path / "clip.wav"
    reserved_input = tmp_path / reserved_name
    source.write_bytes(b"")
    reserved_input.write_bytes(b"input")

    with pytest.raises(SystemExit, match=rf"Reserved {label} collides with an input"):
        build_without_probe(make_config(source, reserved_input))


def test_transcript_output_cannot_collide_with_another_jobs_reserved_state(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "clip.wav").write_bytes(b"")
    (source_root / ".clip.cohere-transcribe.manifest.wav").write_bytes(b"")

    args = make_config(
        source_root,
        output_dir=tmp_path / "out",
        alignment="segment",
        formats=("json",),
    )
    with pytest.raises(SystemExit, match="collides with reserved state path"):
        build_without_probe(args)


@pytest.mark.parametrize(
    ("target_kind", "message"),
    [
        ("input", "Profile path collides with an input audio file"),
        ("output", "Profile path collides with a transcript output"),
        ("reserved", "Profile path collides with a reserved state marker"),
    ],
)
def test_profile_path_collisions_are_rejected_during_planning(
    tmp_path: Path, target_kind: str, message: str
) -> None:
    source = tmp_path / "clip.wav"
    output_root = tmp_path / "out"
    source.write_bytes(b"")
    targets = {
        "input": source,
        "output": output_root / "clip.txt",
        "reserved": output_root / ".clip.cohere-transcribe.manifest.json",
    }

    with pytest.raises(SystemExit, match=message):
        build_without_probe(
            make_config(
                source,
                output_dir=output_root,
                profile_json=targets[target_kind],
            )
        )


@pytest.mark.parametrize("kind", ["symlink", "self-symlink", "directory", "fifo"])
def test_output_path_must_be_a_regular_nonsymlink_file(
    tmp_path: Path, kind: str
) -> None:
    source = tmp_path / "clip.wav"
    output_root = tmp_path / "out"
    output = output_root / "clip.txt"
    source.write_bytes(b"")
    output_root.mkdir()
    create_special_path(output, kind)

    with pytest.raises(SystemExit, match="Output path is not a regular file"):
        build_without_probe(make_config(source, output_dir=output_root))


@pytest.mark.parametrize(
    ("reserved_name", "label", "kind"),
    [
        (".clip.cohere-transcribe.manifest.json", "state marker", "symlink"),
        (
            ".clip.cohere-transcribe.manifest.json",
            "state marker",
            "self-symlink",
        ),
        (".clip.cohere-transcribe.asr.json", "ASR checkpoint", "directory"),
    ],
)
def test_reserved_path_must_be_a_regular_nonsymlink_file(
    tmp_path: Path, reserved_name: str, label: str, kind: str
) -> None:
    source = tmp_path / "clip.wav"
    output_root = tmp_path / "out"
    reserved = output_root / reserved_name
    source.write_bytes(b"")
    output_root.mkdir()
    create_special_path(reserved, kind)

    with pytest.raises(SystemExit, match=rf"Reserved {label} is not a regular file"):
        build_without_probe(make_config(source, output_dir=output_root))


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_profile_path_must_be_a_regular_nonsymlink_file(
    tmp_path: Path, kind: str
) -> None:
    profile = tmp_path / "profile.json"
    create_special_path(profile, kind)
    message = "must not be a symlink" if kind == "symlink" else "not a regular file"

    with pytest.raises(SystemExit, match=message):
        validate_profile_output_path(os.fspath(profile), [])
