"""Transactional output publication and alignment audio reload."""

from __future__ import annotations

import contextlib
import copy
import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from ..audio.decoding import decode_audio
from ..models import (
    SR,
    AudioJob,
    SourceSnapshot,
    SubtitleCue,
    TranscriptionConfig,
    WordTiming,
    default_output_mode,
)
from ..state import create_state_temporary, published_payload
from .rendering import (
    OUTPUT_GENERATORS,
    build_result_content,
    build_result_payload,
    generate_json,
    generate_plain_text,
)


def apply_file_mode(descriptor: int, path: Path, mode: int) -> None:
    """Apply output permissions through the portable API available."""
    fchmod = getattr(os, "fchmod", None)
    if callable(fchmod):
        fchmod(descriptor, mode)
    else:
        os.chmod(path, mode)


def fsync_directories(directories: Iterator[Path]) -> None:
    """Persist directory entries where the platform supports directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    unsupported = {
        errno.EACCES,
        errno.EBADF,
        errno.EINVAL,
        errno.EISDIR,
        errno.ENOTSUP,
        errno.EPERM,
    }
    for directory in dict.fromkeys(path.resolve() for path in directories):
        try:
            descriptor = os.open(directory, flags)
        except OSError as exc:
            if exc.errno in unsupported:
                continue
            raise
        try:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                if exc.errno not in unsupported:
                    raise
        finally:
            os.close(descriptor)


def atomic_write_outputs(
    job: AudioJob,
    cues: Sequence[SubtitleCue],
    words: Sequence[WordTiming] = (),
    transcript_lines: Sequence[str] | None = None,
    *,
    result_payload: dict[str, object] | None = None,
) -> None:
    """Publish one job's formats with rollback if an in-process commit fails."""
    transcript_lines = (
        job.segment_texts if transcript_lines is None else transcript_lines
    )
    output_paths = job.output_paths
    if not output_paths:
        return
    temporary_paths: dict[Path, Path] = {}
    backup_paths: dict[Path, Path | None] = {}
    published: list[Path] = []
    preserved_backups: set[Path] = set()
    publication_paths: list[Path] = list(output_paths.values())
    try:
        for output_format, output_path in output_paths.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
            )
            temporary_path = Path(temporary_name)
            temporary_paths[output_path] = temporary_path
            output_mode = (
                stat.S_IMODE(output_path.stat().st_mode)
                if output_path.exists()
                else default_output_mode()
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    if output_format == "json":
                        handle.write(
                            generate_json(
                                job,
                                words,
                                cues,
                                transcript_lines,
                                payload=result_payload,
                            )
                        )
                    elif output_format == "txt":
                        handle.write(generate_plain_text(transcript_lines))
                    else:
                        handle.write(OUTPUT_GENERATORS[output_format](cues))
                    handle.flush()
                    apply_file_mode(handle.fileno(), temporary_path, output_mode)
                    os.fsync(handle.fileno())
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                raise

        if job.state_path is not None:
            state_temporary = create_state_temporary(
                job.state_path,
                published_payload(
                    job,
                    {
                        output_format: temporary_paths[output_path]
                        for output_format, output_path in output_paths.items()
                    },
                ),
            )
            temporary_paths[job.state_path] = state_temporary
            publication_paths.append(job.state_path)

        for output_path in publication_paths:
            if not output_path.exists():
                backup_paths[output_path] = None
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", suffix=".bak", dir=output_path.parent
            )
            os.close(descriptor)
            backup_path = Path(backup_name)
            backup_paths[output_path] = backup_path
            shutil.copy2(output_path, backup_path)
            with backup_path.open("rb") as backup_handle:
                os.fsync(backup_handle.fileno())

        ensure_source_unchanged(job)
        for output_path in publication_paths:
            os.replace(temporary_paths[output_path], output_path)
            published.append(output_path)
        fsync_directories(output.parent for output in publication_paths)
        job.written.extend(output_paths.values())
        job.published = True
    except BaseException as original_error:
        rollback_errors: list[str] = []
        for output_path in reversed(published):
            rollback_backup = backup_paths.get(output_path)
            try:
                if rollback_backup is None:
                    output_path.unlink(missing_ok=True)
                elif rollback_backup.exists():
                    os.replace(rollback_backup, output_path)
            except BaseException as rollback_error:
                if rollback_backup is not None and rollback_backup.exists():
                    preserved_backups.add(rollback_backup)
                rollback_errors.append(f"{output_path}: {rollback_error}")
        try:
            fsync_directories(output.parent for output in publication_paths)
        except OSError as rollback_error:
            rollback_errors.append(f"directory sync: {rollback_error}")
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Output commit failed and rollback was incomplete ({detail}); "
                f"preserved backups: {sorted(map(os.fspath, preserved_backups))}"
            ) from original_error
        raise
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        for cleanup_backup in backup_paths.values():
            if cleanup_backup is not None and cleanup_backup not in preserved_backups:
                cleanup_backup.unlink(missing_ok=True)


def complete_job_result(
    job: AudioJob,
    cues: Sequence[SubtitleCue],
    words: Sequence[WordTiming] = (),
    transcript_lines: Sequence[str] | None = None,
    *,
    publish_outputs: bool = True,
) -> None:
    """Complete one result through publication and/or detached result capture."""
    transcript_lines = (
        job.segment_texts if transcript_lines is None else transcript_lines
    )
    needs_payload = publish_outputs and "json" in job.output_paths
    payload = (
        build_result_payload(job, words, cues, transcript_lines)
        if needs_payload
        else None
    )

    if publish_outputs and job.output_paths:
        atomic_write_outputs(
            job,
            cues,
            words,
            transcript_lines,
            result_payload=payload,
        )
    else:
        # In-memory completion has no transactional publication step to perform
        # this check, but must offer the same source-consistency guarantee.
        ensure_source_unchanged(job)

    if job.capture_result:
        content = build_result_content(job, words, cues, transcript_lines)
        job.result_payload = copy.deepcopy(content)
    job.result_completed = True


def ensure_source_unchanged(job: AudioJob) -> None:
    current = SourceSnapshot.capture(job.path)
    if current != job.snapshot:
        raise RuntimeError(f"Source changed while processing: {job.path}")


def reload_audio_for_alignment(
    job: AudioJob,
    args: TranscriptionConfig,
) -> np.ndarray:
    ensure_source_unchanged(job)
    if job.audio is not None:
        return job.audio
    audio = decode_audio(
        job.path,
        job.decode_backend or args.audio_backend,
        max_decoded_bytes=int(args.audio_memory_gb * 1024**3),
        duration_hint=job.duration,
    )
    if len(audio) != int(round(job.duration * SR)):
        raise RuntimeError(
            f"Decoded sample count changed between ASR and alignment for {job.path}: "
            f"{len(audio)} != {int(round(job.duration * SR))}"
        )
    return audio
