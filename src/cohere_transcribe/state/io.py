"""Integrity envelopes and atomic I/O shared by durable state files."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..models import AudioJob, SourceSnapshot, default_output_mode

STATE_SCHEMA_VERSION = 1
STATE_SUFFIX = ".cohere-transcribe.manifest.json"
CHECKPOINT_SUFFIX = ".cohere-transcribe.asr.json"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_parent_and_stem(output_paths: Mapping[str, Path]) -> tuple[Path, str]:
    if not output_paths:
        raise ValueError("An output set must contain at least one path")
    paths = list(output_paths.values())
    parent = paths[0].parent
    stem = paths[0].stem
    if any(path.parent != parent or path.stem != stem for path in paths[1:]):
        raise ValueError("All formats in one output set must share a parent and stem")
    return parent, stem


def state_path_for_outputs(output_paths: Mapping[str, Path]) -> Path:
    parent, stem = output_parent_and_stem(output_paths)
    return parent / f".{stem}{STATE_SUFFIX}"


def checkpoint_path_for_outputs(output_paths: Mapping[str, Path]) -> Path:
    parent, stem = output_parent_and_stem(output_paths)
    return parent / f".{stem}{CHECKPOINT_SUFFIX}"


def source_payload(job: AudioJob) -> dict[str, Any]:
    return {
        "canonical_path": os.fspath(job.path.resolve(strict=False)),
        "snapshot": asdict(job.snapshot),
    }


def ensure_generation_id(job: AudioJob) -> str:
    if not job.generation_id:
        job.generation_id = uuid.uuid4().hex
    return job.generation_id


def envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "payload_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        "payload": payload,
    }


def decode_state(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        if path.is_symlink() or not path.is_file():
            return None, "state marker is missing or not a regular file"
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("root is not an object")
        if decoded.get("schema_version") != STATE_SCHEMA_VERSION:
            return None, "state marker schema is unsupported"
        payload = decoded.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
        expected = hashlib.sha256(canonical_json(payload)).hexdigest()
        if decoded.get("payload_sha256") != expected:
            return None, "state marker integrity check failed"
        return payload, ""
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return None, f"state marker is unreadable ({type(exc).__name__}: {exc})"


def match_source_and_generation(
    job: AudioJob, payload: Mapping[str, Any]
) -> tuple[bool, str]:
    if payload.get("source") != source_payload(job):
        return False, "state marker source snapshot does not match"
    generation_id = payload.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        return False, "state marker generation ID is invalid"
    return True, ""


def create_state_temporary(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"State marker must not be a symlink: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else default_output_mode()
    try:
        if callable(getattr(os, "fchmod", None)):
            os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                envelope(payload),
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        with contextlib.suppress(OSError):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_state_atomic(
    job: AudioJob, path: Path | None, payload: Mapping[str, Any]
) -> None:
    if path is None:
        raise RuntimeError("Job does not define a state marker path")
    if SourceSnapshot.capture(job.path) != job.snapshot:
        raise RuntimeError(f"Source changed while processing: {job.path}")
    temporary_path = create_state_temporary(path, payload)
    try:
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
