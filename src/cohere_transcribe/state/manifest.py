"""Verified final-generation manifests for transcript output sets."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..models import AudioJob
from .io import (
    decode_state,
    ensure_generation_id,
    match_source_and_generation,
    sha256_file,
    source_payload,
)


def _matching_manifest(job: AudioJob) -> tuple[dict[str, Any] | None, str]:
    if job.state_path is None:
        return None, "state marker path is unavailable"
    payload, reason = decode_state(job.state_path)
    if payload is None:
        return None, reason
    if payload.get("kind") != "published":
        return None, f"state is {payload.get('kind')!r}, not published"
    if payload.get("asr_contract_key") != job.asr_contract_key:
        return None, "state marker ASR contract does not match"
    if payload.get("render_contract_key") != job.render_contract_key:
        return None, "state marker render contract does not match"
    matched, reason = match_source_and_generation(job, payload)
    if not matched:
        return None, reason
    return payload, ""


def verify_published_outputs(job: AudioJob) -> tuple[bool, str]:
    payload, reason = _matching_manifest(job)
    if payload is None:
        return False, reason
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(job.output_paths):
        return False, "state marker output formats do not match"
    for output_format, output_path in job.output_paths.items():
        record = outputs.get(output_format)
        if not isinstance(record, dict) or record.get("name") != output_path.name:
            return False, f"state marker path for {output_format} does not match"
        try:
            if output_path.is_symlink() or not output_path.is_file():
                return False, f"{output_format} output is missing or not regular"
            if record.get("size") != output_path.stat().st_size or record.get(
                "sha256"
            ) != sha256_file(output_path):
                return False, f"{output_format} output does not match its state marker"
        except OSError as exc:
            return False, f"cannot verify {output_format} output ({exc})"
    job.generation_id = str(payload["generation_id"])
    job.published = True
    return True, ""


def published_payload(
    job: AudioJob, temporary_outputs: Mapping[str, Path]
) -> dict[str, Any]:
    return {
        "kind": "published",
        "generation_id": ensure_generation_id(job),
        "asr_contract_key": job.asr_contract_key,
        "render_contract_key": job.render_contract_key,
        "source": source_payload(job),
        "updated_unix_seconds": time.time(),
        "outputs": {
            output_format: {
                "name": job.output_paths[output_format].name,
                "size": temporary_path.stat().st_size,
                "sha256": sha256_file(temporary_path),
            }
            for output_format, temporary_path in sorted(temporary_outputs.items())
        },
    }
