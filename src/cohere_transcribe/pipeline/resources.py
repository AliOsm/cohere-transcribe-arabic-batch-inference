"""Decoded-audio lifetime and memory partitioning helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ..models import SR, AudioJob


def release_job_audio(jobs: Sequence[AudioJob]) -> None:
    """Release decoded waveforms retained by a set of jobs."""
    for job in jobs:
        job.audio = None


def estimated_decoded_bytes(job: AudioJob, memory_budget: int) -> int:
    """Estimate a job's mono float32 decoded size for group planning."""
    if job.audio is not None:
        return job.audio_bytes
    if job.duration > 0:
        return int(round(job.duration * SR)) * np.dtype(np.float32).itemsize
    if job.duration_hint is not None:
        estimate = int(
            math.ceil(job.duration_hint * SR * np.dtype(np.float32).itemsize)
        )
        return estimate + SR * np.dtype(np.float32).itemsize
    # Unknown-duration inputs are isolated because compressed size cannot bound decoded size.
    return memory_budget


def partition_audio_jobs(
    jobs: Sequence[AudioJob],
    memory_budget: int,
    max_jobs: int | None = None,
) -> list[list[AudioJob]]:
    """Partition jobs using probed or retained decoded-audio sizes."""
    groups: list[list[AudioJob]] = []
    current: list[AudioJob] = []
    current_bytes = 0
    for job in jobs:
        job_bytes = estimated_decoded_bytes(job, memory_budget)
        if current and (
            current_bytes + job_bytes > memory_budget
            or (max_jobs is not None and len(current) >= max_jobs)
        ):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(job)
        current_bytes += job_bytes
    if current:
        groups.append(current)
    return groups
