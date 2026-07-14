"""Audio segmentation and timeline validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models import SR


def segment_audio_auditok(
    audio: np.ndarray,
    min_dur: float,
    max_dur: float,
    max_silence: float,
    energy_threshold: float,
) -> list[tuple[float, float]]:
    from auditok.core import split as auditok_split

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    regions = auditok_split(
        pcm,
        min_dur=min_dur,
        max_dur=max_dur,
        max_silence=max_silence,
        energy_threshold=energy_threshold,
        sampling_rate=SR,
        sample_width=2,
        channels=1,
    )
    return [(float(region.start), float(region.end)) for region in regions]


def segment_audio_fixed(
    audio: np.ndarray, window_seconds: float
) -> list[tuple[float, float]]:
    """Cover the complete waveform with contiguous, non-overlapping windows."""
    total_samples = len(audio)
    if total_samples == 0:
        return []

    window_samples = max(1, int(round(window_seconds * SR)))
    return [
        (start / SR, min(start + window_samples, total_samples) / SR)
        for start in range(0, total_samples, window_samples)
    ]


def validate_processor_single_row_window(processor, window_seconds: float) -> float:
    """Ensure this package, rather than the processor, controls row expansion."""
    feature_extractor = processor.feature_extractor
    max_clip = float(feature_extractor.max_audio_clip_s)
    if not math.isfinite(max_clip) or max_clip <= 0:
        raise RuntimeError("Cohere processor reported an invalid max_audio_clip_s")
    if window_seconds > max_clip + 1e-9:
        raise RuntimeError(
            f"--max-dur {window_seconds:g}s exceeds this Cohere processor's "
            f"single-row limit of {max_clip:g}s"
        )
    return max_clip


def validate_segment_times(
    segments: Sequence[tuple[float, float]],
    duration: float,
    max_duration: float | None = None,
) -> list[tuple[float, float]]:
    """Validate, clamp sub-sample drift, and return ordered non-overlapping spans."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Audio duration must be finite and non-negative")
    tolerance = 2.0 / SR
    validated: list[tuple[float, float]] = []
    previous_end = 0.0
    for index, (raw_start, raw_end) in enumerate(segments):
        start = float(raw_start)
        end = float(raw_end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Segment {index} has non-finite bounds")
        if start < -tolerance or end > duration + tolerance:
            raise ValueError(
                f"Segment {index} lies outside the audio: {start:.6f}..{end:.6f} "
                f"for {duration:.6f}s"
            )
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if end <= start:
            continue
        if start < previous_end - tolerance:
            raise ValueError(f"Segment {index} overlaps or is out of order")
        start = max(start, previous_end) if start < previous_end else start
        if max_duration is not None and end - start > max_duration + tolerance:
            raise ValueError(
                f"Segment {index} is {end - start:.6f}s, exceeding "
                f"the {max_duration:g}s single-row limit"
            )
        validated.append((start, end))
        previous_end = end
    return validated


def sample_timestamps_to_seconds(
    timestamps: Sequence[dict[str, Any]], audio_samples: int
) -> list[tuple[float, float]]:
    duration = audio_samples / SR
    segments: list[tuple[float, float]] = []
    for item in timestamps:
        start = max(0.0, float(item["start"]) / SR)
        end = min(duration, float(item["end"]) / SR)
        if end > start:
            segments.append((start, end))
    return validate_segment_times(segments, duration)


def merge_speech_segments(
    segments: Sequence[tuple[float, float]], max_duration: float
) -> list[tuple[float, float]]:
    """Greedily merge adjacent VAD spans while their full timeline fits."""
    if not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("max_duration must be finite and positive")
    if not segments:
        return []

    merged: list[tuple[float, float]] = []
    current_start, current_end = segments[0]
    if current_start < 0 or current_end < current_start:
        raise ValueError("VAD segments must have non-negative ordered bounds")
    for start, end in segments[1:]:
        if start < current_end or end < start:
            raise ValueError("VAD segments must be sorted and non-overlapping")
        if end - current_start <= max_duration + 1e-9:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged
