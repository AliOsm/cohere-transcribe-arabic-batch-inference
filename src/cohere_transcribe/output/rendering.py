"""Subtitle rendering and machine-readable result provenance."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence

from ..models import (
    ALIGN_MODEL_ID,
    ALIGN_MODEL_REVISION,
    ALIGN_PACKAGE_REPOSITORY,
    ALIGN_PACKAGE_REVISION,
    ASR_MODEL_REVISION,
    MODEL_ID,
    OUTPUT_SCHEMA_VERSION,
    REPETITION_DETECTOR_VERSION,
    SENTENCE_ENDINGS,
    SILERO_VERSION,
    SR,
    UROMAN_VERSION,
    AudioJob,
    SubtitleCue,
    WordTiming,
    package_version,
    runtime_implementation,
)


def build_cues(
    words: Sequence[WordTiming],
    max_chars: int,
    max_duration: float,
    max_gap: float,
    min_cue_duration: float = 0.30,
    media_duration: float | None = None,
) -> list[SubtitleCue]:
    cue_words: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    for word in words:
        if current:
            candidate = " ".join(item["text"] for item in current) + " " + word["text"]
            gap = word["start"] - current[-1]["end"]
            duration = word["end"] - current[0]["start"]
            if len(candidate) > max_chars or duration > max_duration or gap > max_gap:
                cue_words.append(current)
                current = []
        current.append(word)
        if word["text"].endswith(SENTENCE_ENDINGS):
            cue_words.append(current)
            current = []
    if current:
        cue_words.append(current)

    cues: list[SubtitleCue] = [
        {
            "start": max(0.0, items[0]["start"]),
            "end": max(items[0]["start"], items[-1]["end"]),
            "text": " ".join(item["text"] for item in items).strip(),
        }
        for items in cue_words
        if items
    ]
    for index, cue in enumerate(cues):
        if media_duration is not None:
            cue["start"] = min(cue["start"], media_duration)
            cue["end"] = min(max(cue["start"], cue["end"]), media_duration)
        next_start = cues[index + 1]["start"] if index + 1 < len(cues) else math.inf
        media_end = media_duration if media_duration is not None else math.inf
        upper_bound = min(next_start, media_end)
        desired_end = max(cue["end"], cue["start"] + min_cue_duration)
        cue["end"] = max(cue["start"], min(desired_end, upper_bound))
    return cues


def fmt_timestamp(
    seconds: float, include_hours: bool = False, marker: str = "."
) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    if include_hours or hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{marker}{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}{marker}{milliseconds:03d}"


def generate_plain_text(lines: Sequence[str]) -> str:
    return "\n".join(line.strip() for line in lines if line.strip()) + "\n"


def generate_srt(cues: Sequence[SubtitleCue]) -> str:
    return "".join(
        f"{index}\n{fmt_timestamp(cue['start'], True, ',')} --> "
        f"{fmt_timestamp(cue['end'], True, ',')}\n{cue['text']}\n\n"
        for index, cue in enumerate(cues, 1)
    )


def generate_vtt(cues: Sequence[SubtitleCue]) -> str:
    return "WEBVTT\n\n" + "".join(
        f"{fmt_timestamp(cue['start'])} --> {fmt_timestamp(cue['end'])}\n{cue['text']}\n\n"
        for cue in cues
    )


def build_result_content(
    job: AudioJob,
    words: Sequence[WordTiming],
    cues: Sequence[SubtitleCue],
    transcript_lines: Sequence[str],
) -> dict[str, object]:
    """Build the detached transcript content needed by public API results."""
    segments = [
        {
            "segment_index": segment_index,
            "start": start,
            "end": end,
            "text": text.strip(),
        }
        for segment_index, ((start, end), text) in enumerate(
            zip(job.segment_times, job.segment_texts, strict=True)
        )
        if text.strip()
    ]
    return {
        "transcript": [line.strip() for line in transcript_lines if line.strip()],
        "segments": segments,
        "words": [dict(word) for word in words],
        "cues": [dict(cue) for cue in cues],
    }


def build_result_payload(
    job: AudioJob,
    words: Sequence[WordTiming],
    cues: Sequence[SubtitleCue],
    transcript_lines: Sequence[str],
) -> dict[str, object]:
    """Build the complete machine-readable result for one audio source."""
    content = build_result_content(job, words, cues, transcript_lines)
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "implementation": runtime_implementation(),
        "source": {
            "path": os.fspath(job.path),
            "duration_seconds": job.duration,
            "sample_rate": SR,
            "decode_backend": job.decode_backend,
            "decode_fallback_reason": job.decode_fallback_reason,
        },
        "language": job.language,
        "segmentation": job.vad_mode,
        "segmentation_details": {
            "mode": job.vad_mode,
            "requested_engine": job.vad_engine_requested,
            "actual_engine": job.vad_engine_actual,
            "provider": job.vad_provider,
            "provider_options": job.vad_provider_options,
            "fallback_reason": job.vad_fallback_reason,
            "merge": job.vad_merge,
            "parameters": job.segmentation_parameters,
            "speech_spans": [
                {"start": start, "end": end} for start, end in job.speech_spans
            ],
        },
        "timing": job.alignment_mode,
        "models": {
            "asr": {"id": MODEL_ID, "revision": ASR_MODEL_REVISION},
            "vad": (
                {
                    "source": "silero-vad",
                    "source_version": SILERO_VERSION,
                    "distribution": "cohere-transcribe-arabic",
                    "version": package_version("cohere-transcribe-arabic"),
                    "weight_asset": (
                        "cohere_transcribe/vad/silero_vad_v6.onnx"
                        if job.vad_engine_actual == "onnx"
                        else "cohere_transcribe/vad/silero_vad.jit"
                    ),
                    "implementation": {
                        "torch": "packed-sequence-v1",
                        "onnx": "sequence-onnx",
                        "jit": "packaged-torchscript",
                    }.get(job.vad_engine_actual),
                }
                if job.vad_mode == "silero"
                else None
            ),
            "aligner": (
                {
                    "id": ALIGN_MODEL_ID,
                    "revision": ALIGN_MODEL_REVISION,
                    "kernel": {
                        "distribution": "torchaudio",
                        "operation": "torchaudio.functional.forced_align",
                        "version": package_version("torchaudio"),
                    },
                    "utility_package": {
                        "distribution": "cohere-transcribe-arabic",
                        "location": "cohere_transcribe.alignment",
                        "repository": ALIGN_PACKAGE_REPOSITORY,
                        "revision": ALIGN_PACKAGE_REVISION,
                    },
                    "romanizer": {
                        "distribution": "uroman",
                        "version": UROMAN_VERSION,
                    },
                }
                if job.alignment_mode == "word"
                else None
            ),
        },
        "fallback_alignment_segments": job.fallback_alignments,
        "repetition_detector_version": REPETITION_DETECTOR_VERSION,
        "repetition_stopped_segments": sorted(job.repetition_stopped_segments),
        "truncation_retried_segments": sorted(job.truncation_retried_segments),
        "token_limit_segments": sorted(job.token_limit_segments),
        "generated_tokens_by_segment": [
            {"segment_index": index, "tokens": count}
            for index, count in sorted(job.generated_tokens.items())
        ],
        **content,
    }


def generate_json(
    job: AudioJob,
    words: Sequence[WordTiming],
    cues: Sequence[SubtitleCue],
    transcript_lines: Sequence[str],
    *,
    payload: dict[str, object] | None = None,
) -> str:
    """Serialize a result payload, building it only when one was not supplied."""
    if payload is None:
        payload = build_result_payload(job, words, cues, transcript_lines)
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


OUTPUT_GENERATORS = {"srt": generate_srt, "vtt": generate_vtt}
