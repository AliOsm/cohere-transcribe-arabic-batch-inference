"""Immutable public result snapshots built from mutable pipeline jobs."""

from __future__ import annotations

from collections.abc import Sequence

from ..api.types import (
    ResultStatus,
    SubtitleCue,
    TranscriptionOptions,
    TranscriptionProvenance,
    TranscriptionResult,
    TranscriptionRun,
    TranscriptionSegment,
    TranscriptionStatistics,
    TranscriptionWord,
)
from ..config import options_from_config
from ..models import AudioJob, RunStats, TranscriptionConfig


def _statistics(
    stats: RunStats, elapsed: float, jobs: Sequence[AudioJob]
) -> TranscriptionStatistics:
    successful_duration = sum(
        job.duration for job in jobs if not job.skipped and job.error is None
    )
    return TranscriptionStatistics(
        elapsed_seconds=elapsed,
        successful_audio_seconds=successful_duration,
        real_time_factor_x=successful_duration / elapsed if elapsed > 0 else 0.0,
        runtime_import_seconds=stats.runtime_import_seconds,
        serialization_wait_seconds=stats.serialization_wait_seconds,
        input_validation_seconds=stats.input_validation_seconds,
        decode_seconds=stats.decode_seconds,
        vad_seconds=stats.vad_seconds,
        asr_load_seconds=stats.asr_load_seconds,
        asr_seconds=stats.asr_seconds,
        aligner_load_seconds=stats.align_load_seconds,
        emissions_seconds=stats.emissions_seconds,
        viterbi_seconds=stats.viterbi_seconds,
        peak_cuda_allocated_gib=stats.peak_cuda_gib,
        peak_cuda_reserved_gib=stats.peak_cuda_reserved_gib,
        asr_batches=stats.asr_batches,
        asr_processor_rows=stats.asr_processor_rows,
        generated_tokens=stats.asr_generated_tokens,
        oom_retries=stats.asr_oom_retries,
        truncation_retries=stats.asr_truncation_retries,
    )


def _result_from_job(job: AudioJob) -> TranscriptionResult:
    payload = job.result_payload or {}
    transcript = payload.get("transcript")
    if job.skipped:
        status: ResultStatus = "skipped"
        text = None
        duration = job.duration_hint
    else:
        status = "failed" if job.error is not None else "completed"
        lines = (
            transcript
            if isinstance(transcript, list)
            else [text.strip() for text in job.segment_texts if text.strip()]
        )
        joined_text = "\n".join(str(line) for line in lines)
        text = joined_text if joined_text or job.error is None else None
        duration = job.duration if job.duration > 0 else job.duration_hint

    payload_segments = payload.get("segments")
    if not isinstance(payload_segments, list):
        payload_segments = [
            {
                "segment_index": index,
                "start": start,
                "end": end,
                "text": segment_text.strip(),
            }
            for index, ((start, end), segment_text) in enumerate(
                zip(job.segment_times, job.segment_texts, strict=True)
            )
            if segment_text.strip()
        ]
    segments = tuple(
        TranscriptionSegment(
            index=int(segment["segment_index"]),
            start=float(segment["start"]),
            end=float(segment["end"]),
            text=str(segment["text"]),
        )
        for segment in payload_segments
    )
    words = tuple(
        TranscriptionWord(
            start=float(word["start"]),
            end=float(word["end"]),
            text=str(word["text"]),
            segment_index=int(word["segment_index"]),
            segment_word_index=int(word["segment_word_index"]),
            timing_source=str(word["timing_source"]),
        )
        for word in payload.get("words", ())
    )
    cues = tuple(
        SubtitleCue(
            start=float(cue["start"]),
            end=float(cue["end"]),
            text=str(cue["text"]),
        )
        for cue in payload.get("cues", ())
    )
    outputs = tuple(job.output_paths.values()) if job.skipped else tuple(job.written)
    return TranscriptionResult(
        path=job.path,
        relative_path=job.relative_path,
        status=status,
        text=text,
        duration=duration,
        segments=segments,
        words=words,
        cues=cues,
        outputs=outputs,
        error=job.error,
        provenance=TranscriptionProvenance(
            decode_backend=job.decode_backend,
            decode_fallback_reason=job.decode_fallback_reason,
            vad_engine_requested=job.vad_engine_requested,
            vad_engine_actual=job.vad_engine_actual,
            vad_provider=job.vad_provider,
            vad_fallback_reason=job.vad_fallback_reason,
            fallback_alignment_segments=job.fallback_alignments,
            repetition_stopped_segments=tuple(sorted(job.repetition_stopped_segments)),
            truncation_retried_segments=tuple(sorted(job.truncation_retried_segments)),
            token_limit_segments=tuple(sorted(job.token_limit_segments)),
            generated_tokens_by_segment=tuple(sorted(job.generated_tokens.items())),
            resumed_from_asr_checkpoint=job.asr_checkpoint_loaded,
            published=job.published,
        ),
    )


def build_run(
    jobs: Sequence[AudioJob],
    requested_options: TranscriptionOptions,
    args: TranscriptionConfig,
    stats: RunStats,
    elapsed: float,
    errors: Sequence[str] = (),
) -> TranscriptionRun:
    """Freeze mutable pipeline state into a stable, input-ordered API result."""
    ordered_jobs = sorted(jobs, key=lambda job: job.index)
    return TranscriptionRun(
        results=tuple(_result_from_job(job) for job in ordered_jobs),
        requested_options=requested_options,
        resolved_options=options_from_config(
            args, publication_enabled=requested_options.publication is not None
        ),
        statistics=_statistics(stats, elapsed, jobs),
        errors=tuple(errors),
    )


__all__ = ["build_run"]
