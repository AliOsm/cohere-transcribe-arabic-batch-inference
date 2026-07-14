"""Transcript rendering, alignment, and output publication orchestration."""

from __future__ import annotations

import concurrent.futures
import gc
import time
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import torch

from ..alignment.runtime import (
    align_words,
    compute_emissions_streaming,
    load_aligner,
    speech_spans_within_segment,
    uniform_word_timings,
    uniform_word_timings_across_spans,
)
from ..cancellation import raise_if_cancelled, request_cancellation
from ..device import empty_device_cache
from ..models import (
    ALIGNMENT_GC_INTERVAL,
    AudioJob,
    RunStats,
    TranscriptionConfig,
    WordTiming,
    fmt_dur,
    info,
)
from ..pipeline.resources import estimated_decoded_bytes, release_job_audio
from ..runtime.resources import evict_current_asr_owner
from .publication import (
    complete_job_result,
    ensure_source_unchanged,
    reload_audio_for_alignment,
)
from .rendering import build_cues


class PairBudgetPrefetch:
    """Prefetch one next item only when the adjacent pair fits a byte budget."""

    def __init__(
        self,
        items: Sequence[AudioJob],
        fn: Callable[[AudioJob], object],
        estimated_bytes: Sequence[int],
        memory_budget: int,
    ) -> None:
        if len(items) != len(estimated_bytes):
            raise ValueError("items and estimated_bytes must have equal lengths")
        self._items = list(items)
        self._fn = fn
        self._estimated_bytes = list(estimated_bytes)
        self._memory_budget = memory_budget
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._first_future: concurrent.futures.Future | None = None

    def __enter__(self) -> PairBudgetPrefetch:
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audio-reload"
        )
        if self._items:
            self._first_future = self._executor.submit(self._fn, self._items[0])
        return self

    def __iter__(self) -> Iterator[tuple[AudioJob, concurrent.futures.Future]]:
        if not self._items:
            return
        assert self._executor is not None and self._first_future is not None
        current_future = self._first_future
        for index, item in enumerate(self._items):
            next_future: concurrent.futures.Future | None = None
            if index + 1 < len(self._items):
                pair_bytes = (
                    self._estimated_bytes[index] + self._estimated_bytes[index + 1]
                )
                if pair_bytes <= self._memory_budget:
                    next_future = self._executor.submit(
                        self._fn, self._items[index + 1]
                    )
            yield item, current_future
            if index + 1 < len(self._items):
                current_future = next_future or self._executor.submit(
                    self._fn, self._items[index + 1]
                )

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        if self._executor is not None:
            if exc is not None and not isinstance(exc, Exception):
                request_cancellation()
            self._executor.shutdown(wait=True, cancel_futures=exc is not None)


def write_empty_jobs(jobs: Sequence[AudioJob], *, publish_outputs: bool = True) -> None:
    """Publish empty artifacts for jobs that contain no recognized text."""
    for job in jobs:
        if job.error is not None or job.has_text or job.published:
            continue
        try:
            ensure_source_unchanged(job)
            complete_job_result(job, [], [], publish_outputs=publish_outputs)
        except Exception as exc:
            job.error = f"writing empty transcript failed: {exc}"
            info(f"[error] {job.path}: {job.error}")
        finally:
            job.audio = None


def write_segment_timed_outputs(
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    *,
    publish_outputs: bool = True,
) -> None:
    """Write approximate word timings spread within each VAD segment."""
    for job in jobs:
        if job.error is not None or job.published:
            continue
        try:
            words: list[WordTiming] = []
            pairs = zip(job.segment_times, job.segment_texts, strict=True)
            for segment_index, ((start, end), text) in enumerate(pairs):
                spans = speech_spans_within_segment(job.speech_spans, start, end)
                if spans:
                    words.extend(
                        uniform_word_timings_across_spans(
                            text,
                            spans,
                            segment_index,
                            (
                                "uniform_speech_spans"
                                if job.vad_merge
                                else "uniform_segment"
                            ),
                        )
                    )
                else:
                    words.extend(
                        uniform_word_timings(
                            text,
                            start,
                            end,
                            segment_index,
                            "uniform_segment",
                        )
                    )
            cues = build_cues(
                words,
                args.max_chars,
                args.max_cue_dur,
                args.max_gap,
                media_duration=job.duration,
            )
            complete_job_result(job, cues, words, publish_outputs=publish_outputs)
            info(
                f"wrote {job.path.name}: {len(words)} words, {len(cues)} segment-timed cues"
            )
        except Exception as exc:
            job.error = f"segment-timed output failed: {exc}"
            info(f"[error] {job.path}: {job.error}")
        finally:
            job.audio = None


def write_text_only_outputs(
    jobs: Sequence[AudioJob], *, publish_outputs: bool = True
) -> None:
    """Write ASR segment text directly, without constructing timed cues."""
    for job in jobs:
        if job.error is not None or job.published:
            continue
        try:
            lines = [text.strip() for text in job.segment_texts if text.strip()]
            complete_job_result(
                job,
                [],
                transcript_lines=lines,
                publish_outputs=publish_outputs,
            )
            word_count = sum(len(text.split()) for text in lines)
            info(f"wrote {job.path.name}: {word_count} words, text only")
        except Exception as exc:
            job.error = f"text-only output failed: {exc}"
            info(f"[error] {job.path}: {job.error}")
        finally:
            job.audio = None


def report_retained_checkpoint(job: AudioJob) -> None:
    """Report recoverable ASR without mutating the requested output generation."""
    if job.checkpoint_path is not None and job.checkpoint_path.is_file():
        info(f"retained resumable ASR checkpoint for {job.path.name}")


def align_and_write_all(
    jobs: list[AudioJob],
    args: TranscriptionConfig,
    device: str,
    align_dtype: torch.dtype,
    stats: RunStats,
    *,
    publish_outputs: bool = True,
) -> None:
    """Align completed ASR text and publish each job transactionally."""
    write_empty_jobs(jobs, publish_outputs=publish_outputs)
    alignment_jobs = [job for job in jobs if job.error is None and job.has_text]
    if not alignment_jobs:
        return

    # The 2B ASR and 300M aligner are intentionally never resident together.
    # Word-aligned calls trade ASR reuse for a substantially lower memory peak.
    evict_current_asr_owner()

    def reload_fn(job: AudioJob) -> np.ndarray:
        return reload_audio_for_alignment(job, args)

    memory_budget = int(args.audio_memory_gb * 1024**3)
    estimates = [estimated_decoded_bytes(job, memory_budget) for job in alignment_jobs]
    tokenizer = None
    model = None
    try:
        with PairBudgetPrefetch(
            alignment_jobs,
            reload_fn,
            estimates,
            memory_budget,
        ) as prefetch:
            started = time.perf_counter()
            try:
                tokenizer, model = load_aligner(device, align_dtype)
            except Exception as exc:
                stats.align_load_seconds = time.perf_counter() - started
                for job in alignment_jobs:
                    job.error = f"aligner load failed: {exc}"
                    report_retained_checkpoint(job)
                    info(f"[error] {job.path}: {job.error}")
                return
            stats.align_load_seconds = time.perf_counter() - started
            info(f"aligner loaded in {fmt_dur(stats.align_load_seconds)}")
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            for alignment_index, (job, future) in enumerate(prefetch, start=1):
                raise_if_cancelled()
                audio = None
                emissions = None
                try:
                    audio = future.result()
                    raise_if_cancelled()
                    started = time.perf_counter()
                    emissions, stride = compute_emissions_streaming(
                        audio,
                        model,
                        args.align_batch_size,
                        job.path.name,
                    )
                    raise_if_cancelled()
                    stats.emissions_seconds += time.perf_counter() - started

                    started = time.perf_counter()
                    words, fallback_count = align_words(
                        emissions,
                        stride,
                        tokenizer,
                        job.segment_times,
                        job.segment_texts,
                        args.language,
                    )
                    raise_if_cancelled()
                    stats.viterbi_seconds += time.perf_counter() - started
                    job.fallback_alignments = fallback_count
                    cues = build_cues(
                        words,
                        args.max_chars,
                        args.max_cue_dur,
                        args.max_gap,
                        media_duration=job.duration,
                    )
                    complete_job_result(
                        job, cues, words, publish_outputs=publish_outputs
                    )
                    info(
                        f"wrote {job.path.name}: {len(words)} words, {len(cues)} cues"
                        + (
                            f", {fallback_count} approximate segments"
                            if fallback_count
                            else ""
                        )
                    )
                except Exception as exc:
                    job.error = f"alignment/output failed: {exc}"
                    report_retained_checkpoint(job)
                    info(f"[error] {job.path}: {job.error}")
                finally:
                    job.audio = None
                    emissions = None
                    audio = None
                    if alignment_index % ALIGNMENT_GC_INTERVAL == 0:
                        gc.collect()
    finally:
        if device == "cuda" and torch.cuda.is_available():
            stats.peak_cuda_gib = max(
                stats.peak_cuda_gib,
                torch.cuda.max_memory_allocated() / 1024**3,
            )
            stats.peak_cuda_reserved_gib = max(
                stats.peak_cuda_reserved_gib,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
        model = None
        tokenizer = None
        release_job_audio(alignment_jobs)
        gc.collect()
        empty_device_cache(device)
