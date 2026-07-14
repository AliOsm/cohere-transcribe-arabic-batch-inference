"""High-level orchestration for adaptive Cohere ASR inference."""

from __future__ import annotations

import concurrent.futures
import contextlib
import time
from collections import deque
from collections.abc import Sequence

from ..cancellation import cancellable_executor, raise_if_cancelled
from ..models import (
    BAR_FMT,
    AudioJob,
    RunStats,
    SegmentRef,
    TranscriptionConfig,
    info,
)
from ..progress import progress_bar
from .batching import (
    ASRBatchController,
    record_generation_batch,
    record_prepared_batch,
)
from .execution import (
    classify_and_record_asr_failure,
    finish_asr_batch,
    handle_asr_batch_failure,
)
from .generation import (
    ASRGenerationResult,
    PreparedASRBatch,
    generate_asr_batch,
    prepare_asr_batch,
)


def transcribe_group(
    processor,
    model,
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    stats: RunStats,
) -> float:
    refs = [
        SegmentRef(job, segment_index, start, end)
        for job in jobs
        if job.error is None
        for segment_index, (start, end) in enumerate(job.segment_times)
    ]
    refs.sort(key=lambda ref: (-ref.duration, ref.job.index, ref.segment_index))
    if not refs:
        return 0.0

    controller = getattr(model, "_transcribe_batch_controller", None)
    if controller is None:
        controller = ASRBatchController.create(args, model, refs)
        model._transcribe_batch_controller = controller
        info(
            f"ASR batch controller: start {controller.initial_size}, "
            f"cap {controller.max_size}, padded-audio budget "
            f"{controller.audio_budget_seconds:.0f}s, VRAM target "
            f"{controller.target_vram_ratio:.0%}"
            + (
                f" ({controller.memory_budget_bytes / 1024**3:.2f} GiB usable)"
                if controller.memory_budget_bytes
                else ""
            )
        )
    controller.configure_group(args, refs)

    pending: deque[SegmentRef] = deque(refs)
    started = time.perf_counter()
    bar = progress_bar(
        total=len(refs),
        unit="seg",
        desc="transcribing",
        dynamic_ncols=True,
        bar_format=BAR_FMT,
    )
    try:
        with cancellable_executor(
            max_workers=1, thread_name_prefix="feature-prep"
        ) as executor:
            current_refs = controller.take(pending)
            current_future: concurrent.futures.Future | None = executor.submit(
                prepare_asr_batch, processor, current_refs, args
            )
            while current_refs:
                raise_if_cancelled()
                active_refs = [ref for ref in current_refs if ref.job.error is None]
                skipped = len(current_refs) - len(active_refs)
                if skipped:
                    bar.update(skipped)

                prepared: PreparedASRBatch | None = None
                preparation_error = ""
                preparation_kind = ""
                wait_started = time.perf_counter()
                try:
                    assert current_future is not None
                    prepared = current_future.result()
                except Exception as exc:
                    preparation_kind, preparation_error = (
                        classify_and_record_asr_failure(
                            exc,
                            controller,
                        )
                    )
                finally:
                    stats.asr_feature_wait_seconds += time.perf_counter() - wait_started

                if skipped:
                    if prepared is not None:
                        record_prepared_batch(stats, prepared, discarded=True)
                        prepared = None
                    if active_refs:
                        try:
                            prepared = prepare_asr_batch(processor, active_refs, args)
                            record_prepared_batch(stats, prepared)
                            preparation_error = ""
                            preparation_kind = ""
                        except Exception as exc:
                            prepared = None
                            preparation_kind, preparation_error = (
                                classify_and_record_asr_failure(
                                    exc,
                                    controller,
                                )
                            )
                elif prepared is not None:
                    record_prepared_batch(stats, prepared)

                next_refs = controller.take(pending)
                next_future = (
                    executor.submit(prepare_asr_batch, processor, next_refs, args)
                    if next_refs
                    else None
                )

                result: ASRGenerationResult | None = None
                generation_failure = ""
                generation_kind = ""
                if active_refs and prepared is not None and not preparation_error:
                    circuit_error = getattr(controller, "circuit_breaker_error", None)
                    if isinstance(circuit_error, str):
                        generation_kind = "fatal"
                        generation_failure = (
                            f"runtime failure circuit breaker is open: {circuit_error}"
                        )
                    else:
                        try:
                            result = generate_asr_batch(
                                model, prepared, args, args.max_new_tokens
                            )
                            record_generation_batch(stats, prepared, result)
                            controller.record_success(result, len(active_refs))
                        except Exception as exc:
                            generation_kind, generation_failure = (
                                classify_and_record_asr_failure(
                                    exc,
                                    controller,
                                )
                            )

                # Keep tokenizer use serialized with processor feature preparation.
                if next_future is not None:
                    wait_started = time.perf_counter()
                    try:
                        with contextlib.suppress(Exception):
                            next_future.result()
                    finally:
                        stats.asr_feature_wait_seconds += (
                            time.perf_counter() - wait_started
                        )

                if active_refs:
                    if preparation_error:
                        handle_asr_batch_failure(
                            processor,
                            model,
                            active_refs,
                            args,
                            bar,
                            stats,
                            controller,
                            preparation_kind,
                            preparation_error,
                            args.max_new_tokens,
                        )
                    elif generation_kind:
                        handle_asr_batch_failure(
                            processor,
                            model,
                            active_refs,
                            args,
                            bar,
                            stats,
                            controller,
                            generation_kind,
                            generation_failure,
                            args.max_new_tokens,
                        )
                    else:
                        try:
                            assert prepared is not None and result is not None
                            finish_asr_batch(
                                processor,
                                model,
                                prepared,
                                result,
                                args,
                                bar,
                                stats,
                                controller,
                            )
                        except Exception as exc:
                            failure_kind, failure_message = (
                                classify_and_record_asr_failure(
                                    exc,
                                    controller,
                                )
                            )
                            handle_asr_batch_failure(
                                processor,
                                model,
                                active_refs,
                                args,
                                bar,
                                stats,
                                controller,
                                failure_kind,
                                failure_message,
                                args.max_new_tokens,
                            )

                # A failed current batch can lower the persistent controller
                # cap after the next batch has already been prepared. Do not
                # repeat the same known-oversized probe merely because it was
                # one item ahead in the CPU feature pipeline. Account for the
                # discarded preparation work, restore those refs in order, and
                # prepare a batch under the learned cap instead.
                if next_refs and len(next_refs) > controller.current_size:
                    if next_future is not None:
                        try:
                            discarded = next_future.result()
                        except Exception:
                            discarded = None
                        if discarded is not None:
                            record_prepared_batch(stats, discarded, discarded=True)
                            discarded = None
                    pending.extendleft(reversed(next_refs))
                    next_refs = controller.take(pending)
                    next_future = executor.submit(
                        prepare_asr_batch, processor, next_refs, args
                    )
                current_refs = next_refs
                current_future = next_future
    finally:
        bar.close()
    stats.final_batch_size = controller.current_size
    stats.final_batch_cap = controller.max_size
    return time.perf_counter() - started
