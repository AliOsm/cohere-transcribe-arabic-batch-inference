"""Per-batch ASR execution, retry, and failure-isolation policy."""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from dataclasses import dataclass

from tqdm import tqdm

from ..cancellation import raise_if_cancelled
from ..device import empty_device_cache, is_out_of_memory_error
from ..models import INDENT, RunStats, SegmentRef, TranscriptionConfig, info
from ..progress import write as progress_write
from .batching import (
    ASRBatchController,
    balanced_oom_split,
    record_generation_batch,
    record_oom_batch,
    record_prepared_batch,
    runtime_failure_fingerprint,
)
from .generation import (
    ASRGenerationResult,
    PreparedASRBatch,
    decode_asr_batch,
    generate_asr_batch,
    prepare_asr_batch,
)


@dataclass(slots=True)
class _RetryBatchCap:
    """OOM cap scoped to expensive token retries, never normal ASR batches."""

    current_size: int

    def record_oom(self, attempted_rows: int) -> None:
        self.current_size = max(
            1,
            min(self.current_size, attempted_rows // 2 if attempted_rows > 1 else 1),
        )


def apply_generation_metadata(
    prepared: PreparedASRBatch,
    result: ASRGenerationResult,
    included_ref_indices: set[int] | None = None,
) -> None:
    per_ref_tokens: dict[int, int] = {}
    for row_index, count in enumerate(result.row_token_counts):
        ref_index = int(prepared.chunk_index[row_index][0])
        if included_ref_indices is not None and ref_index not in included_ref_indices:
            continue
        per_ref_tokens[ref_index] = max(per_ref_tokens.get(ref_index, 0), count)
    for ref_index, count in per_ref_tokens.items():
        ref = prepared.refs[ref_index]
        ref.job.generated_tokens[ref.segment_index] = count
    for ref_index in result.repetition_ref_indices:
        if included_ref_indices is not None and ref_index not in included_ref_indices:
            continue
        ref = prepared.refs[ref_index]
        ref.job.repetition_stopped_segments.add(ref.segment_index)


def commit_asr_texts(
    refs: Sequence[SegmentRef], texts: Sequence[str], bar: tqdm
) -> None:
    if len(texts) != len(refs):
        raise RuntimeError(f"ASR returned {len(texts)} texts for {len(refs)} segments")
    for ref, text in zip(refs, texts, strict=True):
        ref.job.segment_texts[ref.segment_index] = text
    bar.update(len(refs))


def retry_token_limit(
    model,
    prompt_length: int,
    current_limit: int,
    requested_maximum: int,
) -> int:
    decoder_config = getattr(model.config, "decoder_config", None) or model.config
    max_positions = int(
        getattr(
            decoder_config, "max_position_embeddings", requested_maximum + prompt_length
        )
    )
    positional_cap = max(1, max_positions - prompt_length)
    ceiling = min(requested_maximum, positional_cap)
    proposed = min(ceiling, max(current_limit + 128, current_limit * 2))
    # A tiny final increment would repeat nearly the complete autoregressive
    # decode. Jump directly to the ceiling when fewer than 128 tokens remain.
    return ceiling if ceiling - proposed < 128 else proposed


def finish_asr_batch(
    processor,
    model,
    prepared: PreparedASRBatch,
    result: ASRGenerationResult,
    args: TranscriptionConfig,
    bar: tqdm,
    stats: RunStats,
    controller: ASRBatchController,
    retry_batch_cap: _RetryBatchCap | None = None,
) -> None:
    started = time.perf_counter()
    texts = decode_asr_batch(processor, result.generated, prepared)
    stats.asr_decode_seconds += time.perf_counter() - started

    retry_indices = set(result.truncated_ref_indices)
    next_limit = retry_token_limit(
        model,
        result.prompt_length,
        result.max_new_tokens,
        args.max_retry_tokens,
    )
    can_retry = (
        args.truncation_policy == "retry"
        and retry_indices
        and next_limit > result.max_new_tokens
    )
    if can_retry:
        keep_indices = [
            index for index in range(len(prepared.refs)) if index not in retry_indices
        ]
        if keep_indices:
            apply_generation_metadata(prepared, result, set(keep_indices))
            commit_asr_texts(
                [prepared.refs[index] for index in keep_indices],
                [texts[index] for index in keep_indices],
                bar,
            )
        retry_refs = [prepared.refs[index] for index in sorted(retry_indices)]
        for ref in retry_refs:
            ref.job.truncation_retried_segments.add(ref.segment_index)
        stats.asr_truncation_retries += len(retry_refs)
        info(
            f"[tokens] retrying {len(retry_refs)} segment(s) with "
            f"max_new_tokens={next_limit}"
        )
        retry_batch_cap = retry_batch_cap or _RetryBatchCap(len(retry_refs))
        transcribe_ref_batch(
            processor,
            model,
            retry_refs,
            args,
            bar,
            stats,
            controller,
            max_new_tokens=next_limit,
            retry_batch_cap=retry_batch_cap,
        )
        return

    apply_generation_metadata(prepared, result)
    if retry_indices:
        for ref_index in retry_indices:
            ref = prepared.refs[ref_index]
            ref.job.token_limit_segments.add(ref.segment_index)
    commit_asr_texts(prepared.refs, texts, bar)


def mark_asr_jobs_failed(refs: Sequence[SegmentRef], message: str, bar: tqdm) -> None:
    affected_jobs = {ref.job.index: ref.job for ref in refs}
    for affected in affected_jobs.values():
        if affected.error is None:
            affected.error = f"ASR failed: {message}"
            info(f"[error] {affected.path}: {affected.error}")
    bar.update(len(refs))


def classify_asr_failure(exc: Exception) -> str:
    """Separate invariant implementation failures from data-local failures."""
    message = str(exc).lower()
    if is_out_of_memory_error(exc):
        return "oom"
    if isinstance(
        exc,
        (
            AssertionError,
            AttributeError,
            ImportError,
            IndexError,
            KeyError,
            NotImplementedError,
            TypeError,
        ),
    ):
        return "fatal"
    if any(
        marker in message
        for marker in (
            "cublas_status",
            "cuda error",
            "cudnn_status",
            "device-side assert",
            "driver shutting down",
            "hip error",
            "illegal memory access",
            "mps backend failed",
            "unspecified launch failure",
        )
    ):
        return "fatal"
    return "error"


def classify_and_record_asr_failure(
    exc: Exception,
    controller: ASRBatchController,
) -> tuple[str, str]:
    """Classify a failure and open the circuit only for known fatal states."""
    kind = classify_asr_failure(exc)
    message = f"{type(exc).__name__}: {exc}"
    if kind == "fatal":
        controller.circuit_breaker_error = runtime_failure_fingerprint(exc)
    return kind, message


def handle_asr_batch_failure(
    processor,
    model,
    refs: Sequence[SegmentRef],
    args: TranscriptionConfig,
    bar: tqdm,
    stats: RunStats,
    controller: ASRBatchController,
    failure_kind: str,
    message: str,
    max_new_tokens: int,
    retry_batch_cap: _RetryBatchCap | None = None,
) -> None:
    if failure_kind == "oom":
        stats.asr_oom_retries += 1
        record_oom_batch(stats, model, refs, max_new_tokens)
        learn_base_batch_cap = max_new_tokens <= args.max_new_tokens
        if learn_base_batch_cap:
            controller.record_oom(len(refs))
        elif retry_batch_cap is not None:
            retry_batch_cap.record_oom(len(refs))
        gc.collect()
        empty_device_cache(model.device.type)
    else:
        learn_base_batch_cap = False

    if failure_kind == "fatal":
        mark_asr_jobs_failed(refs, message, bar)
        return

    if len(refs) > 1:
        midpoint = balanced_oom_split(refs) if failure_kind == "oom" else len(refs) // 2
        if failure_kind == "oom":
            cap_note = (
                f"; future cap {controller.current_size}"
                if learn_base_batch_cap
                else (
                    f"; retry cap {retry_batch_cap.current_size}, base ASR cap unchanged"
                    if retry_batch_cap is not None
                    else "; base ASR cap unchanged"
                )
            )
            progress_write(
                f"{INDENT}[oom] ASR retrying batch {len(refs)} as "
                f"{midpoint}+{len(refs) - midpoint}{cap_note}"
            )
        transcribe_ref_batch(
            processor,
            model,
            refs[:midpoint],
            args,
            bar,
            stats,
            controller,
            max_new_tokens=max_new_tokens,
            retry_batch_cap=retry_batch_cap,
        )
        remaining = [ref for ref in refs[midpoint:] if ref.job.error is None]
        bar.update(len(refs) - midpoint - len(remaining))
        if remaining:
            transcribe_ref_batch(
                processor,
                model,
                remaining,
                args,
                bar,
                stats,
                controller,
                max_new_tokens=max_new_tokens,
                retry_batch_cap=retry_batch_cap,
            )
        return
    mark_asr_jobs_failed(refs, message, bar)


def transcribe_ref_batch(
    processor,
    model,
    refs: Sequence[SegmentRef],
    args: TranscriptionConfig,
    bar: tqdm,
    stats: RunStats,
    controller: ASRBatchController,
    max_new_tokens: int,
    retry_batch_cap: _RetryBatchCap | None = None,
) -> None:
    raise_if_cancelled()
    circuit_error = getattr(controller, "circuit_breaker_error", None)
    if isinstance(circuit_error, str):
        mark_asr_jobs_failed(
            refs,
            f"runtime failure circuit breaker is open: {circuit_error}",
            bar,
        )
        return
    if max_new_tokens > args.max_new_tokens and retry_batch_cap is None:
        retry_batch_cap = _RetryBatchCap(len(refs))
    effective_cap = (
        controller.current_size
        if max_new_tokens <= args.max_new_tokens
        else retry_batch_cap.current_size
        if retry_batch_cap is not None
        else len(refs)
    )
    if len(refs) > effective_cap:
        offset = 0
        while offset < len(refs):
            raise_if_cancelled()
            cap = max(
                1,
                controller.current_size
                if max_new_tokens <= args.max_new_tokens
                else retry_batch_cap.current_size
                if retry_batch_cap is not None
                else len(refs),
            )
            capped_refs = list(refs[offset : offset + cap])
            offset += len(capped_refs)
            active_refs = [ref for ref in capped_refs if ref.job.error is None]
            bar.update(len(capped_refs) - len(active_refs))
            if active_refs:
                transcribe_ref_batch(
                    processor,
                    model,
                    active_refs,
                    args,
                    bar,
                    stats,
                    controller,
                    max_new_tokens=max_new_tokens,
                    retry_batch_cap=retry_batch_cap,
                )
        return
    failure_kind = ""
    failure_message = ""
    prepared: PreparedASRBatch | None = None
    result: ASRGenerationResult | None = None
    try:
        prepared = prepare_asr_batch(processor, refs, args)
        raise_if_cancelled()
        record_prepared_batch(stats, prepared)
        result = generate_asr_batch(model, prepared, args, max_new_tokens)
        raise_if_cancelled()
        record_generation_batch(stats, prepared, result)
        if max_new_tokens <= args.max_new_tokens:
            controller.record_success(result, len(refs))
        finish_asr_batch(
            processor,
            model,
            prepared,
            result,
            args,
            bar,
            stats,
            controller,
            retry_batch_cap,
        )
        return
    except Exception as exc:
        failure_kind, failure_message = classify_and_record_asr_failure(
            exc,
            controller,
        )
    finally:
        result = None
        prepared = None

    handle_asr_batch_failure(
        processor,
        model,
        refs,
        args,
        bar,
        stats,
        controller,
        failure_kind,
        failure_message,
        max_new_tokens,
        retry_batch_cap,
    )
