from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

from cohere_transcribe.asr import execution as asr
from cohere_transcribe.asr.batching import (
    ASRBatchController,
    default_asr_batch_size,
    runtime_failure_fingerprint,
)
from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.models import RunStats, SegmentRef


def controller(size: int = 8) -> ASRBatchController:
    return ASRBatchController(
        current_size=size,
        max_size=size,
        audio_budget_seconds=float(size * 30),
        adaptive=False,
        target_vram_ratio=0.9,
    )


def test_default_batch_sizes_are_device_appropriate() -> None:
    assert default_asr_batch_size("cuda") == 24
    assert default_asr_batch_size("mps") == 8
    assert default_asr_batch_size("cpu") == 4


def test_runtime_failure_fingerprint_handles_an_empty_message() -> None:
    assert runtime_failure_fingerprint(RuntimeError()) == "RuntimeError: <no message>"


def test_failure_classifier_recognizes_oom_and_device_fatal_errors() -> None:
    oom_messages = (
        "DefaultCPUAllocator: out of memory",
        "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)",
        "cuDNN error: CUDNN_STATUS_ALLOC_FAILED",
        "cudaErrorMemoryAllocation",
    )
    for message in oom_messages:
        assert asr.classify_asr_failure(RuntimeError(message)) == "oom"
    assert (
        asr.classify_asr_failure(RuntimeError("CUDA error: illegal memory access"))
        == "fatal"
    )


def test_fatal_failure_opens_circuit_before_the_next_top_level_batch() -> None:
    state = controller()
    kind, _message = asr.classify_and_record_asr_failure(
        RuntimeError("CUDA error: illegal memory access"), state
    )
    assert kind == "fatal"
    assert state.circuit_breaker_error is not None

    job = SimpleNamespace(index=0, error=None, path="sample.wav")
    refs = [SegmentRef(job, 0, 0.0, 1.0)]
    args = parse_args(["sample.wav", "--alignment", "segment"])
    validate_args(args)
    with mock.patch.object(asr, "prepare_asr_batch") as prepare:
        asr.transcribe_ref_batch(
            mock.Mock(),
            SimpleNamespace(device=torch.device("cpu")),
            refs,
            args,
            mock.Mock(),
            RunStats(),
            state,
            max_new_tokens=args.max_new_tokens,
        )
    prepare.assert_not_called()
    assert "circuit breaker" in (job.error or "")
    assert (
        asr.classify_asr_failure(RuntimeError("CUDA error: illegal memory access"))
        == "fatal"
    )


def test_repeated_data_local_failures_do_not_poison_untried_healthy_files() -> None:
    jobs = [
        SimpleNamespace(index=index, error=None, path=f"sample-{index}.wav")
        for index in range(5)
    ]
    refs = [
        SegmentRef(job, 0, float(index), float(index + 1))
        for index, job in enumerate(jobs)
    ]
    args = parse_args(["sample.wav", "--alignment", "segment"])
    validate_args(args)
    state = controller()
    attempted: list[list[int]] = []

    def prepare(_processor, batch_refs, _args):
        indices = [ref.job.index for ref in batch_refs]
        attempted.append(indices)
        if any(index < 3 for index in indices):
            raise RuntimeError("malformed sample payload")
        return mock.Mock()

    with (
        mock.patch.object(asr, "prepare_asr_batch", side_effect=prepare),
        mock.patch.object(asr, "record_prepared_batch"),
        mock.patch.object(asr, "generate_asr_batch", return_value=mock.Mock()),
        mock.patch.object(asr, "record_generation_batch"),
        mock.patch.object(asr, "finish_asr_batch") as finish,
    ):
        asr.transcribe_ref_batch(
            mock.Mock(),
            SimpleNamespace(device=torch.device("cpu")),
            refs,
            args,
            mock.Mock(),
            RunStats(),
            state,
            max_new_tokens=args.max_new_tokens,
        )

    assert all(job.error is not None for job in jobs[:3])
    assert all(job.error is None for job in jobs[3:])
    assert [3, 4] in attempted
    assert state.circuit_breaker_error is None
    assert finish.call_count == 1


def test_successful_generation_does_not_hide_repeated_postprocessing_failure() -> None:
    jobs = [
        SimpleNamespace(index=index, error=None, path=f"sample-{index}.wav")
        for index in range(8)
    ]
    refs = [
        SegmentRef(job, 0, float(index), float(index + 1))
        for index, job in enumerate(jobs)
    ]
    args = parse_args(["sample.wav", "--alignment", "segment"])
    validate_args(args)
    state = controller()
    bar = mock.Mock()

    with (
        mock.patch.object(asr, "prepare_asr_batch", return_value=mock.Mock()),
        mock.patch.object(asr, "generate_asr_batch", return_value=mock.Mock()),
        mock.patch.object(asr, "record_prepared_batch"),
        mock.patch.object(asr, "record_generation_batch"),
        mock.patch.object(
            asr,
            "finish_asr_batch",
            side_effect=RuntimeError("tokenizer invariant failed at row 1"),
        ) as finish,
    ):
        asr.transcribe_ref_batch(
            mock.Mock(),
            SimpleNamespace(device=torch.device("cpu")),
            refs,
            args,
            bar,
            RunStats(),
            state,
            max_new_tokens=args.max_new_tokens,
        )

    assert finish.call_count == 15
    assert state.circuit_breaker_error is None
    assert all("tokenizer invariant failed" in (job.error or "") for job in jobs)


def test_one_data_local_failure_does_not_open_circuit_for_healthy_files() -> None:
    jobs = [
        SimpleNamespace(index=index, error=None, path=f"sample-{index}.wav")
        for index in range(8)
    ]
    refs = [
        SegmentRef(job, 0, float(index), float(index + 1))
        for index, job in enumerate(jobs)
    ]
    args = parse_args(["sample.wav", "--alignment", "segment"])
    validate_args(args)
    state = controller()

    def prepare(_processor, batch_refs, _args):
        if any(ref.job.index == 0 for ref in batch_refs):
            raise RuntimeError("bad sample")
        return mock.Mock()

    with (
        mock.patch.object(asr, "prepare_asr_batch", side_effect=prepare),
        mock.patch.object(asr, "record_prepared_batch"),
        mock.patch.object(asr, "generate_asr_batch", return_value=mock.Mock()),
        mock.patch.object(asr, "record_generation_batch"),
        mock.patch.object(asr, "finish_asr_batch") as finish,
    ):
        asr.transcribe_ref_batch(
            mock.Mock(),
            SimpleNamespace(device=torch.device("cpu")),
            refs,
            args,
            mock.Mock(),
            RunStats(),
            state,
            max_new_tokens=args.max_new_tokens,
        )

    assert jobs[0].error == "ASR failed: RuntimeError: bad sample"
    assert all(job.error is None for job in jobs[1:])
    assert state.circuit_breaker_error is None
    assert finish.call_count == 3


def test_recursive_oom_siblings_honor_the_learned_batch_cap() -> None:
    jobs = [
        SimpleNamespace(index=index, error=None, path=f"sample-{index}.wav")
        for index in range(24)
    ]
    refs = [
        SegmentRef(job, 0, float(index), float(index + 1))
        for index, job in enumerate(jobs)
    ]
    args = parse_args(["sample.wav", "--alignment", "segment", "--batch-size", "24"])
    validate_args(args)
    state = controller(24)
    attempts: list[int] = []

    def prepare(_processor, batch_refs, _args):
        return SimpleNamespace(refs=list(batch_refs))

    def generate(_model, prepared, _args, _max_new_tokens):
        batch_size = len(prepared.refs)
        attempts.append(batch_size)
        if batch_size > 6:
            raise torch.OutOfMemoryError("synthetic out of memory")
        return mock.Mock()

    with (
        mock.patch.object(asr, "prepare_asr_batch", side_effect=prepare),
        mock.patch.object(asr, "generate_asr_batch", side_effect=generate),
        mock.patch.object(asr, "record_prepared_batch"),
        mock.patch.object(asr, "record_generation_batch"),
        mock.patch.object(asr, "record_oom_batch"),
        mock.patch.object(asr, "empty_device_cache"),
        mock.patch.object(asr, "finish_asr_batch"),
        mock.patch.object(asr.gc, "collect"),
    ):
        asr.transcribe_ref_batch(
            mock.Mock(),
            SimpleNamespace(device=torch.device("cpu")),
            refs,
            args,
            mock.Mock(),
            RunStats(),
            state,
            max_new_tokens=args.max_new_tokens,
        )

    assert attempts == [24, 12, 6, 6, 6, 6]
    assert state.current_size == 6
    assert all(job.error is None for job in jobs)


def test_high_token_oom_uses_a_retry_local_cap_without_changing_base_cap() -> None:
    jobs = [
        SimpleNamespace(index=index, error=None, path=f"sample-{index}.wav")
        for index in range(24)
    ]
    refs = [
        SegmentRef(job, 0, float(index), float(index + 1))
        for index, job in enumerate(jobs)
    ]
    args = parse_args(["sample.wav", "--alignment", "segment", "--batch-size", "24"])
    validate_args(args)
    state = controller(24)
    attempts: list[int] = []

    def prepare(_processor, batch_refs, _args):
        return SimpleNamespace(refs=list(batch_refs))

    def generate(_model, prepared, _args, _max_new_tokens):
        batch_size = len(prepared.refs)
        attempts.append(batch_size)
        if batch_size > 6:
            raise torch.OutOfMemoryError("synthetic high-token out of memory")
        return mock.Mock()

    with (
        mock.patch.object(asr, "prepare_asr_batch", side_effect=prepare),
        mock.patch.object(asr, "generate_asr_batch", side_effect=generate),
        mock.patch.object(asr, "record_prepared_batch"),
        mock.patch.object(asr, "record_generation_batch"),
        mock.patch.object(asr, "record_oom_batch"),
        mock.patch.object(asr, "empty_device_cache"),
        mock.patch.object(asr, "finish_asr_batch"),
        mock.patch.object(asr.gc, "collect"),
    ):
        asr.transcribe_ref_batch(
            mock.Mock(),
            SimpleNamespace(device=torch.device("cpu")),
            refs,
            args,
            mock.Mock(),
            RunStats(),
            state,
            max_new_tokens=args.max_retry_tokens,
        )

    assert attempts == [24, 12, 6, 6, 6, 6]
    assert (state.current_size, state.max_size) == (24, 24)
    assert all(job.error is None for job in jobs)
