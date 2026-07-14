from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from cohere_transcribe import TranscriptionOptions
from cohere_transcribe.config import config_from_options, validate_args
from cohere_transcribe.models import (
    AudioJob,
    PreparedAudio,
    PreparedGroup,
    PreparedJobResult,
    RunStats,
    SourceSnapshot,
)
from cohere_transcribe.output.pipeline import align_and_write_all
from cohere_transcribe.pipeline.transcription import transcribe_all
from cohere_transcribe.runtime.resources import ModelResources


def test_model_resources_detect_a_fatal_generation_circuit_and_evict() -> None:
    class Controller:
        circuit_breaker_error = "CUDA illegal memory access"

    class Model:
        _transcribe_batch_controller = Controller()

    resources = ModelResources()
    resources.acquire_asr(
        "cpu", torch.float32, loader=lambda *_args: (object(), Model())
    )
    assert resources.asr_circuit_broken
    resources.evict_asr()
    assert not resources.has_asr
    assert not resources.asr_circuit_broken


def test_model_resources_reload_when_device_or_precision_changes() -> None:
    loads: list[tuple[str, torch.dtype]] = []

    def loader(device: str, dtype: torch.dtype):
        loads.append((device, dtype))
        return object(), object()

    resources = ModelResources()
    try:
        first = resources.acquire_asr("cpu", torch.float32, loader=loader)
        second = resources.acquire_asr("cpu", torch.float32, loader=loader)
        third = resources.acquire_asr("cpu", torch.float16, loader=loader)
    finally:
        resources.close()

    assert first[2] is True
    assert second[2] is False
    assert third[2] is True
    assert loads == [("cpu", torch.float32), ("cpu", torch.float16)]


def test_pipeline_submits_first_preparation_before_loading_asr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.pipeline.transcription as pipeline

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="none",
        capture_result=True,
    )
    args = config_from_options(
        [os.fspath(source)], TranscriptionOptions(vad="none", alignment="none")
    )
    validate_args(args)
    order: list[str] = []

    class ImmediateFuture:
        def __init__(self, value) -> None:
            self.value = value

        def result(self):
            return self.value

    class ImmediateExecutor:
        def submit(self, fn, *args):
            order.append("preparation-submitted")
            return ImmediateFuture(fn(*args))

    @contextlib.contextmanager
    def immediate_executor(**_kwargs):
        yield ImmediateExecutor()

    def prepare(group, _args, _workers):
        order.append("preparation-started")
        return PreparedGroup(
            results=[
                PreparedJobResult(
                    job=group[0],
                    prepared=PreparedAudio(
                        audio=np.zeros(16_000, dtype=np.float32),
                        segment_times=[(0.0, 1.0)],
                        speech_spans=[(0.0, 1.0)],
                        decode_seconds=0.0,
                        vad_seconds=0.0,
                        vad_engine="none",
                        decode_backend="test",
                    ),
                )
            ]
        )

    def load(_device, _dtype):
        order.append("asr-load")
        return object(), object()

    monkeypatch.setattr(pipeline, "cancellable_executor", immediate_executor)
    monkeypatch.setattr(pipeline, "prepare_source_group", prepare)
    monkeypatch.setattr(pipeline, "load_asr", load)
    monkeypatch.setattr(
        pipeline, "validate_processor_single_row_window", lambda *_args: 30.0
    )
    monkeypatch.setattr(pipeline, "transcribe_group", lambda *_args: 0.0)
    monkeypatch.setattr(pipeline, "finalize_completed_asr_jobs", lambda *_a, **_k: None)

    resources = ModelResources()
    try:
        transcribe_all(
            [job],
            args,
            "cpu",
            torch.float32,
            RunStats(),
            publish_outputs=False,
            resources=resources,
        )
    finally:
        resources.close()

    assert order[:3] == [
        "preparation-submitted",
        "preparation-started",
        "asr-load",
    ]


def test_pipeline_invalidates_a_cached_model_after_fatal_circuit_breaker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.pipeline.transcription as pipeline

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="none",
        capture_result=True,
    )
    args = config_from_options(
        [os.fspath(source)], TranscriptionOptions(vad="none", alignment="none")
    )
    validate_args(args)

    class Controller:
        circuit_breaker_error: str | None = None

    class Model:
        _transcribe_batch_controller = Controller()

    model = Model()
    prepared = PreparedGroup(
        results=[
            PreparedJobResult(
                job=job,
                prepared=PreparedAudio(
                    audio=np.zeros(16_000, dtype=np.float32),
                    segment_times=[(0.0, 1.0)],
                    speech_spans=[(0.0, 1.0)],
                    decode_seconds=0.0,
                    vad_seconds=0.0,
                    vad_engine="none",
                    decode_backend="test",
                ),
            )
        ]
    )

    def transcribe_group_with_fatal_circuit(*_args):
        model._transcribe_batch_controller.circuit_breaker_error = "fatal CUDA state"
        return 0.0

    monkeypatch.setattr(pipeline, "prepare_source_group", lambda *_args: prepared)
    monkeypatch.setattr(pipeline, "load_asr", lambda *_args: (object(), model))
    monkeypatch.setattr(
        pipeline, "validate_processor_single_row_window", lambda *_args: 30.0
    )
    monkeypatch.setattr(
        pipeline, "transcribe_group", transcribe_group_with_fatal_circuit
    )
    monkeypatch.setattr(pipeline, "finalize_completed_asr_jobs", lambda *_a, **_k: None)

    resources = ModelResources()
    transcribe_all(
        [job],
        args,
        "cpu",
        torch.float32,
        RunStats(),
        publish_outputs=False,
        resources=resources,
    )

    assert not resources.has_asr
    resources.close()


def test_word_alignment_evicts_reusable_asr_before_loading_aligner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.output.pipeline as output_pipeline

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="word",
        duration=1.0,
        segment_times=[(0.0, 1.0)],
        segment_texts=["text"],
        capture_result=True,
    )
    args = config_from_options(
        [os.fspath(source)],
        TranscriptionOptions(vad="none", alignment="word"),
    )
    validate_args(args)
    resources = ModelResources()
    resources.acquire_asr(
        "cpu", torch.float32, loader=lambda *_args: (object(), object())
    )

    def fail_aligner_load(*_args):
        assert not resources.has_asr
        raise RuntimeError("expected test stop")

    monkeypatch.setattr(output_pipeline, "load_aligner", fail_aligner_load)
    monkeypatch.setattr(
        output_pipeline,
        "reload_audio_for_alignment",
        lambda *_args: np.zeros(16_000, dtype=np.float32),
    )

    align_and_write_all(
        [job],
        args,
        "cpu",
        torch.float32,
        RunStats(),
        publish_outputs=False,
    )

    assert not resources.has_asr
    assert job.error == "aligner load failed: expected test stop"
    resources.close()
