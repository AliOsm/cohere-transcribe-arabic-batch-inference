from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import cohere_transcribe
from cohere_transcribe import (
    BatchTranscriptionError,
    Transcriber,
    TranscriptionInputError,
    TranscriptionOptions,
    TranscriptionRun,
    transcribe,
)
from cohere_transcribe.models import AudioJob, RunStats, SourceSnapshot
from cohere_transcribe.output.pipeline import align_and_write_all
from cohere_transcribe.pipeline.transcription import transcribe_all
from cohere_transcribe.runtime.resources import ModelResources

from ._support import patch_cpu_runtime, run_for
from ._support import statistics as zero_statistics


def test_second_reusable_session_takes_exclusive_retained_asr_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    resources_seen: list[ModelResources] = []
    models: list[object] = []

    def loader(_device: str, _dtype: torch.dtype) -> tuple[object, object]:
        model = object()
        models.append(model)
        return object(), model

    def fake_execute(
        _args, requested_options, *, resources: ModelResources, **_kwargs
    ) -> TranscriptionRun:
        resources.acquire_asr("cpu", torch.float32, loader=loader)
        resources_seen.append(resources)
        return run_for(requested_options)

    monkeypatch.setattr(runtime, "execute", fake_execute)
    first = Transcriber()
    second = Transcriber()
    try:
        first.transcribe("first.wav")
        assert resources_seen[0].has_asr

        second.transcribe("second.wav")

        assert len(models) == 2
        assert not resources_seen[0].has_asr
        assert resources_seen[1].has_asr
    finally:
        first.close()
        second.close()


def test_checkpoint_only_word_alignment_evicts_another_sessions_asr_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cohere_transcribe.output.pipeline as output_pipeline
    import cohere_transcribe.pipeline.transcription as transcription_pipeline
    from cohere_transcribe.config import config_from_options, validate_args

    source = tmp_path / "restored.wav"
    source.write_bytes(b"fixture")
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="none",
        alignment_mode="word",
        asr_checkpoint_loaded=True,
        duration=1.0,
        segment_times=[(0.0, 1.0)],
        segment_texts=["restored text"],
        capture_result=True,
    )
    options = TranscriptionOptions(vad="none", alignment="word")
    args = config_from_options([str(source)], options)
    validate_args(args)
    owner = ModelResources()
    checkpoint_session = ModelResources()
    asr_loads = 0
    aligner_reached = False

    def load_asr(_device: str, _dtype: torch.dtype) -> tuple[object, object]:
        nonlocal asr_loads
        asr_loads += 1
        return object(), object()

    def fail_unexpected_asr_load(*_args) -> None:
        raise AssertionError("a restored checkpoint must not load ASR")

    def inspect_aligner_load(*_args) -> None:
        nonlocal aligner_reached
        aligner_reached = True
        assert not owner.has_asr
        assert not checkpoint_session.has_asr
        raise RuntimeError("expected alignment test stop")

    owner.acquire_asr("cpu", torch.float32, loader=load_asr)
    assert owner.has_asr
    monkeypatch.setattr(transcription_pipeline, "load_asr", fail_unexpected_asr_load)
    monkeypatch.setattr(
        output_pipeline,
        "reload_audio_for_alignment",
        lambda *_args: torch.zeros(16_000).numpy(),
    )
    monkeypatch.setattr(output_pipeline, "load_aligner", inspect_aligner_load)
    try:
        transcribe_all(
            [job],
            args,
            "cpu",
            torch.float32,
            RunStats(),
            publish_outputs=False,
            resources=checkpoint_session,
        )
        assert owner.has_asr
        assert not checkpoint_session.has_asr

        align_and_write_all(
            [job],
            args,
            "cpu",
            torch.float32,
            RunStats(),
            publish_outputs=False,
        )

        assert aligner_reached
        assert asr_loads == 1
        assert not owner.has_asr
        assert job.error == "aligner load failed: expected alignment test stop"
    finally:
        checkpoint_session.close()
        owner.close()


def test_lazy_runtime_import_failure_is_a_typed_api_error() -> None:
    root = Path(__file__).resolve().parents[2]
    script = f"""
import builtins
import sys
sys.path.insert(0, {str(root / "src")!r})
import cohere_transcribe
assert 'torch' not in sys.modules
real_import = builtins.__import__
def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.endswith('runtime.engine'):
        raise ModuleNotFoundError("No module named 'torch'")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = blocked_import
try:
    cohere_transcribe.transcribe('unused.wav')
except cohere_transcribe.TranscriptionRuntimeError as exc:
    assert isinstance(exc.__cause__, ModuleNotFoundError)
    assert "No module named 'torch'" in str(exc)
else:
    raise AssertionError('lazy runtime import failure escaped its typed API boundary')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_callback_failure_after_completion_is_typed_and_does_not_relabel_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = patch_cpu_runtime(monkeypatch)
    from cohere_transcribe.output.pipeline import write_segment_timed_outputs

    callback_error_type = getattr(cohere_transcribe, "ProgressCallbackError", None)
    assert callback_error_type is not None, (
        "ProgressCallbackError must be exported by the public API"
    )

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    jobs_seen: list[AudioJob] = []

    def fake_transcribe(jobs, args, *_args, publish_outputs: bool, **_kwargs) -> None:
        jobs_seen.extend(jobs)
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = ["completed text"]
        write_segment_timed_outputs(jobs, args, publish_outputs=publish_outputs)

    def fail_after_completion(event) -> None:
        if event.message and "wrote clip.wav" in event.message:
            raise RuntimeError("application callback failed")

    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )

    with pytest.raises(callback_error_type, match="application callback failed"):
        transcribe(
            source,
            options=TranscriptionOptions(vad="none"),
            progress=fail_after_completion,
        )

    assert len(jobs_seen) == 1
    job = jobs_seen[0]
    assert job.result_completed
    assert job.result_payload is not None
    assert job.result_payload["transcript"] == ["completed text"]
    assert job.error is None
    assert not job.published
    assert job.written == []


def test_first_api_call_includes_runtime_setup_in_elapsed_and_import_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = patch_cpu_runtime(monkeypatch)
    import cohere_transcribe._environment as environment
    from cohere_transcribe.output.pipeline import write_text_only_outputs

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    setup_delay = 0.04

    def delayed_environment_setup() -> None:
        time.sleep(setup_delay)

    def fake_transcribe(
        jobs, _args, *_unused, publish_outputs: bool, **_kwargs
    ) -> None:
        for job in jobs:
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.segment_texts = ["text"]
        write_text_only_outputs(jobs, publish_outputs=publish_outputs)

    monkeypatch.setattr(
        environment, "configure_runtime_environment", delayed_environment_setup
    )
    monkeypatch.setattr(
        runtime.transcription_pipeline, "transcribe_all", fake_transcribe
    )

    run = transcribe(
        source,
        options=TranscriptionOptions(vad="none", alignment="none"),
    )

    assert run.statistics.runtime_import_seconds >= setup_delay * 0.8
    assert run.statistics.elapsed_seconds >= setup_delay * 0.8
    assert (
        run.statistics.elapsed_seconds
        >= run.statistics.runtime_import_seconds
        + run.statistics.input_validation_seconds
    )


def test_reusable_batch_error_receives_final_facade_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    def fake_execute(_args, requested_options, **_kwargs) -> TranscriptionRun:
        time.sleep(0.01)
        run = TranscriptionRun(
            results=(),
            requested_options=requested_options,
            resolved_options=requested_options,
            statistics=zero_statistics(),
            errors=("profile publication failed",),
        )
        return replace(
            run,
            statistics=replace(
                run.statistics,
                successful_audio_seconds=2.0,
            ),
        )

    monkeypatch.setattr(runtime, "execute", fake_execute)
    with (
        Transcriber() as session,
        pytest.raises(BatchTranscriptionError) as captured,
    ):
        session.transcribe("failed.wav", raise_on_error=True)

    statistics = captured.value.run.statistics
    assert statistics.elapsed_seconds >= 0.008
    assert statistics.real_time_factor_x == pytest.approx(
        2.0 / statistics.elapsed_seconds
    )


def test_symlink_loop_is_reported_as_an_input_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_cpu_runtime(monkeypatch)
    loop = tmp_path / "loop.wav"
    loop.symlink_to(loop.name)

    with pytest.raises(TranscriptionInputError, match=r"[Ss]ymlink loop"):
        transcribe(loop, options=TranscriptionOptions(device="cpu"))
