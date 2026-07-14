from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import replace

import pytest
import torch

from cohere_transcribe import (
    Transcriber,
    TranscriberBusyError,
    TranscriberClosedError,
    TranscriptionRun,
    TranscriptionRuntimeError,
    transcribe,
)
from cohere_transcribe.runtime.resources import ModelResources

from ._support import patch_execute, result, run_for


def test_reusable_session_retains_compatible_asr_and_close_releases_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[tuple[str, torch.dtype]] = []
    resource_ids: list[int] = []
    resources_seen: list[ModelResources] = []

    def loader(device: str, dtype: torch.dtype):
        loads.append((device, dtype))
        return object(), object()

    def fake_execute(_args, requested_options, *, resources, **_kwargs):
        resources.acquire_asr("cpu", torch.float32, loader=loader)
        resource_ids.append(id(resources))
        resources_seen.append(resources)
        return run_for(requested_options, result("done.wav"))

    patch_execute(monkeypatch, fake_execute)
    session = Transcriber()
    session.transcribe("first.wav")
    session.transcribe("second.wav")

    assert loads == [("cpu", torch.float32)]
    assert len(set(resource_ids)) == 1
    assert resources_seen[-1].has_asr
    session.close()
    assert not resources_seen[-1].has_asr
    with pytest.raises(TranscriberClosedError):
        session.transcribe("third.wav")
    session.close()


def test_one_shot_transcribe_always_closes_its_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[ModelResources] = []

    def fake_execute(_args, requested_options, *, resources, **_kwargs):
        resources.acquire_asr(
            "cpu", torch.float32, loader=lambda *_args: (object(), object())
        )
        captured.append(resources)
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    transcribe("one.wav")

    assert len(captured) == 1
    assert not captured[0].has_asr
    with pytest.raises(RuntimeError, match="closed"):
        captured[0].acquire_asr(
            "cpu", torch.float32, loader=lambda *_args: (object(), object())
        )


def test_context_manager_closes_after_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources_seen: list[ModelResources] = []

    def fake_execute(_args, _requested_options, *, resources, **_kwargs):
        resources.acquire_asr(
            "cpu", torch.float32, loader=lambda *_args: (object(), object())
        )
        resources_seen.append(resources)
        raise RuntimeError("unexpected failure")

    patch_execute(monkeypatch, fake_execute)
    with pytest.raises(TranscriptionRuntimeError, match="unexpected failure"):
        transcribe("one.wav")
    assert resources_seen and not resources_seen[0].has_asr


def test_reentrant_call_on_the_same_session_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session: Transcriber

    def fake_execute(_args, requested_options, **_kwargs):
        with pytest.raises(TranscriberBusyError):
            session.transcribe("nested.wav")
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    session = Transcriber()
    try:
        session.transcribe("outer.wav")
    finally:
        session.close()


def test_same_thread_reentry_through_another_session_is_rejected_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_attempts = 0
    outer = Transcriber()
    inner = Transcriber()

    def fake_execute(args, requested_options, **_kwargs):
        nonlocal nested_attempts
        if args.audio == ["outer.wav"]:
            nested_attempts += 1
            with pytest.raises(TranscriberBusyError, match="process"):
                inner.transcribe("nested.wav")
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    try:
        outer.transcribe("outer.wav")
        recovered = inner.transcribe("after.wav")
    finally:
        outer.close()
        inner.close()

    assert nested_attempts == 1
    assert recovered.ok


def test_reentrant_close_is_rejected_without_closing_the_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    session = Transcriber()

    def fake_execute(_args, requested_options, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            with pytest.raises(TranscriberBusyError, match="active"):
                session.close()
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    try:
        first = session.transcribe("first.wav")
        second = session.transcribe("second.wav")
    finally:
        session.close()

    assert first.ok and second.ok
    assert calls == 2


def test_close_racing_first_use_never_leaks_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    created: list[FakeSession] = []

    class FakeSession:
        def __init__(self, options, _progress) -> None:
            self.options = options
            self.closed = False
            created.append(self)

        def transcribe(self, _audio, *, raise_on_error=False, **_timing):
            del raise_on_error
            return run_for(self.options)

        def close(self) -> None:
            self.closed = True

    class FirstCloseGapLock:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.close_thread: threading.Thread | None = None
            self.close_released = threading.Event()
            self.resume_close = threading.Event()
            self.intercepted = False

        def __enter__(self):
            self.lock.acquire()
            return self

        def __exit__(self, *_args) -> None:
            self.lock.release()
            if threading.current_thread() is self.close_thread and not self.intercepted:
                self.intercepted = True
                self.close_released.set()
                assert self.resume_close.wait(timeout=2)

    monkeypatch.setattr(runtime, "_TranscriberSession", FakeSession)
    session = Transcriber()
    gap_lock = FirstCloseGapLock()
    session._lock = gap_lock  # type: ignore[attr-defined]

    closer = threading.Thread(target=session.close)
    gap_lock.close_thread = closer
    closer.start()
    assert gap_lock.close_released.wait(timeout=2)
    try:
        with contextlib.suppress(TranscriberBusyError, TranscriberClosedError):
            session.transcribe("racing.wav")
    finally:
        gap_lock.resume_close.set()
        closer.join(timeout=2)

    assert not closer.is_alive()
    assert all(implementation.closed for implementation in created)


def test_transcribe_and_second_close_are_rejected_while_close_is_in_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    close_started = threading.Event()
    release_close = threading.Event()

    class BlockingSession:
        def __init__(self, options, _progress) -> None:
            self.options = options

        def transcribe(self, _audio, *, raise_on_error=False, **_timing):
            del raise_on_error
            return run_for(self.options)

        def close(self) -> None:
            close_started.set()
            assert release_close.wait(timeout=2)

    monkeypatch.setattr(runtime, "_TranscriberSession", BlockingSession)
    session = Transcriber()
    session.transcribe("materialize.wav")
    first_errors: list[BaseException] = []
    second_errors: list[BaseException] = []

    def close_into(errors: list[BaseException]) -> None:
        try:
            session.close()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=close_into, args=(first_errors,))
    first.start()
    assert close_started.wait(timeout=2)
    with pytest.raises(TranscriberBusyError, match="clos"):
        session.transcribe("racing.wav")

    second = threading.Thread(target=close_into, args=(second_errors,))
    second.start()
    second.join(timeout=1)
    release_close.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert first_errors == []
    assert len(second_errors) == 1
    assert isinstance(second_errors[0], TranscriberBusyError)
    with pytest.raises(TranscriberClosedError):
        session.transcribe("after-close.wav")


def test_failed_resource_close_rolls_back_closing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cohere_transcribe.runtime.engine as runtime

    close_calls = 0

    class FailingOnceSession:
        def __init__(self, options, _progress) -> None:
            self.options = options

        def transcribe(self, _audio, *, raise_on_error=False, **_timing):
            del raise_on_error
            return run_for(self.options)

        def close(self) -> None:
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise RuntimeError("resource close failed")

    monkeypatch.setattr(runtime, "_TranscriberSession", FailingOnceSession)
    session = Transcriber()
    session.transcribe("before.wav")

    with pytest.raises(RuntimeError, match="resource close failed"):
        session.close()
    assert session.transcribe("after-failed-close.wav").ok
    session.close()

    assert close_calls == 2
    with pytest.raises(TranscriberClosedError):
        session.transcribe("after-successful-close.wav")


def test_process_runtime_serializes_two_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    entered = threading.Barrier(2)
    results: list[TranscriptionRun] = []
    failures: list[BaseException] = []

    def fake_execute(_args, requested_options, **kwargs):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        run = run_for(requested_options)
        return replace(
            run,
            statistics=replace(
                run.statistics,
                serialization_wait_seconds=kwargs["serialization_wait_seconds"],
            ),
        )

    def worker(session: Transcriber, path: str) -> None:
        try:
            entered.wait(timeout=2)
            results.append(session.transcribe(path))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            failures.append(exc)

    patch_execute(monkeypatch, fake_execute)
    first = Transcriber()
    second = Transcriber()
    threads = [
        threading.Thread(target=worker, args=(first, "first.wav")),
        threading.Thread(target=worker, args=(second, "second.wav")),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        first.close()
        second.close()

    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    assert len(results) == 2
    assert maximum_active == 1
    assert max(run.statistics.serialization_wait_seconds for run in results) >= 0.02


def test_same_session_concurrent_calls_never_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    guard = threading.Lock()
    start = threading.Barrier(2)
    outcomes: list[TranscriptionRun | BaseException] = []

    def fake_execute(_args, requested_options, **kwargs):
        nonlocal active, maximum_active
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        run = run_for(requested_options)
        return replace(
            run,
            statistics=replace(
                run.statistics,
                serialization_wait_seconds=kwargs["serialization_wait_seconds"],
            ),
        )

    def worker(path: str) -> None:
        try:
            start.wait(timeout=2)
            outcomes.append(session.transcribe(path))
        except BaseException as exc:
            outcomes.append(exc)

    patch_execute(monkeypatch, fake_execute)
    session = Transcriber()
    threads = [
        threading.Thread(target=worker, args=("first.wav",)),
        threading.Thread(target=worker, args=("second.wav",)),
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
    finally:
        session.close()

    assert all(not thread.is_alive() for thread in threads)
    assert maximum_active == 1
    assert len(outcomes) == 2
    assert all(
        isinstance(outcome, (TranscriptionRun, TranscriberBusyError))
        for outcome in outcomes
    )
    waits = [
        outcome.statistics.serialization_wait_seconds
        for outcome in outcomes
        if isinstance(outcome, TranscriptionRun)
    ]
    assert len(waits) == 2
    assert max(waits) >= 0.02
