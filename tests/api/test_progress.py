from __future__ import annotations

import threading
import time

import pytest

from cohere_transcribe import (
    ProgressEvent,
    Transcriber,
    TranscriberBusyError,
    transcribe,
)
from cohere_transcribe.models import info
from cohere_transcribe.progress import progress_bar

from ._support import patch_execute, run_for


@pytest.mark.parametrize("operation", ["transcribe", "close"])
def test_worker_progress_callback_reentry_is_rejected_without_blocking(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    outcomes: list[BaseException] = []
    callback_finished = threading.Event()
    session: Transcriber

    def callback(_event: ProgressEvent) -> None:
        try:
            if operation == "transcribe":
                session.transcribe("nested.wav")
            else:
                session.close()
        except BaseException as exc:
            outcomes.append(exc)
        finally:
            callback_finished.set()

    def fake_execute(_args, requested_options, **_kwargs):
        worker = threading.Thread(target=info, args=("worker progress",), daemon=True)
        worker.start()
        assert callback_finished.wait(timeout=2), "progress callback reentry blocked"
        worker.join(timeout=2)
        assert not worker.is_alive()
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    session = Transcriber(progress=callback)
    try:
        run = session.transcribe("outer.wav")
    finally:
        session.close()

    assert run.ok
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], TranscriberBusyError)


def test_api_is_quiet_by_default_and_progress_events_are_ordered(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[ProgressEvent] = []

    def fake_execute(_args, requested_options, **_kwargs):
        info("preparing")
        for _ in progress_bar(range(3), desc="ASR", total=3):
            pass
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    transcribe("quiet.wav")
    assert capsys.readouterr() == ("", "")

    transcribe("progress.wav", progress=events.append)
    assert events[0] == ProgressEvent(stage="message", message="    preparing")
    assert events[1:] == [
        ProgressEvent(stage="ASR", current=0, total=3),
        ProgressEvent(stage="ASR", current=1, total=3),
        ProgressEvent(stage="ASR", current=2, total=3),
        ProgressEvent(stage="ASR", current=3, total=3),
    ]


def test_progress_callback_is_never_invoked_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    lock = threading.Lock()
    received: list[str] = []

    def callback(event: ProgressEvent) -> None:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.005)
        if event.message is not None:
            received.append(event.message)
        with lock:
            active -= 1

    def fake_execute(_args, requested_options, **_kwargs):
        threads = [
            threading.Thread(target=info, args=(f"message-{index}",))
            for index in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
        return run_for(requested_options)

    patch_execute(monkeypatch, fake_execute)
    transcribe("progress.wav", progress=callback)

    assert maximum_active == 1
    assert sorted(received) == [f"    message-{index}" for index in range(6)]
