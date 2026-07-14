from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from cohere_transcribe.cancellation import (
    TerminationRequested,
    cancellable_executor,
    cancellation_requested,
    cancellation_signal_handlers,
    registered_process,
    request_cancellation,
    reset_cancellation,
)


class FakeProcess:
    def __init__(self) -> None:
        self.killed = False

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        self.killed = True

    def wait(self, _timeout: float | None = None) -> int:
        return 0


def setup_function() -> None:
    reset_cancellation()


def teardown_function() -> None:
    reset_cancellation()


def test_request_cancellation_terminates_registered_process() -> None:
    process = FakeProcess()
    with registered_process(process):
        request_cancellation()
        assert cancellation_requested()
        assert process.killed


def test_cancellable_executor_signals_worker_before_waiting() -> None:
    worker_observed = threading.Event()

    def worker() -> None:
        while not cancellation_requested():
            time.sleep(0.001)
        worker_observed.set()

    with (
        pytest.raises(KeyboardInterrupt),
        cancellable_executor(max_workers=1, thread_name_prefix="test") as executor,
    ):
        executor.submit(worker)
        raise KeyboardInterrupt
    assert worker_observed.is_set()


def test_ordinary_executor_error_does_not_poison_later_fallback_work() -> None:
    with (
        pytest.raises(RuntimeError, match="backend unavailable"),
        cancellable_executor(max_workers=1, thread_name_prefix="test"),
    ):
        raise RuntimeError("backend unavailable")

    assert not cancellation_requested()


def test_sigterm_runs_cleanup_aware_exception_path() -> None:
    with pytest.raises(TerminationRequested), cancellation_signal_handlers():
        os.kill(os.getpid(), signal.SIGTERM)
    assert cancellation_requested()
