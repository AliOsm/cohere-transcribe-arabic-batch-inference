"""Process-wide cooperative cancellation for bounded worker pipelines."""

from __future__ import annotations

import concurrent.futures
import contextlib
import signal
import subprocess
import threading
from collections.abc import Iterator


class TerminationRequested(BaseException):
    """Raised on SIGTERM so normal cleanup runs before exit status 143."""


_cancelled = threading.Event()
_process_lock = threading.RLock()
_active_processes: set[subprocess.Popen] = set()


def reset_cancellation() -> None:
    """Reset state at the start of a new in-process CLI invocation."""
    with _process_lock:
        if _active_processes:
            raise RuntimeError("cannot reset cancellation while decoders are active")
        _cancelled.clear()


def cancellation_requested() -> bool:
    return _cancelled.is_set()


def raise_if_cancelled() -> None:
    if cancellation_requested():
        raise KeyboardInterrupt


def request_cancellation() -> None:
    """Signal workers and terminate registered external decoder processes."""
    _cancelled.set()
    with _process_lock:
        processes = tuple(_active_processes)
    for process in processes:
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.kill()


def terminate_process(process: subprocess.Popen) -> None:
    """Kill and reap a child process while preserving the caller's exception."""
    with contextlib.suppress(OSError):
        process.kill()
    with contextlib.suppress(Exception):
        process.wait(timeout=5)


@contextlib.contextmanager
def registered_process(process: subprocess.Popen) -> Iterator[None]:
    with _process_lock:
        _active_processes.add(process)
    try:
        if cancellation_requested():
            terminate_process(process)
            raise KeyboardInterrupt
        yield
    finally:
        with _process_lock:
            _active_processes.discard(process)


@contextlib.contextmanager
def cancellable_executor(
    *, max_workers: int, thread_name_prefix: str
) -> Iterator[concurrent.futures.ThreadPoolExecutor]:
    """Cancel queued futures and active decoders before waiting on interruption."""
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=thread_name_prefix
    )
    try:
        yield executor
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, TerminationRequested)):
            request_cancellation()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


@contextlib.contextmanager
def cancellation_signal_handlers() -> Iterator[None]:
    """Translate process termination signals into cleanup-aware exceptions."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handle_interrupt(_signal_number, _frame) -> None:
        request_cancellation()
        raise KeyboardInterrupt

    def handle_termination(_signal_number, _frame) -> None:
        request_cancellation()
        raise TerminationRequested

    previous_interrupt = signal.signal(signal.SIGINT, handle_interrupt)
    previous_termination = signal.signal(signal.SIGTERM, handle_termination)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_interrupt)
        signal.signal(signal.SIGTERM, previous_termination)
