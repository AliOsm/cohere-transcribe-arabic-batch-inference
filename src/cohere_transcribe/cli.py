"""Command-line adapter for the shared transcription runtime."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict

from ._environment import configure_runtime_environment
from .api.types import TranscriptionError
from .cancellation import (
    TerminationRequested,
    cancellation_signal_handlers,
    reset_cancellation,
)
from .config import options_from_config, parse_args, validate_args
from .models import INDENT, default_output_mode
from .preflight import preflight_runtime
from .state import release_all_output_locks


def configure_cli_environment() -> None:
    """Set CLI defaults without replacing caller-provided runtime configuration."""
    configure_runtime_environment()


def _main(argv: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    validate_args(args)
    requested_configuration = asdict(args)
    requested_options = options_from_config(args, publication_enabled=True)

    # Establish process defaults before importing PyTorch and runtime modules.
    configure_cli_environment()
    default_output_mode()
    runtime_import_started = time.perf_counter()
    from .runtime.engine import execute

    runtime_import_seconds = time.perf_counter() - runtime_import_started
    try:
        run = execute(
            args,
            requested_options,
            requested_configuration=requested_configuration,
            resources=None,
            publication_enabled=True,
            console=True,
            runtime_import_seconds=runtime_import_seconds,
            started=started,
            preflight=preflight_runtime,
        )
    except TranscriptionError as exc:
        raise SystemExit(str(exc)) from exc
    return 0 if run.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    reset_cancellation()
    with cancellation_signal_handlers():
        try:
            return _main(argv)
        finally:
            release_all_output_locks()


def cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print(
            f"\n{INDENT}Interrupted; the active output commit was rolled back. "
            "Files completed earlier remain published.",
            flush=True,
        )
        return 130
    except TerminationRequested:
        print(
            f"\n{INDENT}Termination requested; active work was cancelled and "
            "completed files remain published.",
            flush=True,
        )
        return 143


if __name__ == "__main__":
    raise SystemExit(cli())
