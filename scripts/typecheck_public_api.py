"""Strict type-checking fixture for the installed public API."""

from pathlib import Path

from cohere_transcribe import (
    ProgressEvent,
    PublicationOptions,
    Transcriber,
    TranscriptionOptions,
    TranscriptionRun,
    transcribe,
)


def report_progress(event: ProgressEvent) -> None:
    if event.total is not None and event.current is not None:
        current: int = event.current
        total: int = event.total
        _ = current, total


def consume_public_api() -> tuple[TranscriptionRun, TranscriptionRun]:
    options = TranscriptionOptions(
        language="ar",
        alignment="segment",
        publication=PublicationOptions(formats=("txt", "json")),
    )
    single = transcribe(Path("clip.wav"), options=options, progress=report_progress)
    with Transcriber(options, progress=report_progress) as session:
        batch = session.transcribe(["first.wav", Path("nested")])
    return single, batch
