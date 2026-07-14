"""Stable, dependency-light types for the public Python API."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, overload

AudioInput = str | os.PathLike[str] | Sequence[str | os.PathLike[str]]
Language = Literal["ar", "en"]
OutputFormat = Literal["txt", "srt", "vtt", "json"]
ResultStatus = Literal["completed", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class PublicationOptions:
    """Optional durable output settings for a Python transcription run."""

    formats: tuple[OutputFormat, ...] | None = None
    output_dir: str | os.PathLike[str] | None = None
    existing: Literal["error", "overwrite", "skip"] = "error"
    profile_json: str | os.PathLike[str] | None = None

    def __post_init__(self) -> None:
        if self.formats is not None:
            formats = tuple(dict.fromkeys(self.formats))
            if not formats:
                raise ValueError("formats must contain at least one output format")
            unsupported = set(formats).difference(("txt", "srt", "vtt", "json"))
            if unsupported:
                raise ValueError(
                    "Unsupported output format(s): " + ", ".join(sorted(unsupported))
                )
            object.__setattr__(self, "formats", formats)
        if self.existing not in {"error", "overwrite", "skip"}:
            raise ValueError("existing must be 'error', 'overwrite', or 'skip'")
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.profile_json is not None:
            object.__setattr__(self, "profile_json", Path(self.profile_json))


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    """Complete transcription configuration shared with the command-line interface."""

    language: Language = "ar"
    text_only: bool = False
    recursive: bool = True
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    audio_backend: Literal["auto", "torchcodec", "ffmpeg", "librosa"] = "auto"
    audio_memory_gb: float = 4.0
    preprocess_workers: int | None = None
    pipeline_preparation: bool = True
    vad: Literal["silero", "auditok", "none"] = "silero"
    vad_engine: Literal["auto", "torch", "onnx", "jit"] = "auto"
    vad_batch_size: int = 16
    vad_block_frames: int = 512
    vad_threads: int | None = None
    vad_merge: bool = False
    min_dur: float = 0.5
    max_dur: float = 30.0
    max_silence: float = 0.6
    energy_threshold: float = 50.0
    vad_threshold: float = 0.5
    min_silence_ms: int = 300
    speech_pad_ms: int = 60
    batch_size: int | None = None
    batch_max_size: int | None = None
    batch_audio_seconds: float | None = None
    batch_vram_target: float = 0.9
    adaptive_batch: bool = False
    pin_memory: bool = False
    max_new_tokens: int = 445
    max_retry_tokens: int = 896
    truncation_policy: Literal["retry", "warn"] = "retry"
    stop_repetition_loops: bool = True
    alignment: Literal["word", "segment", "none"] = "segment"
    align_batch_size: int = 4
    align_dtype: Literal["fp32", "fp16"] = "fp32"
    max_chars: int = 80
    max_cue_dur: float = 6.0
    max_gap: float = 0.6
    publication: PublicationOptions | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    """Text generated for one ASR input segment."""

    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionWord:
    """One word with either CTC or approximate timing."""

    start: float
    end: float
    text: str
    segment_index: int
    segment_word_index: int
    timing_source: str


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    """One rendered subtitle cue."""

    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionProvenance:
    """Per-file decoder, VAD, generation, and alignment provenance."""

    decode_backend: str | None = None
    decode_fallback_reason: str | None = None
    vad_engine_requested: str | None = None
    vad_engine_actual: str | None = None
    vad_provider: str | None = None
    vad_fallback_reason: str | None = None
    fallback_alignment_segments: int = 0
    repetition_stopped_segments: tuple[int, ...] = ()
    truncation_retried_segments: tuple[int, ...] = ()
    token_limit_segments: tuple[int, ...] = ()
    generated_tokens_by_segment: tuple[tuple[int, int], ...] = ()
    resumed_from_asr_checkpoint: bool = False
    published: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Immutable result for one expanded input audio file."""

    path: Path
    relative_path: Path
    status: ResultStatus
    text: str | None
    duration: float | None
    segments: tuple[TranscriptionSegment, ...] = ()
    words: tuple[TranscriptionWord, ...] = ()
    cues: tuple[SubtitleCue, ...] = ()
    outputs: tuple[Path, ...] = ()
    error: str | None = None
    provenance: TranscriptionProvenance = field(default_factory=TranscriptionProvenance)


@dataclass(frozen=True, slots=True)
class TranscriptionStatistics:
    """Stable high-level performance and resource statistics for one run."""

    elapsed_seconds: float
    successful_audio_seconds: float
    real_time_factor_x: float
    runtime_import_seconds: float
    serialization_wait_seconds: float
    input_validation_seconds: float
    decode_seconds: float
    vad_seconds: float
    asr_load_seconds: float
    asr_seconds: float
    aligner_load_seconds: float
    emissions_seconds: float
    viterbi_seconds: float
    peak_cuda_allocated_gib: float
    peak_cuda_reserved_gib: float
    asr_batches: int
    asr_processor_rows: int
    generated_tokens: int
    oom_retries: int
    truncation_retries: int


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A serialized message or bounded progress update from a run."""

    stage: str
    message: str | None = None
    current: int | None = None
    total: int | None = None


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass(frozen=True, slots=True)
class TranscriptionRun(Sequence[TranscriptionResult]):
    """Immutable, sequence-like result of one API call."""

    results: tuple[TranscriptionResult, ...]
    requested_options: TranscriptionOptions
    resolved_options: TranscriptionOptions
    statistics: TranscriptionStatistics
    errors: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[TranscriptionResult]:
        return iter(self.results)

    @overload
    def __getitem__(self, index: int) -> TranscriptionResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TranscriptionResult, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> TranscriptionResult | tuple[TranscriptionResult, ...]:
        return self.results[index]

    @property
    def single(self) -> TranscriptionResult:
        """Return the only expanded result, or fail when cardinality is not one."""
        if len(self.results) != 1:
            raise ValueError(
                f"Expected exactly one expanded audio file, found {len(self.results)}"
            )
        return self.results[0]

    @property
    def successful(self) -> tuple[TranscriptionResult, ...]:
        return tuple(result for result in self.results if result.status == "completed")

    @property
    def failed(self) -> tuple[TranscriptionResult, ...]:
        return tuple(result for result in self.results if result.status == "failed")

    @property
    def skipped(self) -> tuple[TranscriptionResult, ...]:
        return tuple(result for result in self.results if result.status == "skipped")

    @property
    def ok(self) -> bool:
        return not self.failed and not self.errors


class TranscriptionError(Exception):
    """Base exception for Python API failures."""


class TranscriptionConfigurationError(TranscriptionError, ValueError):
    """Invalid or unsupported transcription configuration."""


class TranscriptionInputError(TranscriptionError, ValueError):
    """Invalid audio input or output planning request."""


class TranscriptionRuntimeError(TranscriptionError):
    """Dependency, device, model, or execution initialization failure."""


class ProgressCallbackError(TranscriptionError, RuntimeError):
    """A user-provided progress callback raised an exception."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(
            f"Progress callback failed: {type(original).__name__}: {original}"
        )


class TranscriberClosedError(TranscriptionError, RuntimeError):
    """Operation attempted after a transcriber was closed."""


class TranscriberBusyError(TranscriptionError, RuntimeError):
    """A reentrant transcription call was attempted."""


class BatchTranscriptionError(TranscriptionError):
    """Aggregate failure that retains every completed per-file result."""

    def __init__(self, run: TranscriptionRun) -> None:
        self.run = run
        count = len(run.failed)
        if count and run.errors:
            message = (
                f"{count} transcription file(s) failed; run errors: {len(run.errors)}"
            )
        elif count:
            message = f"{count} transcription file(s) failed"
        else:
            message = f"Transcription run failed with {len(run.errors)} run error(s)"
        super().__init__(message)


__all__ = [
    "AudioInput",
    "BatchTranscriptionError",
    "Language",
    "OutputFormat",
    "ProgressCallback",
    "ProgressCallbackError",
    "ProgressEvent",
    "PublicationOptions",
    "ResultStatus",
    "SubtitleCue",
    "TranscriberBusyError",
    "TranscriberClosedError",
    "TranscriptionConfigurationError",
    "TranscriptionError",
    "TranscriptionInputError",
    "TranscriptionOptions",
    "TranscriptionProvenance",
    "TranscriptionResult",
    "TranscriptionRun",
    "TranscriptionRuntimeError",
    "TranscriptionSegment",
    "TranscriptionStatistics",
    "TranscriptionWord",
]
