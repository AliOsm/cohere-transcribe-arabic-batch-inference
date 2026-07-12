#!/usr/bin/env python3
"""Production batch transcription with bounded preprocessing and optional timestamps.

Inputs may be audio files, directories, or a mixture of both. The Cohere ASR model is loaded once
for the run, work is batched across files, and decoded audio is kept within a configurable group
target. Word timing uses MMS CTC forced alignment; segment timing and plain-text modes skip it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import errno
import gc
import importlib
import io
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypedDict

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from tqdm import tqdm


_PROCESS_UMASK = os.umask(0)
os.umask(_PROCESS_UMASK)
DEFAULT_OUTPUT_MODE = 0o666 & ~_PROCESS_UMASK


MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
ALIGN_MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"
# These are the exact revisions used by the accuracy and throughput evaluation.
ASR_MODEL_REVISION = "0a8193caa4f3f92131471ab08824e488141cb392"
ALIGN_MODEL_REVISION = "49402e9577b1158620820667c218cd494cc44486"
ALIGN_PACKAGE_REPOSITORY = "https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
ALIGN_PACKAGE_REVISION = "c344f5bc900323aa434a7cb200b7c629d463bd02"
OUTPUT_SCHEMA_VERSION = 4
PROFILE_SCHEMA_VERSION = 3
SR = 16_000
ALIGN_WINDOW_S = 30
ALIGN_CONTEXT_S = 2
# The exact one-row limit is read from the pinned processor at runtime. The
# current Cohere feature extractor uses 35 s chunks and starts its quiet-boundary
# path above 30 s, but a 30-35 s clip still remains one processor row.
ASR_FIXED_MIN_S = 1.0
# The projection/mask hot-path patches below depend on Transformers internals.
# Keep the accepted range narrow and widen it only after parity tests pass.
MIN_TRANSFORMERS_VERSION = "5.13.0"
MAX_TRANSFORMERS_VERSION = "5.14.0"
PIPELINE_GROUP_MAX_BYTES = 512 * 1024**2
PIPELINE_GROUP_MAX_JOBS = 128
ALIGNMENT_GC_INTERVAL = 64
FFMPEG_DECODE_TIMEOUT_S = 3_600
OUTPUT_PATH_DISPLAY_LIMIT = 20

REPETITION_DETECTOR_VERSION = 3
REPETITION_MIN_GENERATED_TOKENS = 96
REPETITION_REPEATS = 4
REPETITION_MIN_PERIOD = 8
REPETITION_MAX_PERIOD = 32
# Checking less often can append extra loop tokens and changes output semantics.
REPETITION_CHECK_INTERVAL = 1

ISO3 = {"ar": "ara", "en": "eng"}
SENTENCE_ENDINGS = tuple(".!?؟،؛…")
AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".wave",
    ".webm",
    ".wma",
}

INDENT = "    "
BAR_FMT = (
    INDENT + "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}]"
)


def fmt_dur(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def info(message: str) -> None:
    tqdm.write(f"{INDENT}{message}")


# Domain and configuration


@dataclass(frozen=True)
class SourceSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> "SourceSnapshot":
        stat = path.stat()
        return cls(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


@dataclass(slots=True)
class AudioJob:
    index: int
    path: Path
    relative_path: Path
    snapshot: SourceSnapshot
    duration_hint: float | None
    language: str
    vad_mode: str
    alignment_mode: str
    output_paths: dict[str, Path] = field(default_factory=dict)
    audio: np.ndarray | None = None
    duration: float = 0.0
    segment_times: list[tuple[float, float]] = field(default_factory=list)
    # Raw speech spans are retained so approximate segment timing can preserve
    # internal pauses even when --vad-merge joins several spans for ASR.
    speech_spans: list[tuple[float, float]] = field(default_factory=list)
    segment_texts: list[str] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)
    fallback_alignments: int = 0
    repetition_stopped_segments: set[int] = field(default_factory=set)
    truncation_retried_segments: set[int] = field(default_factory=set)
    token_limit_segments: set[int] = field(default_factory=set)
    generated_tokens: dict[int, int] = field(default_factory=dict)
    decode_backend: str | None = None
    vad_engine_requested: str | None = None
    vad_engine_actual: str | None = None
    vad_provider: str | None = None
    vad_provider_options: dict[str, dict[str, str]] | None = None
    vad_fallback_reason: str | None = None
    vad_merge: bool = False
    segmentation_parameters: dict[str, int | float] = field(default_factory=dict)
    error: str | None = None

    @property
    def audio_bytes(self) -> int:
        return 0 if self.audio is None else int(self.audio.nbytes)

    @property
    def has_text(self) -> bool:
        return any(text.strip() for text in self.segment_texts)


@dataclass(slots=True)
class PreparedAudio:
    audio: np.ndarray
    segment_times: list[tuple[float, float]]
    speech_spans: list[tuple[float, float]]
    decode_seconds: float
    vad_seconds: float
    vad_engine: str
    decode_backend: str
    vad_provider: str | None = None
    vad_provider_options: dict[str, dict[str, str]] | None = None
    vad_fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentRef:
    job: AudioJob
    segment_index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class RunStats:
    decode_seconds: float = 0.0
    vad_seconds: float = 0.0
    asr_load_seconds: float = 0.0
    asr_seconds: float = 0.0
    asr_feature_seconds: float = 0.0
    asr_feature_wait_seconds: float = 0.0
    asr_h2d_seconds: float = 0.0
    asr_generate_seconds: float = 0.0
    asr_decode_seconds: float = 0.0
    align_load_seconds: float = 0.0
    emissions_seconds: float = 0.0
    viterbi_seconds: float = 0.0
    peak_cuda_gib: float = 0.0
    peak_cuda_reserved_gib: float = 0.0
    cuda_total_gib: float = 0.0
    cuda_free_start_gib: float = 0.0
    cuda_free_end_gib: float = 0.0
    asr_batches: int = 0
    asr_processor_rows: int = 0
    asr_generated_tokens: int = 0
    asr_valid_feature_frames: int = 0
    asr_padded_feature_frames: int = 0
    asr_oom_retries: int = 0
    asr_truncation_retries: int = 0
    asr_discarded_feature_batches: int = 0
    pin_memory_fallbacks: int = 0
    effective_batch_min: int = 0
    effective_batch_max: int = 0
    final_batch_size: int = 0
    final_batch_cap: int = 0
    batch_history: list[dict[str, Any]] = field(default_factory=list)


class WordTiming(TypedDict):
    start: float
    end: float
    text: str
    segment_index: int
    segment_word_index: int
    timing_source: str


class SubtitleCue(TypedDict):
    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionConfig:
    audio: list[str]
    language: str
    formats: list[str] | None
    text_only: bool
    output_dir: str | None
    recursive: bool
    existing: str
    device: str
    dtype: str
    audio_backend: str
    audio_memory_gb: float
    preprocess_workers: int | None
    pipeline_preparation: bool
    vad: str
    vad_engine: str
    vad_merge: bool
    min_dur: float
    max_dur: float
    max_silence: float
    energy_threshold: float
    vad_threshold: float
    min_silence_ms: int
    speech_pad_ms: int
    batch_size: int | None
    batch_max_size: int | None
    batch_audio_seconds: float | None
    batch_vram_target: float
    adaptive_batch: bool
    pin_memory: bool
    max_new_tokens: int
    max_retry_tokens: int
    truncation_policy: str
    stop_repetition_loops: bool
    alignment: str
    align_batch_size: int
    align_dtype: str
    max_chars: int
    max_cue_dur: float
    max_gap: float
    profile_json: str | None


class BoundedPrefetch:
    """Keep at most ``depth`` preprocessing results resident ahead of the consumer."""

    def __init__(
        self,
        items: Sequence[AudioJob],
        fn: Callable[[AudioJob], object],
        workers: int,
        depth: int | None = None,
        refill_before_yield: bool = True,
    ) -> None:
        self._items = iter(items)
        self._fn = fn
        self._workers = max(1, workers)
        self._depth = max(1, depth or self._workers)
        self._refill_before_yield = refill_before_yield
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._pending: deque[tuple[AudioJob, concurrent.futures.Future]] = deque()

    def __enter__(self) -> "BoundedPrefetch":
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._workers, thread_name_prefix="audio-prep"
        )
        for _ in range(self._depth):
            if not self._submit_one():
                break
        return self

    def _submit_one(self) -> bool:
        assert self._executor is not None
        try:
            item = next(self._items)
        except StopIteration:
            return False
        self._pending.append((item, self._executor.submit(self._fn, item)))
        return True

    def __iter__(self) -> Iterator[tuple[AudioJob, concurrent.futures.Future]]:
        while self._pending:
            item, future = self._pending.popleft()
            if self._refill_before_yield:
                self._submit_one()
            yield item, future
            if not self._refill_before_yield:
                self._submit_one()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exc is not None)


class PairBudgetPrefetch:
    """Prefetch one next item only when the adjacent pair fits a byte budget."""

    def __init__(
        self,
        items: Sequence[AudioJob],
        fn: Callable[[AudioJob], object],
        estimated_bytes: Sequence[int],
        memory_budget: int,
    ) -> None:
        if len(items) != len(estimated_bytes):
            raise ValueError("items and estimated_bytes must have equal lengths")
        self._items = list(items)
        self._fn = fn
        self._estimated_bytes = list(estimated_bytes)
        self._memory_budget = memory_budget
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._first_future: concurrent.futures.Future | None = None

    def __enter__(self) -> "PairBudgetPrefetch":
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audio-reload"
        )
        if self._items:
            self._first_future = self._executor.submit(self._fn, self._items[0])
        return self

    def __iter__(self) -> Iterator[tuple[AudioJob, concurrent.futures.Future]]:
        if not self._items:
            return
        assert self._executor is not None and self._first_future is not None
        current_future = self._first_future
        for index, item in enumerate(self._items):
            next_future: concurrent.futures.Future | None = None
            if index + 1 < len(self._items):
                pair_bytes = (
                    self._estimated_bytes[index] + self._estimated_bytes[index + 1]
                )
                if pair_bytes <= self._memory_budget:
                    next_future = self._executor.submit(
                        self._fn, self._items[index + 1]
                    )
            yield item, current_future
            if index + 1 < len(self._items):
                current_future = next_future or self._executor.submit(
                    self._fn, self._items[index + 1]
                )

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exc is not None)


# Input discovery and audio preparation


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=duration:format=duration",
        "-of",
        "json",
        os.fspath(path),
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
        payload = json.loads(result.stdout)
        values = [stream.get("duration") for stream in payload.get("streams", [])]
        values.append(payload.get("format", {}).get("duration"))
        for value in values:
            if value not in (None, "N/A"):
                duration = float(value)
                if math.isfinite(duration) and duration >= 0:
                    return duration
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def expand_inputs(inputs: Sequence[str], recursive: bool) -> list[tuple[Path, Path]]:
    expanded: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for raw in inputs:
        source = Path(raw).expanduser()
        try:
            source = source.resolve(strict=True)
        except FileNotFoundError as exc:
            raise SystemExit(f"Input does not exist: {source}") from exc

        if source.is_file():
            candidates = [(source, Path(source.name))]
        elif source.is_dir():
            iterator = source.rglob("*") if recursive else source.iterdir()
            paths = sorted(
                (
                    path
                    for path in iterator
                    if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
                ),
                key=lambda path: os.fspath(path).casefold(),
            )
            candidates = [(path.resolve(), path.relative_to(source)) for path in paths]
        else:
            raise SystemExit(f"Input is not a regular file or directory: {source}")

        for path, relative_path in candidates:
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            expanded.append((canonical, relative_path))

    if not expanded:
        raise SystemExit("No audio files found in the supplied inputs.")
    return expanded


def segmentation_parameters(args: TranscriptionConfig) -> dict[str, int | float]:
    """Return the behavior-affecting segmentation settings for output provenance."""
    parameters: dict[str, int | float] = {
        "max_duration_seconds": args.max_dur,
    }
    if args.vad == "silero":
        parameters.update(
            {
                "min_duration_seconds": args.min_dur,
                "threshold": args.vad_threshold,
                "min_silence_ms": args.min_silence_ms,
                "speech_pad_ms": args.speech_pad_ms,
            }
        )
    elif args.vad == "auditok":
        parameters.update(
            {
                "min_duration_seconds": args.min_dur,
                "max_silence_seconds": args.max_silence,
                "energy_threshold": args.energy_threshold,
            }
        )
    return parameters


def build_jobs(args: TranscriptionConfig) -> list[AudioJob]:
    entries = expand_inputs(args.audio, args.recursive)
    if args.formats is None:
        raise RuntimeError("Output formats must be normalized before building jobs")
    output_root = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    )
    if output_root is not None:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SystemExit(
                f"Cannot create output directory {output_root}: {exc}"
            ) from exc

    jobs: list[AudioJob] = []
    claimed_outputs: dict[Path, Path] = {}
    input_paths = {path for path, _ in entries}
    profile_candidate = (
        Path(args.profile_json).expanduser().resolve(strict=False)
        if args.profile_json is not None
        else None
    )
    if profile_candidate in input_paths:
        raise SystemExit(
            f"Profile path collides with an input audio file: {profile_candidate}"
        )
    for path, relative_path in entries:
        if output_root is None:
            parent = path.parent
        else:
            parent = output_root / relative_path.parent
        output_paths = {
            fmt: parent / f"{relative_path.stem}.{fmt}" for fmt in args.formats
        }
        for output in output_paths.values():
            key = output.resolve(strict=False)
            previous = claimed_outputs.get(key)
            if previous is not None and previous != path:
                raise SystemExit(
                    "Output collision detected before model loading:\n"
                    f"  {previous}\n  {path}\n  -> {output}\n"
                    "Use separate output directories or preserve distinct relative paths."
                )
            if key in input_paths:
                raise SystemExit(
                    f"Output path collides with an input audio file: {output}"
                )
            if key == profile_candidate:
                raise SystemExit(
                    f"Profile path collides with a transcript output: {output}"
                )
            claimed_outputs[key] = path
            try:
                output.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise SystemExit(
                    f"Cannot create output directory {output.parent}: {exc}"
                ) from exc
            if output.is_symlink() or (output.exists() and not output.is_file()):
                raise SystemExit(f"Output path is not a regular file: {output}")
            if not os.access(output.parent, os.W_OK):
                raise SystemExit(f"Output directory is not writable: {output.parent}")

        existing_outputs = [
            output for output in output_paths.values() if output.exists()
        ]
        if existing_outputs and args.existing == "error":
            paths = "\n".join(f"  {output}" for output in existing_outputs)
            raise SystemExit(
                f"Output already exists:\n{paths}\n"
                "Use --existing overwrite to replace it or --existing skip to keep complete sets."
            )
        if existing_outputs and args.existing == "skip":
            if len(existing_outputs) == len(output_paths):
                info(f"skipping {path}: all requested outputs already exist")
                continue
            info(f"rebuilding {path}: requested output set is incomplete")

        jobs.append(
            AudioJob(
                index=len(jobs),
                path=path,
                relative_path=relative_path,
                snapshot=SourceSnapshot.capture(path),
                duration_hint=None,
                language=args.language,
                vad_mode=args.vad,
                alignment_mode=args.alignment,
                output_paths=output_paths,
                vad_engine_requested=(
                    args.vad_engine if args.vad == "silero" else None
                ),
                vad_merge=args.vad == "silero" and args.vad_merge,
                segmentation_parameters=segmentation_parameters(args),
            )
        )

    if not jobs:
        return []

    probe_workers = min(len(jobs), 8, max(1, (os.cpu_count() or 2) // 2))
    if probe_workers == 1:
        jobs[0].duration_hint = probe_duration(jobs[0].path)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=probe_workers, thread_name_prefix="duration-probe"
        ) as executor:
            durations = executor.map(probe_duration, (job.path for job in jobs))
            for job, duration in zip(jobs, durations, strict=True):
                job.duration_hint = duration
    return jobs


def resolved_transformers_audio_backend(backend: str) -> str:
    """Resolve Transformers' ``auto`` backend before decoding for provenance."""
    if backend != "auto":
        return backend
    from packaging.version import Version
    from transformers.utils import is_torchcodec_available

    if is_torchcodec_available():
        try:
            if Version(importlib_metadata.version("torchcodec")) >= Version("0.3.0"):
                return "torchcodec"
        except importlib_metadata.PackageNotFoundError:
            pass
    return "librosa"


def load_audio_ffmpeg(path: Path) -> np.ndarray:
    """Decode to mono float32 PCM without retaining a second full-size bytes copy."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is not installed; use --audio-backend librosa or install ffmpeg"
        )
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "0",
        "-i",
        os.fspath(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "pipe:1",
    ]
    output = io.BytesIO()
    timed_out = threading.Event()
    with tempfile.TemporaryFile() as stderr_handle:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_handle,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"Could not launch FFmpeg for {path}: {exc}") from exc
        assert process.stdout is not None

        def terminate_on_timeout() -> None:
            if process.poll() is None:
                timed_out.set()
                with contextlib.suppress(OSError):
                    process.kill()

        timer = threading.Timer(FFMPEG_DECODE_TIMEOUT_S, terminate_on_timeout)
        timer.daemon = True
        timer.start()
        try:
            while True:
                chunk = process.stdout.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
            returncode = process.wait()
        except BaseException:
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(Exception):
                process.wait(timeout=5)
            raise
        finally:
            timer.cancel()
            process.stdout.close()

        if timed_out.is_set():
            raise RuntimeError(
                f"FFmpeg exceeded the {FFMPEG_DECODE_TIMEOUT_S}s decode timeout for {path}"
            )
        if returncode:
            stderr_handle.seek(0)
            message = " ".join(
                stderr_handle.read().decode("utf-8", errors="replace").split()
            )
            raise RuntimeError(f"FFmpeg failed for {path}: {message[-500:]}")

    view = output.getbuffer()
    if len(view) % np.dtype("<f4").itemsize:
        raise RuntimeError(
            f"FFmpeg returned an incomplete float32 sample for {path}: {len(view)} bytes"
        )
    # The ndarray retains the memoryview/BytesIO owner, so this is zero-copy.
    return np.frombuffer(view, dtype="<f4")


def decode_audio_resolved(path: Path, backend: str) -> tuple[np.ndarray, str]:
    """Decode audio and return the concrete backend used for this file."""
    if backend == "ffmpeg":
        audio = load_audio_ffmpeg(path)
        resolved_backend = "ffmpeg"
    else:
        from transformers.audio_utils import load_audio

        concrete_backend = resolved_transformers_audio_backend(backend)
        try:
            audio = load_audio(
                os.fspath(path), sampling_rate=SR, backend=concrete_backend
            )
        except (ImportError, OSError, RuntimeError, ValueError):
            if backend != "auto" or not shutil.which("ffmpeg"):
                raise
            audio = load_audio_ffmpeg(path)
            resolved_backend = "ffmpeg"
        else:
            resolved_backend = concrete_backend
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio after decoding, got shape {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError("Decoded audio contains NaN or infinite samples")
    return np.ascontiguousarray(audio), resolved_backend


def decode_audio(path: Path, backend: str) -> np.ndarray:
    audio, _ = decode_audio_resolved(path, backend)
    return audio


def segment_audio_auditok(
    audio: np.ndarray,
    min_dur: float,
    max_dur: float,
    max_silence: float,
    energy_threshold: float,
) -> list[tuple[float, float]]:
    from auditok.core import split as auditok_split

    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    regions = auditok_split(
        pcm,
        min_dur=min_dur,
        max_dur=max_dur,
        max_silence=max_silence,
        energy_threshold=energy_threshold,
        sampling_rate=SR,
        sample_width=2,
        channels=1,
    )
    return [(float(region.start), float(region.end)) for region in regions]


def segment_audio_fixed(
    audio: np.ndarray, window_seconds: float
) -> list[tuple[float, float]]:
    """Cover the complete waveform with contiguous, non-overlapping windows."""
    total_samples = len(audio)
    if total_samples == 0:
        return []

    window_samples = max(1, int(round(window_seconds * SR)))
    return [
        (start / SR, min(start + window_samples, total_samples) / SR)
        for start in range(0, total_samples, window_samples)
    ]


def validate_processor_single_row_window(processor, window_seconds: float) -> float:
    """Ensure this script, rather than the processor, controls row expansion."""
    feature_extractor = processor.feature_extractor
    max_clip = float(feature_extractor.max_audio_clip_s)
    if not math.isfinite(max_clip) or max_clip <= 0:
        raise RuntimeError("Cohere processor reported an invalid max_audio_clip_s")
    if window_seconds > max_clip + 1e-9:
        raise RuntimeError(
            f"--max-dur {window_seconds:g}s exceeds this Cohere processor's "
            f"single-row limit of {max_clip:g}s"
        )
    return max_clip


def validate_segment_times(
    segments: Sequence[tuple[float, float]],
    duration: float,
    max_duration: float | None = None,
) -> list[tuple[float, float]]:
    """Validate, clamp sub-sample drift, and return ordered non-overlapping spans."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Audio duration must be finite and non-negative")
    tolerance = 2.0 / SR
    validated: list[tuple[float, float]] = []
    previous_end = 0.0
    for index, (raw_start, raw_end) in enumerate(segments):
        start = float(raw_start)
        end = float(raw_end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"Segment {index} has non-finite bounds")
        if start < -tolerance or end > duration + tolerance:
            raise ValueError(
                f"Segment {index} lies outside the audio: {start:.6f}..{end:.6f} "
                f"for {duration:.6f}s"
            )
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
        if end <= start:
            continue
        if start < previous_end - tolerance:
            raise ValueError(f"Segment {index} overlaps or is out of order")
        start = max(start, previous_end) if start < previous_end else start
        if max_duration is not None and end - start > max_duration + tolerance:
            raise ValueError(
                f"Segment {index} is {end - start:.6f}s, exceeding "
                f"the {max_duration:g}s single-row limit"
            )
        validated.append((start, end))
        previous_end = end
    return validated


@dataclass(slots=True)
class SileroRuntime:
    model: Any
    engine: str
    runner: Callable[[np.ndarray, Any, TranscriptionConfig], Sequence[dict[str, Any]]]
    provider: str | None = None
    provider_options: dict[str, dict[str, str]] | None = None


class SileroBackendUnavailable(RuntimeError):
    """An optional ONNX backend failed for an environmental reason."""


_vad_thread_local = threading.local()


def onnx_provider_details(
    model: Any,
) -> tuple[str | None, dict[str, dict[str, str]] | None]:
    for candidate in (
        model,
        getattr(model, "session", None),
        getattr(model, "ort_session", None),
        getattr(model, "_session", None),
    ):
        if candidate is None or not hasattr(candidate, "get_providers"):
            continue
        try:
            providers = candidate.get_providers()
        except Exception:
            continue
        if providers:
            raw_options = None
            if hasattr(candidate, "get_provider_options"):
                try:
                    raw_options = candidate.get_provider_options()
                except Exception:
                    raw_options = None
            options = (
                {
                    str(provider): {
                        str(key): str(value) for key, value in values.items()
                    }
                    for provider, values in raw_options.items()
                }
                if isinstance(raw_options, dict)
                else None
            )
            return ",".join(map(str, providers)), options
    return None, None


def is_onnxruntime_failure(exc: BaseException) -> bool:
    """Recognize ONNX Runtime failures without importing it for JIT-only runs."""
    return exc.__class__.__module__.startswith("onnxruntime.")


def build_silero_jit_runtime() -> SileroRuntime:
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=False)

    def run(
        audio: np.ndarray, active_model: Any, args: TranscriptionConfig
    ) -> Sequence[dict[str, Any]]:
        return get_speech_timestamps(
            torch.from_numpy(audio),
            active_model,
            sampling_rate=SR,
            threshold=args.vad_threshold,
            min_speech_duration_ms=int(round(args.min_dur * 1000)),
            max_speech_duration_s=args.max_dur,
            min_silence_duration_ms=args.min_silence_ms,
            speech_pad_ms=args.speech_pad_ms,
            return_seconds=False,
        )

    return SileroRuntime(model=model, engine="jit", runner=run)


def build_silero_onnx_runtime() -> SileroRuntime:
    try:
        module = importlib.import_module("transcribe_assets.vectorized_silero")
    except ImportError as exc:
        raise SileroBackendUnavailable(str(exc)) from exc
    model_class = getattr(module, "VectorizedSileroVAD")
    timestamps_fn = getattr(module, "get_speech_timestamps")
    try:
        model = model_class()
    except Exception as exc:
        if isinstance(exc, (ImportError, OSError)) or is_onnxruntime_failure(exc):
            raise SileroBackendUnavailable(str(exc)) from exc
        raise

    def run(
        audio: np.ndarray, active_model: Any, args: TranscriptionConfig
    ) -> Sequence[dict[str, Any]]:
        try:
            return timestamps_fn(
                audio,
                active_model,
                sampling_rate=SR,
                threshold=args.vad_threshold,
                min_speech_duration_ms=int(round(args.min_dur * 1000)),
                max_speech_duration_s=args.max_dur,
                min_silence_duration_ms=args.min_silence_ms,
                speech_pad_ms=args.speech_pad_ms,
            )
        except Exception as exc:
            if isinstance(exc, (ImportError, OSError)) or is_onnxruntime_failure(exc):
                raise SileroBackendUnavailable(str(exc)) from exc
            raise

    provider, provider_options = onnx_provider_details(model)

    return SileroRuntime(
        model=model,
        engine="onnx",
        runner=run,
        provider=provider,
        provider_options=provider_options,
    )


def get_silero_runtime(requested_engine: str) -> SileroRuntime:
    cache = getattr(_vad_thread_local, "runtimes", None)
    if cache is None:
        cache = {}
        _vad_thread_local.runtimes = cache
    if requested_engine in cache:
        return cache[requested_engine]

    if requested_engine in {"auto", "onnx"}:
        try:
            runtime = build_silero_onnx_runtime()
        except SileroBackendUnavailable as exc:
            if requested_engine == "onnx":
                raise
            _vad_thread_local.onnx_fallback_error = f"{type(exc).__name__}: {exc}"
        else:
            cache[requested_engine] = runtime
            return runtime

    runtime = build_silero_jit_runtime()
    cache[requested_engine] = runtime
    return runtime


def sample_timestamps_to_seconds(
    timestamps: Sequence[dict[str, Any]], audio_samples: int
) -> list[tuple[float, float]]:
    duration = audio_samples / SR
    segments: list[tuple[float, float]] = []
    for item in timestamps:
        start = max(0.0, float(item["start"]) / SR)
        end = min(duration, float(item["end"]) / SR)
        if end > start:
            segments.append((start, end))
    return validate_segment_times(segments, duration)


def segment_audio_silero(
    audio: np.ndarray, args: TranscriptionConfig
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    str,
    str | None,
    dict[str, dict[str, str]] | None,
    str | None,
]:
    runtime = get_silero_runtime(args.vad_engine)
    fallback_reason = getattr(_vad_thread_local, "onnx_fallback_error", None)
    if (
        args.vad_engine == "auto"
        and runtime.engine == "jit"
        and fallback_reason
        and not getattr(_vad_thread_local, "onnx_fallback_reported", False)
    ):
        info(f"[warn] ONNX Silero unavailable ({fallback_reason}); using TorchScript")
        _vad_thread_local.onnx_fallback_reported = True
    try:
        timestamps = runtime.runner(audio, runtime.model, args)
    except SileroBackendUnavailable as exc:
        if args.vad_engine != "auto" or runtime.engine != "onnx":
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        _vad_thread_local.onnx_fallback_error = fallback_reason
        _vad_thread_local.onnx_fallback_reported = True
        info(
            f"[warn] ONNX Silero failed ({type(exc).__name__}: {exc}); "
            "falling back to TorchScript"
        )
        runtime = build_silero_jit_runtime()
        cache = getattr(_vad_thread_local, "runtimes", None)
        if cache is None:
            cache = {}
            _vad_thread_local.runtimes = cache
        cache["auto"] = runtime
        timestamps = runtime.runner(audio, runtime.model, args)

    raw_segments = sample_timestamps_to_seconds(timestamps, len(audio))
    segments = (
        merge_speech_segments(raw_segments, args.max_dur)
        if args.vad_merge
        else list(raw_segments)
    )
    segments = validate_segment_times(
        segments, len(audio) / SR, max_duration=args.max_dur
    )
    engine = runtime.engine + ("+merge" if args.vad_merge else "")
    return (
        segments,
        raw_segments,
        engine,
        runtime.provider,
        runtime.provider_options,
        fallback_reason,
    )


def merge_speech_segments(
    segments: Sequence[tuple[float, float]], max_duration: float
) -> list[tuple[float, float]]:
    """Greedily merge adjacent VAD spans while their full timeline fits."""
    if not math.isfinite(max_duration) or max_duration <= 0:
        raise ValueError("max_duration must be finite and positive")
    if not segments:
        return []

    merged: list[tuple[float, float]] = []
    current_start, current_end = segments[0]
    if current_start < 0 or current_end < current_start:
        raise ValueError("VAD segments must have non-negative ordered bounds")
    for start, end in segments[1:]:
        if start < current_end or end < start:
            raise ValueError("VAD segments must be sorted and non-overlapping")
        if end - current_start <= max_duration + 1e-9:
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def prepare_audio(job: AudioJob, args: TranscriptionConfig) -> PreparedAudio:
    started = time.perf_counter()
    audio, decode_backend = decode_audio_resolved(job.path, args.audio_backend)
    decode_seconds = time.perf_counter() - started
    duration = len(audio) / SR

    started = time.perf_counter()
    provider: str | None = None
    provider_options: dict[str, dict[str, str]] | None = None
    fallback_reason: str | None = None
    if args.vad == "silero":
        (
            segment_times,
            speech_spans,
            engine,
            provider,
            provider_options,
            fallback_reason,
        ) = segment_audio_silero(audio, args)
    elif args.vad == "auditok":
        segment_times = segment_audio_auditok(
            audio, args.min_dur, args.max_dur, args.max_silence, args.energy_threshold
        )
        segment_times = validate_segment_times(
            segment_times, duration, max_duration=args.max_dur
        )
        speech_spans = list(segment_times)
        engine = "auditok"
    else:
        segment_times = segment_audio_fixed(audio, args.max_dur)
        segment_times = validate_segment_times(
            segment_times, duration, max_duration=args.max_dur
        )
        speech_spans = list(segment_times)
        engine = "none (fixed windows)"
    vad_seconds = time.perf_counter() - started
    return PreparedAudio(
        audio=audio,
        segment_times=segment_times,
        speech_spans=speech_spans,
        decode_seconds=decode_seconds,
        vad_seconds=vad_seconds,
        vad_engine=engine,
        decode_backend=decode_backend,
        vad_provider=provider,
        vad_provider_options=provider_options,
        vad_fallback_reason=fallback_reason,
    )


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda was requested, but CUDA is not available to PyTorch"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "--device mps was requested, but MPS is not available to PyTorch"
        )
    return requested


def empty_device_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ASR inference


class MemoizedEncoderProjection(torch.nn.Module):
    """Project each encoder output once across its autoregressive decode."""

    def __init__(self, projection: torch.nn.Module) -> None:
        super().__init__()
        self.projection = projection
        self._source: torch.Tensor | None = None
        self._projected: torch.Tensor | None = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source is not self._source:
            self._source = source
            self._projected = self.projection(source)
        assert self._projected is not None
        return self._projected

    def clear(self) -> None:
        self._source = None
        self._projected = None


def clear_encoder_projection_cache(model) -> None:
    projection = model.model.decoder.proj
    if isinstance(projection, MemoizedEncoderProjection):
        projection.clear()


def prepare_encoder_attention_mask_once(_module, _inputs, output):
    """Convert the encoder padding mask once instead of once per decoder token."""
    mask = getattr(output, "attention_mask", None)
    if mask is None or mask.ndim != 2:
        return output
    # The stock decoder checks this on CUDA for every token. One check here also
    # preserves its mask-free SDPA path for batches without encoder padding.
    if bool(mask.all()):
        output.attention_mask = None
    else:
        output.attention_mask = mask.to(dtype=torch.bool)[:, None, None, :]
    return output


def load_asr(
    device: str,
    dtype: torch.dtype,
    revision: str | None = ASR_MODEL_REVISION,
    projection_cache: bool = True,
    encoder_attention_mask_cache: bool = True,
):
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="sdpa",
        revision=revision,
    )
    if projection_cache:
        try:
            projection = model.model.decoder.proj
        except AttributeError as exc:
            raise RuntimeError(
                f"{MODEL_ID}@{revision} is incompatible with the encoder-projection cache"
            ) from exc
        model.model.decoder.proj = MemoizedEncoderProjection(projection)
    if encoder_attention_mask_cache:
        try:
            encoder = model.model.encoder
        except AttributeError as exc:
            raise RuntimeError(
                f"{MODEL_ID}@{revision} is incompatible with the encoder-mask cache"
            ) from exc
        encoder.register_forward_hook(prepare_encoder_attention_mask_once)
    model.to(device)
    model.eval()
    return processor, model


def reassemble_chunk_texts(
    chunk_texts: Sequence[str],
    audio_chunk_index: Sequence[tuple[int, int | None]],
    separator: str,
    expected_samples: int,
) -> list[str]:
    if len(chunk_texts) != len(audio_chunk_index):
        raise RuntimeError(
            "Processor chunk metadata does not match decoded ASR rows: "
            f"{len(audio_chunk_index)} indices for {len(chunk_texts)} texts"
        )
    if expected_samples < 0:
        raise ValueError("expected_samples must be non-negative")

    grouped: dict[int, list[tuple[int, str]]] = {}
    direct: dict[int, str] = {}
    for metadata, text in zip(audio_chunk_index, chunk_texts, strict=True):
        if len(metadata) != 2:
            raise RuntimeError(f"Invalid audio_chunk_index row: {metadata!r}")
        raw_sample_index, raw_chunk_index = metadata
        sample_index = int(raw_sample_index)
        if not 0 <= sample_index < expected_samples:
            raise RuntimeError(
                f"Processor returned sample index {sample_index}; expected 0..{expected_samples - 1}"
            )
        chunk_index = None if raw_chunk_index is None else int(raw_chunk_index)
        if chunk_index is not None and chunk_index < 0:
            raise RuntimeError(f"Processor returned negative chunk index {chunk_index}")
        if chunk_index is None:
            if sample_index in direct or sample_index in grouped:
                raise RuntimeError(
                    f"Processor returned duplicate rows for sample {sample_index}"
                )
            direct[sample_index] = text.strip()
        else:
            if sample_index in direct:
                raise RuntimeError(
                    f"Processor mixed direct and chunked rows for sample {sample_index}"
                )
            grouped.setdefault(sample_index, []).append((chunk_index, text))

    outputs = [""] * expected_samples
    for sample_index, text in direct.items():
        outputs[sample_index] = text
    for sample_index, items in grouped.items():
        items.sort(key=lambda item: item[0])
        indices = [chunk_index for chunk_index, _ in items]
        if indices != list(range(len(items))):
            raise RuntimeError(
                f"Processor returned non-contiguous chunk indices for sample {sample_index}: {indices}"
            )
        parts = [text.strip() for _, text in items if text and text.strip()]
        outputs[sample_index] = separator.join(parts)
    missing = [
        index
        for index in range(expected_samples)
        if index not in direct and index not in grouped
    ]
    if missing:
        raise RuntimeError(
            f"Processor returned no ASR row for sample indices: {missing}"
        )
    return outputs


@dataclass(slots=True)
class PreparedASRBatch:
    refs: list[SegmentRef]
    model_inputs: dict[str, torch.Tensor]
    chunk_index: list[tuple[int, int | None]]
    prepare_seconds: float
    valid_feature_frames: int
    padded_feature_frames: int
    pin_memory_fallbacks: int = 0


@dataclass(slots=True)
class ASRGenerationResult:
    generated: torch.Tensor
    row_token_counts: list[int]
    truncated_ref_indices: set[int]
    repetition_ref_indices: set[int]
    max_new_tokens: int
    prompt_length: int
    elapsed_seconds: float
    h2d_seconds: float = 0.0
    baseline_reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


@dataclass(slots=True)
class ASRBatchController:
    current_size: int
    max_size: int
    audio_budget_seconds: float
    adaptive: bool
    target_vram_ratio: float
    total_vram_bytes: int = 0
    memory_budget_bytes: int = 0
    initial_size: int = 1
    oom_count: int = 0
    growth_cooldown: int = 0

    @classmethod
    def create(
        cls,
        args: TranscriptionConfig,
        model,
        refs: Sequence[SegmentRef],
    ) -> "ASRBatchController":
        device_type = model.device.type
        default_initial = 24 if device_type == "cuda" else 8
        initial = args.batch_size or (
            min(default_initial, args.batch_max_size)
            if args.batch_max_size is not None
            else default_initial
        )
        total_vram = 0
        memory_budget = 0
        if device_type == "cuda":
            total_vram = int(
                torch.cuda.get_device_properties(model.device).total_memory
            )
            free_vram, _reported_total = torch.cuda.mem_get_info(model.device)
            baseline_reserved = int(torch.cuda.memory_reserved(model.device))
            # Respect both the user-selected fraction of the physical GPU and
            # memory already consumed by other processes. Keep 5% of currently
            # free memory outside the PyTorch budget as a fragmentation margin.
            memory_budget = max(
                baseline_reserved,
                min(
                    int(args.batch_vram_target * total_vram),
                    baseline_reserved + int(0.95 * free_vram),
                ),
            )

        if not args.adaptive_batch:
            maximum = initial
        elif args.batch_max_size is not None:
            maximum = args.batch_max_size
        elif args.batch_size is not None:
            maximum = args.batch_size
        elif device_type == "cuda":
            # A cautious upper search bound. The controller approaches it only
            # after measured successful batches and never jumps by more than 25%.
            total_gib = total_vram / 1024**3
            maximum = max(initial, min(128, int(total_gib * 4)))
        else:
            maximum = initial
        maximum = max(initial, maximum)

        longest = max((ref.duration for ref in refs), default=1.0)
        audio_budget = args.batch_audio_seconds or initial * max(longest, 0.25)
        return cls(
            current_size=initial,
            max_size=maximum,
            audio_budget_seconds=audio_budget,
            adaptive=args.adaptive_batch,
            target_vram_ratio=args.batch_vram_target,
            total_vram_bytes=total_vram,
            memory_budget_bytes=memory_budget,
            initial_size=initial,
        )

    def configure_group(
        self, args: TranscriptionConfig, refs: Sequence[SegmentRef]
    ) -> None:
        """Refresh only the group-local frame budget; retain learned row caps."""
        longest = max((ref.duration for ref in refs), default=1.0)
        self.audio_budget_seconds = (
            args.batch_audio_seconds
            if args.batch_audio_seconds is not None
            else self.initial_size * max(longest, 0.25)
        )

    def take(self, pending: deque[SegmentRef]) -> list[SegmentRef]:
        if not pending:
            return []
        longest = max(pending[0].duration, 1.0 / SR)
        frame_limited = max(1, int(self.audio_budget_seconds / longest))
        count = min(len(pending), self.current_size, frame_limited)
        return [pending.popleft() for _ in range(count)]

    def record_oom(self, attempted_rows: int) -> None:
        self.oom_count += 1
        self.growth_cooldown = max(self.growth_cooldown, 2)
        if attempted_rows <= 1:
            self.current_size = 1
            return
        self.max_size = min(self.max_size, attempted_rows - 1)
        self.current_size = max(1, min(self.max_size, attempted_rows // 2))

    def record_success(self, result: ASRGenerationResult, attempted_rows: int) -> None:
        if self.growth_cooldown:
            self.growth_cooldown -= 1
            return
        if (
            not self.adaptive
            or self.current_size >= self.max_size
            or attempted_rows <= 0
            or attempted_rows < self.current_size
            or self.total_vram_bytes <= 0
            or result.peak_reserved_bytes <= 0
        ):
            return
        memory_budget = self.memory_budget_bytes or int(
            self.target_vram_ratio * self.total_vram_bytes
        )
        headroom_bytes = memory_budget - result.peak_reserved_bytes
        headroom = headroom_bytes / self.total_vram_bytes
        if headroom <= 0.05:
            return
        factor = 1.25 if headroom >= 0.20 else 1.125
        proposed = max(
            self.current_size + 1, int(math.ceil(self.current_size * factor))
        )

        incremental = max(
            1, result.peak_reserved_bytes - result.baseline_reserved_bytes
        )
        available_incremental = max(
            0,
            memory_budget - result.baseline_reserved_bytes,
        )
        memory_estimate = int(attempted_rows * available_incremental / incremental)
        if memory_estimate > 0:
            proposed = min(proposed, max(self.current_size, memory_estimate))
        self.current_size = min(self.max_size, proposed)


class RepetitionLoopStoppingCriteria:
    """Stop a row after four repeated 8-32-token blocks in a long decode."""

    def __init__(
        self,
        prompt_length: int,
        min_generated_tokens: int = REPETITION_MIN_GENERATED_TOKENS,
        repeats: int = REPETITION_REPEATS,
        min_period: int = REPETITION_MIN_PERIOD,
        max_period: int = REPETITION_MAX_PERIOD,
        eos_token_ids: Sequence[int] = (),
    ) -> None:
        self.prompt_length = prompt_length
        self.min_generated_tokens = min_generated_tokens
        self.repeats = repeats
        self.min_period = min_period
        self.max_period = max_period
        self.eos_token_ids = tuple(eos_token_ids)
        self._triggered_mask: torch.BoolTensor | None = None
        self._pattern_cache: dict[
            tuple[str, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._eos_cache: dict[str, torch.Tensor] = {}

    @property
    def triggered_rows(self) -> set[int]:
        if self._triggered_mask is None:
            return set()
        rows = self._triggered_mask.nonzero(as_tuple=False).flatten().tolist()
        return set(rows)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _scores: torch.FloatTensor | None,
        **_kwargs,
    ) -> torch.BoolTensor:
        generated = input_ids[:, self.prompt_length :]
        generated_tokens = generated.shape[1]
        if self._triggered_mask is None:
            self._triggered_mask = torch.zeros(
                generated.shape[0], dtype=torch.bool, device=generated.device
            )
        if generated_tokens < self.min_generated_tokens:
            return self._triggered_mask
        if (
            generated_tokens != self.min_generated_tokens
            and (generated_tokens - self.min_generated_tokens)
            % REPETITION_CHECK_INTERVAL
        ):
            return self._triggered_mask

        largest_period = min(self.max_period, generated_tokens // self.repeats)
        if largest_period < self.min_period:
            return self._triggered_mask
        span = self.repeats * largest_period
        cache_key = (str(generated.device), largest_period)
        cached = self._pattern_cache.get(cache_key)
        if cached is None:
            periods = torch.arange(
                self.min_period, largest_period + 1, device=generated.device
            )
            positions = torch.arange(span, device=generated.device)
            base_indices = positions.unsqueeze(0) % periods.unsqueeze(1)
            relevant = positions.unsqueeze(0) < self.repeats * periods.unsqueeze(1)
            cached = (base_indices, relevant)
            self._pattern_cache[cache_key] = cached
        base_indices, relevant = cached

        reverse_tail = generated[:, -span:].flip(dims=(1,))
        expected = reverse_tail[:, base_indices]
        actual = reverse_tail[:, None, :]
        matches = (actual == expected) | ~relevant.unsqueeze(0)
        newly_done = matches.all(dim=2).any(dim=1)
        if self.eos_token_ids:
            device_key = str(generated.device)
            eos_ids = self._eos_cache.get(device_key)
            if eos_ids is None:
                eos_ids = generated.new_tensor(self.eos_token_ids)
                self._eos_cache[device_key] = eos_ids
            eos_seen = (generated.unsqueeze(-1) == eos_ids).any(dim=(1, 2))
            newly_done &= ~eos_seen
        self._triggered_mask |= newly_done
        return self._triggered_mask


def repetition_stopping_criteria(
    decoder_input_ids: torch.Tensor,
    enabled: bool,
    eos_token_ids: Sequence[int] = (),
) -> list[RepetitionLoopStoppingCriteria] | None:
    if not enabled:
        return None
    return [
        RepetitionLoopStoppingCriteria(
            prompt_length=decoder_input_ids.shape[1],
            eos_token_ids=eos_token_ids,
        )
    ]


def maybe_pin_tensor(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


def prepare_asr_batch(
    processor,
    refs: Sequence[SegmentRef],
    args: TranscriptionConfig,
) -> PreparedASRBatch:
    started = time.perf_counter()
    waveforms: list[np.ndarray] = []
    for ref in refs:
        if ref.job.audio is None:
            raise RuntimeError(f"Decoded audio was released before ASR: {ref.job.path}")
        first = int(round(ref.start * SR))
        last = int(round(ref.end * SR))
        waveform = ref.job.audio[first:last]
        if waveform.size == 0:
            raise ValueError(
                f"Segment {ref.segment_index} of {ref.job.path} has no audio samples"
            )
        waveforms.append(waveform)

    with torch.inference_mode():
        inputs = processor(
            audio=waveforms,
            sampling_rate=SR,
            return_tensors="pt",
            return_attention_mask=True,
            language=args.language,
        )
    chunk_index = [
        (int(sample_index), None if chunk_index is None else int(chunk_index))
        for sample_index, chunk_index in inputs["audio_chunk_index"]
    ]
    sample_indices = sorted(sample_index for sample_index, _ in chunk_index)
    if len(chunk_index) != len(refs) or sample_indices != list(range(len(refs))):
        raise RuntimeError(
            "The Cohere processor expanded one or more script-controlled segments "
            "into multiple model rows. Lower --max-dur or use a compatible "
            "Transformers release so adaptive batching and OOM recovery remain exact."
        )
    model_inputs = {
        "input_features": inputs["input_features"],
        "decoder_input_ids": inputs["decoder_input_ids"],
    }
    if inputs.get("attention_mask") is not None:
        model_inputs["attention_mask"] = inputs["attention_mask"]

    pin = args.pin_memory and args.device == "cuda"
    pin_memory_fallbacks = 0
    if pin:
        pinned_inputs: dict[str, torch.Tensor] = {}
        for name, tensor in model_inputs.items():
            pinned = maybe_pin_tensor(tensor, True)
            if not pinned.is_pinned():
                pin_memory_fallbacks += 1
            pinned_inputs[name] = pinned
        model_inputs = pinned_inputs

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is None:
        rows, frames = model_inputs["input_features"].shape[:2]
        valid_frames = padded_frames = int(rows * frames)
    else:
        valid_frames = int(attention_mask.sum().item())
        padded_frames = int(attention_mask.numel())
    return PreparedASRBatch(
        refs=list(refs),
        model_inputs=model_inputs,
        chunk_index=chunk_index,
        prepare_seconds=time.perf_counter() - started,
        valid_feature_frames=valid_frames,
        padded_feature_frames=padded_frames,
        pin_memory_fallbacks=pin_memory_fallbacks,
    )


def generation_eos_token_ids(model) -> tuple[int, ...]:
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (eos_token_id,)
    return tuple(int(token_id) for token_id in eos_token_id)


def analyze_generated_rows(
    generated: torch.Tensor,
    prompt_length: int,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    pad_token_id: int | None,
    repetition_rows: set[int],
    chunk_index: Sequence[tuple[int, int | None]],
) -> tuple[list[int], set[int], set[int]]:
    generated_part = generated[:, prompt_length:]
    token_counts: list[int] = []
    truncated_refs: set[int] = set()
    repetition_refs: set[int] = set()
    eos_set = set(eos_token_ids)
    for row_index, row in enumerate(generated_part.tolist()):
        count = len(row)
        saw_eos = False
        for token_index, token_id in enumerate(row):
            if token_id in eos_set:
                count = token_index + 1
                saw_eos = True
                break
            if (
                row_index in repetition_rows
                and pad_token_id is not None
                and token_id == pad_token_id
            ):
                count = token_index
                break
        token_counts.append(count)
        ref_index = int(chunk_index[row_index][0])
        if row_index in repetition_rows:
            repetition_refs.add(ref_index)
        elif not saw_eos and len(row) >= max_new_tokens:
            truncated_refs.add(ref_index)
    return token_counts, truncated_refs, repetition_refs


def generate_asr_batch(
    model,
    prepared: PreparedASRBatch,
    args: TranscriptionConfig,
    max_new_tokens: int,
) -> ASRGenerationResult:
    is_cuda = model.device.type == "cuda"
    non_blocking = bool(is_cuda and args.pin_memory)
    baseline_reserved = 0
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(model.device)
        baseline_reserved = int(torch.cuda.memory_reserved(model.device))
    started = time.perf_counter()

    h2d_start = torch.cuda.Event(enable_timing=True) if is_cuda else None
    h2d_end = torch.cuda.Event(enable_timing=True) if is_cuda else None
    if h2d_start is not None:
        h2d_start.record()

    model_inputs = {
        "input_features": prepared.model_inputs["input_features"].to(
            model.device,
            dtype=model.dtype,
            non_blocking=non_blocking,
        ),
        "decoder_input_ids": prepared.model_inputs["decoder_input_ids"].to(
            model.device, non_blocking=non_blocking
        ),
    }
    if "attention_mask" in prepared.model_inputs:
        model_inputs["attention_mask"] = prepared.model_inputs["attention_mask"].to(
            model.device, non_blocking=non_blocking
        )
    if h2d_end is not None:
        h2d_end.record()

    eos_token_ids = generation_eos_token_ids(model)
    stopping_criteria = repetition_stopping_criteria(
        model_inputs["decoder_input_ids"],
        args.stop_repetition_loops,
        eos_token_ids,
    )
    prompt_length = int(model_inputs["decoder_input_ids"].shape[1])
    clear_encoder_projection_cache(model)
    try:
        with torch.inference_mode():
            generated_device = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=stopping_criteria,
            )
        # Moving the generated ids to host is the synchronization point, so
        # separate cuda.synchronize() calls would only add launch overhead.
        generated = generated_device.detach().cpu()
        del generated_device
    finally:
        clear_encoder_projection_cache(model)

    repetition_rows = (
        stopping_criteria[0].triggered_rows if stopping_criteria else set()
    )
    token_counts, truncated_refs, repetition_refs = analyze_generated_rows(
        generated=generated,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        pad_token_id=model.generation_config.pad_token_id,
        repetition_rows=repetition_rows,
        chunk_index=prepared.chunk_index,
    )
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(model.device)) if is_cuda else 0
    )
    peak_reserved = int(torch.cuda.max_memory_reserved(model.device)) if is_cuda else 0
    h2d_seconds = (
        float(h2d_start.elapsed_time(h2d_end)) / 1000.0
        if h2d_start is not None and h2d_end is not None
        else 0.0
    )
    return ASRGenerationResult(
        generated=generated,
        row_token_counts=token_counts,
        truncated_ref_indices=truncated_refs,
        repetition_ref_indices=repetition_refs,
        max_new_tokens=max_new_tokens,
        prompt_length=prompt_length,
        elapsed_seconds=time.perf_counter() - started,
        h2d_seconds=h2d_seconds,
        baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
    )


def decode_asr_batch(
    processor,
    generated: torch.Tensor,
    prepared: PreparedASRBatch,
) -> list[str]:
    chunk_texts = processor.batch_decode(generated, skip_special_tokens=True)
    return reassemble_chunk_texts(
        chunk_texts,
        prepared.chunk_index,
        " ",
        len(prepared.refs),
    )


def record_prepared_batch(stats: RunStats, prepared: PreparedASRBatch) -> None:
    stats.asr_feature_seconds += prepared.prepare_seconds
    stats.asr_processor_rows += len(prepared.chunk_index)
    stats.asr_valid_feature_frames += prepared.valid_feature_frames
    stats.asr_padded_feature_frames += prepared.padded_feature_frames
    stats.pin_memory_fallbacks += prepared.pin_memory_fallbacks


def record_generation_batch(
    stats: RunStats,
    prepared: PreparedASRBatch,
    result: ASRGenerationResult,
) -> None:
    stats.asr_batches += 1
    stats.asr_generate_seconds += result.elapsed_seconds
    stats.asr_h2d_seconds += result.h2d_seconds
    stats.asr_generated_tokens += sum(result.row_token_counts)
    stats.peak_cuda_gib = max(
        stats.peak_cuda_gib, result.peak_allocated_bytes / 1024**3
    )
    stats.peak_cuda_reserved_gib = max(
        stats.peak_cuda_reserved_gib, result.peak_reserved_bytes / 1024**3
    )
    rows = len(prepared.refs)
    stats.effective_batch_min = (
        rows if stats.effective_batch_min == 0 else min(stats.effective_batch_min, rows)
    )
    stats.effective_batch_max = max(stats.effective_batch_max, rows)
    stats.batch_history.append(
        {
            "segments": rows,
            "processor_rows": len(prepared.chunk_index),
            "max_new_tokens": result.max_new_tokens,
            "generated_tokens": sum(result.row_token_counts),
            "generated_tokens_by_row": list(result.row_token_counts),
            "prepare_seconds": prepared.prepare_seconds,
            "h2d_seconds": result.h2d_seconds,
            "generate_seconds": result.elapsed_seconds,
            "padded_audio_seconds": (
                len(prepared.refs)
                * max((ref.duration for ref in prepared.refs), default=0.0)
            ),
            "padding_ratio": (
                0.0
                if prepared.padded_feature_frames == 0
                else 1.0
                - prepared.valid_feature_frames / prepared.padded_feature_frames
            ),
            "peak_allocated_gib": result.peak_allocated_bytes / 1024**3,
            "peak_reserved_gib": result.peak_reserved_bytes / 1024**3,
        }
    )


def apply_generation_metadata(
    prepared: PreparedASRBatch,
    result: ASRGenerationResult,
) -> None:
    per_ref_tokens: dict[int, int] = {}
    for row_index, count in enumerate(result.row_token_counts):
        ref_index = int(prepared.chunk_index[row_index][0])
        per_ref_tokens[ref_index] = max(per_ref_tokens.get(ref_index, 0), count)
    for ref_index, count in per_ref_tokens.items():
        ref = prepared.refs[ref_index]
        ref.job.generated_tokens[ref.segment_index] = max(
            ref.job.generated_tokens.get(ref.segment_index, 0), count
        )
    for ref_index in result.repetition_ref_indices:
        ref = prepared.refs[ref_index]
        ref.job.repetition_stopped_segments.add(ref.segment_index)


def commit_asr_texts(
    refs: Sequence[SegmentRef], texts: Sequence[str], bar: tqdm
) -> None:
    if len(texts) != len(refs):
        raise RuntimeError(f"ASR returned {len(texts)} texts for {len(refs)} segments")
    for ref, text in zip(refs, texts, strict=True):
        ref.job.segment_texts[ref.segment_index] = text
    bar.update(len(refs))


def retry_token_limit(
    model,
    prompt_length: int,
    current_limit: int,
    requested_maximum: int,
) -> int:
    decoder_config = getattr(model.config, "decoder_config", None) or model.config
    max_positions = int(
        getattr(
            decoder_config, "max_position_embeddings", requested_maximum + prompt_length
        )
    )
    positional_cap = max(1, max_positions - prompt_length)
    ceiling = min(requested_maximum, positional_cap)
    return min(ceiling, max(current_limit + 128, current_limit * 2))


def finish_asr_batch(
    processor,
    model,
    prepared: PreparedASRBatch,
    result: ASRGenerationResult,
    args: TranscriptionConfig,
    bar: tqdm,
    stats: RunStats,
    controller: ASRBatchController,
) -> None:
    started = time.perf_counter()
    texts = decode_asr_batch(processor, result.generated, prepared)
    stats.asr_decode_seconds += time.perf_counter() - started
    apply_generation_metadata(prepared, result)

    retry_indices = set(result.truncated_ref_indices)
    next_limit = retry_token_limit(
        model,
        result.prompt_length,
        result.max_new_tokens,
        args.max_retry_tokens,
    )
    can_retry = (
        args.truncation_policy == "retry"
        and retry_indices
        and next_limit > result.max_new_tokens
    )
    if can_retry:
        keep_indices = [
            index for index in range(len(prepared.refs)) if index not in retry_indices
        ]
        if keep_indices:
            commit_asr_texts(
                [prepared.refs[index] for index in keep_indices],
                [texts[index] for index in keep_indices],
                bar,
            )
        retry_refs = [prepared.refs[index] for index in sorted(retry_indices)]
        for ref in retry_refs:
            ref.job.truncation_retried_segments.add(ref.segment_index)
        stats.asr_truncation_retries += len(retry_refs)
        info(
            f"[tokens] retrying {len(retry_refs)} segment(s) with "
            f"max_new_tokens={next_limit}"
        )
        transcribe_ref_batch(
            processor,
            model,
            retry_refs,
            args,
            bar,
            stats,
            controller,
            max_new_tokens=next_limit,
        )
        return

    if retry_indices:
        for ref_index in retry_indices:
            ref = prepared.refs[ref_index]
            ref.job.token_limit_segments.add(ref.segment_index)
    commit_asr_texts(prepared.refs, texts, bar)


def mark_asr_jobs_failed(refs: Sequence[SegmentRef], message: str, bar: tqdm) -> None:
    affected_jobs = {ref.job.index: ref.job for ref in refs}
    for affected in affected_jobs.values():
        if affected.error is None:
            affected.error = f"ASR failed: {message}"
            info(f"[error] {affected.path}: {affected.error}")
    bar.update(len(refs))


def balanced_oom_split(refs: Sequence[SegmentRef]) -> int:
    if len(refs) < 2:
        return 1
    first_duration = refs[0].duration
    best_index = len(refs) // 2
    best_cost = float("inf")
    for index in range(1, len(refs)):
        left_cost = index * first_duration
        right_cost = (len(refs) - index) * refs[index].duration
        cost = max(left_cost, right_cost)
        if cost < best_cost:
            best_cost = cost
            best_index = index
    return best_index


def handle_asr_batch_failure(
    processor,
    model,
    refs: Sequence[SegmentRef],
    args: TranscriptionConfig,
    bar: tqdm,
    stats: RunStats,
    controller: ASRBatchController,
    failure_kind: str,
    message: str,
    max_new_tokens: int,
) -> None:
    if failure_kind == "oom":
        stats.asr_oom_retries += 1
        if model.device.type == "cuda" and torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(model.device) / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved(model.device) / 1024**3
            stats.peak_cuda_gib = max(stats.peak_cuda_gib, peak_allocated)
            stats.peak_cuda_reserved_gib = max(
                stats.peak_cuda_reserved_gib, peak_reserved
            )
            stats.batch_history.append(
                {
                    "event": "oom",
                    "segments": len(refs),
                    "max_new_tokens": max_new_tokens,
                    "peak_allocated_gib": peak_allocated,
                    "peak_reserved_gib": peak_reserved,
                }
            )
        learn_base_batch_cap = max_new_tokens <= args.max_new_tokens
        if learn_base_batch_cap:
            controller.record_oom(len(refs))
        gc.collect()
        empty_device_cache(model.device.type)
    else:
        learn_base_batch_cap = False

    if len(refs) > 1:
        midpoint = balanced_oom_split(refs) if failure_kind == "oom" else len(refs) // 2
        if failure_kind == "oom":
            cap_note = (
                f"; future cap {controller.current_size}"
                if learn_base_batch_cap
                else "; base ASR cap unchanged"
            )
            tqdm.write(
                f"{INDENT}[oom] ASR retrying batch {len(refs)} as "
                f"{midpoint}+{len(refs) - midpoint}{cap_note}"
            )
        transcribe_ref_batch(
            processor,
            model,
            refs[:midpoint],
            args,
            bar,
            stats,
            controller,
            max_new_tokens=max_new_tokens,
        )
        remaining = [ref for ref in refs[midpoint:] if ref.job.error is None]
        bar.update(len(refs) - midpoint - len(remaining))
        if remaining:
            transcribe_ref_batch(
                processor,
                model,
                remaining,
                args,
                bar,
                stats,
                controller,
                max_new_tokens=max_new_tokens,
            )
        return
    mark_asr_jobs_failed(refs, message, bar)


def transcribe_ref_batch(
    processor,
    model,
    refs: Sequence[SegmentRef],
    args: TranscriptionConfig,
    bar: tqdm,
    stats: RunStats,
    controller: ASRBatchController,
    max_new_tokens: int,
) -> None:
    failure_kind = ""
    failure_message = ""
    prepared: PreparedASRBatch | None = None
    result: ASRGenerationResult | None = None
    try:
        prepared = prepare_asr_batch(processor, refs, args)
        record_prepared_batch(stats, prepared)
        result = generate_asr_batch(model, prepared, args, max_new_tokens)
        record_generation_batch(stats, prepared, result)
        if max_new_tokens <= args.max_new_tokens:
            controller.record_success(result, len(refs))
        finish_asr_batch(
            processor, model, prepared, result, args, bar, stats, controller
        )
        return
    except torch.OutOfMemoryError:
        failure_kind = "oom"
        failure_message = "device out of memory on a single segment"
    except Exception as exc:
        failure_kind = "error"
        failure_message = f"{type(exc).__name__}: {exc}"
    finally:
        result = None
        prepared = None

    handle_asr_batch_failure(
        processor,
        model,
        refs,
        args,
        bar,
        stats,
        controller,
        failure_kind,
        failure_message,
        max_new_tokens,
    )


def transcribe_group(
    processor,
    model,
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    stats: RunStats,
) -> float:
    refs = [
        SegmentRef(job, segment_index, start, end)
        for job in jobs
        if job.error is None
        for segment_index, (start, end) in enumerate(job.segment_times)
    ]
    refs.sort(key=lambda ref: (-ref.duration, ref.job.index, ref.segment_index))
    if not refs:
        return 0.0

    controller = getattr(model, "_transcribe_batch_controller", None)
    if controller is None:
        controller = ASRBatchController.create(args, model, refs)
        model._transcribe_batch_controller = controller
        info(
            f"ASR batch controller: start {controller.initial_size}, "
            f"cap {controller.max_size}, padded-audio budget "
            f"{controller.audio_budget_seconds:.0f}s, VRAM target "
            f"{controller.target_vram_ratio:.0%}"
            + (
                f" ({controller.memory_budget_bytes / 1024**3:.2f} GiB usable)"
                if controller.memory_budget_bytes
                else ""
            )
        )
    controller.configure_group(args, refs)

    pending: deque[SegmentRef] = deque(refs)
    started = time.perf_counter()
    bar = tqdm(
        total=len(refs),
        unit="seg",
        desc="transcribing",
        dynamic_ncols=True,
        bar_format=BAR_FMT,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="feature-prep"
        ) as executor:
            current_refs = controller.take(pending)
            current_future: concurrent.futures.Future | None = executor.submit(
                prepare_asr_batch, processor, current_refs, args
            )
            while current_refs:
                active_refs = [ref for ref in current_refs if ref.job.error is None]
                skipped = len(current_refs) - len(active_refs)
                if skipped:
                    bar.update(skipped)

                prepared: PreparedASRBatch | None = None
                preparation_error = ""
                wait_started = time.perf_counter()
                try:
                    assert current_future is not None
                    prepared = current_future.result()
                    record_prepared_batch(stats, prepared)
                except Exception as exc:
                    preparation_error = f"{type(exc).__name__}: {exc}"
                finally:
                    stats.asr_feature_wait_seconds += time.perf_counter() - wait_started

                if skipped and active_refs:
                    try:
                        prepared = prepare_asr_batch(processor, active_refs, args)
                        record_prepared_batch(stats, prepared)
                        preparation_error = ""
                    except Exception as exc:
                        prepared = None
                        preparation_error = f"{type(exc).__name__}: {exc}"

                next_refs = controller.take(pending)
                next_future = (
                    executor.submit(prepare_asr_batch, processor, next_refs, args)
                    if next_refs
                    else None
                )

                result: ASRGenerationResult | None = None
                generation_failure = ""
                generation_kind = ""
                if active_refs and prepared is not None and not preparation_error:
                    try:
                        result = generate_asr_batch(
                            model, prepared, args, args.max_new_tokens
                        )
                        record_generation_batch(stats, prepared, result)
                        controller.record_success(result, len(active_refs))
                    except torch.OutOfMemoryError:
                        generation_kind = "oom"
                        generation_failure = "device out of memory on a single segment"
                    except Exception as exc:
                        generation_kind = "error"
                        generation_failure = f"{type(exc).__name__}: {exc}"

                # Keep tokenizer use serialized with processor feature preparation.
                if next_future is not None:
                    wait_started = time.perf_counter()
                    try:
                        with contextlib.suppress(Exception):
                            next_future.result()
                    finally:
                        stats.asr_feature_wait_seconds += (
                            time.perf_counter() - wait_started
                        )

                if active_refs:
                    if preparation_error:
                        handle_asr_batch_failure(
                            processor,
                            model,
                            active_refs,
                            args,
                            bar,
                            stats,
                            controller,
                            "error",
                            preparation_error,
                            args.max_new_tokens,
                        )
                    elif generation_kind:
                        handle_asr_batch_failure(
                            processor,
                            model,
                            active_refs,
                            args,
                            bar,
                            stats,
                            controller,
                            generation_kind,
                            generation_failure,
                            args.max_new_tokens,
                        )
                    else:
                        try:
                            assert prepared is not None and result is not None
                            finish_asr_batch(
                                processor,
                                model,
                                prepared,
                                result,
                                args,
                                bar,
                                stats,
                                controller,
                            )
                        except Exception as exc:
                            handle_asr_batch_failure(
                                processor,
                                model,
                                active_refs,
                                args,
                                bar,
                                stats,
                                controller,
                                "error",
                                f"{type(exc).__name__}: {exc}",
                                args.max_new_tokens,
                            )

                # A failed current batch can lower the persistent controller
                # cap after the next batch has already been prepared. Do not
                # repeat the same known-oversized probe merely because it was
                # one item ahead in the CPU feature pipeline. Account for the
                # discarded preparation work, restore those refs in order, and
                # prepare a batch under the learned cap instead.
                if next_refs and len(next_refs) > controller.current_size:
                    if next_future is not None:
                        try:
                            discarded = next_future.result()
                        except Exception:
                            discarded = None
                        if discarded is not None:
                            record_prepared_batch(stats, discarded)
                            stats.asr_discarded_feature_batches += 1
                            discarded = None
                    pending.extendleft(reversed(next_refs))
                    next_refs = controller.take(pending)
                    next_future = executor.submit(
                        prepare_asr_batch, processor, next_refs, args
                    )
                current_refs = next_refs
                current_future = next_future
    finally:
        bar.close()
    stats.final_batch_size = controller.current_size
    stats.final_batch_cap = controller.max_size
    return time.perf_counter() - started


# Forced alignment


def load_aligner(
    device: str,
    dtype: torch.dtype,
    revision: str | None = ALIGN_MODEL_REVISION,
):
    from transformers import AutoTokenizer, Wav2Vec2ForCTC

    tokenizer = AutoTokenizer.from_pretrained(
        ALIGN_MODEL_ID,
        revision=revision,
        word_delimiter_token=None,
    )
    model = Wav2Vec2ForCTC.from_pretrained(
        ALIGN_MODEL_ID,
        dtype=dtype,
        attn_implementation="sdpa",
        revision=revision,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def alignment_frame_geometry(model) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    raw_ratio = getattr(config, "inputs_to_logits_ratio", None)
    if isinstance(raw_ratio, bool) or not isinstance(raw_ratio, (int, np.integer)):
        raise RuntimeError(
            "Aligner config must define an integer inputs_to_logits_ratio"
        )
    ratio = int(raw_ratio)
    if ratio <= 0:
        raise RuntimeError("Aligner inputs_to_logits_ratio must be positive")

    window_samples = ALIGN_WINDOW_S * SR
    context_samples = ALIGN_CONTEXT_S * SR
    if window_samples % ratio or context_samples % ratio:
        raise RuntimeError(
            "Alignment window and context must be divisible by the model input stride"
        )
    return ratio, window_samples // ratio, context_samples // ratio


def build_alignment_window_batch(
    audio: np.ndarray,
    window_indices: range,
    window_samples: int,
    context_samples: int,
) -> np.ndarray:
    """Build the same zero-padded windows as ctc_forced_aligner, one batch at a time."""
    input_samples = window_samples + 2 * context_samples
    batch: np.ndarray = np.zeros((len(window_indices), input_samples), dtype=np.float32)
    audio_samples = len(audio)
    for row, window_index in enumerate(window_indices):
        requested_start = window_index * window_samples - context_samples
        requested_end = requested_start + input_samples
        source_start = max(0, requested_start)
        source_end = min(audio_samples, requested_end)
        if source_end <= source_start:
            continue
        destination_start = source_start - requested_start
        destination_end = destination_start + source_end - source_start
        batch[row, destination_start:destination_end] = audio[source_start:source_end]
    return batch


def _compute_emissions_streaming(audio: np.ndarray, model, batch_size: int, label: str):
    window_samples = ALIGN_WINDOW_S * SR
    context_samples = ALIGN_CONTEXT_S * SR
    ratio, window_frames, context_frames = alignment_frame_geometry(model)
    total_windows = math.ceil(len(audio) / window_samples)
    extension_samples = total_windows * window_samples - len(audio)
    extension_frames = extension_samples // ratio
    emissions: np.ndarray | None = None
    write_offset = 0
    first_window = 0
    learned_batch_size = int(
        getattr(model, "_transcribe_align_batch_size", max(1, batch_size))
    )
    current_batch_size = max(1, min(batch_size, learned_batch_size))

    bar = tqdm(
        total=total_windows,
        unit="win",
        desc=f"emissions {label}",
        dynamic_ncols=True,
        bar_format=BAR_FMT,
    )
    try:
        while first_window < total_windows:
            window_count = min(current_batch_size, total_windows - first_window)
            indices = range(first_window, first_window + window_count)
            input_batch = build_alignment_window_batch(
                audio, indices, window_samples, context_samples
            )
            values = None
            logits = None
            log_probs_tensor = None
            try:
                values = torch.from_numpy(input_batch).to(
                    model.device, dtype=model.dtype
                )
                with torch.inference_mode():
                    logits = model(values).logits
                    required_frames = context_frames + window_frames
                    if logits.shape[1] < required_frames:
                        raise RuntimeError(
                            "Aligner returned too few frames for the configured window"
                        )
                    logits = logits[
                        :, context_frames : context_frames + window_frames, :
                    ]
                    # Stable in FP32 on the accelerator; avoids exp overflow and
                    # transfers only normalized, cropped frames to host memory.
                    log_probs_tensor = torch.log_softmax(logits.float(), dim=-1)
                batch_log_probs = log_probs_tensor.cpu().numpy()
            except torch.OutOfMemoryError:
                values = None
                logits = None
                log_probs_tensor = None
                del input_batch
                gc.collect()
                if model.device.type == "cuda":
                    torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                model._transcribe_align_batch_size = current_batch_size
                info(
                    f"[oom] emissions continuing {label} with batch "
                    f"{current_batch_size} (completed {first_window}/{total_windows} windows)"
                )
                continue
            finally:
                values = None
                logits = None
                log_probs_tensor = None

            del input_batch
            batch_log_probs = batch_log_probs.reshape(-1, batch_log_probs.shape[-1])
            expected_batch_frames = window_count * window_frames
            if batch_log_probs.shape[0] != expected_batch_frames:
                raise RuntimeError(
                    "Aligner returned an unexpected number of frames for its windows"
                )
            if emissions is None:
                frame_count = total_windows * window_frames - extension_frames
                if frame_count <= 0:
                    raise RuntimeError("Aligner produced no usable CTC frames")
                emissions = np.zeros(
                    (frame_count, batch_log_probs.shape[-1] + 1),
                    dtype=np.float32,
                )
            if first_window + window_count == total_windows and extension_frames:
                batch_log_probs = batch_log_probs[:-extension_frames]

            next_offset = write_offset + len(batch_log_probs)
            if next_offset > len(emissions):
                raise RuntimeError(
                    "Aligner produced more CTC frames than the first window predicted"
                )
            emissions[write_offset:next_offset, :-1] = batch_log_probs
            write_offset = next_offset
            first_window += window_count
            del batch_log_probs
            bar.update(window_count)
    finally:
        bar.close()

    if emissions is None or write_offset != len(emissions):
        raise RuntimeError(
            f"Aligner emission assembly mismatch: wrote {write_offset} frames, "
            f"expected {0 if emissions is None else len(emissions)}"
        )
    stride = ratio * 1000 / SR
    return emissions, stride


def compute_emissions_streaming(
    audio: np.ndarray,
    model,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, float]:
    if len(audio) == 0:
        raise ValueError("Cannot compute CTC emissions for empty audio")
    return _compute_emissions_streaming(audio, model, max(1, batch_size), label)


@dataclass(slots=True)
class AlignmentVocabulary:
    dictionary: dict[str, int]
    index_to_token: dict[int, str]
    blank_id: int


def build_alignment_vocabulary(tokenizer, emission_classes: int) -> AlignmentVocabulary:
    if emission_classes < 2:
        raise ValueError("Aligner must expose at least one token and one <star> column")
    dictionary = {
        key.lower(): int(value) for key, value in tokenizer.get_vocab().items()
    }
    star_id = emission_classes - 1
    if star_id in dictionary.values():
        raise ValueError(
            "The reserved <star> emission column collides with the tokenizer vocabulary"
        )
    dictionary["<star>"] = star_id
    raw_blank_id = dictionary.get("<blank>", tokenizer.pad_token_id)
    if raw_blank_id is None:
        raise ValueError("Aligner tokenizer does not define a blank or pad token")
    blank_id = int(raw_blank_id)
    if not 0 <= blank_id < emission_classes:
        raise ValueError("blank must be within the emissions vocabulary")
    index_to_token = {value: key for key, value in dictionary.items()}
    if blank_id not in index_to_token:
        index_to_token[blank_id] = "<blank>"
    return AlignmentVocabulary(dictionary, index_to_token, blank_id)


def get_alignments_safe(
    emissions: np.ndarray,
    tokens: Sequence[str],
    vocabulary: AlignmentVocabulary,
):
    """Run the maintained TorchAudio CTC op, bypassing the package ctypes extension."""
    from ctc_forced_aligner import merge_repeats
    from torchaudio.functional import forced_align

    token_indices = [
        vocabulary.dictionary[token]
        for token in " ".join(tokens).split(" ")
        if token in vocabulary.dictionary
    ]
    if not token_indices:
        raise ValueError("Transcript produced no aligner vocabulary tokens")
    if vocabulary.blank_id in token_indices:
        raise ValueError(
            f"targets array should not contain blank index ({vocabulary.blank_id})"
        )
    if max(token_indices) >= emissions.shape[-1] or min(token_indices) < 0:
        raise ValueError("targets values must be within the emissions vocabulary")

    targets = torch.tensor([token_indices], dtype=torch.int64)
    log_probs = torch.from_numpy(
        np.ascontiguousarray(emissions[None], dtype=np.float32)
    )
    path, scores = forced_align(log_probs, targets, blank=vocabulary.blank_id)
    path_values = path.squeeze(0).tolist()
    score_values = scores.squeeze(0).numpy()
    return (
        merge_repeats(path_values, vocabulary.index_to_token),
        score_values,
        vocabulary.index_to_token[vocabulary.blank_id],
    )


def uniform_word_timings(
    text: str,
    start: float,
    end: float,
    segment_index: int,
    timing_source: str,
) -> list[WordTiming]:
    """Keep every ASR token when precise timing is unavailable."""
    tokens = text.strip().split()
    if not tokens:
        return []
    token_duration = max(0.0, end - start) / len(tokens)
    return [
        {
            "start": start + token_index * token_duration,
            "end": start + (token_index + 1) * token_duration,
            "text": token,
            "segment_index": segment_index,
            "segment_word_index": token_index,
            "timing_source": timing_source,
        }
        for token_index, token in enumerate(tokens)
    ]


def proportional_token_counts(
    token_count: int, spans: Sequence[tuple[float, float]]
) -> list[int]:
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    durations = [max(0.0, end - start) for start, end in spans]
    total = sum(durations)
    if token_count == 0 or total <= 0:
        return [0] * len(spans)
    exact = [token_count * duration / total for duration in durations]
    counts = [int(math.floor(value)) for value in exact]
    remaining = token_count - sum(counts)
    order = sorted(
        range(len(spans)),
        key=lambda index: (exact[index] - counts[index], durations[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def uniform_word_timings_across_spans(
    text: str,
    spans: Sequence[tuple[float, float]],
    segment_index: int,
    timing_source: str,
) -> list[WordTiming]:
    """Distribute words over speech spans without stretching them across silence."""
    tokens = text.strip().split()
    valid_spans = [(start, end) for start, end in spans if end > start]
    if not tokens or not valid_spans:
        return []
    counts = proportional_token_counts(len(tokens), valid_spans)
    words: list[WordTiming] = []
    token_offset = 0
    for (start, end), count in zip(valid_spans, counts, strict=True):
        if count <= 0:
            continue
        token_duration = (end - start) / count
        for local_index in range(count):
            token_index = token_offset + local_index
            words.append(
                {
                    "start": start + local_index * token_duration,
                    "end": start + (local_index + 1) * token_duration,
                    "text": tokens[token_index],
                    "segment_index": segment_index,
                    "segment_word_index": token_index,
                    "timing_source": timing_source,
                }
            )
        token_offset += count
    if token_offset != len(tokens):
        raise RuntimeError("Speech-span token allocation did not preserve every token")
    return words


def speech_spans_within_segment(
    speech_spans: Sequence[tuple[float, float]], start: float, end: float
) -> list[tuple[float, float]]:
    return [
        (max(start, speech_start), min(end, speech_end))
        for speech_start, speech_end in speech_spans
        if speech_end > start
        and speech_start < end
        and min(end, speech_end) > max(start, speech_start)
    ]


def align_words(
    emissions: np.ndarray,
    stride: float,
    tokenizer,
    segment_times: Sequence[tuple[float, float]],
    segment_texts: Sequence[str],
    language: str,
) -> tuple[list[WordTiming], int]:
    from ctc_forced_aligner import get_spans, postprocess_results, preprocess_text

    iso_language = ISO3.get(language, language)
    frame_count = emissions.shape[0]
    words: list[WordTiming] = []
    fallback_count = 0
    pairs = list(zip(segment_times, segment_texts, strict=True))
    vocabulary = build_alignment_vocabulary(tokenizer, emissions.shape[-1])
    bar = tqdm(
        pairs, unit="seg", desc="aligning", dynamic_ncols=True, bar_format=BAR_FMT
    )
    for segment_index, ((start, end), text) in enumerate(bar):
        text = text.strip()
        if not text:
            continue
        first_frame = max(0, int(round(start * 1000 / stride)))
        last_frame = min(frame_count, int(round(end * 1000 / stride)))
        if last_frame - first_frame < 2:
            fallback_count += 1
            tqdm.write(
                f"{INDENT}[warn] segment {segment_index} is shorter than two CTC frames; "
                "using uniform word timing"
            )
            words.extend(
                uniform_word_timings(
                    text,
                    start,
                    end,
                    segment_index,
                    "uniform_fallback",
                )
            )
            continue
        try:
            tokens_starred, text_starred = preprocess_text(
                text, romanize=True, language=iso_language
            )
            segments, scores, blank = get_alignments_safe(
                emissions[first_frame:last_frame], tokens_starred, vocabulary
            )
            spans = get_spans(tokens_starred, segments, blank)
            results = postprocess_results(text_starred, spans, stride, scores)
            expected_tokens = text.split()
            aligned_tokens = [word["text"] for word in results]
            if aligned_tokens != expected_tokens:
                raise ValueError(
                    "forced alignment did not preserve the complete ASR transcript "
                    f"({len(aligned_tokens)}/{len(expected_tokens)} words)"
                )
        except Exception as exc:
            fallback_count += 1
            tqdm.write(
                f"{INDENT}[warn] align failed on segment {segment_index}: {repr(exc)[:100]}"
            )
            words.extend(
                uniform_word_timings(
                    text,
                    start,
                    end,
                    segment_index,
                    "uniform_fallback",
                )
            )
            continue
        for word_index, word in enumerate(results):
            absolute_start = min(end, max(start, start + float(word["start"])))
            absolute_end = min(
                end,
                max(absolute_start, start + float(word["end"])),
            )
            words.append(
                {
                    "start": absolute_start,
                    "end": absolute_end,
                    "text": word["text"],
                    "segment_index": segment_index,
                    "segment_word_index": word_index,
                    "timing_source": "ctc",
                }
            )
    return words, fallback_count


# Output rendering and publication


def build_cues(
    words: Sequence[WordTiming],
    max_chars: int,
    max_duration: float,
    max_gap: float,
    min_cue_duration: float = 0.30,
    media_duration: float | None = None,
) -> list[SubtitleCue]:
    cue_words: list[list[WordTiming]] = []
    current: list[WordTiming] = []
    for word in words:
        if current:
            candidate = " ".join(item["text"] for item in current) + " " + word["text"]
            gap = word["start"] - current[-1]["end"]
            duration = word["end"] - current[0]["start"]
            if len(candidate) > max_chars or duration > max_duration or gap > max_gap:
                cue_words.append(current)
                current = []
        current.append(word)
        if word["text"].endswith(SENTENCE_ENDINGS):
            cue_words.append(current)
            current = []
    if current:
        cue_words.append(current)

    cues: list[SubtitleCue] = [
        {
            "start": max(0.0, items[0]["start"]),
            "end": max(items[0]["start"], items[-1]["end"]),
            "text": " ".join(item["text"] for item in items).strip(),
        }
        for items in cue_words
        if items
    ]
    for index, cue in enumerate(cues):
        if media_duration is not None:
            cue["start"] = min(cue["start"], media_duration)
            cue["end"] = min(max(cue["start"], cue["end"]), media_duration)
        next_start = cues[index + 1]["start"] if index + 1 < len(cues) else math.inf
        media_end = media_duration if media_duration is not None else math.inf
        upper_bound = min(next_start, media_end)
        desired_end = max(cue["end"], cue["start"] + min_cue_duration)
        cue["end"] = max(cue["start"], min(desired_end, upper_bound))
    return cues


def fmt_timestamp(
    seconds: float, include_hours: bool = False, marker: str = "."
) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    if include_hours or hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{marker}{milliseconds:03d}"
    return f"{minutes:02d}:{seconds:02d}{marker}{milliseconds:03d}"


def generate_plain_text(lines: Sequence[str]) -> str:
    return "\n".join(line.strip() for line in lines if line.strip()) + "\n"


def generate_srt(cues: Sequence[SubtitleCue]) -> str:
    return "".join(
        f"{index}\n{fmt_timestamp(cue['start'], True, ',')} --> "
        f"{fmt_timestamp(cue['end'], True, ',')}\n{cue['text']}\n\n"
        for index, cue in enumerate(cues, 1)
    )


def generate_vtt(cues: Sequence[SubtitleCue]) -> str:
    return "WEBVTT\n\n" + "".join(
        f"{fmt_timestamp(cue['start'])} --> {fmt_timestamp(cue['end'])}\n{cue['text']}\n\n"
        for cue in cues
    )


def generate_json(
    job: AudioJob,
    words: Sequence[WordTiming],
    cues: Sequence[SubtitleCue],
    transcript_lines: Sequence[str],
) -> str:
    segments = [
        {"start": start, "end": end, "text": text.strip()}
        for (start, end), text in zip(job.segment_times, job.segment_texts, strict=True)
        if text.strip()
    ]
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "source": {
            "path": os.fspath(job.path),
            "duration_seconds": job.duration,
            "sample_rate": SR,
            "decode_backend": job.decode_backend,
        },
        "language": job.language,
        "segmentation": job.vad_mode,
        "segmentation_details": {
            "mode": job.vad_mode,
            "requested_engine": job.vad_engine_requested,
            "actual_engine": job.vad_engine_actual,
            "provider": job.vad_provider,
            "provider_options": job.vad_provider_options,
            "fallback_reason": job.vad_fallback_reason,
            "merge": job.vad_merge,
            "parameters": job.segmentation_parameters,
            "speech_spans": [
                {"start": start, "end": end} for start, end in job.speech_spans
            ],
        },
        "timing": job.alignment_mode,
        "models": {
            "asr": {"id": MODEL_ID, "revision": ASR_MODEL_REVISION},
            "aligner": (
                {
                    "id": ALIGN_MODEL_ID,
                    "revision": ALIGN_MODEL_REVISION,
                    "package_repository": ALIGN_PACKAGE_REPOSITORY,
                    "package_revision": ALIGN_PACKAGE_REVISION,
                    "romanizer": "uroman",
                }
                if job.alignment_mode == "word"
                else None
            ),
        },
        "fallback_alignment_segments": job.fallback_alignments,
        "repetition_detector_version": REPETITION_DETECTOR_VERSION,
        "repetition_stopped_segments": sorted(job.repetition_stopped_segments),
        "truncation_retried_segments": sorted(job.truncation_retried_segments),
        "token_limit_segments": sorted(job.token_limit_segments),
        "generated_tokens_by_segment": [
            {"segment_index": index, "tokens": count}
            for index, count in sorted(job.generated_tokens.items())
        ],
        "transcript": [line.strip() for line in transcript_lines if line.strip()],
        "segments": segments,
        "words": words,
        "cues": cues,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


OUTPUT_GENERATORS = {"srt": generate_srt, "vtt": generate_vtt}


def fsync_directories(directories: Iterator[Path]) -> None:
    """Persist directory entries where the platform supports directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    for directory in dict.fromkeys(path.resolve() for path in directories):
        descriptor = os.open(directory, flags)
        try:
            try:
                os.fsync(descriptor)
            except OSError as exc:
                if exc.errno not in {errno.EBADF, errno.EINVAL, errno.ENOTSUP}:
                    raise
        finally:
            os.close(descriptor)


def atomic_write_outputs(
    job: AudioJob,
    cues: Sequence[SubtitleCue],
    words: Sequence[WordTiming] = (),
    transcript_lines: Sequence[str] | None = None,
    only_formats: set[str] | None = None,
) -> None:
    """Publish one job's formats with rollback if an in-process commit fails."""
    transcript_lines = (
        job.segment_texts if transcript_lines is None else transcript_lines
    )
    output_paths = {
        output_format: output_path
        for output_format, output_path in job.output_paths.items()
        if only_formats is None or output_format in only_formats
    }
    if not output_paths:
        return
    temporary_paths: dict[Path, Path] = {}
    backup_paths: dict[Path, Path | None] = {}
    published: list[Path] = []
    preserved_backups: set[Path] = set()
    try:
        for output_format, output_path in output_paths.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
            )
            temporary_path = Path(temporary_name)
            temporary_paths[output_path] = temporary_path
            output_mode = (
                stat.S_IMODE(output_path.stat().st_mode)
                if output_path.exists()
                else DEFAULT_OUTPUT_MODE
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    if output_format == "json":
                        handle.write(generate_json(job, words, cues, transcript_lines))
                    elif output_format == "txt":
                        handle.write(generate_plain_text(transcript_lines))
                    else:
                        handle.write(OUTPUT_GENERATORS[output_format](cues))
                    handle.flush()
                    os.fchmod(handle.fileno(), output_mode)
                    os.fsync(handle.fileno())
            except BaseException:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                raise

        for output_path in output_paths.values():
            if not output_path.exists():
                backup_paths[output_path] = None
                continue
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.", suffix=".bak", dir=output_path.parent
            )
            os.close(descriptor)
            backup_path = Path(backup_name)
            backup_paths[output_path] = backup_path
            shutil.copy2(output_path, backup_path)
            with backup_path.open("rb") as backup_handle:
                os.fsync(backup_handle.fileno())

        ensure_source_unchanged(job)
        for output_path in output_paths.values():
            os.replace(temporary_paths[output_path], output_path)
            published.append(output_path)
        fsync_directories(output.parent for output in output_paths.values())
        job.written.extend(output_paths.values())
    except BaseException as original_error:
        rollback_errors: list[str] = []
        for output_path in reversed(published):
            rollback_backup = backup_paths.get(output_path)
            try:
                if rollback_backup is None:
                    output_path.unlink(missing_ok=True)
                elif rollback_backup.exists():
                    os.replace(rollback_backup, output_path)
            except BaseException as rollback_error:
                if rollback_backup is not None and rollback_backup.exists():
                    preserved_backups.add(rollback_backup)
                rollback_errors.append(f"{output_path}: {rollback_error}")
        try:
            fsync_directories(output.parent for output in output_paths.values())
        except OSError as rollback_error:
            rollback_errors.append(f"directory sync: {rollback_error}")
        if rollback_errors:
            detail = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Output commit failed and rollback was incomplete ({detail}); "
                f"preserved backups: {sorted(map(os.fspath, preserved_backups))}"
            ) from original_error
        raise
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)
        for cleanup_backup in backup_paths.values():
            if cleanup_backup is not None and cleanup_backup not in preserved_backups:
                cleanup_backup.unlink(missing_ok=True)


def ensure_source_unchanged(job: AudioJob) -> None:
    current = SourceSnapshot.capture(job.path)
    if current != job.snapshot:
        raise RuntimeError(f"Source changed while processing: {job.path}")


def reload_audio_for_alignment(
    job: AudioJob,
    args: TranscriptionConfig,
) -> np.ndarray:
    ensure_source_unchanged(job)
    if job.audio is not None:
        return job.audio
    audio = decode_audio(job.path, job.decode_backend or args.audio_backend)
    if len(audio) != int(round(job.duration * SR)):
        raise RuntimeError(
            f"Decoded sample count changed between ASR and alignment for {job.path}: "
            f"{len(audio)} != {int(round(job.duration * SR))}"
        )
    return audio


# CLI and validation


class CompactDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if (
            action.default is None
            or action.default is False
            or action.default == argparse.SUPPRESS
        ):
            return help_text
        return super()._get_help_string(action) or help_text


def parse_args(argv: Sequence[str] | None = None) -> TranscriptionConfig:
    parser = argparse.ArgumentParser(
        description="Batch Arabic/English transcription with optional timestamp alignment.",
        formatter_class=CompactDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="One or more audio files or directories.",
    )
    parser.add_argument(
        "--language",
        default="ar",
        choices=["ar", "en"],
        help="Spoken language tag.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        choices=["txt", "srt", "vtt", "json"],
        help="Outputs to write (default: txt srt vtt; text-only: txt).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root; relative structure is preserved for directory inputs.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recurse into input directories.",
    )
    parser.add_argument(
        "--existing",
        default="error",
        choices=["error", "overwrite", "skip"],
        help="error, replace outputs, or skip complete sets and rebuild partial sets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="ASR model precision; CPU runs in FP32.",
    )
    parser.add_argument(
        "--audio-backend",
        default="auto",
        choices=["librosa", "torchcodec", "auto", "ffmpeg"],
        help="auto matches transcribe.py and falls back to FFmpeg for unsupported containers.",
    )
    parser.add_argument(
        "--audio-memory-gb",
        type=float,
        default=4.0,
        help="Decoded-audio group target; one file and decoder transients may exceed it.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help="Concurrent decode/VAD workers; auto uses 1 for one file and at most 2 otherwise.",
    )
    parser.add_argument(
        "--pipeline-preparation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlap bounded next-group decode/VAD with multi-file GPU ASR.",
    )

    group = parser.add_argument_group("segmentation (VAD)")
    group.add_argument(
        "--vad",
        default="silero",
        choices=["silero", "auditok", "none"],
        help=(
            "silero neural VAD, auditok energy VAD, or none for contiguous "
            "fixed windows that retain silence"
        ),
    )
    group.add_argument(
        "--vad-engine",
        default="auto",
        choices=["auto", "onnx", "jit"],
        help="Silero runtime; auto prefers faster ONNX and falls back to TorchScript.",
    )
    group.add_argument(
        "--vad-merge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Greedily merge adjacent Silero speech spans up to --max-dur, "
            "retaining intervening silence."
        ),
    )
    group.add_argument(
        "--min-dur",
        type=float,
        default=0.5,
        help="Minimum speech duration in seconds for Silero/Auditok.",
    )
    group.add_argument(
        "--max-dur",
        type=float,
        default=30.0,
        help="Maximum segment or fixed-window duration in seconds.",
    )
    group.add_argument(
        "--max-silence",
        type=float,
        default=0.6,
        help="Maximum internal silence in seconds for Auditok.",
    )
    group.add_argument(
        "--energy-threshold",
        type=float,
        default=50.0,
        help="Auditok log-energy threshold in decibels.",
    )
    group.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero speech-probability threshold.",
    )
    group.add_argument(
        "--min-silence-ms",
        type=int,
        default=300,
        help="Silero silence required to split a segment, in milliseconds.",
    )
    group.add_argument(
        "--speech-pad-ms",
        type=int,
        default=60,
        help="Silero padding added around speech, in milliseconds.",
    )

    group = parser.add_argument_group("transcription")
    group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Initial ASR segment count; auto uses the measured RTX 3060 optimum "
            "of 24 on CUDA and 8 elsewhere."
        ),
    )
    group.add_argument(
        "--batch-max-size",
        type=int,
        default=None,
        help="Upper row cap for adaptive batching; explicit --batch-size is the default cap.",
    )
    group.add_argument(
        "--batch-audio-seconds",
        type=float,
        default=None,
        help=(
            "Maximum padded audio seconds per batch; auto derives a fresh bounded "
            "budget for each decoded-audio group."
        ),
    )
    group.add_argument(
        "--batch-vram-target",
        type=float,
        default=0.90,
        help="Target fraction of total CUDA memory for adaptive batch growth.",
    )
    group.add_argument(
        "--adaptive-batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimentally grow successful batches from VRAM headroom; persistent "
            "OOM learning remains active when disabled."
        ),
    )
    group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Pin prepared CPU tensors and use nonblocking CUDA transfers; this is "
            "opt-in because the extra host copy was not faster on the RTX 3060."
        ),
    )
    group.add_argument(
        "--max-new-tokens",
        type=int,
        default=445,
        help="Initial decoder token limit per processor row.",
    )
    group.add_argument(
        "--max-retry-tokens",
        type=int,
        default=896,
        help="Maximum token limit for automatic retry of rows that end without EOS.",
    )
    group.add_argument(
        "--truncation-policy",
        default="retry",
        choices=["retry", "warn"],
        help="Retry token-limited rows or keep them and emit a warning.",
    )
    group.add_argument(
        "--stop-repetition-loops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop conservative 8-32-token periodic loops after 96 generated tokens.",
    )
    group = parser.add_argument_group("alignment")
    mode = group.add_mutually_exclusive_group()
    mode.add_argument(
        "--alignment",
        default="word",
        choices=["word", "segment", "none"],
        help="word uses MMS forced alignment; segment is approximate; none writes plain text.",
    )
    mode.add_argument(
        "--text-only",
        action="store_true",
        help="Alias for --alignment none.",
    )
    group.add_argument(
        "--align-batch-size",
        type=int,
        default=4,
        help="Maximum MMS emission windows per GPU batch; halves automatically on OOM.",
    )
    group.add_argument(
        "--align-dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="FP32 preserves timestamp parity; FP16 is faster but can shift timestamps slightly.",
    )

    group = parser.add_argument_group("subtitle cues")
    group.add_argument(
        "--max-chars",
        type=int,
        default=80,
        help="Maximum subtitle cue length in characters.",
    )
    group.add_argument(
        "--max-cue-dur",
        type=float,
        default=6.0,
        help="Maximum subtitle cue duration in seconds.",
    )
    group.add_argument(
        "--max-gap",
        type=float,
        default=0.6,
        help="Maximum inter-word gap inside one cue, in seconds.",
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help="Optional path for exact timings, batch history, versions, and memory telemetry.",
    )
    namespace = parser.parse_args(argv)
    return TranscriptionConfig(**vars(namespace))


def validate_args(args: TranscriptionConfig) -> None:
    text_mode = args.text_only or args.alignment == "none"
    args.formats = (
        (["txt"] if text_mode else ["txt", "srt", "vtt"])
        if args.formats is None
        else list(dict.fromkeys(args.formats))
    )
    if args.text_only:
        args.alignment = "none"
    if text_mode and args.formats != ["txt"]:
        raise SystemExit("Plain-text mode supports only --formats txt")
    if not math.isfinite(args.audio_memory_gb) or args.audio_memory_gb <= 0:
        raise SystemExit("--audio-memory-gb must be positive")
    if args.preprocess_workers is not None and args.preprocess_workers <= 0:
        raise SystemExit("--preprocess-workers must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.batch_max_size is not None and args.batch_max_size <= 0:
        raise SystemExit("--batch-max-size must be positive")
    if (
        args.batch_size is not None
        and args.batch_max_size is not None
        and args.batch_max_size < args.batch_size
    ):
        raise SystemExit("--batch-max-size must be at least --batch-size")
    if args.batch_max_size is not None and not args.adaptive_batch:
        raise SystemExit("--batch-max-size requires --adaptive-batch")
    if args.batch_audio_seconds is not None and (
        not math.isfinite(args.batch_audio_seconds) or args.batch_audio_seconds <= 0
    ):
        raise SystemExit("--batch-audio-seconds must be finite and positive")
    if (
        not math.isfinite(args.batch_vram_target)
        or not 0.50 <= args.batch_vram_target <= 0.98
    ):
        raise SystemExit("--batch-vram-target must be between 0.50 and 0.98")
    if args.align_batch_size <= 0:
        raise SystemExit("--align-batch-size must be positive")
    if (
        not math.isfinite(args.min_dur)
        or not math.isfinite(args.max_dur)
        or args.min_dur < 0
        or args.max_dur <= 0
        or args.min_dur > args.max_dur
    ):
        raise SystemExit("Require 0 <= --min-dur <= --max-dur")
    if args.vad == "none" and args.max_dur < ASR_FIXED_MIN_S:
        raise SystemExit(
            f"--vad none requires --max-dur >= {ASR_FIXED_MIN_S:g} second to bound "
            "segment count and memory use"
        )
    if not math.isfinite(args.vad_threshold) or not 0 <= args.vad_threshold <= 1:
        raise SystemExit("--vad-threshold must be between 0 and 1")
    if args.min_silence_ms < 0 or args.speech_pad_ms < 0:
        raise SystemExit("--min-silence-ms and --speech-pad-ms must be non-negative")
    if (
        not math.isfinite(args.max_silence)
        or not math.isfinite(args.energy_threshold)
        or args.max_silence < 0
        or args.energy_threshold < 0
    ):
        raise SystemExit(
            "Auditok silence and energy thresholds must be finite and non-negative"
        )
    if args.vad == "auditok" and args.min_dur <= 0:
        raise SystemExit("--vad auditok requires --min-dur > 0")
    if getattr(args, "vad_merge", False) and args.vad != "silero":
        raise SystemExit("--vad-merge is supported only with --vad silero")
    if args.vad == "auditok" and args.max_silence >= args.max_dur:
        raise SystemExit("--vad auditok requires --max-silence < --max-dur")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.max_retry_tokens < args.max_new_tokens:
        raise SystemExit("--max-retry-tokens must be at least --max-new-tokens")
    if (
        args.max_chars <= 0
        or not math.isfinite(args.max_cue_dur)
        or not math.isfinite(args.max_gap)
        or args.max_cue_dur <= 0
        or args.max_gap < 0
    ):
        raise SystemExit(
            "Subtitle cue limits must be positive (and --max-gap non-negative)"
        )


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def validate_alignment_package_source() -> None:
    try:
        distribution = importlib_metadata.distribution("ctc-forced-aligner")
        direct_url_text = distribution.read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except (importlib_metadata.PackageNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(
            "Cannot verify the official ctc-forced-aligner installation. "
            "Reinstall it with: pip install -r requirements.txt"
        ) from exc

    if not isinstance(direct_url, dict):
        raise SystemExit(
            "ctc-forced-aligner was not installed from the pinned official Git source. "
            "Reinstall it with: pip install -r requirements.txt"
        )
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict):
        raise SystemExit("ctc-forced-aligner installation is missing Git provenance")
    repository = str(direct_url.get("url", "")).rstrip("/")
    if repository != ALIGN_PACKAGE_REPOSITORY.rstrip("/"):
        raise SystemExit(
            "ctc-forced-aligner is installed from an unexpected repository: "
            f"{repository or 'unknown'}"
        )
    if (
        vcs_info.get("vcs") != "git"
        or vcs_info.get("commit_id") != ALIGN_PACKAGE_REVISION
    ):
        raise SystemExit(
            "ctc-forced-aligner is not installed at the evaluated Git revision "
            f"{ALIGN_PACKAGE_REVISION}"
        )


def release_pair(version_text: str) -> tuple[int, int] | None:
    from packaging.version import InvalidVersion, Version

    try:
        release = Version(version_text.split("+", 1)[0]).release
    except InvalidVersion:
        return None
    if len(release) < 2:
        return None
    return int(release[0]), int(release[1])


def preflight_forced_align() -> None:
    """Verify that TorchAudio's maintained CPU CTC operation is callable."""
    try:
        from torchaudio.functional import forced_align

        emissions = torch.log_softmax(
            torch.tensor([[[4.0, 0.0], [0.0, 4.0]]], dtype=torch.float32), dim=-1
        )
        targets = torch.tensor([[1]], dtype=torch.int64)
        path, scores = forced_align(emissions, targets, blank=0)
        if path.shape != (1, 2) or scores.shape != (1, 2):
            raise RuntimeError(
                f"unexpected forced_align output shapes {path.shape}, {scores.shape}"
            )
    except Exception as exc:
        raise SystemExit(
            "TorchAudio forced alignment is unavailable or incompatible with this "
            f"PyTorch build ({type(exc).__name__}: {exc}). Install matching torch "
            "and torchaudio releases."
        ) from exc


def preflight_runtime(args: TranscriptionConfig) -> None:
    """Import only dependencies required by the selected execution path."""
    required: list[tuple[str, str, str]] = [
        (
            "transformers",
            "Cohere ASR",
            f"transformers>={MIN_TRANSFORMERS_VERSION},<{MAX_TRANSFORMERS_VERSION}",
        ),
    ]
    if args.vad == "silero":
        if args.vad_engine in {"auto", "jit"}:
            required.append(("silero_vad", "Silero VAD", "silero-vad"))
        if args.vad_engine == "onnx":
            required.append(
                (
                    "transcribe_assets.vectorized_silero",
                    "vectorized ONNX Silero VAD",
                    "the local transcribe_assets package",
                )
            )
            required.append(("onnxruntime", "ONNX Silero VAD", "onnxruntime"))
    elif args.vad == "auditok":
        required.append(("auditok.core", "Auditok VAD", "auditok"))
    if args.alignment == "word":
        required.extend(
            [
                ("ctc_forced_aligner", "word alignment", "-r requirements.txt"),
                ("torchaudio", "word alignment", "torchaudio"),
                ("uroman", "word-alignment romanization", "-r requirements.txt"),
            ]
        )
    if args.audio_backend == "torchcodec":
        required.append(("torchcodec", "TorchCodec audio decoding", "torchcodec>=0.3"))
    elif args.audio_backend == "librosa":
        required.append(("librosa", "librosa audio decoding", "librosa"))

    for module_name, feature, package in required:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise SystemExit(
                f"Cannot initialize {feature}: import {module_name!r} failed ({exc}).\n"
                f"  Install a compatible build with: pip install {package}"
            ) from exc

    from packaging.version import Version

    transformers_version = package_version("transformers")
    if (
        transformers_version is None
        or Version(transformers_version) < Version(MIN_TRANSFORMERS_VERSION)
        or Version(transformers_version) >= Version(MAX_TRANSFORMERS_VERSION)
    ):
        raise SystemExit(
            "The optimized Cohere hot path is validated only with "
            f"transformers>={MIN_TRANSFORMERS_VERSION},<{MAX_TRANSFORMERS_VERSION}; "
            f"found {transformers_version or 'unknown'}"
        )

    if args.alignment == "word":
        validate_alignment_package_source()
        torch_pair = release_pair(torch.__version__)
        torchaudio_version = package_version("torchaudio")
        audio_pair = release_pair(torchaudio_version or "")
        if (
            torch_pair is not None
            and audio_pair is not None
            and torch_pair != audio_pair
        ):
            raise SystemExit(
                "PyTorch and TorchAudio must use matching major/minor releases for "
                f"forced alignment; found torch {torch.__version__} and "
                f"torchaudio {torchaudio_version}"
            )
        preflight_forced_align()
    if args.audio_backend == "ffmpeg" and not shutil.which("ffmpeg"):
        raise SystemExit(
            "--audio-backend ffmpeg requires the ffmpeg executable on PATH"
        )


# Pipeline orchestration


def attach_prepared(job: AudioJob, prepared: PreparedAudio, stats: RunStats) -> None:
    job.audio = prepared.audio
    job.duration = len(prepared.audio) / SR
    job.segment_times = prepared.segment_times
    job.speech_spans = prepared.speech_spans
    job.segment_texts = [""] * len(prepared.segment_times)
    job.decode_backend = prepared.decode_backend
    job.vad_engine_actual = prepared.vad_engine.removesuffix("+merge")
    job.vad_provider = prepared.vad_provider
    job.vad_provider_options = prepared.vad_provider_options
    job.vad_fallback_reason = prepared.vad_fallback_reason
    stats.decode_seconds += prepared.decode_seconds
    stats.vad_seconds += prepared.vad_seconds
    speech_seconds = sum(end - start for start, end in job.segment_times)
    speech_percent = 0.0 if job.duration == 0 else speech_seconds / job.duration * 100
    coverage_label = (
        "selected audio"
        if prepared.vad_engine.endswith("+merge")
        or prepared.vad_engine.startswith("none")
        else "speech"
    )
    info(
        f"prepared {job.path.name}: {fmt_dur(job.duration)}, {len(job.segment_times)} segments, "
        f"{fmt_dur(speech_seconds)} {coverage_label} ({speech_percent:.0f}%), "
        f"decode={prepared.decode_backend}, VAD={prepared.vad_engine}"
        + (f" [{prepared.vad_provider}]" if prepared.vad_provider else "")
    )


def release_job_audio(jobs: Sequence[AudioJob]) -> None:
    for job in jobs:
        job.audio = None


def process_asr_group(
    processor,
    model,
    jobs: Sequence[AudioJob],
    args: TranscriptionConfig,
    stats: RunStats,
    keep_audio: bool,
) -> float:
    try:
        return transcribe_group(processor, model, jobs, args, stats)
    finally:
        if not keep_audio:
            release_job_audio(jobs)


def estimated_decoded_bytes(job: AudioJob, memory_budget: int) -> int:
    if job.audio is not None:
        return job.audio_bytes
    if job.duration > 0:
        return int(round(job.duration * SR)) * np.dtype(np.float32).itemsize
    if job.duration_hint is not None:
        estimate = int(
            math.ceil(job.duration_hint * SR * np.dtype(np.float32).itemsize)
        )
        return estimate + SR * np.dtype(np.float32).itemsize
    # Unknown-duration inputs are isolated because compressed size cannot bound decoded size.
    return memory_budget


def partition_audio_jobs(
    jobs: Sequence[AudioJob],
    memory_budget: int,
    max_jobs: int | None = None,
) -> list[list[AudioJob]]:
    groups: list[list[AudioJob]] = []
    current: list[AudioJob] = []
    current_bytes = 0
    for job in jobs:
        job_bytes = estimated_decoded_bytes(job, memory_budget)
        if current and (
            current_bytes + job_bytes > memory_budget
            or (max_jobs is not None and len(current) >= max_jobs)
        ):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(job)
        current_bytes += job_bytes
    if current:
        groups.append(current)
    return groups


def partition_prepared_jobs(
    jobs: Sequence[AudioJob], memory_budget: int
) -> list[list[AudioJob]]:
    groups: list[list[AudioJob]] = []
    current: list[AudioJob] = []
    current_bytes = 0
    for job in jobs:
        if current and current_bytes + job.audio_bytes > memory_budget:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(job)
        current_bytes += job.audio_bytes
    if current:
        groups.append(current)
    return groups


def transcribe_all(
    jobs: list[AudioJob],
    args: TranscriptionConfig,
    device: str,
    dtype: torch.dtype,
    stats: RunStats,
) -> None:
    memory_budget = int(args.audio_memory_gb * 1024**3)
    expected_bytes = [estimated_decoded_bytes(job, memory_budget) for job in jobs]
    retain_audio = args.alignment == "word" and sum(expected_bytes) <= memory_budget
    pipeline_enabled = args.pipeline_preparation and len(jobs) > 1
    group_budget = memory_budget
    group_max_jobs = None
    if pipeline_enabled:
        group_budget = max(1, min(memory_budget // 2, PIPELINE_GROUP_MAX_BYTES))
        group_max_jobs = PIPELINE_GROUP_MAX_JOBS
    source_groups = partition_audio_jobs(jobs, group_budget, group_max_jobs)
    planned_group_bytes = [
        sum(estimated_decoded_bytes(job, memory_budget) for job in group)
        for group in source_groups
    ]
    workers = args.preprocess_workers
    if workers is None:
        workers = (
            1
            if len(jobs) == 1
            else min(2, len(jobs), max(1, (os.cpu_count() or 2) // 2))
        )
    workers = min(workers, len(jobs), max(1, os.cpu_count() or 1))

    cache_strategy = (
        "retain through alignment"
        if retain_audio
        else "bounded groups + re-decode"
        if args.alignment == "word"
        else "release after ASR"
    )
    info(
        f"audio cache: {cache_strategy} "
        f"(group target {group_budget / 1024**3:.2f} GiB); preprocess workers: {workers}; "
        f"next-group preparation: {'on' if pipeline_enabled else 'off'}"
    )

    def prepare_fn(job: AudioJob) -> PreparedAudio:
        return prepare_audio(job, args)

    processor = None
    model = None
    retained_processed_jobs: list[AudioJob] = []

    def load_model_once() -> None:
        nonlocal processor, model
        if model is not None:
            return
        started = time.perf_counter()
        processor, model = load_asr(
            device,
            dtype,
        )
        max_clip = validate_processor_single_row_window(processor, args.max_dur)
        info(f"processor single-row audio limit: {max_clip:g}s")
        stats.asr_load_seconds = time.perf_counter() - started
        info(f"ASR loaded in {fmt_dur(stats.asr_load_seconds)}")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def collect_pending(pending) -> list[AudioJob]:
        prepared_group: list[AudioJob] = []
        for job, future in pending:
            try:
                prepared = future.result()
                attach_prepared(job, prepared, stats)
                prepared_group.append(job)
                del prepared
            except Exception as exc:
                job.error = f"audio preparation failed: {exc}"
                info(f"[error] {job.path}: {job.error}")
        return prepared_group

    def enforce_actual_memory_budget() -> None:
        nonlocal retain_audio
        resident_bytes = sum(job.audio_bytes for job in jobs)
        if retain_audio and resident_bytes > memory_budget:
            retain_audio = False
            release_job_audio(retained_processed_jobs)
            retained_processed_jobs.clear()
            info("decoded audio exceeded the estimate; switching to bounded groups")

    def process_prepared_group(prepared_group: Sequence[AudioJob]) -> None:
        if not prepared_group:
            return
        assert processor is not None and model is not None
        actual_groups = partition_prepared_jobs(prepared_group, memory_budget)
        if len(actual_groups) > 1 or any(
            job.audio_bytes > memory_budget for job in prepared_group
        ):
            info(
                "decoded audio exceeded its planned group; processing smaller actual-size groups"
            )
        for actual_group in actual_groups:
            stats.asr_seconds += process_asr_group(
                processor,
                model,
                actual_group,
                args,
                stats,
                keep_audio=retain_audio,
            )
        if retain_audio:
            retained = [
                job for job in prepared_group if job.error is None and job.has_text
            ]
            retained_processed_jobs.extend(retained)
            retained_ids = {job.index for job in retained}
            release_job_audio(
                [job for job in prepared_group if job.index not in retained_ids]
            )

    def submit_group(executor, group: Sequence[AudioJob]):
        return [(job, executor.submit(prepare_fn, job)) for job in group]

    completed = False
    try:
        if not pipeline_enabled:
            for source_group in source_groups:
                group_workers = min(workers, len(source_group))
                with BoundedPrefetch(
                    source_group, prepare_fn, workers=group_workers, depth=group_workers
                ) as prefetch:
                    load_model_once()
                    prepared_group = collect_pending(prefetch)
                enforce_actual_memory_budget()
                process_prepared_group(prepared_group)
        else:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="audio-prep"
            )
            try:
                pending = submit_group(executor, source_groups[0])
                load_model_once()

                for group_index, _ in enumerate(source_groups):
                    prepared_group = collect_pending(pending)
                    enforce_actual_memory_budget()

                    next_pending = None
                    if group_index + 1 < len(source_groups):
                        resident_bytes = (
                            sum(job.audio_bytes for job in jobs)
                            if retain_audio
                            else sum(job.audio_bytes for job in prepared_group)
                        )
                        can_overlap = (
                            resident_bytes + planned_group_bytes[group_index + 1]
                            <= memory_budget
                        )
                        if can_overlap:
                            next_pending = submit_group(
                                executor,
                                source_groups[group_index + 1],
                            )

                    process_prepared_group(prepared_group)

                    if group_index + 1 < len(source_groups):
                        pending = next_pending or submit_group(
                            executor,
                            source_groups[group_index + 1],
                        )
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        completed = True
    finally:
        if device == "cuda" and torch.cuda.is_available():
            stats.peak_cuda_gib = max(
                stats.peak_cuda_gib,
                torch.cuda.max_memory_allocated() / 1024**3,
            )
            stats.peak_cuda_reserved_gib = max(
                stats.peak_cuda_reserved_gib,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
        model = None
        processor = None
        if not completed:
            release_job_audio(jobs)
        gc.collect()
        empty_device_cache(device)


def write_empty_jobs(jobs: Sequence[AudioJob]) -> None:
    for job in jobs:
        if job.error is not None or job.has_text:
            continue
        try:
            ensure_source_unchanged(job)
            atomic_write_outputs(job, [], [])
        except Exception as exc:
            job.error = f"writing empty transcript failed: {exc}"
            info(f"[error] {job.path}: {job.error}")
        finally:
            job.audio = None


def write_segment_timed_outputs(
    jobs: Sequence[AudioJob], args: TranscriptionConfig
) -> None:
    """Write approximate word timings spread within each VAD segment."""
    for job in jobs:
        if job.error is not None:
            continue
        try:
            words: list[WordTiming] = []
            pairs = zip(job.segment_times, job.segment_texts, strict=True)
            for segment_index, ((start, end), text) in enumerate(pairs):
                spans = speech_spans_within_segment(job.speech_spans, start, end)
                if spans:
                    words.extend(
                        uniform_word_timings_across_spans(
                            text,
                            spans,
                            segment_index,
                            (
                                "uniform_speech_spans"
                                if job.vad_merge
                                else "uniform_segment"
                            ),
                        )
                    )
                else:
                    words.extend(
                        uniform_word_timings(
                            text,
                            start,
                            end,
                            segment_index,
                            "uniform_segment",
                        )
                    )
            cues = build_cues(
                words,
                args.max_chars,
                args.max_cue_dur,
                args.max_gap,
                media_duration=job.duration,
            )
            atomic_write_outputs(job, cues, words)
            info(
                f"wrote {job.path.name}: {len(words)} words, {len(cues)} segment-timed cues"
            )
        except Exception as exc:
            job.error = f"segment-timed output failed: {exc}"
            info(f"[error] {job.path}: {job.error}")
        finally:
            job.audio = None


def write_text_only_outputs(jobs: Sequence[AudioJob]) -> None:
    """Write ASR segment text directly, without constructing words, cues, or timestamps."""
    for job in jobs:
        if job.error is not None:
            continue
        try:
            lines = [text.strip() for text in job.segment_texts if text.strip()]
            atomic_write_outputs(job, [], transcript_lines=lines)
            word_count = sum(len(text.split()) for text in lines)
            info(f"wrote {job.path.name}: {word_count} words, text only")
        except Exception as exc:
            job.error = f"text-only output failed: {exc}"
            info(f"[error] {job.path}: {job.error}")
        finally:
            job.audio = None


def preserve_plain_transcript(job: AudioJob) -> str | None:
    """Publish canonical ASR text when a requested timing stage fails."""
    if "txt" not in job.output_paths:
        return None
    try:
        atomic_write_outputs(job, [], only_formats={"txt"})
    except Exception as exc:
        return str(exc)
    info(f"preserved untimed transcript for {job.path.name}")
    return None


def align_and_write_all(
    jobs: list[AudioJob],
    args: TranscriptionConfig,
    device: str,
    align_dtype: torch.dtype,
    stats: RunStats,
) -> None:
    write_empty_jobs(jobs)
    alignment_jobs = [job for job in jobs if job.error is None and job.has_text]
    if not alignment_jobs:
        return

    def reload_fn(job: AudioJob) -> np.ndarray:
        return reload_audio_for_alignment(job, args)

    memory_budget = int(args.audio_memory_gb * 1024**3)
    estimates = [estimated_decoded_bytes(job, memory_budget) for job in alignment_jobs]
    tokenizer = None
    model = None
    try:
        with PairBudgetPrefetch(
            alignment_jobs,
            reload_fn,
            estimates,
            memory_budget,
        ) as prefetch:
            started = time.perf_counter()
            try:
                tokenizer, model = load_aligner(device, align_dtype)
            except Exception as exc:
                stats.align_load_seconds = time.perf_counter() - started
                for job in alignment_jobs:
                    recovery_error = preserve_plain_transcript(job)
                    job.error = f"aligner load failed: {exc}"
                    if recovery_error:
                        job.error += f"; preserving TXT also failed: {recovery_error}"
                    info(f"[error] {job.path}: {job.error}")
                return
            stats.align_load_seconds = time.perf_counter() - started
            info(f"aligner loaded in {fmt_dur(stats.align_load_seconds)}")
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            for alignment_index, (job, future) in enumerate(prefetch, start=1):
                audio = None
                emissions = None
                try:
                    audio = future.result()
                    started = time.perf_counter()
                    emissions, stride = compute_emissions_streaming(
                        audio,
                        model,
                        args.align_batch_size,
                        job.path.name,
                    )
                    stats.emissions_seconds += time.perf_counter() - started

                    started = time.perf_counter()
                    words, fallback_count = align_words(
                        emissions,
                        stride,
                        tokenizer,
                        job.segment_times,
                        job.segment_texts,
                        args.language,
                    )
                    stats.viterbi_seconds += time.perf_counter() - started
                    job.fallback_alignments = fallback_count
                    cues = build_cues(
                        words,
                        args.max_chars,
                        args.max_cue_dur,
                        args.max_gap,
                        media_duration=job.duration,
                    )
                    atomic_write_outputs(job, cues, words)
                    info(
                        f"wrote {job.path.name}: {len(words)} words, {len(cues)} cues"
                        + (
                            f", {fallback_count} approximate segments"
                            if fallback_count
                            else ""
                        )
                    )
                except Exception as exc:
                    job.error = f"alignment/output failed: {exc}"
                    recovery_error = preserve_plain_transcript(job)
                    if recovery_error:
                        job.error += f"; preserving TXT also failed: {recovery_error}"
                    info(f"[error] {job.path}: {job.error}")
                finally:
                    job.audio = None
                    emissions = None
                    audio = None
                    if alignment_index % ALIGNMENT_GC_INTERVAL == 0:
                        gc.collect()
    finally:
        if device == "cuda" and torch.cuda.is_available():
            stats.peak_cuda_gib = max(
                stats.peak_cuda_gib,
                torch.cuda.max_memory_allocated() / 1024**3,
            )
            stats.peak_cuda_reserved_gib = max(
                stats.peak_cuda_reserved_gib,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
        model = None
        tokenizer = None
        release_job_audio(alignment_jobs)
        gc.collect()
        empty_device_cache(device)


def validate_profile_output_path(
    path_text: str | None, jobs: Sequence[AudioJob]
) -> Path | None:
    if path_text is None:
        return None
    supplied_path = Path(path_text).expanduser()
    if supplied_path.is_symlink():
        raise SystemExit(f"Profile path must not be a symlink: {supplied_path}")
    path = supplied_path.resolve(strict=False)
    source_paths = {job.path.resolve() for job in jobs}
    output_paths = {
        output.resolve(strict=False)
        for job in jobs
        for output in job.output_paths.values()
    }
    if path in source_paths:
        raise SystemExit(f"Profile path collides with an input audio file: {path}")
    if path in output_paths:
        raise SystemExit(f"Profile path collides with a transcript output: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(
            f"Cannot create profile directory {path.parent}: {exc}"
        ) from exc
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise SystemExit(f"Profile path is not a regular file: {path}")
    if not os.access(path.parent, os.W_OK):
        raise SystemExit(f"Profile directory is not writable: {path.parent}")
    return path


def runtime_environment(device: str, dtype: torch.dtype) -> dict[str, Any]:
    package_names = (
        "torch",
        "torchaudio",
        "transformers",
        "numpy",
        "tqdm",
        "onnxruntime",
        "silero-vad",
        "ctc-forced-aligner",
        "uroman",
        "torchcodec",
        "librosa",
        "auditok",
    )
    environment: dict[str, Any] = {
        "python": sys.version.splitlines()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "dtype": str(dtype),
        "packages": {
            package: version
            for package in package_names
            if (version := package_version(package)) is not None
        },
        "torch_cuda_build": torch.version.cuda,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if device == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        allocator_backend = None
        get_allocator_backend = getattr(torch.cuda, "get_allocator_backend", None)
        if callable(get_allocator_backend):
            with contextlib.suppress(Exception):
                allocator_backend = get_allocator_backend()
        environment["cuda"] = {
            "device_index": index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_gib": total_bytes / 1024**3,
            "free_memory_at_profile_gib": free_bytes / 1024**3,
            "driver_visible_device_count": torch.cuda.device_count(),
            "cudnn_version": torch.backends.cudnn.version(),
            "allocator_backend": allocator_backend,
        }
    return environment


def build_profile_payload(
    args: TranscriptionConfig,
    stats: RunStats,
    jobs: Sequence[AudioJob],
    elapsed: float,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    successful = [job for job in jobs if job.error is None]
    successful_duration = sum(job.duration for job in successful)
    padded_frames = stats.asr_padded_feature_frames
    padding_ratio = (
        0.0
        if padded_frames == 0
        else 1.0 - stats.asr_valid_feature_frames / padded_frames
    )
    segment_durations = sorted(
        end - start for job in jobs for start, end in job.segment_times
    )
    duration_quantiles = (
        {
            "min": segment_durations[0],
            "p50": float(np.quantile(segment_durations, 0.50)),
            "p90": float(np.quantile(segment_durations, 0.90)),
            "p99": float(np.quantile(segment_durations, 0.99)),
            "max": segment_durations[-1],
        }
        if segment_durations
        else None
    )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "created_unix_seconds": time.time(),
        "models": {
            "asr": {"id": MODEL_ID, "revision": ASR_MODEL_REVISION},
            "aligner": (
                {
                    "id": ALIGN_MODEL_ID,
                    "revision": ALIGN_MODEL_REVISION,
                    "package_repository": ALIGN_PACKAGE_REPOSITORY,
                    "package_revision": ALIGN_PACKAGE_REVISION,
                    "romanizer": "uroman",
                }
                if args.alignment == "word"
                else None
            ),
        },
        "environment": runtime_environment(device, dtype),
        "configuration": asdict(args),
        "run": {
            "elapsed_seconds": elapsed,
            "successful_files": len(successful),
            "failed_files": len(jobs) - len(successful),
            "successful_audio_seconds": successful_duration,
            "real_time_factor_x": (
                successful_duration / elapsed if elapsed > 0 else 0.0
            ),
        },
        "timings": {
            "decode_worker_seconds": stats.decode_seconds,
            "vad_worker_seconds": stats.vad_seconds,
            "asr_load_seconds": stats.asr_load_seconds,
            "asr_wall_seconds": stats.asr_seconds,
            "asr_feature_worker_seconds": stats.asr_feature_seconds,
            "asr_feature_wait_seconds": stats.asr_feature_wait_seconds,
            "asr_h2d_seconds": stats.asr_h2d_seconds,
            "asr_generate_seconds": stats.asr_generate_seconds,
            "asr_decode_seconds": stats.asr_decode_seconds,
            "aligner_load_seconds": stats.align_load_seconds,
            "emissions_seconds": stats.emissions_seconds,
            "viterbi_seconds": stats.viterbi_seconds,
        },
        "asr": {
            "batches": stats.asr_batches,
            "processor_rows": stats.asr_processor_rows,
            "generated_tokens": stats.asr_generated_tokens,
            "valid_feature_frames": stats.asr_valid_feature_frames,
            "padded_feature_frames": stats.asr_padded_feature_frames,
            "padding_ratio": padding_ratio,
            "effective_batch_min": stats.effective_batch_min,
            "effective_batch_max": stats.effective_batch_max,
            "final_batch_size": stats.final_batch_size,
            "final_batch_cap": stats.final_batch_cap,
            "oom_retries": stats.asr_oom_retries,
            "truncation_retries": stats.asr_truncation_retries,
            "discarded_feature_batches": stats.asr_discarded_feature_batches,
            "pin_memory_fallbacks": stats.pin_memory_fallbacks,
            "segment_duration_seconds": duration_quantiles,
            "batch_history": stats.batch_history,
        },
        "cuda_memory": {
            "total_gib": stats.cuda_total_gib,
            "free_start_gib": stats.cuda_free_start_gib,
            "free_end_gib": stats.cuda_free_end_gib,
            "peak_allocated_gib": stats.peak_cuda_gib,
            "peak_reserved_gib": stats.peak_cuda_reserved_gib,
        },
        "files": [
            {
                "path": os.fspath(job.path),
                "relative_path": os.fspath(job.relative_path),
                "duration_seconds": job.duration,
                "segment_count": len(job.segment_times),
                "raw_speech_span_count": len(job.speech_spans),
                "raw_speech_seconds": sum(
                    end - start for start, end in job.speech_spans
                ),
                "selected_audio_seconds": sum(
                    end - start for start, end in job.segment_times
                ),
                "decode_backend": job.decode_backend,
                "vad_engine": job.vad_engine_actual,
                "vad_provider": job.vad_provider,
                "vad_provider_options": job.vad_provider_options,
                "vad_fallback_reason": job.vad_fallback_reason,
                "generated_tokens": sum(job.generated_tokens.values()),
                "repetition_stopped_segments": sorted(job.repetition_stopped_segments),
                "truncation_retried_segments": sorted(job.truncation_retried_segments),
                "token_limit_segments": sorted(job.token_limit_segments),
                "fallback_alignment_segments": job.fallback_alignments,
                "outputs": [os.fspath(path) for path in job.written],
                "error": job.error,
            }
            for job in jobs
        ],
    }


def write_profile_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        output_mode = (
            stat.S_IMODE(path.stat().st_mode) if path.exists() else DEFAULT_OUTPUT_MODE
        )
        os.fchmod(descriptor, output_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directories(iter((path.parent,)))
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    assert args.formats is not None
    started = time.perf_counter()

    print("\n[1/4] Validating inputs and outputs", flush=True)
    jobs = build_jobs(args)
    if not jobs:
        print(f"{INDENT}All inputs were skipped; no model was loaded.", flush=True)
        return 0
    profile_path = validate_profile_output_path(args.profile_json, jobs)
    preflight_runtime(args)

    device = pick_device(args.device)
    # Store the resolved device so worker-side pinning does not treat --device auto
    # as a CPU path after CUDA has already been selected.
    args.device = device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    if device == "cpu":
        if args.dtype != "fp32":
            info(
                f"CPU inference uses FP32; ignoring requested {args.dtype.upper()} precision"
            )
        dtype = torch.float32
    if (
        device == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise SystemExit("This CUDA device does not support BF16; use --dtype fp16")
    if device == "mps" and dtype == torch.bfloat16:
        try:
            probe = torch.zeros(1, device="mps", dtype=torch.bfloat16)
            del probe
        except (RuntimeError, TypeError) as exc:
            raise SystemExit(
                "This MPS device/runtime does not support BF16; use --dtype fp16"
            ) from exc
    if args.alignment == "word" and args.align_dtype == "fp16" and device != "cuda":
        raise SystemExit("--align-dtype fp16 is supported only with CUDA")
    align_dtype = torch.float16 if args.align_dtype == "fp16" else torch.float32

    stats = RunStats()
    if device == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        stats.cuda_total_gib = total_bytes / 1024**3
        stats.cuda_free_start_gib = free_bytes / 1024**3

    default_initial_batch = 24 if device == "cuda" else 8
    initial_batch = args.batch_size or (
        min(default_initial_batch, args.batch_max_size)
        if args.batch_max_size is not None
        else default_initial_batch
    )
    if not args.adaptive_batch:
        batch_label = f"{initial_batch} fixed"
    elif args.batch_size is not None and args.batch_max_size is None:
        batch_label = f"{initial_batch} cap + OOM learning"
    else:
        cap_label = (
            str(args.batch_max_size)
            if args.batch_max_size is not None
            else "VRAM-derived"
        )
        batch_label = f"adaptive {initial_batch}→{cap_label}"

    total_hint = sum(job.duration_hint or 0.0 for job in jobs)
    vad_label = {
        "silero": (f"Silero {args.vad_engine}{' + merge' if args.vad_merge else ''}"),
        "auditok": "auditok",
        "none": f"no VAD ({args.max_dur:g}s fixed windows)",
    }[args.vad]
    print(
        f"{INDENT}{len(jobs)} file(s), {fmt_dur(total_hint)} probed audio | "
        f"{device} / {dtype} | ASR batch {batch_label} (length sorted) | "
        f"{vad_label} | {'text only' if args.alignment == 'none' else args.alignment + ' timing'}",
        flush=True,
    )

    print("\n[2/4] Loading ASR + preparing audio", flush=True)
    transcribe_all(jobs, args, device, dtype, stats)
    prepared_duration = sum(job.duration for job in jobs if job.duration > 0)
    segment_count = sum(len(job.segment_times) for job in jobs if job.error is None)
    print(
        f"{INDENT}ASR done: {segment_count} segments in {fmt_dur(stats.asr_seconds)} "
        f"(decode {fmt_dur(stats.decode_seconds)}, VAD {fmt_dur(stats.vad_seconds)} worker-time)",
        flush=True,
    )
    if args.alignment == "word":
        print("\n[3/4] Forced alignment + transactional outputs", flush=True)
        align_and_write_all(jobs, args, device, align_dtype, stats)
    elif args.alignment == "segment":
        print("\n[3/4] Segment-timed transactional outputs", flush=True)
        write_segment_timed_outputs(jobs, args)
    else:
        print("\n[3/4] Text-only transactional outputs", flush=True)
        write_text_only_outputs(jobs)

    if device == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        stats.cuda_free_end_gib = free_bytes / 1024**3

    print("\n[4/4] Summary", flush=True)
    elapsed = time.perf_counter() - started
    successful = [job for job in jobs if job.error is None]
    failures = [job for job in jobs if job.error is not None]
    successful_duration = sum(job.duration for job in successful)
    fallback_count = sum(job.fallback_alignments for job in successful)
    repetition_stop_count = sum(
        len(job.repetition_stopped_segments) for job in successful
    )
    token_limit_count = sum(len(job.token_limit_segments) for job in successful)
    rtfx = successful_duration / elapsed if elapsed > 0 else 0.0
    memory_label = (
        f", CUDA peak {stats.peak_cuda_gib:.2f} GiB allocated / "
        f"{stats.peak_cuda_reserved_gib:.2f} GiB reserved"
        if device == "cuda"
        else ""
    )
    print(
        f"{INDENT}{len(successful)}/{len(jobs)} files finished in {fmt_dur(elapsed)} "
        f"(RTFx {rtfx:.1f}{memory_label})",
        flush=True,
    )
    print(
        f"{INDENT}ASR load {fmt_dur(stats.asr_load_seconds)} | "
        f"ASR wall {fmt_dur(stats.asr_seconds)} | "
        f"feature worker {stats.asr_feature_seconds:.3f}s | "
        f"feature wait {stats.asr_feature_wait_seconds:.3f}s | "
        f"H2D {stats.asr_h2d_seconds:.3f}s | "
        f"generate {stats.asr_generate_seconds:.3f}s | "
        f"decode text {stats.asr_decode_seconds:.3f}s",
        flush=True,
    )
    print(
        f"{INDENT}aligner load {fmt_dur(stats.align_load_seconds)} | "
        f"emissions {fmt_dur(stats.emissions_seconds)} | "
        f"Viterbi {fmt_dur(stats.viterbi_seconds)}",
        flush=True,
    )
    padding_ratio = (
        0.0
        if stats.asr_padded_feature_frames == 0
        else 1.0 - stats.asr_valid_feature_frames / stats.asr_padded_feature_frames
    )
    if stats.asr_batches:
        print(
            f"{INDENT}{stats.asr_batches} generation batch(es), "
            f"{stats.asr_processor_rows} processor row(s), "
            f"batch range {stats.effective_batch_min}-{stats.effective_batch_max}, "
            f"feature padding {padding_ratio:.1%}, "
            f"{stats.asr_generated_tokens} generated token(s)",
            flush=True,
        )
    if stats.asr_oom_retries or stats.asr_truncation_retries:
        print(
            f"{INDENT}adaptive retries: {stats.asr_oom_retries} OOM, "
            f"{stats.asr_truncation_retries} token-limit segment(s)",
            flush=True,
        )
    if fallback_count:
        print(
            f"{INDENT}warning: {fallback_count} segment(s) used approximate word timing",
            flush=True,
        )
    if repetition_stop_count:
        triggered = [
            f"{job.relative_path}:{segment_index}"
            for job in successful
            for segment_index in sorted(job.repetition_stopped_segments)
        ]
        provenance_hint = (
            "see JSON provenance for segment indices"
            if "json" in args.formats
            else "segments "
            + ", ".join(triggered[:10])
            + (f" (+{len(triggered) - 10} more)" if len(triggered) > 10 else "")
        )
        print(
            f"{INDENT}warning: repetition guard stopped {repetition_stop_count} segment(s); "
            f"{provenance_hint}",
            flush=True,
        )
    if token_limit_count:
        limited = [
            f"{job.relative_path}:{segment_index}"
            for job in successful
            for segment_index in sorted(job.token_limit_segments)
        ]
        provenance_hint = (
            "see JSON provenance for segment indices"
            if "json" in args.formats
            else "segments "
            + ", ".join(limited[:10])
            + (f" (+{len(limited) - 10} more)" if len(limited) > 10 else "")
        )
        print(
            f"{INDENT}warning: {token_limit_count} segment(s) reached the final "
            f"decoder token limit without EOS; {provenance_hint}",
            flush=True,
        )

    profile_error: str | None = None
    if profile_path is not None:
        try:
            payload = build_profile_payload(args, stats, jobs, elapsed, device, dtype)
            write_profile_json(profile_path, payload)
            print(f"{INDENT}{profile_path}", flush=True)
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"
            print(
                f"{INDENT}[error] writing performance profile failed: {profile_error}",
                flush=True,
            )

    written_paths = [path for job in jobs for path in job.written]
    for path in written_paths[:OUTPUT_PATH_DISPLAY_LIMIT]:
        print(f"{INDENT}{path}", flush=True)
    if len(written_paths) > OUTPUT_PATH_DISPLAY_LIMIT:
        print(
            f"{INDENT}... and {len(written_paths) - OUTPUT_PATH_DISPLAY_LIMIT} more output files",
            flush=True,
        )
    if failures and prepared_duration > successful_duration:
        attempted_rtfx = prepared_duration / elapsed if elapsed > 0 else 0.0
        print(
            f"{INDENT}attempted {fmt_dur(prepared_duration)} of audio "
            f"(RTFx {attempted_rtfx:.1f})",
            flush=True,
        )
    if failures:
        print(f"{INDENT}Failures:", flush=True)
        for job in failures:
            print(f"{INDENT}- {job.path}: {job.error}", flush=True)
    return 1 if failures or profile_error is not None else 0


def cli() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print(f"\n{INDENT}Interrupted; temporary outputs were rolled back.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
