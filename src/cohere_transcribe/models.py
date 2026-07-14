"""Shared constants, domain models, console helpers, and implementation provenance."""

from __future__ import annotations

import functools
import hashlib
import os
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ._version import __version__
from .progress import write as progress_write

MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"
MODEL_PAGE_URL = f"https://huggingface.co/{MODEL_ID}"
ALIGN_MODEL_ID = "MahmoudAshraf/mms-300m-1130-forced-aligner"
# These are the exact revisions used by the accuracy and throughput evaluation.
ASR_MODEL_REVISION = "0a8193caa4f3f92131471ab08824e488141cb392"
ALIGN_MODEL_REVISION = "49402e9577b1158620820667c218cd494cc44486"
ALIGN_PACKAGE_REPOSITORY = "https://github.com/MahmoudAshraf97/ctc-forced-aligner.git"
ALIGN_PACKAGE_REVISION = "11855d1de76af2b490dd2e8e2db2661805ae90a0"
OUTPUT_SCHEMA_VERSION = 7
PROFILE_SCHEMA_VERSION = 8
SR = 16_000
ALIGN_WINDOW_S = 30
ALIGN_CONTEXT_S = 2
# The exact one-row limit is read from the pinned processor at runtime. The
# current Cohere feature extractor uses 35 s chunks and starts its quiet-boundary
# path above 30 s, but a 30-35 s clip still remains one processor row.
ASR_FIXED_MIN_S = 1.0
# The projection/mask hot-path patches below depend on Transformers internals.
# Admit only the exact release used by the parity and throughput validation.
TRANSFORMERS_VERSION = "5.13.1"
SILERO_VERSION = "6.2.1"
UROMAN_VERSION = "1.3.1.1"
PIPELINE_GROUP_MAX_BYTES = 512 * 1024**2
PIPELINE_GROUP_MAX_JOBS = 128
DEFAULT_TORCH_VAD_BATCH_SIZE = 16
DEFAULT_TORCH_VAD_BLOCK_FRAMES = 512
MAX_TORCH_VAD_PADDED_FRAMES = 32_768
ALIGNMENT_GC_INTERVAL = 64
FFMPEG_DECODE_TIMEOUT_S = 3_600
OUTPUT_PATH_DISPLAY_LIMIT = 20


def is_model_access_error(error: BaseException) -> bool:
    """Recognize gated-model authorization failures through wrapper exceptions."""
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GatedRepoError):
            return True
        if isinstance(current, HfHubHTTPError):
            response = getattr(current, "response", None)
            if getattr(response, "status_code", None) in {401, 403}:
                return True
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "access to model is restricted",
                "access to this resource is disabled",
                "cannot access gated repo",
                "gated repo",
                "not in the authorized list",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def model_access_message(error: BaseException | None = None) -> str:
    """Return actionable instructions for the gated Cohere model."""
    message = (
        f"Cannot access the gated Cohere ASR model {MODEL_ID}. Accept the model "
        f"terms at {MODEL_PAGE_URL}, then authenticate with `hf auth login` or set "
        "HF_TOKEN for the same Hugging Face account."
    )
    if error is not None:
        detail = str(error).splitlines()[0].strip()
        if detail:
            message += f" ({type(error).__name__}: {detail})"
    return message


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
    progress_write(f"{INDENT}{message}")


@functools.cache
def default_output_mode() -> int:
    """Return the process-umask-adjusted mode used for newly published files."""
    process_umask = os.umask(0)
    os.umask(process_umask)
    return 0o666 & ~process_umask


def package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def release_pair(version_text: str) -> tuple[int, int] | None:
    from packaging.version import InvalidVersion, Version

    try:
        release = Version(version_text.split("+", 1)[0]).release
    except InvalidVersion:
        return None
    if len(release) < 2:
        return None
    return int(release[0]), int(release[1])


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@functools.cache
def runtime_implementation() -> dict[str, Any]:
    """Identify the exact installed code and model asset that produced an output."""
    root = Path(__file__).resolve().parent
    behavior_suffixes = {".jit", ".lst", ".onnx", ".py"}
    artifacts = {
        path.relative_to(root.parent).as_posix(): path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in behavior_suffixes
    }
    return {
        "package_version": __version__,
        "artifacts_sha256": {
            name: file_sha256(path) for name, path in artifacts.items()
        },
    }


# Domain and configuration


@dataclass(frozen=True)
class SourceSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def capture(cls, path: Path) -> SourceSnapshot:
        stat = path.stat()
        return cls(
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )


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
    state_path: Path | None = None
    checkpoint_path: Path | None = None
    asr_contract_key: str = ""
    render_contract_key: str = ""
    generation_id: str = ""
    output_lock: object | None = None
    asr_checkpoint_loaded: bool = False
    skipped: bool = False
    published: bool = False
    capture_result: bool = False
    result_completed: bool = False
    result_payload: dict[str, Any] | None = None
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
    decode_fallback_reason: str | None = None
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
    decode_fallback_reason: str | None = None
    vad_provider: str | None = None
    vad_provider_options: dict[str, dict[str, str]] | None = None
    vad_fallback_reason: str | None = None


@dataclass(slots=True)
class DecodedAudio:
    audio: np.ndarray
    decode_backend: str
    decode_seconds: float
    decode_fallback_reason: str | None = None


@dataclass(slots=True)
class VadBatchMetrics:
    model_load_seconds: float = 0.0
    inference_seconds: float = 0.0
    postprocess_seconds: float = 0.0
    prepared_groups: int = 0
    model_calls: int = 0
    valid_frames: int = 0
    padded_frames: int = 0
    max_files_per_call: int = 0
    effective_block_frames: int = 0


@dataclass(slots=True)
class PreparedJobResult:
    job: AudioJob
    prepared: PreparedAudio | None = None
    error: BaseException | None = None


@dataclass(slots=True)
class PreparedGroup:
    results: list[PreparedJobResult]
    vad_metrics: VadBatchMetrics = field(default_factory=VadBatchMetrics)


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
    runtime_import_seconds: float = 0.0
    serialization_wait_seconds: float = 0.0
    input_validation_seconds: float = 0.0
    decode_seconds: float = 0.0
    vad_seconds: float = 0.0
    vad_model_load_seconds: float = 0.0
    vad_inference_seconds: float = 0.0
    vad_postprocess_seconds: float = 0.0
    vad_prepared_groups: int = 0
    vad_model_calls: int = 0
    vad_valid_frames: int = 0
    vad_padded_frames: int = 0
    vad_max_files_per_call: int = 0
    vad_effective_block_frames: int = 0
    preparation_wait_seconds: float = 0.0
    asr_load_seconds: float = 0.0
    asr_seconds: float = 0.0
    asr_feature_seconds: float = 0.0
    asr_feature_wait_seconds: float = 0.0
    asr_h2d_seconds: float = 0.0
    asr_generation_call_seconds: float = 0.0
    asr_generate_device_seconds: float = 0.0
    asr_generation_analysis_seconds: float = 0.0
    asr_decode_seconds: float = 0.0
    align_load_seconds: float = 0.0
    emissions_seconds: float = 0.0
    viterbi_seconds: float = 0.0
    post_asr_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    progressive_output_seconds: float = 0.0
    asr_checkpoint_resumed_files: int = 0
    asr_checkpoint_written_files: int = 0
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
    asr_discarded_feature_seconds: float = 0.0
    asr_discarded_processor_rows: int = 0
    asr_discarded_valid_feature_frames: int = 0
    asr_discarded_padded_feature_frames: int = 0
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
    vad_batch_size: int
    vad_block_frames: int
    vad_threads: int | None
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
