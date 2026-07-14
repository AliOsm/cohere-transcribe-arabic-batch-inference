"""Command-line argument parsing and semantic validation."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from dataclasses import fields
from numbers import Real

from ._version import __version__
from .api.types import PublicationOptions, TranscriptionOptions
from .models import (
    ASR_FIXED_MIN_S,
    DEFAULT_TORCH_VAD_BATCH_SIZE,
    DEFAULT_TORCH_VAD_BLOCK_FRAMES,
    MAX_TORCH_VAD_PADDED_FRAMES,
    TranscriptionConfig,
)

OPTION_CHOICES: dict[str, frozenset[str]] = {
    "language": frozenset(("ar", "en")),
    "existing": frozenset(("error", "overwrite", "skip")),
    "device": frozenset(("auto", "mps", "cuda", "cpu")),
    "dtype": frozenset(("auto", "bf16", "fp16", "fp32")),
    "audio_backend": frozenset(("auto", "torchcodec", "ffmpeg", "librosa")),
    "vad": frozenset(("silero", "auditok", "none")),
    "vad_engine": frozenset(("auto", "torch", "onnx", "jit")),
    "truncation_policy": frozenset(("retry", "warn")),
    "alignment": frozenset(("word", "segment", "none")),
    "align_dtype": frozenset(("fp32", "fp16")),
}
OUTPUT_FORMATS = frozenset(("txt", "srt", "vtt", "json"))
INTEGER_OPTIONS = (
    "preprocess_workers",
    "vad_batch_size",
    "vad_block_frames",
    "vad_threads",
    "min_silence_ms",
    "speech_pad_ms",
    "batch_size",
    "batch_max_size",
    "max_new_tokens",
    "max_retry_tokens",
    "align_batch_size",
    "max_chars",
)
REAL_OPTIONS = (
    "audio_memory_gb",
    "min_dur",
    "max_dur",
    "max_silence",
    "energy_threshold",
    "vad_threshold",
    "batch_audio_seconds",
    "batch_vram_target",
    "max_cue_dur",
    "max_gap",
)


def config_from_options(
    audio: Sequence[str], options: TranscriptionOptions
) -> TranscriptionConfig:
    """Create the mutable internal configuration used by the execution engine."""
    if not isinstance(options, TranscriptionOptions):
        raise TypeError("options must be a TranscriptionOptions instance")
    option_names = {
        item.name for item in fields(TranscriptionOptions) if item.name != "publication"
    }
    values = {name: getattr(options, name) for name in option_names}
    publication = options.publication
    return TranscriptionConfig(
        audio=list(audio),
        formats=(
            list(publication.formats)
            if publication is not None and publication.formats is not None
            else None
        ),
        output_dir=(
            os.fspath(publication.output_dir)
            if publication is not None and publication.output_dir is not None
            else None
        ),
        existing=publication.existing if publication is not None else "error",
        profile_json=(
            os.fspath(publication.profile_json)
            if publication is not None and publication.profile_json is not None
            else None
        ),
        **values,
    )


def options_from_config(
    args: TranscriptionConfig, *, publication_enabled: bool
) -> TranscriptionOptions:
    """Snapshot a mutable runtime configuration as immutable public options."""
    option_names = {
        item.name for item in fields(TranscriptionOptions) if item.name != "publication"
    }
    values = {name: getattr(args, name) for name in option_names}
    publication = None
    if publication_enabled:
        publication = PublicationOptions(
            formats=tuple(args.formats) if args.formats is not None else None,
            output_dir=args.output_dir,
            existing=args.existing,
            profile_json=args.profile_json,
        )
    return TranscriptionOptions(publication=publication, **values)


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
        "--version",
        action="version",
        version=f"cohere-transcribe {__version__}",
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
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="ASR model precision; auto uses BF16 or FP16 on accelerators and FP32 on CPU.",
    )
    parser.add_argument(
        "--audio-backend",
        default="auto",
        choices=["librosa", "torchcodec", "auto", "ffmpeg"],
        help="auto prefers working TorchCodec and retries per-file failures with FFmpeg; Librosa is explicit-only.",
    )
    parser.add_argument(
        "--audio-memory-gb",
        type=float,
        default=4.0,
        help=(
            "Hard decoded-PCM limit per file and target for each prepared group; "
            "decoder implementation transients may add overhead."
        ),
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help=(
            "Concurrent audio decode workers; non-packed engines also run one VAD "
            "instance per worker. Auto uses 1 for one file and at most 2 otherwise."
        ),
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
        choices=["auto", "torch", "onnx", "jit"],
        help=(
            "Silero runtime; torch enables packed independent-file CPU batches, "
            "auto selects packed Torch, then falls back to sequence ONNX or "
            "the packaged TorchScript model."
        ),
    )
    group.add_argument(
        "--vad-batch-size",
        type=int,
        default=DEFAULT_TORCH_VAD_BATCH_SIZE,
        help="Maximum independent files per packed Torch VAD model call.",
    )
    group.add_argument(
        "--vad-block-frames",
        type=int,
        default=DEFAULT_TORCH_VAD_BLOCK_FRAMES,
        help="Maximum 32 ms frames per file in one packed Torch VAD model call.",
    )
    group.add_argument(
        "--vad-threads",
        type=int,
        default=None,
        help=(
            "Process-wide PyTorch CPU thread count; auto preserves the platform "
            "default. This also affects ASR CPU feature preparation."
        ),
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
            "Initial ASR segment count; auto starts at 24 on CUDA, 8 on MPS, "
            "and 4 on CPU."
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
        default="segment",
        choices=["word", "segment", "none"],
        help=(
            "Timestamp mode: segment is the fast default, word uses optional MMS "
            "forced alignment, and none writes plain text."
        ),
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
        help="Target subtitle cue length; a single indivisible word may exceed it.",
    )
    group.add_argument(
        "--max-cue-dur",
        type=float,
        default=6.0,
        help="Target cue duration; one word interval is never split.",
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
    boolean_options = (
        "text_only",
        "recursive",
        "pipeline_preparation",
        "vad_merge",
        "adaptive_batch",
        "pin_memory",
        "stop_repetition_loops",
    )
    for option in boolean_options:
        if hasattr(args, option) and not isinstance(getattr(args, option), bool):
            raise SystemExit(f"--{option.replace('_', '-')} must be a boolean")
    for option in INTEGER_OPTIONS:
        if not hasattr(args, option):
            continue
        value = getattr(args, option)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise SystemExit(f"--{option.replace('_', '-')} must be an integer")
    for option in REAL_OPTIONS:
        if not hasattr(args, option):
            continue
        value = getattr(args, option)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real)
        ):
            raise SystemExit(f"--{option.replace('_', '-')} must be a real number")
    for option, choices in OPTION_CHOICES.items():
        if not hasattr(args, option):
            continue
        value = getattr(args, option)
        if value not in choices:
            rendered = ", ".join(sorted(choices))
            raise SystemExit(f"--{option.replace('_', '-')} must be one of: {rendered}")
    if args.formats is not None:
        if not args.formats:
            raise SystemExit("--formats requires at least one output format")
        unsupported_formats = set(args.formats).difference(OUTPUT_FORMATS)
        if unsupported_formats:
            raise SystemExit(
                "Unsupported --formats value(s): "
                + ", ".join(sorted(unsupported_formats))
            )
    if args.text_only and args.alignment == "word":
        raise SystemExit("--text-only conflicts with --alignment word")
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
    packed_vad = args.vad == "silero" and args.vad_engine in {"auto", "torch"}
    if packed_vad:
        if args.vad_batch_size <= 0 or args.vad_block_frames <= 0:
            raise SystemExit("--vad-batch-size and --vad-block-frames must be positive")
        if args.vad_batch_size * args.vad_block_frames > MAX_TORCH_VAD_PADDED_FRAMES:
            raise SystemExit(
                "--vad-batch-size * --vad-block-frames must not exceed "
                f"{MAX_TORCH_VAD_PADDED_FRAMES:,} frames"
            )
    if args.vad_threads is not None and args.vad_threads <= 0:
        raise SystemExit("--vad-threads must be positive")
    if args.vad_threads is not None and (
        args.vad != "silero" or args.vad_engine not in {"auto", "torch"}
    ):
        raise SystemExit("--vad-threads applies only to packed Torch Silero VAD")
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
    if args.alignment == "word" and args.align_batch_size <= 0:
        raise SystemExit("--align-batch-size must be positive")
    if not math.isfinite(args.max_dur) or args.max_dur <= 0:
        raise SystemExit("--max-dur must be finite and positive")
    if not math.isfinite(args.min_dur):
        raise SystemExit("--min-dur must be finite")
    if args.vad != "none" and (args.min_dur < 0 or args.min_dur > args.max_dur):
        raise SystemExit("Require 0 <= --min-dur <= --max-dur")
    if args.vad == "none" and args.max_dur < ASR_FIXED_MIN_S:
        raise SystemExit(
            f"--vad none requires --max-dur >= {ASR_FIXED_MIN_S:g} second to bound "
            "segment count and memory use"
        )
    if args.vad == "silero":
        if not 0 <= args.vad_threshold <= 1:
            raise SystemExit("--vad-threshold must be between 0 and 1")
        if args.min_silence_ms < 0 or args.speech_pad_ms < 0:
            raise SystemExit(
                "--min-silence-ms and --speech-pad-ms must be non-negative"
            )
    elif args.vad == "auditok":
        if args.max_silence < 0 or args.energy_threshold < 0:
            raise SystemExit(
                "Auditok silence and energy thresholds must be finite and non-negative"
            )
        if args.min_dur <= 0:
            raise SystemExit("--vad auditok requires --min-dur > 0")
    if args.vad_merge and args.vad != "silero":
        raise SystemExit("--vad-merge is supported only with --vad silero")
    if args.vad == "auditok" and args.max_silence >= args.max_dur:
        raise SystemExit("--vad auditok requires --max-silence < --max-dur")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.max_retry_tokens < args.max_new_tokens:
        raise SystemExit("--max-retry-tokens must be at least --max-new-tokens")
    if not all(
        math.isfinite(value)
        for value in (
            args.vad_threshold,
            args.max_silence,
            args.energy_threshold,
            args.max_cue_dur,
            args.max_gap,
        )
    ):
        raise SystemExit("All numeric thresholds and cue limits must be finite")
    if args.alignment != "none" and (
        args.max_chars <= 0
        or not math.isfinite(args.max_cue_dur)
        or not math.isfinite(args.max_gap)
        or args.max_cue_dur <= 0
        or args.max_gap < 0
    ):
        raise SystemExit(
            "Subtitle cue limits must be positive (and --max-gap non-negative)"
        )
