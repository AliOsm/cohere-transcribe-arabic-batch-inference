"""Stable fingerprints for resumable ASR and published render contracts."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import Any

from .._version import __version__
from ..models import (
    ALIGN_MODEL_REVISION,
    ASR_MODEL_REVISION,
    OUTPUT_SCHEMA_VERSION,
    SILERO_VERSION,
    TRANSFORMERS_VERSION,
    TranscriptionConfig,
    file_sha256,
)
from .io import canonical_json

CONTRACT_SCHEMA_VERSION = 2

_ASR_IMPLEMENTATION_FILES = (
    "asr/batching.py",
    "asr/execution.py",
    "asr/generation.py",
    "asr/model.py",
    "asr/orchestration.py",
    "audio/backends.py",
    "audio/decoding.py",
    "audio/preparation.py",
    "audio/segmentation.py",
    "models.py",
    "pipeline/resources.py",
    "pipeline/transcription.py",
    "vad/runtime.py",
    "vad/silero_vad.jit",
    "vad/silero_vad_v6.onnx",
    "vad/torch_silero.py",
    "vad/vectorized_silero.py",
)

_RENDER_IMPLEMENTATION_FILES = (
    "alignment/alignment_utils.py",
    "alignment/norm_config.py",
    "alignment/punctuations.lst",
    "alignment/runtime.py",
    "alignment/text_utils.py",
    "audio/backends.py",
    "audio/decoding.py",
    "models.py",
    "output/pipeline.py",
    "output/publication.py",
    "output/rendering.py",
    "pipeline/resources.py",
)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


@cache
def _implementation_fingerprint(files: tuple[str, ...]) -> str:
    root = Path(__file__).resolve().parents[1]
    missing = [
        relative_path for relative_path in files if not (root / relative_path).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Implementation fingerprint references missing package artifacts: "
            + ", ".join(missing)
        )
    return _fingerprint(
        {
            "package_version": __version__,
            "artifacts_sha256": {
                relative_path: file_sha256(root / relative_path)
                for relative_path in files
            },
        }
    )


def asr_contract_key(args: TranscriptionConfig) -> str:
    """Identify settings that can affect ASR segment text or boundaries."""
    configuration = {
        "language": args.language,
        "device": args.device,
        "dtype": args.dtype,
        "audio_backend": args.audio_backend,
        "vad": args.vad,
        "vad_engine": args.vad_engine if args.vad == "silero" else None,
        "vad_batch_size": args.vad_batch_size if args.vad == "silero" else None,
        "vad_block_frames": args.vad_block_frames if args.vad == "silero" else None,
        "vad_threads": args.vad_threads if args.vad == "silero" else None,
        "vad_merge": args.vad_merge if args.vad == "silero" else None,
        "min_dur": args.min_dur if args.vad != "none" else None,
        "max_dur": args.max_dur,
        "max_silence": args.max_silence if args.vad == "auditok" else None,
        "energy_threshold": (args.energy_threshold if args.vad == "auditok" else None),
        "vad_threshold": args.vad_threshold if args.vad == "silero" else None,
        "min_silence_ms": args.min_silence_ms if args.vad == "silero" else None,
        "speech_pad_ms": args.speech_pad_ms if args.vad == "silero" else None,
        "batch_size": args.batch_size,
        "batch_max_size": args.batch_max_size,
        "batch_audio_seconds": args.batch_audio_seconds,
        "batch_vram_target": args.batch_vram_target,
        "adaptive_batch": args.adaptive_batch,
        "pin_memory": args.pin_memory,
        "audio_memory_gb": args.audio_memory_gb,
        "pipeline_preparation": args.pipeline_preparation,
        "max_new_tokens": args.max_new_tokens,
        "max_retry_tokens": args.max_retry_tokens,
        "truncation_policy": args.truncation_policy,
        "stop_repetition_loops": args.stop_repetition_loops,
    }
    return _fingerprint(
        {
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "configuration": configuration,
            "models": {
                "asr_revision": ASR_MODEL_REVISION,
                "silero_version": SILERO_VERSION,
                "transformers_version": TRANSFORMERS_VERSION,
            },
            "implementation_sha256": _implementation_fingerprint(
                _ASR_IMPLEMENTATION_FILES
            ),
        }
    )


def render_contract_key(args: TranscriptionConfig) -> str:
    """Identify settings that affect artifacts rendered from completed ASR."""
    configuration = {
        "alignment": args.alignment,
        "align_batch_size": args.align_batch_size,
        "align_dtype": args.align_dtype,
        "formats": sorted(args.formats or ()),
        "max_chars": args.max_chars,
        "max_cue_dur": args.max_cue_dur,
        "max_gap": args.max_gap,
    }
    return _fingerprint(
        {
            "contract_schema_version": CONTRACT_SCHEMA_VERSION,
            "configuration": configuration,
            "models": {
                "align_revision": (
                    ALIGN_MODEL_REVISION if args.alignment == "word" else None
                ),
            },
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "implementation_sha256": _implementation_fingerprint(
                _RENDER_IMPLEMENTATION_FILES
            ),
        }
    )
