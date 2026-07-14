"""Adaptive ASR batch sizing and batch-level performance telemetry."""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..models import SR, RunStats, SegmentRef, TranscriptionConfig
from .generation import ASRGenerationResult, PreparedASRBatch


def default_asr_batch_size(device_type: str) -> int:
    if device_type == "cuda":
        return 24
    if device_type == "mps":
        return 8
    return 4


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
    growth_cooldown: int = 0
    circuit_breaker_error: str | None = None

    @classmethod
    def create(
        cls,
        args: TranscriptionConfig,
        model,
        refs: Sequence[SegmentRef],
    ) -> ASRBatchController:
        device_type = model.device.type
        default_initial = default_asr_batch_size(device_type)
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


def runtime_failure_fingerprint(exc: Exception) -> str:
    """Normalize volatile details in a fatal backend error for reporting."""
    message = (" ".join(str(exc).split()) or "<no message>")[:500]
    message = re.sub(r"0x[0-9a-fA-F]+", "0x*", message)
    message = re.sub(r"\b\d+\b", "#", message)
    return f"{type(exc).__name__}: {message}"


def record_prepared_batch(
    stats: RunStats, prepared: PreparedASRBatch, *, discarded: bool = False
) -> None:
    stats.asr_feature_seconds += prepared.prepare_seconds
    stats.pin_memory_fallbacks += prepared.pin_memory_fallbacks
    if discarded:
        stats.asr_discarded_feature_batches += 1
        stats.asr_discarded_feature_seconds += prepared.prepare_seconds
        stats.asr_discarded_processor_rows += len(prepared.chunk_index)
        stats.asr_discarded_valid_feature_frames += prepared.valid_feature_frames
        stats.asr_discarded_padded_feature_frames += prepared.padded_feature_frames
        return
    stats.asr_processor_rows += len(prepared.chunk_index)
    stats.asr_valid_feature_frames += prepared.valid_feature_frames
    stats.asr_padded_feature_frames += prepared.padded_feature_frames


def record_generation_batch(
    stats: RunStats,
    prepared: PreparedASRBatch,
    result: ASRGenerationResult,
) -> None:
    stats.asr_batches += 1
    stats.asr_generation_call_seconds += result.call_wall_seconds
    stats.asr_generate_device_seconds += result.device_generate_seconds
    stats.asr_generation_analysis_seconds += result.analysis_seconds
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
            "generation_call_wall_seconds": result.call_wall_seconds,
            "generate_device_seconds": result.device_generate_seconds,
            "generation_analysis_seconds": result.analysis_seconds,
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


def record_oom_batch(
    stats: RunStats,
    model,
    refs: Sequence[SegmentRef],
    max_new_tokens: int,
) -> None:
    """Record device memory peaks for a failed ASR generation attempt."""
    if model.device.type != "cuda" or not torch.cuda.is_available():
        return
    peak_allocated = torch.cuda.max_memory_allocated(model.device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(model.device) / 1024**3
    stats.peak_cuda_gib = max(stats.peak_cuda_gib, peak_allocated)
    stats.peak_cuda_reserved_gib = max(stats.peak_cuda_reserved_gib, peak_reserved)
    stats.batch_history.append(
        {
            "event": "oom",
            "segments": len(refs),
            "max_new_tokens": max_new_tokens,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
        }
    )


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
