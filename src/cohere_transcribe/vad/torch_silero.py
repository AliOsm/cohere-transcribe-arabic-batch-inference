"""Bounded, state-preserving Torch inference for offline Silero VAD audio.

The packaged Silero TorchScript model evaluates one 32 ms frame per call. This
module reconstructs the public v5/v6 network so the stateless encoder can process
many frames together while an LSTM preserves temporal state. Independent audio
files use separate recurrent states and variable-length packed sequences.

This module returns frame probabilities only. Timestamp thresholding remains the
caller's responsibility so it can share one segmentation implementation with
other Silero runtimes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_sequence,
    unpack_sequence,
)

from ..cancellation import raise_if_cancelled

EXPECTED_SILERO_VERSION = "6.2.1"
SUPPORTED_SAMPLE_RATES = (8_000, 16_000)
STATE_SIZE = 128


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """Memory and work limits for a packed inference call.

    One Silero frame represents 32 ms at either supported sample rate. Both the
    frame and optional duration limits apply; the smaller effective limit wins.
    ``max_padded_frames`` bounds the dense tensor created before packing.
    """

    block_frames: int = 512
    max_files: int = 16
    max_valid_frames: int = 8_192
    max_padded_frames: int = 8_192
    max_audio_seconds: float | None = 300.0

    def effective_valid_frames(self) -> int:
        if self.max_audio_seconds is None:
            return self.max_valid_frames
        duration_frames = math.floor(self.max_audio_seconds * 1_000 / 32 + 1e-9)
        return min(self.max_valid_frames, duration_frames)

    def validate(self) -> None:
        integer_values = {
            "block_frames": self.block_frames,
            "max_files": self.max_files,
            "max_valid_frames": self.max_valid_frames,
            "max_padded_frames": self.max_padded_frames,
        }
        for name, value in integer_values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_audio_seconds is not None and (
            not np.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0
        ):
            raise ValueError("max_audio_seconds must be finite and positive or None")
        effective_frames = self.effective_valid_frames()
        if effective_frames < 1:
            raise ValueError("max_audio_seconds is shorter than one Silero frame")
        if self.block_frames > effective_frames:
            raise ValueError("block_frames exceeds the effective valid-frame limit")
        if self.block_frames > self.max_padded_frames:
            raise ValueError("block_frames exceeds max_padded_frames")


@dataclass(frozen=True, slots=True)
class BatchExecutionStats:
    """Work performed by the most recent probability request."""

    files: int = 0
    frames: int = 0
    model_calls: int = 0
    valid_frames: int = 0
    padded_frames: int = 0
    max_files_per_call: int = 0


class _Encoder(nn.Module):
    def __init__(self, sampling_rate: int) -> None:
        super().__init__()
        filter_length = int(sampling_rate / 62.5)
        hop_length = filter_length // 2
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.register_buffer(
            "forward_basis_buffer",
            torch.empty((2 * (hop_length + 1), 1, filter_length)),
        )
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(hop_length + 1, 128, 3, padding=1),
                nn.Conv1d(128, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 64, 3, stride=2, padding=1),
                nn.Conv1d(64, 128, 3, padding=1),
            ]
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Use Silero's packaged STFT basis instead of regenerating a Hann basis.
        # The right reflection pad is the exact operation in the v6 TorchScript.
        padded = F.pad(frames, (0, self.filter_length // 4), mode="reflect")
        transformed = F.conv1d(
            padded.unsqueeze(1),
            self.forward_basis_buffer,
            stride=self.hop_length,
        )
        cutoff = self.hop_length + 1
        real = transformed[:, :cutoff]
        imaginary = transformed[:, cutoff:]
        encoded = torch.sqrt(real.pow(2) + imaginary.pow(2))
        for layer in self.layers:
            encoded = F.relu(layer(encoded))
        return encoded


class _SequenceModel(nn.Module):
    def __init__(self, sampling_rate: int) -> None:
        super().__init__()
        self.encoder = _Encoder(sampling_rate)
        self.recurrent = nn.LSTM(STATE_SIZE, STATE_SIZE)
        self.output = nn.Conv1d(STATE_SIZE, 1, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        lengths: Sequence[int],
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        encoded = self.encoder(inputs).squeeze(-1)
        sequences = encoded.split(tuple(lengths))
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths=list(lengths),
            batch_first=True,
            enforce_sorted=False,
        )
        decoded, (hidden, cell) = self.recurrent(packed, (hidden, cell))
        decoded_sequences = unpack_sequence(decoded)
        flat = torch.cat(decoded_sequences)
        probabilities = torch.sigmoid(self.output(F.relu(flat).unsqueeze(-1))).view(-1)
        return probabilities.split(tuple(lengths)), hidden, cell


def _weight_mapping(prefix: str) -> dict[str, str]:
    mapping = {
        "encoder.forward_basis_buffer": prefix + ".stft.forward_basis_buffer",
        "recurrent.weight_ih_l0": prefix + ".decoder.rnn.weight_ih",
        "recurrent.weight_hh_l0": prefix + ".decoder.rnn.weight_hh",
        "recurrent.bias_ih_l0": prefix + ".decoder.rnn.bias_ih",
        "recurrent.bias_hh_l0": prefix + ".decoder.rnn.bias_hh",
        "output.weight": prefix + ".decoder.decoder.2.weight",
        "output.bias": prefix + ".decoder.decoder.2.bias",
    }
    for index in range(4):
        for parameter in ("weight", "bias"):
            mapping[f"encoder.layers.{index}.{parameter}"] = (
                f"{prefix}.encoder.{index}.reparam_conv.{parameter}"
            )
    return mapping


def _packaged_model_path(expected_version: str) -> Path:
    if expected_version != EXPECTED_SILERO_VERSION:
        raise RuntimeError(
            "unsupported Silero model version: "
            f"expected {EXPECTED_SILERO_VERSION}, got {expected_version}"
        )
    path = Path(__file__).with_name("silero_vad.jit")
    if not path.is_file():
        raise RuntimeError(f"packaged Silero TorchScript model is missing: {path}")
    return path


def _load_sequence_model(sampling_rate: int, expected_version: str) -> _SequenceModel:
    if sampling_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"sampling_rate must be one of {SUPPORTED_SAMPLE_RATES}")

    # Loading the packaged data file avoids importing silero_vad.model, which
    # changes PyTorch's process-wide intra-op thread count.
    source = torch.jit.load(
        str(_packaged_model_path(expected_version)), map_location="cpu"
    ).eval()
    source_state = source.state_dict()
    prefix = "_model" if sampling_rate == 16_000 else "_model_8k"
    mapping = _weight_mapping(prefix)
    model = _SequenceModel(sampling_rate)

    state_names = set(model.state_dict())
    if state_names != set(mapping):
        missing = sorted(state_names - set(mapping))
        extra = sorted(set(mapping) - state_names)
        raise RuntimeError(
            f"incomplete Silero parameter mapping; missing={missing}, extra={extra}"
        )
    missing_sources = sorted(set(mapping.values()) - set(source_state))
    if missing_sources:
        raise RuntimeError(f"Silero source model is missing weights: {missing_sources}")

    target_state = model.state_dict()
    remapped: dict[str, torch.Tensor] = {}
    for target_name, source_name in mapping.items():
        source_tensor = source_state[source_name]
        expected = target_state[target_name]
        if source_tensor.shape != expected.shape:
            raise RuntimeError(
                f"unexpected shape for {source_name}: expected {tuple(expected.shape)}, "
                f"got {tuple(source_tensor.shape)}"
            )
        if source_tensor.dtype != torch.float32:
            raise RuntimeError(
                f"unexpected dtype for {source_name}: expected float32, "
                f"got {source_tensor.dtype}"
            )
        if not torch.isfinite(source_tensor).all():
            raise RuntimeError(f"non-finite values in Silero weight {source_name}")
        remapped[target_name] = source_tensor.detach().clone()
    model.load_state_dict(remapped, strict=True)
    return model.eval().requires_grad_(False)


@dataclass(slots=True)
class _Stream:
    index: int
    audio: np.ndarray
    frame_count: int
    probabilities: np.ndarray
    hidden: torch.Tensor
    cell: torch.Tensor
    context: torch.Tensor
    cursor: int = 0

    @property
    def remaining(self) -> int:
        return self.frame_count - self.cursor


class TorchSileroSequenceVAD:
    """CPU Silero v6 inference for bounded long files and independent batches."""

    def __init__(
        self,
        *,
        sampling_rate: int = 16_000,
        limits: BatchLimits | None = None,
        expected_version: str = EXPECTED_SILERO_VERSION,
    ) -> None:
        self.frame_samples = 512 if sampling_rate == 16_000 else 256
        self.context_samples = 64 if sampling_rate == 16_000 else 32
        self.limits = limits or BatchLimits()
        self.limits.validate()
        self._effective_valid_frames = self.limits.effective_valid_frames()
        self._model = _load_sequence_model(sampling_rate, expected_version)
        self.last_stats = BatchExecutionStats()

    def speech_probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Return one probability per zero-padded frame for one recording."""
        return self.speech_probabilities_batch([audio])[0]

    def speech_probabilities_batch(
        self, audios: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """Return independent probability sequences in input order.

        Work is grouped by similar block length and bounded by ``BatchLimits``.
        Hidden state, cell state, and waveform context are retained independently
        for every file across temporal blocks.
        """
        raise_if_cancelled()
        streams = [
            self._make_stream(index, audio) for index, audio in enumerate(audios)
        ]
        active = [stream for stream in streams if stream.frame_count]
        model_calls = valid_frames = padded_frames = max_files_per_call = 0

        with torch.inference_mode():
            while active:
                pending = sorted(
                    active,
                    key=lambda stream: (
                        -min(self.limits.block_frames, stream.remaining),
                        stream.index,
                    ),
                )
                while pending:
                    group, pending = self._take_group(pending)
                    raise_if_cancelled()
                    valid, padded = self._run_group(group)
                    raise_if_cancelled()
                    model_calls += 1
                    valid_frames += valid
                    padded_frames += padded
                    max_files_per_call = max(max_files_per_call, len(group))
                active = [stream for stream in active if stream.remaining]

        self.last_stats = BatchExecutionStats(
            files=len(streams),
            frames=sum(stream.frame_count for stream in streams),
            model_calls=model_calls,
            valid_frames=valid_frames,
            padded_frames=padded_frames,
            max_files_per_call=max_files_per_call,
        )
        return [stream.probabilities for stream in streams]

    def _make_stream(self, index: int, audio: np.ndarray) -> _Stream:
        waveform = np.asarray(audio)
        if waveform.ndim != 1:
            raise ValueError(
                f"Silero VAD expects mono audio, got shape {waveform.shape} at index {index}"
            )
        if (
            waveform.dtype != np.float32
            or not waveform.flags.c_contiguous
            or not waveform.flags.writeable
        ):
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
            if not waveform.flags.writeable:
                waveform = waveform.copy()
        finite_block = 1_048_576
        for start in range(0, waveform.size, finite_block):
            raise_if_cancelled()
            if not np.isfinite(waveform[start : start + finite_block]).all():
                raise ValueError(
                    f"Silero VAD audio contains non-finite values at index {index}"
                )
        raise_if_cancelled()
        frame_count = (waveform.size + self.frame_samples - 1) // self.frame_samples
        state = torch.zeros((1, 1, STATE_SIZE), dtype=torch.float32)
        return _Stream(
            index=index,
            audio=waveform,
            frame_count=frame_count,
            probabilities=np.empty(frame_count, dtype=np.float32),
            hidden=state,
            cell=state.clone(),
            context=torch.zeros(self.context_samples, dtype=torch.float32),
        )

    def _take_group(
        self, pending: list[_Stream]
    ) -> tuple[list[_Stream], list[_Stream]]:
        group: list[_Stream] = []
        deferred: list[_Stream] = []
        valid_frames = 0
        longest = 0

        for stream in pending:
            frames = min(self.limits.block_frames, stream.remaining)
            next_longest = max(longest, frames)
            file_count = len(group) + 1
            fits = (
                file_count <= self.limits.max_files
                and valid_frames + frames <= self._effective_valid_frames
                and file_count * next_longest <= self.limits.max_padded_frames
            )
            if fits:
                group.append(stream)
                valid_frames += frames
                longest = next_longest
            else:
                deferred.append(stream)

        if not group:
            raise RuntimeError(
                "Silero batch limits cannot accommodate one temporal block"
            )
        return group, deferred

    def _run_group(self, streams: Sequence[_Stream]) -> tuple[int, int]:
        lengths = [
            min(self.limits.block_frames, stream.remaining) for stream in streams
        ]
        total_frames = sum(lengths)
        inputs = torch.empty(
            (total_frames, self.context_samples + self.frame_samples),
            dtype=torch.float32,
        )
        next_contexts: list[torch.Tensor] = []
        row_start = 0

        for stream, frame_count in zip(streams, lengths, strict=True):
            rows = inputs[row_start : row_start + frame_count]
            rows[:, self.context_samples :].zero_()
            sample_start = stream.cursor * self.frame_samples
            sample_end = min(
                stream.audio.size, sample_start + frame_count * self.frame_samples
            )
            source = torch.from_numpy(stream.audio[sample_start:sample_end])
            complete_frames, remaining_samples = divmod(
                source.numel(), self.frame_samples
            )
            if complete_frames:
                complete_samples = complete_frames * self.frame_samples
                rows[:complete_frames, self.context_samples :].copy_(
                    source[:complete_samples].view(complete_frames, self.frame_samples)
                )
            if remaining_samples:
                rows[
                    complete_frames,
                    self.context_samples : self.context_samples + remaining_samples,
                ].copy_(source[-remaining_samples:])
            rows[0, : self.context_samples].copy_(stream.context)
            if frame_count > 1:
                rows[1:, : self.context_samples].copy_(
                    rows[:-1, -self.context_samples :]
                )
            next_contexts.append(rows[-1, -self.context_samples :].clone())
            row_start += frame_count

        hidden = torch.cat([stream.hidden for stream in streams], dim=1)
        cell = torch.cat([stream.cell for stream in streams], dim=1)
        outputs, hidden, cell = self._model(inputs, hidden, cell, lengths)

        expected_state_shape = (1, len(streams), STATE_SIZE)
        if hidden.shape != expected_state_shape or cell.shape != expected_state_shape:
            raise RuntimeError(
                "Silero returned invalid recurrent-state shapes: "
                f"hidden={tuple(hidden.shape)}, cell={tuple(cell.shape)}, "
                f"expected={expected_state_shape}"
            )
        if not torch.isfinite(hidden).all() or not torch.isfinite(cell).all():
            raise RuntimeError("Silero returned non-finite recurrent state")
        if len(outputs) != len(streams):
            raise RuntimeError(
                f"Silero returned {len(outputs)} outputs for {len(streams)} streams"
            )

        for offset, (stream, probabilities, frame_count, context) in enumerate(
            zip(streams, outputs, lengths, next_contexts, strict=True)
        ):
            if probabilities.shape != (frame_count,):
                raise RuntimeError(
                    "Silero returned an invalid probability shape: "
                    f"expected {(frame_count,)}, got {tuple(probabilities.shape)}"
                )
            if (
                probabilities.dtype != torch.float32
                or not torch.isfinite(probabilities).all()
                or torch.any((probabilities < 0) | (probabilities > 1))
            ):
                raise RuntimeError("Silero returned invalid speech probabilities")
            destination = stream.probabilities[
                stream.cursor : stream.cursor + frame_count
            ]
            destination[:] = probabilities.detach().numpy()
            stream.cursor += frame_count
            stream.hidden = hidden[:, offset : offset + 1].contiguous()
            stream.cell = cell[:, offset : offset + 1].contiguous()
            stream.context = context
        return total_frames, len(streams) * max(lengths)
