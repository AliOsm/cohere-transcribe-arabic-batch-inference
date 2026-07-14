from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cohere_transcribe.cancellation import request_cancellation, reset_cancellation
from cohere_transcribe.vad.torch_silero import BatchLimits, TorchSileroSequenceVAD


def packaged_torchscript() -> torch.jit.ScriptModule:
    path = (
        Path(__file__).resolve().parents[1] / "src/cohere_transcribe/vad/silero_vad.jit"
    )
    return torch.jit.load(str(path), map_location="cpu").eval()


def canonical_probabilities(
    source: torch.jit.ScriptModule, audio: np.ndarray, sampling_rate: int = 16_000
) -> np.ndarray:
    frame_samples = 512 if sampling_rate == 16_000 else 256
    context_samples = 64 if sampling_rate == 16_000 else 32
    frame_count = (audio.size + frame_samples - 1) // frame_samples
    if not frame_count:
        return np.empty(0, dtype=np.float32)

    padded = torch.zeros(frame_count * frame_samples, dtype=torch.float32)
    padded[: audio.size].copy_(torch.from_numpy(audio))
    frames = padded.view(frame_count, frame_samples)
    context = torch.zeros(context_samples, dtype=torch.float32)
    state = torch.zeros((2, 1, 128), dtype=torch.float32)
    model = source._model if sampling_rate == 16_000 else source._model_8k
    outputs: list[float] = []
    with torch.inference_mode():
        for frame in frames:
            model_input = torch.cat((context, frame)).unsqueeze(0)
            probability, state = model(model_input, state)
            outputs.append(float(probability.item()))
            context = frame[-context_samples:]
    return np.asarray(outputs, dtype=np.float32)


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[int, ...], int, int]] = []

    def forward(self, inputs, hidden, cell, lengths):
        self.calls.append((tuple(lengths), inputs.shape[0], inputs.shape[1]))
        outputs = tuple(
            torch.full((length,), float(hidden[0, index, 0]) / 10)
            for index, length in enumerate(lengths)
        )
        return outputs, hidden + 1, cell + 1


class CancellingRecordingModel(RecordingModel):
    def forward(self, inputs, hidden, cell, lengths):
        result = super().forward(inputs, hidden, cell, lengths)
        request_cancellation()
        return result


class TorchSileroSequenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = packaged_torchscript()

    def test_loading_does_not_change_global_thread_count(self) -> None:
        before = torch.get_num_threads()
        TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=2,
                max_files=2,
                max_valid_frames=4,
                max_padded_frames=4,
                max_audio_seconds=1.0,
            )
        )
        self.assertEqual(torch.get_num_threads(), before)

    def test_variable_file_batch_matches_canonical_v6(self) -> None:
        random = np.random.default_rng(47)
        audios = [
            (random.standard_normal(samples) * 0.03).astype(np.float32)
            for samples in (1, 512, 513, 2_307, 6_151, 15_872)
        ]
        runtime = TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=5,
                max_files=3,
                max_valid_frames=12,
                max_padded_frames=12,
                max_audio_seconds=1.0,
            )
        )

        actual = runtime.speech_probabilities_batch(audios)
        expected = [canonical_probabilities(self.source, audio) for audio in audios]

        self.assertEqual([len(item) for item in actual], [1, 1, 2, 5, 13, 31])
        for candidate, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(candidate, reference, rtol=0, atol=2e-6)

    def test_8khz_weights_and_probabilities_match_canonical_v6(self) -> None:
        random = np.random.default_rng(8)
        audios = [
            (random.standard_normal(samples) * 0.03).astype(np.float32)
            for samples in (1, 256, 257, 2_900)
        ]
        runtime = TorchSileroSequenceVAD(
            sampling_rate=8_000,
            limits=BatchLimits(
                block_frames=4,
                max_files=3,
                max_valid_frames=9,
                max_padded_frames=9,
                max_audio_seconds=1.0,
            ),
        )

        actual = runtime.speech_probabilities_batch(audios)
        expected = [
            canonical_probabilities(self.source, audio, sampling_rate=8_000)
            for audio in audios
        ]
        for candidate, reference in zip(actual, expected, strict=True):
            np.testing.assert_allclose(candidate, reference, rtol=0, atol=2e-6)

    def test_long_file_blocking_preserves_probabilities(self) -> None:
        audio = (
            np.random.default_rng(9).standard_normal(41 * 512 + 103) * 0.03
        ).astype(np.float32)
        small_blocks = TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=3,
                max_files=1,
                max_valid_frames=3,
                max_padded_frames=3,
                max_audio_seconds=1.0,
            )
        )
        one_block = TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=64,
                max_files=1,
                max_valid_frames=64,
                max_padded_frames=64,
                max_audio_seconds=3.0,
            )
        )

        blocked = small_blocks.speech_probabilities(audio)
        complete = one_block.speech_probabilities(audio)

        np.testing.assert_allclose(blocked, complete, rtol=0, atol=2e-6)

    def test_cancellation_before_temporal_group_skips_model_work(self) -> None:
        runtime = TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=2,
                max_files=1,
                max_valid_frames=2,
                max_padded_frames=2,
                max_audio_seconds=1.0,
            )
        )
        recorder = RecordingModel()
        runtime._model = recorder
        reset_cancellation()
        self.addCleanup(reset_cancellation)
        request_cancellation()

        with self.assertRaises(KeyboardInterrupt):
            runtime.speech_probabilities(np.zeros(5 * 512, dtype=np.float32))

        self.assertEqual(recorder.calls, [])

    def test_cancellation_after_temporal_group_stops_before_next_group(self) -> None:
        runtime = TorchSileroSequenceVAD(
            limits=BatchLimits(
                block_frames=2,
                max_files=1,
                max_valid_frames=2,
                max_padded_frames=2,
                max_audio_seconds=1.0,
            )
        )
        recorder = CancellingRecordingModel()
        runtime._model = recorder
        reset_cancellation()
        self.addCleanup(reset_cancellation)

        with self.assertRaises(KeyboardInterrupt):
            runtime.speech_probabilities(np.zeros(5 * 512, dtype=np.float32))

        self.assertEqual(len(recorder.calls), 1)

    def test_scheduler_enforces_caps_and_preserves_per_file_state(self) -> None:
        limits = BatchLimits(
            block_frames=4,
            max_files=2,
            max_valid_frames=7,
            max_padded_frames=7,
            max_audio_seconds=1.0,
        )
        runtime = TorchSileroSequenceVAD(limits=limits)
        recorder = RecordingModel()
        runtime._model = recorder
        audios = [
            np.full(frames * 512, index / 10, dtype=np.float32)
            for index, frames in enumerate((10, 7, 3), start=1)
        ]

        outputs = runtime.speech_probabilities_batch(audios)

        for lengths, valid_frames, input_width in recorder.calls:
            self.assertLessEqual(len(lengths), limits.max_files)
            self.assertLessEqual(valid_frames, limits.max_valid_frames)
            self.assertLessEqual(len(lengths) * max(lengths), limits.max_padded_frames)
            self.assertEqual(input_width, 576)
        np.testing.assert_allclose(outputs[0], [0] * 4 + [0.1] * 4 + [0.2] * 2)
        np.testing.assert_allclose(outputs[1], [0] * 4 + [0.1] * 3)
        np.testing.assert_array_equal(outputs[2], [0] * 3)

    def test_empty_input_and_input_validation(self) -> None:
        runtime = TorchSileroSequenceVAD()
        empty = runtime.speech_probabilities(np.empty(0, dtype=np.float32))
        self.assertEqual(empty.dtype, np.float32)
        self.assertEqual(empty.shape, (0,))

        read_only = np.zeros(512, dtype=np.float32)
        read_only.flags.writeable = False
        self.assertEqual(runtime.speech_probabilities(read_only).shape, (1,))

        with self.assertRaisesRegex(ValueError, "mono audio"):
            runtime.speech_probabilities(np.zeros((2, 512), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            runtime.speech_probabilities(
                np.asarray([0.0, np.nan, 0.0], dtype=np.float32)
            )

        extreme = np.full(512, np.finfo(np.float32).max, dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "Silero returned"):
            runtime.speech_probabilities(extreme)

    def test_limits_reject_unbounded_or_impossible_configurations(self) -> None:
        invalid = [
            BatchLimits(block_frames=0),
            BatchLimits(max_files=0),
            BatchLimits(max_audio_seconds=float("inf")),
            BatchLimits(block_frames=10, max_valid_frames=9),
            BatchLimits(block_frames=10, max_padded_frames=9),
            BatchLimits(max_audio_seconds=0.01),
        ]
        for limits in invalid:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                limits.validate()

    def test_packaged_model_version_is_checked(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported Silero model version"):
            TorchSileroSequenceVAD(expected_version="0.0.invalid")


if __name__ == "__main__":
    unittest.main()
