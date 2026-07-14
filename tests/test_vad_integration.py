from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from cohere_transcribe.audio import preparation
from cohere_transcribe.audio.preparation import (
    _preparation_thread_local,
    prepare_source_group,
    prepare_torch_silero_group,
)
from cohere_transcribe.config import parse_args
from cohere_transcribe.models import (
    AudioJob,
    DecodedAudio,
    PreparedAudio,
    SourceSnapshot,
)
from cohere_transcribe.vad.runtime import (
    SileroBackendUnavailable,
    SileroRuntime,
    build_silero_jit_runtime,
)
from cohere_transcribe.vad.vectorized_silero import (
    get_speech_timestamps_from_probabilities,
)


def make_job(index: int) -> AudioJob:
    return AudioJob(
        index=index,
        path=Path(f"{index}.wav"),
        relative_path=Path(f"{index}.wav"),
        snapshot=SourceSnapshot(0, index, 0, 0, 0),
        duration_hint=0.064,
        language="ar",
        vad_mode="silero",
        alignment_mode="segment",
        vad_engine_requested="torch",
    )


class PoisonBatchModel:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.limits = types.SimpleNamespace(block_frames=512)
        self.last_stats = types.SimpleNamespace(
            model_calls=0,
            valid_frames=0,
            padded_frames=0,
            max_files_per_call=0,
        )

    def speech_probabilities_batch(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        self.calls.append(len(audios))
        if any(audio[0] == 2 for audio in audios):
            raise RuntimeError("poison input")
        outputs = [
            np.full((len(audio) + 511) // 512, 0.1, dtype=np.float32)
            for audio in audios
        ]
        frames = sum(map(len, outputs))
        self.last_stats = types.SimpleNamespace(
            model_calls=1,
            valid_frames=frames,
            padded_frames=frames,
            max_files_per_call=len(audios),
        )
        return outputs


class OomAboveTwoModel(PoisonBatchModel):
    def speech_probabilities_batch(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        if len(audios) > 2:
            self.calls.append(len(audios))
            raise RuntimeError("DefaultCPUAllocator: out of memory")
        self.calls.append(len(audios))
        outputs = [
            np.full((len(audio) + 511) // 512, 0.1, dtype=np.float32)
            for audio in audios
        ]
        frames = sum(map(len, outputs))
        self.last_stats = types.SimpleNamespace(
            model_calls=1,
            valid_frames=frames,
            padded_frames=frames,
            max_files_per_call=len(audios),
        )
        return outputs


class OomLargeBlockModel(OomAboveTwoModel):
    def __init__(self, block_frames: int) -> None:
        super().__init__()
        self.limits = types.SimpleNamespace(block_frames=block_frames)

    def speech_probabilities_batch(self, audios: list[np.ndarray]) -> list[np.ndarray]:
        if self.limits.block_frames > 128:
            raise RuntimeError("DefaultCPUAllocator: out of memory")
        return super().speech_probabilities_batch(audios)


class TorchVadIntegrationTest(unittest.TestCase):
    def tearDown(self) -> None:
        for attribute in (
            "torch_vad_retry_cap",
            "torch_vad_retry_reported",
            "torch_vad_retry_block",
        ):
            if hasattr(_preparation_thread_local, attribute):
                delattr(_preparation_thread_local, attribute)

    def test_packed_failure_isolated_to_one_file(self) -> None:
        args = parse_args(
            ["placeholder.wav", "--vad-engine", "torch", "--alignment", "segment"]
        )
        model = PoisonBatchModel()
        runtime = SileroRuntime(
            model=model,
            engine="torch",
            runner=lambda *_args: [],
            provider="CPU",
            load_seconds=0.05,
        )
        jobs = [make_job(index) for index in range(3)]

        def decode(job, _args):
            return DecodedAudio(
                audio=np.full(1024, job.index, dtype=np.float32),
                decode_backend="test",
                decode_seconds=0.01,
            )

        with (
            mock.patch.object(preparation, "decode_job", side_effect=decode),
            mock.patch.object(preparation, "get_silero_runtime", return_value=runtime),
        ):
            group = prepare_torch_silero_group(jobs, args, 2)

        self.assertIsNotNone(group.results[0].prepared)
        self.assertIsNotNone(group.results[1].prepared)
        self.assertIsNone(group.results[0].error)
        self.assertIsNone(group.results[1].error)
        self.assertIsNone(group.results[2].prepared)
        self.assertRegex(str(group.results[2].error), "poison input")
        self.assertEqual(model.calls, [3, 1, 2, 1, 1])
        self.assertEqual(group.vad_metrics.prepared_groups, 1)
        self.assertEqual(group.vad_metrics.model_calls, 2)

    def test_safe_jit_loader_preserves_torch_threads(self) -> None:
        before = torch.get_num_threads()
        runtime = build_silero_jit_runtime()
        after = torch.get_num_threads()
        args = parse_args(
            ["placeholder.wav", "--vad-engine", "jit", "--alignment", "segment"]
        )
        timestamps = runtime.runner(
            np.zeros(1024, dtype=np.float32), runtime.model, args
        )
        self.assertEqual(after, before)
        self.assertEqual(timestamps, [])

    def test_cpu_oom_lowers_pack_cap_for_later_groups(self) -> None:
        args = parse_args(
            ["placeholder.wav", "--vad-engine", "torch", "--alignment", "segment"]
        )
        model = OomAboveTwoModel()
        runtime = SileroRuntime(
            model=model,
            engine="torch",
            runner=lambda *_args: [],
            provider="CPU",
        )

        def decode(job, _args):
            return DecodedAudio(
                audio=np.zeros(1024, dtype=np.float32),
                decode_backend="test",
                decode_seconds=0.01,
            )

        with (
            mock.patch.object(preparation, "decode_job", side_effect=decode),
            mock.patch.object(preparation, "get_silero_runtime", return_value=runtime),
        ):
            first = prepare_torch_silero_group(
                [make_job(index) for index in range(5)], args, 2
            )
            first_calls = list(model.calls)
            model.calls.clear()
            second = prepare_torch_silero_group(
                [make_job(index) for index in range(4)], args, 2
            )

        self.assertTrue(all(result.prepared is not None for result in first.results))
        self.assertTrue(all(result.prepared is not None for result in second.results))
        self.assertEqual(first_calls, [5, 2, 2, 1])
        self.assertEqual(model.calls, [2, 2])

    def test_precomputed_timestamp_input_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            get_speech_timestamps_from_probabilities(
                512.0, np.asarray([0.1], dtype=np.float32)
            )
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            get_speech_timestamps_from_probabilities(
                512, np.asarray([1.1], dtype=np.float32)
            )

    def test_timestamp_postprocessing_checks_cancellation_periodically(self) -> None:
        checks = 0

        def cancel_during_state_machine() -> None:
            nonlocal checks
            checks += 1
            if checks == 3:
                raise KeyboardInterrupt

        probabilities = np.zeros(8_193, dtype=np.float32)
        with (
            mock.patch(
                "cohere_transcribe.vad.vectorized_silero.raise_if_cancelled",
                side_effect=cancel_during_state_machine,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            get_speech_timestamps_from_probabilities(
                len(probabilities) * 512, probabilities
            )

        self.assertEqual(checks, 3)

    def test_single_file_oom_persistently_lowers_temporal_block(self) -> None:
        args = parse_args(
            ["placeholder.wav", "--vad-engine", "torch", "--alignment", "segment"]
        )
        requested_blocks: list[int] = []

        def runtime_for(_engine, config):
            requested_blocks.append(config.vad_block_frames)
            return SileroRuntime(
                model=OomLargeBlockModel(config.vad_block_frames),
                engine="torch",
                runner=lambda *_args: [],
                provider="CPU",
            )

        def decode(_job, _args):
            return DecodedAudio(
                audio=np.zeros(1024, dtype=np.float32),
                decode_backend="test",
                decode_seconds=0.01,
            )

        with (
            mock.patch.object(preparation, "decode_job", side_effect=decode),
            mock.patch.object(
                preparation, "get_silero_runtime", side_effect=runtime_for
            ),
        ):
            group = prepare_torch_silero_group([make_job(0)], args, 1)

        self.assertIsNotNone(group.results[0].prepared)
        self.assertEqual(requested_blocks, [512, 256, 128])
        self.assertEqual(group.vad_metrics.effective_block_frames, 128)

    def test_auto_falls_back_to_onnx_without_failing_the_group(self) -> None:
        args = parse_args(
            ["placeholder.wav", "--vad-engine", "auto", "--alignment", "segment"]
        )
        args.vad_engine = "torch"
        jobs = [make_job(0), make_job(1)]
        for job in jobs:
            job.vad_engine_requested = "auto"

        def prepared(_job, config):
            self.assertEqual(config.vad_engine, "auto")
            return PreparedAudio(
                audio=np.zeros(1024, dtype=np.float32),
                segment_times=[],
                speech_spans=[],
                decode_seconds=0.01,
                vad_seconds=0.01,
                vad_engine="onnx",
                decode_backend="test",
                vad_provider="CPUExecutionProvider",
            )

        with (
            mock.patch.object(
                preparation,
                "prepare_torch_silero_group",
                side_effect=SileroBackendUnavailable("missing weights"),
            ),
            mock.patch.object(preparation, "prepare_audio", side_effect=prepared),
        ):
            group = prepare_source_group(jobs, args, 2)

        self.assertTrue(all(result.prepared is not None for result in group.results))
        self.assertTrue(
            all(
                result.prepared.vad_fallback_reason
                == "SileroBackendUnavailable: missing weights"
                for result in group.results
                if result.prepared is not None
            )
        )


if __name__ == "__main__":
    unittest.main()
