from __future__ import annotations

import errno
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from cohere_transcribe.alignment.runtime import compute_emissions_streaming
from cohere_transcribe.asr import execution as asr
from cohere_transcribe.asr.batching import record_prepared_batch
from cohere_transcribe.asr.execution import (
    handle_asr_batch_failure,
    retry_token_limit,
)
from cohere_transcribe.asr.generation import PreparedASRBatch, generate_asr_batch
from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.models import (
    AudioJob,
    RunStats,
    SegmentRef,
    SourceSnapshot,
    file_sha256,
)
from cohere_transcribe.output import publication as outputs
from cohere_transcribe.output.publication import (
    atomic_write_outputs,
    fsync_directories,
)
from cohere_transcribe.output.rendering import (
    generate_json,
)
from cohere_transcribe.profiling import build_profile_payload
from cohere_transcribe.vad.runtime import build_silero_torch_runtime
from cohere_transcribe.vad.torch_silero import BatchLimits


def make_job(source: Path | None = None) -> AudioJob:
    path = source or Path("sample.wav")
    snapshot = (
        SourceSnapshot.capture(path)
        if source is not None
        else SourceSnapshot(0, 0, 0, 0, 0)
    )
    return AudioJob(
        index=0,
        path=path,
        relative_path=Path("sample.wav"),
        snapshot=snapshot,
        duration_hint=2.0,
        language="ar",
        vad_mode="silero",
        alignment_mode="segment",
        duration=2.0,
        segment_times=[(0.0, 1.0), (1.0, 2.0)],
        segment_texts=["", "مرحبا"],
        vad_engine_requested="auto",
        vad_engine_actual="torch",
    )


class RuntimeRegressionTest(unittest.TestCase):
    def test_non_power_of_two_vad_limit_is_exact(self) -> None:
        limits = BatchLimits(
            block_frames=2001,
            max_files=1,
            max_valid_frames=2001,
            max_padded_frames=2001,
            max_audio_seconds=None,
        )
        limits.validate()
        self.assertEqual(limits.effective_valid_frames(), 2001)

        args = parse_args(
            [
                "sample.wav",
                "--vad-batch-size",
                "1",
                "--vad-block-frames",
                "2001",
                "--alignment",
                "segment",
            ]
        )
        validate_args(args)
        runtime = build_silero_torch_runtime(args)
        self.assertEqual(runtime.model.limits.effective_valid_frames(), 2001)

    def test_default_token_retry_jumps_to_ceiling(self) -> None:
        model = mock.Mock()
        model.config = mock.Mock()
        model.config.decoder_config = None
        model.config.max_position_embeddings = 2000
        self.assertEqual(retry_token_limit(model, 10, 445, 896), 896)

    def test_inactive_mode_options_do_not_reject_valid_configuration(self) -> None:
        fixed = parse_args(
            [
                "sample.wav",
                "--vad",
                "none",
                "--min-dur",
                "-100",
                "--vad-batch-size",
                "100000",
                "--text-only",
                "--max-chars",
                "0",
            ]
        )
        validate_args(fixed)
        onnx = parse_args(
            [
                "sample.wav",
                "--vad-engine",
                "onnx",
                "--vad-batch-size",
                "100000",
                "--alignment",
                "segment",
            ]
        )
        validate_args(onnx)

    def test_json_segments_preserve_original_indices_and_kernel_provenance(
        self,
    ) -> None:
        job = make_job()
        job.decode_backend = "ffmpeg"
        job.decode_fallback_reason = "RuntimeError: unsupported container"
        words = [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "مرحبا",
                "segment_index": 1,
                "segment_word_index": 0,
                "timing_source": "uniform_segment",
            }
        ]
        payload = json.loads(generate_json(job, words, [], ["مرحبا"]))
        self.assertEqual(payload["schema_version"], 7)
        self.assertEqual(payload["source"]["decode_backend"], "ffmpeg")
        self.assertEqual(
            payload["source"]["decode_fallback_reason"],
            "RuntimeError: unsupported container",
        )
        self.assertEqual(payload["segments"][0]["segment_index"], 1)
        self.assertEqual(payload["words"][0]["segment_index"], 1)
        self.assertEqual(
            payload["models"]["aligner"],
            None,
        )
        self.assertEqual(
            payload["implementation"]["artifacts_sha256"][
                "cohere_transcribe/models.py"
            ],
            file_sha256(Path(__file__).parents[1] / "src/cohere_transcribe/models.py"),
        )

        job.alignment_mode = "word"
        word_payload = json.loads(generate_json(job, words, [], ["مرحبا"]))
        self.assertEqual(
            word_payload["models"]["aligner"]["kernel"]["operation"],
            "torchaudio.functional.forced_align",
        )

    def test_profile_separates_requested_and_resolved_configuration(self) -> None:
        args = parse_args(["sample.wav", "--alignment", "segment"])
        validate_args(args)
        requested = asdict(args)
        args.device = "cpu"
        args.vad_engine = "torch"
        job = make_job()
        job.decode_backend = "ffmpeg"
        job.decode_fallback_reason = "RuntimeError: unsupported container"
        profile = build_profile_payload(
            args,
            requested,
            RunStats(),
            [job],
            1.0,
            "cpu",
            torch.float32,
        )
        self.assertEqual(profile["schema_version"], 8)
        self.assertEqual(profile["configuration"]["device"], "auto")
        self.assertEqual(profile["configuration"]["vad_engine"], "auto")
        self.assertEqual(profile["resolved_configuration"]["device"], "cpu")
        self.assertEqual(profile["resolved_configuration"]["vad_engine"], "torch")
        self.assertEqual(profile["files"][0]["decode_backend"], "ffmpeg")
        self.assertEqual(
            profile["files"][0]["decode_fallback_reason"],
            "RuntimeError: unsupported container",
        )

    def test_profile_distinguishes_all_segments_from_inferred_this_run(self) -> None:
        args = parse_args(["sample.wav", "--alignment", "segment"])
        validate_args(args)
        fresh = make_job()
        fresh.generated_tokens = {1: 7}
        resumed = make_job()
        resumed.index = 1
        resumed.duration = 2.0
        resumed.segment_times = [(0.0, 2.0)]
        resumed.segment_texts = ["resumed"]
        resumed.generated_tokens = {0: 9}
        resumed.asr_checkpoint_loaded = True
        stats = RunStats(asr_checkpoint_resumed_files=1)

        profile = build_profile_payload(
            args,
            asdict(args),
            stats,
            [fresh, resumed],
            1.0,
            "cpu",
            torch.float32,
        )

        self.assertNotIn("segment_duration_seconds", profile["asr"])
        self.assertEqual(profile["asr"]["all_segment_duration_seconds"]["max"], 2.0)
        self.assertEqual(
            profile["asr"]["inferred_segment_duration_seconds"],
            {"min": 1.0, "p50": 1.0, "p90": 1.0, "p99": 1.0, "max": 1.0},
        )
        self.assertEqual(profile["asr"]["checkpoint_resumed_files"], 1)

    def test_alignment_redecode_reuses_the_concrete_backend(self) -> None:
        job = make_job()
        job.audio = None
        job.decode_backend = "ffmpeg"
        args = parse_args(
            ["sample.wav", "--audio-backend", "auto", "--alignment", "word"]
        )
        expected = np.zeros(2 * 16_000, dtype=np.float32)

        with (
            mock.patch.object(outputs, "ensure_source_unchanged"),
            mock.patch.object(
                outputs, "decode_audio", return_value=expected
            ) as decoder,
        ):
            actual = outputs.reload_audio_for_alignment(job, args)

        self.assertIs(actual, expected)
        decoder.assert_called_once_with(
            job.path,
            "ffmpeg",
            max_decoded_bytes=4 * 1024**3,
            duration_hint=2.0,
        )

    def test_alignment_geometry_covers_short_and_cross_window_audio(self) -> None:
        class FakeAligner:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(inputs_to_logits_ratio=320)

            @staticmethod
            def __call__(values):
                frames = values.shape[1] // 320
                return SimpleNamespace(
                    logits=torch.zeros((len(values), frames, 3), dtype=torch.float32)
                )

        for samples in (16_000, 31 * 16_000):
            emissions, stride = compute_emissions_streaming(
                np.zeros(samples, dtype=np.float32),
                FakeAligner(),
                2,
                "test",
            )
            self.assertEqual(len(emissions), (samples + 319) // 320)
            self.assertEqual(emissions.shape[1], 4)
            self.assertEqual(stride, 20.0)

    def test_discarded_features_are_not_counted_as_generation_rows(self) -> None:
        prepared = PreparedASRBatch(
            refs=[],
            model_inputs={},
            chunk_index=[(0, None), (1, None)],
            prepare_seconds=0.25,
            valid_feature_frames=10,
            padded_feature_frames=12,
        )
        stats = RunStats()
        record_prepared_batch(stats, prepared, discarded=True)
        self.assertEqual(stats.asr_processor_rows, 0)
        self.assertEqual(stats.asr_discarded_processor_rows, 2)
        self.assertEqual(stats.asr_discarded_feature_batches, 1)
        self.assertEqual(stats.asr_feature_seconds, 0.25)

    def test_generation_telemetry_separates_call_and_analysis(self) -> None:
        class FakeModel:
            device = torch.device("cpu")
            dtype = torch.float32
            generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0)
            model = SimpleNamespace(decoder=SimpleNamespace(proj=torch.nn.Identity()))

            @staticmethod
            def generate(**inputs):
                prompts = inputs["decoder_input_ids"]
                eos = torch.full((len(prompts), 1), 2, dtype=torch.int64)
                return torch.cat((prompts, eos), dim=1)

        args = parse_args(["sample.wav", "--alignment", "segment"])
        validate_args(args)
        prepared = PreparedASRBatch(
            refs=[],
            model_inputs={
                "input_features": torch.zeros((1, 2, 2)),
                "decoder_input_ids": torch.ones((1, 1), dtype=torch.int64),
            },
            chunk_index=[(0, None)],
            prepare_seconds=0.0,
            valid_feature_frames=2,
            padded_feature_frames=2,
        )
        result = generate_asr_batch(FakeModel(), prepared, args, 10)
        self.assertEqual(result.row_token_counts, [1])
        self.assertGreaterEqual(
            result.call_wall_seconds, result.device_generate_seconds
        )
        self.assertGreaterEqual(result.analysis_seconds, 0.0)

    def test_invariant_asr_failure_does_not_bisect_batch(self) -> None:
        job = make_job()
        refs = [
            SegmentRef(job, index, float(index), float(index + 1)) for index in range(2)
        ]
        model = SimpleNamespace(device=torch.device("cpu"))
        bar = mock.Mock()
        with mock.patch.object(asr, "transcribe_ref_batch") as retry:
            handle_asr_batch_failure(
                None,
                model,
                refs,
                parse_args(["sample.wav", "--alignment", "segment"]),
                bar,
                RunStats(),
                mock.Mock(),
                "fatal",
                "TypeError: incompatible model API",
                445,
            )
        retry.assert_not_called()
        self.assertIn("incompatible model API", job.error or "")
        bar.update.assert_called_once_with(2)

    def test_unsupported_directory_fsync_is_best_effort(self) -> None:
        with mock.patch.object(
            outputs.os,
            "open",
            side_effect=OSError(errno.EACCES, "directory handles unsupported"),
        ):
            fsync_directories(iter((Path("."),)))

    def test_source_change_rejects_transactional_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sample.wav"
            source.write_bytes(b"before")
            job = make_job(source)
            output = root / "sample.txt"
            job.output_paths = {"txt": output}
            source.write_bytes(b"after and larger")
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                atomic_write_outputs(job, [], transcript_lines=["text"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
