from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from cohere_transcribe.alignment.runtime import (
    ALIGN_CONTEXT_S,
    ALIGN_WINDOW_S,
    _compute_emissions_streaming,
    align_words,
    build_alignment_window_batch,
)
from cohere_transcribe.asr.generation import (
    RepetitionLoopStoppingCriteria,
    reassemble_chunk_texts,
)
from cohere_transcribe.config import (
    parse_args,
    validate_args,
)
from cohere_transcribe.inputs import build_jobs
from cohere_transcribe.models import (
    ASR_MODEL_REVISION,
    MODEL_ID,
    SR,
    AudioJob,
    RunStats,
    SourceSnapshot,
)
from cohere_transcribe.output.pipeline import align_and_write_all
from cohere_transcribe.output.publication import atomic_write_outputs
from cohere_transcribe.state import release_output_locks
from cohere_transcribe.vad.runtime import get_silero_runtime
from cohere_transcribe.vad.vectorized_silero import (
    VectorizedSileroVAD,
)
from cohere_transcribe.vad.vectorized_silero import (
    get_speech_timestamps as get_vectorized_speech_timestamps,
)


def make_job(
    source: Path,
    output_paths: dict[str, Path],
    *,
    segment_texts: list[str] | None = None,
) -> AudioJob:
    texts = ["النص الاصلي"] if segment_texts is None else segment_texts
    job = AudioJob(
        index=0,
        path=source,
        relative_path=Path(source.name),
        snapshot=SourceSnapshot.capture(source),
        duration_hint=1.0,
        language="ar",
        vad_mode="silero",
        alignment_mode="word",
        output_paths=output_paths,
        duration=1.0,
        segment_times=[(0.0, 1.0)] if texts else [],
        segment_texts=texts,
    )
    return job


def fake_alignment_tokenizer() -> SimpleNamespace:
    return SimpleNamespace(
        get_vocab=lambda: {"<blank>": 0, "a": 1, "b": 2},
        pad_token_id=0,
    )


class CliAndPlanningTest(unittest.TestCase):
    def test_input_is_required(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args([])

    def test_plain_text_mode_has_one_output(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)

        self.assertEqual(args.alignment, "none")
        self.assertEqual(args.formats, ["txt"])

    def test_auditok_rejects_invalid_duration_settings(self) -> None:
        args = parse_args(["audio.wav", "--vad", "auditok", "--min-dur", "0"])
        with self.assertRaisesRegex(SystemExit, r"min-dur > 0"):
            validate_args(args)

        args = parse_args(
            [
                "audio.wav",
                "--vad",
                "auditok",
                "--max-dur",
                "2",
                "--max-silence",
                "2",
            ]
        )
        with self.assertRaisesRegex(SystemExit, r"max-silence < --max-dur"):
            validate_args(args)

    def test_existing_output_policy_is_preflighted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clip.wav"
            source.write_bytes(b"not needed for planning")
            output_root = root / "outputs"
            output_root.mkdir()
            (output_root / "clip.txt").write_text("old", encoding="utf-8")

            args = parse_args(
                [
                    os.fspath(source),
                    "--alignment",
                    "none",
                    "--output-dir",
                    os.fspath(output_root),
                ]
            )
            validate_args(args)
            with self.assertRaisesRegex(SystemExit, r"Output already exists"):
                build_jobs(args)

            args.existing = "skip"
            jobs = build_jobs(args)
            self.assertEqual(len(jobs), 1)
            release_output_locks(jobs)


class ChunkReassemblyTest(unittest.TestCase):
    def test_direct_and_single_chunk_rows_are_mapped_to_samples(self) -> None:
        result = reassemble_chunk_texts(
            ["second", "first", ""],
            [(1, 0), (0, None), (2, None)],
            3,
        )
        self.assertEqual(result, ["first", "second", ""])

    def test_metadata_count_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"does not match"):
            reassemble_chunk_texts(["only one"], [(0, None), (1, None)], 2)

    def test_missing_duplicate_and_expanded_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"no ASR row"):
            reassemble_chunk_texts(["first"], [(0, None)], 2)
        with self.assertRaisesRegex(RuntimeError, r"duplicate rows"):
            reassemble_chunk_texts(["first", "again"], [(0, None), (0, 0)], 2)
        with self.assertRaisesRegex(RuntimeError, r"expanded sample"):
            reassemble_chunk_texts(["chunk"], [(0, 1)], 1)


class VectorizedSileroRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        from cohere_transcribe.vad.runtime import _vad_thread_local

        for name in (
            "runtimes",
            "onnx_fallback_error",
            "onnx_fallback_reported",
        ):
            if hasattr(_vad_thread_local, name):
                delattr(_vad_thread_local, name)

    def tearDown(self) -> None:
        from cohere_transcribe.vad.runtime import _vad_thread_local

        for name in (
            "runtimes",
            "onnx_fallback_error",
            "onnx_fallback_reported",
        ):
            if hasattr(_vad_thread_local, name):
                delattr(_vad_thread_local, name)

    def test_onnx_sessions_are_cached_per_preprocessing_thread(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none", "--vad-engine", "onnx"])
        validate_args(args)
        created: list[object] = []

        def construct() -> object:
            session = object()
            created.append(session)
            return session

        worker_result: list[object] = []
        with mock.patch(
            "cohere_transcribe.vad.vectorized_silero.VectorizedSileroVAD",
            side_effect=construct,
        ):
            first = get_silero_runtime("onnx", args)
            second = get_silero_runtime("onnx", args)

            thread = threading.Thread(
                target=lambda: worker_result.append(
                    get_silero_runtime("onnx", args).model
                )
            )
            thread.start()
            thread.join()

        self.assertIs(first, second)
        self.assertEqual((first.engine, second.engine), ("onnx", "onnx"))
        self.assertEqual(len(created), 2)
        self.assertIsNot(first.model, worker_result[0])

    def test_auto_falls_back_to_existing_jit_loader(self) -> None:
        from cohere_transcribe.vad.runtime import SileroBackendUnavailable

        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)
        jit_runtime = SimpleNamespace(model=object(), engine="jit")
        with (
            mock.patch(
                "cohere_transcribe.vad.runtime.build_silero_onnx_runtime",
                side_effect=SileroBackendUnavailable("ONNX unavailable"),
            ),
            mock.patch(
                "cohere_transcribe.vad.runtime.build_silero_jit_runtime",
                return_value=jit_runtime,
            ),
        ):
            runtime = get_silero_runtime("auto", args)

        self.assertIs(runtime.model, jit_runtime.model)
        self.assertEqual(runtime.engine, "jit")

    def test_auto_does_not_hide_onnx_programming_errors(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)
        with (
            mock.patch(
                "cohere_transcribe.vad.runtime.build_silero_onnx_runtime",
                side_effect=TypeError("bad wrapper schema"),
            ),
            self.assertRaisesRegex(TypeError, "bad wrapper schema"),
        ):
            get_silero_runtime("auto", args)

    def test_timestamp_state_machine_matches_silero_621(self) -> None:
        from silero_vad import get_speech_timestamps

        class VectorProbabilities:
            def __init__(self, probabilities: np.ndarray) -> None:
                self.probabilities = probabilities

            def speech_probabilities(self, audio: np.ndarray) -> np.ndarray:
                return self.probabilities

        class SequentialProbabilities:
            def __init__(self, probabilities: np.ndarray) -> None:
                self.probabilities = probabilities

            def reset_states(self) -> None:
                self.index = 0

            def __call__(self, chunk: torch.Tensor, sampling_rate: int) -> torch.Tensor:
                del chunk, sampling_rate
                value = self.probabilities[self.index]
                self.index += 1
                return torch.tensor(value)

        random = np.random.default_rng(7)
        for frames, remainder, max_duration in (
            (31, 512, 30.0),
            (97, 173, 2.0),
            (181, 500, 1.0),
        ):
            probabilities = random.random(frames).astype(np.float32)
            audio = np.zeros((frames - 1) * 512 + remainder, dtype=np.float32)
            options = {
                "sampling_rate": 16_000,
                "threshold": 0.5,
                "min_speech_duration_ms": 500,
                "max_speech_duration_s": max_duration,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 60,
            }
            expected = get_speech_timestamps(
                torch.from_numpy(audio),
                SequentialProbabilities(probabilities),
                **options,
            )
            actual = get_vectorized_speech_timestamps(
                audio,
                VectorProbabilities(probabilities),
                **options,
            )
            self.assertEqual(actual, expected)

    def test_sequence_export_is_bit_exact_to_frame_onnx(self) -> None:
        from silero_vad import load_silero_vad

        random = np.random.default_rng(11)
        audio = (random.standard_normal(512 * 12) * 0.03).astype(np.float32)
        frame_model = load_silero_vad(onnx=True)
        frame_model.reset_states()
        expected = np.asarray(
            [
                frame_model(torch.from_numpy(audio[start : start + 512]), 16_000).item()
                for start in range(0, len(audio), 512)
            ],
            dtype=np.float32,
        )

        actual = VectorizedSileroVAD().speech_probabilities(audio)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(actual), 12)

    def test_sequence_chunks_preserve_recurrent_state_and_context(self) -> None:
        random = np.random.default_rng(17)
        audio = (random.standard_normal(512 * 12 + 173) * 0.03).astype(np.float32)

        expected = VectorizedSileroVAD().speech_probabilities(audio)
        with mock.patch(
            "cohere_transcribe.vad.vectorized_silero.MAX_SEQUENCE_FRAMES", 5
        ):
            actual = VectorizedSileroVAD().speech_probabilities(audio)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(actual), 13)


class RepetitionGuardTest(unittest.TestCase):
    def test_triggered_rows_are_auditable(self) -> None:
        criterion = RepetitionLoopStoppingCriteria(prompt_length=2)
        repeated = torch.arange(8, dtype=torch.long).repeat(12)
        input_ids = torch.cat([torch.zeros(2, dtype=torch.long), repeated]).unsqueeze(0)

        done = criterion(input_ids, None)

        self.assertEqual(done.tolist(), [True])
        self.assertEqual(criterion.triggered_rows, {0})

    def test_rows_that_already_reached_eos_are_not_reported(self) -> None:
        criterion = RepetitionLoopStoppingCriteria(
            prompt_length=2,
            eos_token_ids=(99,),
        )
        repeated = torch.tensor([0, 1, 99, 3, 4, 5, 6, 7]).repeat(12)
        input_ids = torch.cat([torch.zeros(2, dtype=torch.long), repeated]).unsqueeze(0)

        done = criterion(input_ids, None)

        self.assertEqual(done.tolist(), [False])
        self.assertEqual(criterion.triggered_rows, set())


class AlignmentPreservationTest(unittest.TestCase):
    def test_subframe_segment_uses_complete_uniform_fallback(self) -> None:
        words, fallback_count = align_words(
            np.zeros((1, 32), dtype=np.float32),
            20,
            fake_alignment_tokenizer(),
            [(0.0, 0.01)],
            ["كلمتان هنا"],
            "ar",
        )

        self.assertEqual([word["text"] for word in words], ["كلمتان", "هنا"])
        self.assertEqual(
            {word["timing_source"] for word in words}, {"uniform_fallback"}
        )
        self.assertEqual(fallback_count, 1)


class EmissionAndAlignmentRegressionTest(unittest.TestCase):
    def test_bounded_assembly_is_bit_exact_with_previous_full_concat(self) -> None:
        class FakeAligner:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(inputs_to_logits_ratio=320)

            def __call__(self, values: torch.Tensor) -> SimpleNamespace:
                batch_size = values.shape[0]
                frames = torch.linspace(-0.2, 0.2, 1_700).view(1, -1, 1)
                vocabulary = torch.linspace(-1.0, 1.0, 7).view(1, 1, -1)
                logits = (frames + vocabulary).expand(batch_size, -1, -1).clone()
                return SimpleNamespace(logits=logits)

        def time_to_frame(seconds: float) -> int:
            return int(round(seconds * 50))

        def previous_assembly(
            audio: np.ndarray, model: FakeAligner
        ) -> tuple[np.ndarray, int]:
            window_samples = ALIGN_WINDOW_S * SR
            context_samples = ALIGN_CONTEXT_S * SR
            total_windows = math.ceil(len(audio) / window_samples)
            extension_samples = total_windows * window_samples - len(audio)
            batches: list[np.ndarray] = []
            for first_window in range(0, total_windows, 2):
                indices = range(first_window, min(first_window + 2, total_windows))
                inputs = build_alignment_window_batch(
                    audio,
                    indices,
                    window_samples,
                    context_samples,
                )
                batches.append(model(torch.from_numpy(inputs)).logits.numpy())
            emissions = np.concatenate(batches, axis=0)
            context_frames = time_to_frame(ALIGN_CONTEXT_S)
            window_frames = ALIGN_WINDOW_S * 50
            emissions = emissions[:, context_frames : context_frames + window_frames, :]
            emissions = emissions.reshape(-1, emissions.shape[-1])
            extension_frames = time_to_frame(extension_samples / SR)
            if extension_frames:
                emissions = emissions[:-extension_frames]
            emissions = torch.log_softmax(
                torch.from_numpy(emissions).float(), dim=-1
            ).numpy()
            emissions = np.concatenate(
                [emissions, np.zeros((emissions.shape[0], 1))],
                axis=1,
            ).astype(np.float32)
            stride = 20.0
            return emissions, stride

        audio = np.linspace(-0.1, 0.1, 65 * SR, dtype=np.float32)
        model = FakeAligner()
        expected, expected_stride = previous_assembly(audio, model)
        with contextlib.redirect_stderr(io.StringIO()):
            actual, actual_stride = _compute_emissions_streaming(
                audio, model, 2, "test"
            )

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_stride, expected_stride)

    def test_partial_aligner_result_falls_back_to_complete_asr_text(self) -> None:
        with (
            mock.patch(
                "cohere_transcribe.alignment.text_utils.preprocess_text",
                return_value=(["a", "b"], ["كلمتان", "هنا"]),
            ),
            mock.patch(
                "cohere_transcribe.alignment.alignment_utils.get_spans",
                return_value=[],
            ),
            mock.patch(
                "cohere_transcribe.alignment.text_utils.postprocess_results",
                return_value=[{"start": 0.0, "end": 0.5, "text": "كلمتان"}],
            ),
            mock.patch(
                "cohere_transcribe.alignment.runtime.get_alignments_safe",
                return_value=([], "<blank>"),
            ),
        ):
            words, fallback_count = align_words(
                np.zeros((100, 32), dtype=np.float32),
                20,
                fake_alignment_tokenizer(),
                [(0.0, 1.0)],
                ["كلمتان هنا"],
                "ar",
            )

        self.assertEqual([word["text"] for word in words], ["كلمتان", "هنا"])
        self.assertEqual(fallback_count, 1)


class OutputTransactionTest(unittest.TestCase):
    def test_txt_and_json_use_canonical_asr_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clip.wav"
            source.write_bytes(b"audio snapshot")
            txt_path = root / "clip.txt"
            json_path = root / "clip.json"
            job = make_job(source, {"txt": txt_path, "json": json_path})
            job.vad_engine_requested = "auto"
            job.vad_engine_actual = "onnx"
            job.vad_merge = True
            job.segmentation_parameters = {
                "max_duration_seconds": 30.0,
                "threshold": 0.5,
            }
            cues = [{"start": 0.0, "end": 1.0, "text": "wrong aligned text"}]

            atomic_write_outputs(job, cues, [])

            self.assertEqual(txt_path.read_text(encoding="utf-8"), "النص الاصلي\n")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["transcript"], ["النص الاصلي"])
            self.assertEqual(
                payload["models"]["asr"],
                {"id": MODEL_ID, "revision": ASR_MODEL_REVISION},
            )
            self.assertEqual(payload["schema_version"], 7)
            self.assertEqual(payload["repetition_stopped_segments"], [])
            self.assertEqual(
                payload["segmentation_details"],
                {
                    "mode": "silero",
                    "requested_engine": "auto",
                    "actual_engine": "onnx",
                    "provider": None,
                    "provider_options": None,
                    "fallback_reason": None,
                    "merge": True,
                    "parameters": {
                        "max_duration_seconds": 30.0,
                        "threshold": 0.5,
                    },
                    "speech_spans": [],
                },
            )

    def test_keyboard_interrupt_restores_every_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clip.wav"
            source.write_bytes(b"audio snapshot")
            txt_path = root / "clip.txt"
            srt_path = root / "clip.srt"
            txt_path.write_text("old text\n", encoding="utf-8")
            srt_path.write_text("old subtitles\n", encoding="utf-8")
            job = make_job(source, {"txt": txt_path, "srt": srt_path})
            real_replace = os.replace

            def interrupt_second_publish(source_path, destination_path):
                source_path = Path(source_path)
                destination_path = Path(destination_path)
                if source_path.suffix == ".tmp" and destination_path == srt_path:
                    raise KeyboardInterrupt
                return real_replace(source_path, destination_path)

            with (
                mock.patch(
                    "cohere_transcribe.output.publication.os.replace",
                    side_effect=interrupt_second_publish,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                atomic_write_outputs(
                    job,
                    [{"start": 0.0, "end": 1.0, "text": "new text"}],
                    [],
                )

            self.assertEqual(txt_path.read_text(encoding="utf-8"), "old text\n")
            self.assertEqual(srt_path.read_text(encoding="utf-8"), "old subtitles\n")
            self.assertEqual(job.written, [])
            self.assertEqual(list(root.glob(".*.tmp")), [])
            self.assertEqual(list(root.glob(".*.bak")), [])

    def test_aligner_load_failure_does_not_publish_a_partial_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clip.wav"
            source.write_bytes(b"audio snapshot")
            txt_path = root / "clip.txt"
            srt_path = root / "clip.srt"
            txt_path.write_text("old text\n", encoding="utf-8")
            srt_path.write_text("old subtitles\n", encoding="utf-8")
            job = make_job(source, {"txt": txt_path, "srt": srt_path})
            job.audio = np.zeros(SR, dtype=np.float32)
            args = parse_args([os.fspath(source), "--alignment", "word"])
            validate_args(args)

            with mock.patch(
                "cohere_transcribe.output.pipeline.load_aligner",
                side_effect=RuntimeError("model unavailable"),
            ):
                align_and_write_all(
                    [job],
                    args,
                    "cpu",
                    torch.float32,
                    RunStats(),
                )

            self.assertEqual(txt_path.read_text(encoding="utf-8"), "old text\n")
            self.assertEqual(srt_path.read_text(encoding="utf-8"), "old subtitles\n")
            self.assertIn("aligner load failed", job.error or "")
            self.assertIsNone(job.audio)


if __name__ == "__main__":
    unittest.main()
