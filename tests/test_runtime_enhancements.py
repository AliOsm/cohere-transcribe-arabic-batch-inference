from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from cohere_transcribe.alignment.runtime import (
    _compute_emissions_streaming,
    uniform_word_timings_across_spans,
)
from cohere_transcribe.asr.batching import ASRBatchController
from cohere_transcribe.asr.execution import (
    apply_generation_metadata,
    finish_asr_batch,
    retry_token_limit,
    transcribe_ref_batch,
)
from cohere_transcribe.asr.generation import (
    ASRGenerationResult,
    PreparedASRBatch,
    RepetitionLoopStoppingCriteria,
    analyze_generated_rows,
)
from cohere_transcribe.asr.model import (
    MemoizedEncoderProjection,
    prepare_encoder_attention_mask_once,
)
from cohere_transcribe.audio.segmentation import sample_timestamps_to_seconds
from cohere_transcribe.config import (
    parse_args,
    validate_args,
)
from cohere_transcribe.inputs import build_jobs
from cohere_transcribe.models import SR, SegmentRef
from cohere_transcribe.preflight import preflight_forced_align, preflight_runtime
from cohere_transcribe.profiling import (
    validate_profile_output_path,
    write_profile_json,
)
from cohere_transcribe.vad.runtime import (
    SileroBackendUnavailable,
    SileroRuntime,
    get_silero_runtime,
    onnx_provider_details,
    segment_audio_silero,
)


def refs_with_duration(count: int, duration: float) -> list[SegmentRef]:
    job = SimpleNamespace(index=0)
    return [SegmentRef(job, index, 0.0, duration) for index in range(count)]


class AdaptiveBatchControllerTest(unittest.TestCase):
    def test_batch_max_requires_explicit_adaptive_mode(self) -> None:
        args = parse_args(
            ["audio.wav", "--alignment", "none", "--batch-max-size", "48"]
        )
        with self.assertRaisesRegex(SystemExit, "requires --adaptive-batch"):
            validate_args(args)

    def test_static_default_and_persistent_oom_learning(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none", "--batch-size", "4"])
        validate_args(args)
        model = SimpleNamespace(device=torch.device("cpu"))
        refs = refs_with_duration(12, 2.0)

        controller = ASRBatchController.create(args, model, refs)
        self.assertFalse(controller.adaptive)
        self.assertEqual((controller.current_size, controller.max_size), (4, 4))

        controller.record_oom(4)
        self.assertEqual((controller.current_size, controller.max_size), (2, 3))
        pending = deque(refs)
        self.assertEqual(len(controller.take(pending)), 2)

    def test_automatic_audio_budget_is_refreshed_for_each_group(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none", "--batch-size", "4"])
        validate_args(args)
        model = SimpleNamespace(device=torch.device("cpu"))
        controller = ASRBatchController.create(args, model, refs_with_duration(4, 0.5))
        self.assertEqual(controller.audio_budget_seconds, 2.0)

        controller.configure_group(args, refs_with_duration(4, 30.0))
        self.assertEqual(controller.audio_budget_seconds, 120.0)


class DecoderSafetyTest(unittest.TestCase):
    def test_full_budget_without_eos_is_reported_per_reference(self) -> None:
        generated = torch.tensor(
            [
                [7, 8, 10, 3, 2, 2],
                [7, 8, 20, 21, 22, 23],
                [7, 8, 30, 31, 2, 2],
            ]
        )
        counts, truncated, repetition = analyze_generated_rows(
            generated=generated,
            prompt_length=2,
            max_new_tokens=4,
            eos_token_ids=(3,),
            pad_token_id=2,
            repetition_rows={2},
            chunk_index=[(0, None), (1, None), (2, None)],
        )

        self.assertEqual(counts, [2, 4, 2])
        self.assertEqual(truncated, {1})
        self.assertEqual(repetition, {2})

    def test_retry_limit_uses_top_level_context_when_decoder_config_is_none(
        self,
    ) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(
                decoder_config=None,
                max_position_embeddings=100,
            )
        )
        self.assertEqual(retry_token_limit(model, 10, 70, 200), 90)

    def test_repetition_guard_checks_every_generated_token(self) -> None:
        criterion = RepetitionLoopStoppingCriteria(prompt_length=1)
        generated = torch.arange(8, dtype=torch.long).repeat(12)
        input_ids = torch.cat((torch.zeros(1, dtype=torch.long), generated)).unsqueeze(
            0
        )
        self.assertEqual(criterion(input_ids, None).tolist(), [True])

    def test_only_truncated_references_are_retried(self) -> None:
        job = SimpleNamespace(
            segment_texts=["", ""],
            generated_tokens={},
            repetition_stopped_segments=set(),
            truncation_retried_segments=set(),
            token_limit_segments=set(),
        )
        refs = [
            SegmentRef(job, 0, 0.0, 1.0),
            SegmentRef(job, 1, 1.0, 2.0),
        ]
        prepared = PreparedASRBatch(
            refs=refs,
            model_inputs={},
            chunk_index=[(0, None), (1, None)],
            prepare_seconds=0.0,
            valid_feature_frames=2,
            padded_feature_frames=2,
        )
        result = ASRGenerationResult(
            generated=torch.tensor([[1, 2, 3], [1, 2, 4]]),
            row_token_counts=[2, 445],
            truncated_ref_indices={1},
            repetition_ref_indices=set(),
            max_new_tokens=445,
            prompt_length=10,
            call_wall_seconds=0.1,
            device_generate_seconds=0.1,
            analysis_seconds=0.0,
        )
        processor = SimpleNamespace(
            batch_decode=lambda *_args, **_kwargs: ["ok", "cut"]
        )
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024))
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)
        bar = mock.Mock()

        def complete_retry(*_args, **_kwargs) -> None:
            retry_prepared = PreparedASRBatch(
                refs=[refs[1]],
                model_inputs={},
                chunk_index=[(0, None)],
                prepare_seconds=0.0,
                valid_feature_frames=1,
                padded_feature_frames=1,
            )
            retry_result = ASRGenerationResult(
                generated=torch.tensor([[1, 2, 3]]),
                row_token_counts=[180],
                truncated_ref_indices=set(),
                repetition_ref_indices=set(),
                max_new_tokens=896,
                prompt_length=10,
                call_wall_seconds=0.1,
                device_generate_seconds=0.1,
                analysis_seconds=0.0,
            )
            apply_generation_metadata(retry_prepared, retry_result)

        with mock.patch(
            "cohere_transcribe.asr.execution.transcribe_ref_batch",
            side_effect=complete_retry,
        ) as retry:
            finish_asr_batch(
                processor,
                model,
                prepared,
                result,
                args,
                bar,
                SimpleNamespace(
                    asr_decode_seconds=0.0,
                    asr_truncation_retries=0,
                ),
                mock.Mock(),
            )

        self.assertEqual(job.segment_texts, ["ok", ""])
        self.assertEqual(job.generated_tokens, {0: 2, 1: 180})
        self.assertEqual(job.truncation_retried_segments, {1})
        bar.update.assert_called_once_with(1)
        self.assertEqual(retry.call_args.args[2], [refs[1]])
        self.assertEqual(retry.call_args.kwargs["max_new_tokens"], 896)

    def test_high_token_retry_does_not_grow_base_batch_controller(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)
        controller = mock.Mock()
        prepared = mock.Mock()
        result = mock.Mock()
        refs = refs_with_duration(1, 1.0)

        with (
            mock.patch(
                "cohere_transcribe.asr.execution.prepare_asr_batch",
                return_value=prepared,
            ),
            mock.patch(
                "cohere_transcribe.asr.execution.generate_asr_batch",
                return_value=result,
            ),
            mock.patch("cohere_transcribe.asr.execution.record_prepared_batch"),
            mock.patch("cohere_transcribe.asr.execution.record_generation_batch"),
            mock.patch("cohere_transcribe.asr.execution.finish_asr_batch"),
        ):
            transcribe_ref_batch(
                mock.Mock(),
                SimpleNamespace(device=torch.device("cpu")),
                refs,
                args,
                mock.Mock(),
                mock.Mock(),
                controller,
                max_new_tokens=890,
            )

        controller.record_success.assert_not_called()


class ModelHotPathTest(unittest.TestCase):
    def test_encoder_projection_is_reused_only_for_the_same_tensor(self) -> None:
        class CountingProjection(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return value + 1

        projection = CountingProjection()
        memoized = MemoizedEncoderProjection(projection)
        first_source = torch.zeros(1, 2, 3)

        first = memoized(first_source)
        second = memoized(first_source)
        third = memoized(first_source.clone())

        self.assertIs(first, second)
        self.assertEqual(projection.calls, 2)
        torch.testing.assert_close(third, torch.ones_like(third))

    def test_encoder_mask_is_rewritten_once_for_decoder_sdpa(self) -> None:
        full = SimpleNamespace(attention_mask=torch.ones(2, 3, dtype=torch.int64))
        prepare_encoder_attention_mask_once(None, (), full)
        self.assertIsNone(full.attention_mask)

        padded = SimpleNamespace(attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]))
        prepare_encoder_attention_mask_once(None, (), padded)
        self.assertEqual(padded.attention_mask.shape, (2, 1, 1, 3))
        self.assertEqual(padded.attention_mask.dtype, torch.bool)


class VadRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        from cohere_transcribe.vad.runtime import _vad_thread_local

        for name in (
            "runtimes",
            "onnx_fallback_error",
            "onnx_fallback_reported",
        ):
            if hasattr(_vad_thread_local, name):
                delattr(_vad_thread_local, name)

    tearDown = setUp

    def test_explicit_jit_never_imports_custom_onnx_module(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none", "--vad-engine", "jit"])
        validate_args(args)
        with mock.patch(
            "cohere_transcribe.vad.runtime.build_silero_onnx_runtime"
        ) as onnx:
            runtime = get_silero_runtime("jit", args)
        self.assertEqual(runtime.engine, "jit")
        onnx.assert_not_called()

    def test_auto_falls_back_only_for_backend_unavailability(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)
        jit_runtime = SileroRuntime(object(), "jit", lambda *_args: [])
        with (
            mock.patch(
                "cohere_transcribe.vad.runtime.build_silero_onnx_runtime",
                side_effect=SileroBackendUnavailable("missing runtime"),
            ),
            mock.patch(
                "cohere_transcribe.vad.runtime.build_silero_jit_runtime",
                return_value=jit_runtime,
            ),
        ):
            runtime = get_silero_runtime("auto", args)
        self.assertEqual(runtime.engine, "jit")
        self.assertIs(runtime.model, jit_runtime.model)

    def test_auto_falls_back_when_onnxruntime_dependency_is_missing(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)
        jit_runtime = SileroRuntime(object(), "jit", lambda *_args: [])
        with (
            mock.patch(
                "cohere_transcribe.vad.vectorized_silero.VectorizedSileroVAD",
                side_effect=ModuleNotFoundError("No module named 'onnxruntime'"),
            ),
            mock.patch(
                "cohere_transcribe.vad.runtime.build_silero_jit_runtime",
                return_value=jit_runtime,
            ),
        ):
            runtime = get_silero_runtime("auto", args)
        self.assertEqual(runtime.engine, "jit")
        self.assertIs(runtime.model, jit_runtime.model)

    def test_provider_names_and_options_are_recorded(self) -> None:
        session = SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"],
            get_provider_options=lambda: {
                "CPUExecutionProvider": {"arena_extend_strategy": "kNextPowerOfTwo"}
            },
        )
        provider, options = onnx_provider_details(SimpleNamespace(session=session))
        self.assertEqual(provider, "CPUExecutionProvider")
        self.assertEqual(
            options,
            {"CPUExecutionProvider": {"arena_extend_strategy": "kNextPowerOfTwo"}},
        )

    def test_vad_sample_indices_are_not_rounded_to_100ms(self) -> None:
        spans = sample_timestamps_to_seconds(
            [{"start": 1, "end": 513}], audio_samples=1024
        )
        self.assertEqual(spans, [(1 / SR, 513 / SR)])

    def test_runtime_fallback_reason_is_retained_for_later_files(self) -> None:
        from cohere_transcribe.vad.runtime import _vad_thread_local

        def fail_onnx(*_args) -> list[dict[str, int]]:
            raise SileroBackendUnavailable("session failed")

        onnx_runtime = SileroRuntime(object(), "onnx", fail_onnx)
        jit_runtime = SileroRuntime(object(), "jit", lambda *_args: [])
        _vad_thread_local.runtimes = {("auto", 16, 512): onnx_runtime}
        args = parse_args(["audio.wav", "--alignment", "none"])
        validate_args(args)

        with mock.patch(
            "cohere_transcribe.vad.runtime.build_silero_jit_runtime",
            return_value=jit_runtime,
        ):
            first = segment_audio_silero(np.zeros(SR, dtype=np.float32), args)
            second = segment_audio_silero(np.zeros(SR, dtype=np.float32), args)

        self.assertIn("session failed", first[-1] or "")
        self.assertEqual(second[-1], first[-1])


class ApproximateTimingTest(unittest.TestCase):
    def test_words_are_distributed_across_speech_not_known_silence(self) -> None:
        words = uniform_word_timings_across_spans(
            "one two three four",
            [(0.0, 1.0), (3.0, 4.0)],
            0,
            "uniform_speech_spans",
        )
        self.assertEqual(
            [word["text"] for word in words], ["one", "two", "three", "four"]
        )
        self.assertEqual(words[1]["end"], 1.0)
        self.assertEqual(words[2]["start"], 3.0)


class AlignmentNumericsTest(unittest.TestCase):
    def test_large_logits_produce_finite_emissions(self) -> None:
        class FakeAligner:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(inputs_to_logits_ratio=320)

            def __call__(self, values: torch.Tensor) -> SimpleNamespace:
                row = torch.tensor([1000.0, 999.0, -1000.0]).view(1, 1, 3)
                frames = values.shape[1] // 320
                return SimpleNamespace(logits=row.expand(values.shape[0], frames, 3))

        audio = np.zeros(30 * SR, dtype=np.float32)
        with contextlib.redirect_stderr(io.StringIO()):
            emissions, _stride = _compute_emissions_streaming(
                audio, FakeAligner(), 2, "finite"
            )
        self.assertTrue(np.isfinite(emissions).all())

    def test_alignment_oom_batch_cap_persists_on_model(self) -> None:
        class FakeAligner:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(inputs_to_logits_ratio=320)

            def __init__(self) -> None:
                self.batch_calls: list[int] = []

            def __call__(self, values: torch.Tensor) -> SimpleNamespace:
                self.batch_calls.append(values.shape[0])
                if values.shape[0] > 2:
                    raise RuntimeError("DefaultCPUAllocator: cannot allocate memory")
                return SimpleNamespace(
                    logits=torch.zeros(
                        values.shape[0], values.shape[1] // 320, 3, dtype=torch.float32
                    )
                )

        model = FakeAligner()
        with contextlib.redirect_stderr(io.StringIO()):
            _compute_emissions_streaming(
                np.zeros(120 * SR, dtype=np.float32), model, 4, "first"
            )
            _compute_emissions_streaming(
                np.zeros(60 * SR, dtype=np.float32), model, 4, "second"
            )
        self.assertEqual(model.batch_calls[0:2], [4, 2])
        self.assertEqual(model.batch_calls[-1], 2)
        self.assertEqual(model._transcribe_align_batch_size, 2)

    def test_alignment_retries_host_batch_allocation_failure(self) -> None:
        class FakeAligner:
            device = torch.device("cpu")
            dtype = torch.float32
            config = SimpleNamespace(inputs_to_logits_ratio=320)

            def __call__(self, values: torch.Tensor) -> SimpleNamespace:
                return SimpleNamespace(
                    logits=torch.zeros(
                        values.shape[0], values.shape[1] // 320, 3, dtype=torch.float32
                    )
                )

        from cohere_transcribe.alignment import runtime as alignment_runtime

        original = alignment_runtime.build_alignment_window_batch
        calls: list[int] = []

        def allocate(audio, indices, window_samples, context_samples):
            calls.append(len(indices))
            if len(calls) == 1:
                raise MemoryError("synthetic host allocation failure")
            return original(audio, indices, window_samples, context_samples)

        model = FakeAligner()
        with (
            mock.patch.object(
                alignment_runtime,
                "build_alignment_window_batch",
                side_effect=allocate,
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            _compute_emissions_streaming(
                np.zeros(120 * SR, dtype=np.float32), model, 4, "host-oom"
            )

        self.assertEqual(calls[:2], [4, 2])
        self.assertEqual(model._transcribe_align_batch_size, 2)


class PreflightAndProfileTest(unittest.TestCase):
    def test_missing_forced_align_fails_preflight(self) -> None:
        with (
            mock.patch("torchaudio.functional.forced_align", None),
            self.assertRaisesRegex(SystemExit, "forced alignment is unavailable"),
        ):
            preflight_forced_align()

    def test_explicit_onnx_preflight_does_not_require_jit_package(self) -> None:
        args = parse_args(["audio.wav", "--alignment", "none", "--vad-engine", "onnx"])
        validate_args(args)
        imported: list[str] = []

        def record_import(name: str):
            imported.append(name)
            return object()

        with mock.patch(
            "cohere_transcribe.preflight.importlib.import_module",
            side_effect=record_import,
        ):
            preflight_runtime(args)

        self.assertNotIn("silero_vad", imported)
        self.assertIn("cohere_transcribe.vad.vectorized_silero", imported)
        self.assertIn("onnxruntime", imported)

    def test_profile_cannot_overwrite_skipped_input_or_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clip.wav"
            source.write_bytes(b"audio")
            output = root / "clip.txt"
            output.write_text("existing\n", encoding="utf-8")

            for profile in (source, output):
                args = parse_args(
                    [
                        os.fspath(source),
                        "--alignment",
                        "none",
                        "--existing",
                        "skip",
                        "--profile-json",
                        os.fspath(profile),
                    ]
                )
                validate_args(args)
                with self.assertRaisesRegex(SystemExit, "Profile path collides"):
                    build_jobs(args)

    def test_profile_is_atomically_written_as_finite_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            write_profile_json(path, {"seconds": 1.25})
            self.assertEqual(
                path.read_text(encoding="utf-8"), '{\n  "seconds": 1.25\n}\n'
            )
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_profile_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "unrelated.json"
            target.write_text("keep\n", encoding="utf-8")
            link = root / "profile.json"
            link.symlink_to(target)

            with self.assertRaisesRegex(SystemExit, "must not be a symlink"):
                validate_profile_output_path(os.fspath(link), [])
            self.assertEqual(target.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
