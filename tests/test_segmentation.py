from __future__ import annotations

import unittest
from argparse import Namespace
from itertools import pairwise

import numpy as np

from cohere_transcribe.audio.segmentation import (
    merge_speech_segments,
    segment_audio_auditok,
    segment_audio_fixed,
    validate_processor_single_row_window,
)
from cohere_transcribe.config import validate_args
from cohere_transcribe.models import ASR_FIXED_MIN_S, SR


def valid_args(**overrides) -> Namespace:
    values = {
        "formats": None,
        "text_only": True,
        "alignment": "segment",
        "audio_memory_gb": 4.0,
        "preprocess_workers": None,
        "vad_engine": "auto",
        "vad_batch_size": 16,
        "vad_block_frames": 512,
        "vad_threads": None,
        "batch_size": None,
        "batch_max_size": None,
        "batch_audio_seconds": None,
        "batch_vram_target": 0.9,
        "adaptive_batch": False,
        "pin_memory": False,
        "align_batch_size": 4,
        "vad": "none",
        "vad_merge": False,
        "min_dur": 0.5,
        "max_dur": 30.0,
        "vad_threshold": 0.5,
        "min_silence_ms": 300,
        "speech_pad_ms": 60,
        "max_silence": 0.6,
        "energy_threshold": 50.0,
        "max_new_tokens": 445,
        "max_retry_tokens": 896,
        "truncation_policy": "retry",
        "max_chars": 80,
        "max_cue_dur": 6.0,
        "max_gap": 0.6,
    }
    values.update(overrides)
    return Namespace(**values)


class FixedWindowSegmentationTest(unittest.TestCase):
    def test_vad_cut_and_merge_retains_timeline_gaps(self) -> None:
        segments = [(0.2, 4.0), (4.4, 9.8), (15.0, 24.0), (24.5, 31.0)]

        self.assertEqual(
            merge_speech_segments(segments, 10.0),
            [(0.2, 9.8), (15.0, 24.0), (24.5, 31.0)],
        )

    def test_vad_cut_and_merge_rejects_overlapping_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted and non-overlapping"):
            merge_speech_segments([(0.0, 2.0), (1.0, 3.0)], 30.0)

    def test_windows_cover_waveform_once_without_gaps(self) -> None:
        total_samples = 65 * SR + SR // 4
        audio = np.zeros(total_samples, dtype=np.float32)

        segments = segment_audio_fixed(audio, 30.0)

        self.assertEqual(segments, [(0.0, 30.0), (30.0, 60.0), (60.0, 65.25)])
        sample_ranges = [
            (int(round(start * SR)), int(round(end * SR))) for start, end in segments
        ]
        self.assertEqual(sample_ranges[0][0], 0)
        self.assertEqual(sample_ranges[-1][1], total_samples)
        self.assertTrue(
            all(left[1] == right[0] for left, right in pairwise(sample_ranges))
        )

    def test_empty_waveform_has_no_windows(self) -> None:
        self.assertEqual(segment_audio_fixed(np.empty(0, dtype=np.float32), 30.0), [])

    def test_fractional_window_is_sample_accurate(self) -> None:
        audio = np.zeros(5_000, dtype=np.float32)
        window_seconds = 1_001 / SR

        segments = segment_audio_fixed(audio, window_seconds)

        self.assertEqual(
            [(round(start * SR), round(end * SR)) for start, end in segments],
            [
                (0, 1_001),
                (1_001, 2_002),
                (2_002, 3_003),
                (3_003, 4_004),
                (4_004, 5_000),
            ],
        )

    def test_none_mode_allows_runtime_checked_35_second_windows(self) -> None:
        args = valid_args(max_dur=35.0)
        validate_args(args)
        self.assertEqual(args.max_dur, 35.0)

    def test_none_mode_rejects_pathologically_small_windows(self) -> None:
        with self.assertRaisesRegex(SystemExit, r"--max-dur >= 1 second"):
            validate_args(valid_args(min_dur=0.0, max_dur=ASR_FIXED_MIN_S - 0.01))

    def test_loaded_processor_limit_is_checked(self) -> None:
        class FeatureExtractor:
            max_audio_clip_s = 35.0
            overlap_chunk_second = 5.0

        class Processor:
            feature_extractor = FeatureExtractor()

        self.assertEqual(validate_processor_single_row_window(Processor(), 35.0), 35.0)
        with self.assertRaisesRegex(RuntimeError, r"35.1s exceeds.*35s"):
            validate_processor_single_row_window(Processor(), 35.1)

    def test_other_vad_engines_keep_existing_long_window_behavior(self) -> None:
        args = valid_args(vad="silero", max_dur=31.0)
        validate_args(args)
        self.assertEqual(args.alignment, "none")


class AuditokSegmentationTest(unittest.TestCase):
    def test_tone_between_silence_uses_the_supported_core_api(self) -> None:
        audio = np.zeros(SR, dtype=np.float32)
        start_sample = int(0.2 * SR)
        end_sample = int(0.7 * SR)
        phase = np.arange(end_sample - start_sample, dtype=np.float32)
        audio[start_sample:end_sample] = 0.5 * np.sin(2 * np.pi * 440 * phase / SR)

        segments = segment_audio_auditok(
            audio,
            min_dur=0.1,
            max_dur=2.0,
            max_silence=0.05,
            energy_threshold=50.0,
        )

        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0][0], 0.2, places=2)
        self.assertAlmostEqual(segments[0][1], 0.75, places=2)


if __name__ == "__main__":
    unittest.main()
