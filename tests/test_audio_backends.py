from __future__ import annotations

import importlib.util
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from cohere_transcribe import inputs
from cohere_transcribe.audio import backends as audio_backends
from cohere_transcribe.audio import decoding
from cohere_transcribe.cancellation import request_cancellation, reset_cancellation


class AudioBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_cancellation()
        audio_backends.probe_torchcodec.cache_clear()

    def tearDown(self) -> None:
        reset_cancellation()
        audio_backends.probe_torchcodec.cache_clear()

    def test_auto_prefers_working_torchcodec(self) -> None:
        with (
            mock.patch.object(
                audio_backends, "torchcodec_is_usable", return_value=True
            ),
            mock.patch.object(
                audio_backends.shutil, "which", return_value="/usr/bin/ffmpeg"
            ),
        ):
            self.assertEqual(audio_backends.resolve_audio_backend("auto"), "torchcodec")

    def test_auto_uses_ffmpeg_without_working_torchcodec(self) -> None:
        with (
            mock.patch.object(
                audio_backends, "torchcodec_is_usable", return_value=False
            ),
            mock.patch.object(
                audio_backends.shutil, "which", return_value="/usr/bin/ffmpeg"
            ),
        ):
            self.assertEqual(audio_backends.resolve_audio_backend("auto"), "ffmpeg")

    def test_auto_rejects_missing_decoders(self) -> None:
        with (
            mock.patch.object(
                audio_backends, "torchcodec_is_usable", return_value=False
            ),
            mock.patch.object(audio_backends.shutil, "which", return_value=None),
            self.assertRaisesRegex(RuntimeError, "working TorchCodec.*ffmpeg"),
        ):
            audio_backends.resolve_audio_backend("auto")

    def test_broken_torchcodec_is_not_usable(self) -> None:
        with mock.patch.object(
            audio_backends.importlib,
            "import_module",
            side_effect=RuntimeError("incompatible native wheel"),
        ):
            self.assertFalse(audio_backends.torchcodec_is_usable())

    def test_supported_torchcodec_is_usable(self) -> None:
        with (
            mock.patch.object(
                audio_backends.importlib, "import_module", return_value=object()
            ),
            mock.patch.object(
                audio_backends.importlib_metadata,
                "version",
                return_value="0.14.0+cpu",
            ),
        ):
            self.assertTrue(audio_backends.torchcodec_is_usable())

    def test_too_old_torchcodec_is_not_usable(self) -> None:
        with (
            mock.patch.object(
                audio_backends.importlib, "import_module", return_value=object()
            ),
            mock.patch.object(
                audio_backends.importlib_metadata, "version", return_value="0.2.1"
            ),
        ):
            self.assertFalse(audio_backends.torchcodec_is_usable())

    def test_torchcodec_without_distribution_metadata_is_not_usable(self) -> None:
        with (
            mock.patch.object(
                audio_backends.importlib, "import_module", return_value=object()
            ),
            mock.patch.object(
                audio_backends.importlib_metadata,
                "version",
                side_effect=audio_backends.importlib_metadata.PackageNotFoundError,
            ),
        ):
            self.assertFalse(audio_backends.torchcodec_is_usable())

    def test_duration_hint_is_rejected_before_decoder_initialization(self) -> None:
        resolver = mock.Mock(return_value="torchcodec")
        with (
            mock.patch.object(decoding, "resolve_audio_backend", resolver),
            self.assertRaisesRegex(decoding.DecodedAudioLimitError, "per-file limit"),
        ):
            decoding.decode_audio_resolved(
                Path("oversized.wav"),
                "auto",
                duration_hint=10.0,
                max_decoded_bytes=1024,
            )
        resolver.assert_not_called()

    def test_ffmpeg_is_not_spawned_after_cancellation(self) -> None:
        request_cancellation()
        with (
            mock.patch.object(decoding.subprocess, "Popen") as popen,
            self.assertRaises(KeyboardInterrupt),
        ):
            decoding.load_audio_ffmpeg(Path("cancelled.wav"))
        popen.assert_not_called()

    def test_ffprobe_is_killed_and_reaped_after_communicate_error(self) -> None:
        process = mock.Mock()
        process.stdout = mock.Mock()
        process.stderr = mock.Mock()
        process.communicate.side_effect = OSError("pipe read failed")
        with (
            mock.patch.object(inputs.shutil, "which", return_value="/usr/bin/ffprobe"),
            mock.patch.object(inputs.subprocess, "Popen", return_value=process),
        ):
            self.assertIsNone(inputs.probe_duration(Path("broken.wav")))

        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)
        process.stdout.close.assert_called_once_with()
        process.stderr.close.assert_called_once_with()

    def test_injected_decoder_output_is_checked_against_limit(self) -> None:
        with (
            mock.patch.object(decoding, "resolve_audio_backend", return_value="ffmpeg"),
            mock.patch.object(
                decoding,
                "load_audio_ffmpeg",
                return_value=np.zeros(1024, dtype=np.float32),
            ),
            self.assertRaisesRegex(decoding.DecodedAudioLimitError, "per-file limit"),
        ):
            decoding.decode_audio_resolved(
                Path("oversized.wav"),
                "ffmpeg",
                max_decoded_bytes=1024,
            )

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_real_ffmpeg_stream_is_stopped_at_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\0\0" * 16_000)
            with self.assertRaisesRegex(
                decoding.DecodedAudioLimitError, "FFmpeg output.*per-file limit"
            ):
                decoding.load_audio_ffmpeg(path, max_decoded_bytes=1024)

    @unittest.skipUnless(
        importlib.util.find_spec("torchcodec") is not None,
        "TorchCodec is not installed",
    )
    def test_real_torchcodec_metadata_is_checked_before_pcm_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(b"\0\0" * 16_000)
            with self.assertRaisesRegex(
                decoding.DecodedAudioLimitError, "per-file limit"
            ):
                decoding.load_audio_torchcodec(path, max_decoded_bytes=1024)

    @unittest.skipUnless(
        importlib.util.find_spec("torchcodec") is not None,
        "TorchCodec is not installed",
    )
    def test_torchcodec_range_bounds_output_when_metadata_underreports(self) -> None:
        decoder = mock.Mock()
        decoder.metadata.duration_seconds = 0.001
        decoder.get_samples_played_in_range.return_value.data = torch.zeros(
            (1, 257), dtype=torch.float32
        )

        with (
            mock.patch("torchcodec.decoders.AudioDecoder", return_value=decoder),
            self.assertRaisesRegex(
                decoding.DecodedAudioLimitError, "exceeding the.*per-file limit"
            ),
        ):
            decoding.load_audio_torchcodec(
                Path("underreported.wav"), max_decoded_bytes=1024
            )

        decoder.get_all_samples.assert_not_called()
        decoder.get_samples_played_in_range.assert_called_once_with(
            start_seconds=0.0,
            stop_seconds=257 / 16_000,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torchcodec") is not None,
        "TorchCodec is not installed",
    )
    def test_torchcodec_bounds_the_playable_stream_timeline(self) -> None:
        decoder = mock.Mock()
        decoder.metadata.duration_seconds_from_header = None
        decoder.metadata.duration_seconds = 6.0
        decoder.metadata.begin_stream_seconds = 5.0
        decoder.get_samples_played_in_range.return_value.data = torch.zeros(
            (1, 16_000), dtype=torch.float32
        )

        with mock.patch("torchcodec.decoders.AudioDecoder", return_value=decoder):
            decoded = decoding.load_audio_torchcodec(
                Path("offset.wav"), max_decoded_bytes=2 * 16_000 * 4
            )

        self.assertEqual(len(decoded), 16_000)
        decoder.get_samples_played_in_range.assert_called_once_with(
            start_seconds=5.0,
            stop_seconds=5.0 + (2 * 16_000 + 1) / 16_000,
        )

    @unittest.skipUnless(
        shutil.which("ffmpeg")
        and shutil.which("ffprobe")
        and importlib.util.find_spec("torchcodec") is not None,
        "FFmpeg, FFprobe, and TorchCodec are required",
    )
    def test_offset_media_uses_playable_duration_and_preserves_decoder_parity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offset.mkv"
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:sample_rate=16000:duration=1",
                    "-c:a",
                    "pcm_s16le",
                    "-output_ts_offset",
                    "5",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )

            self.assertEqual(inputs.probe_duration(path), 1.0)
            byte_limit = 2 * 16_000 * np.dtype(np.float32).itemsize
            ffmpeg_audio = decoding.load_audio_ffmpeg(path, byte_limit)
            torchcodec_audio = decoding.load_audio_torchcodec(path, byte_limit)

        self.assertEqual(len(ffmpeg_audio), 16_000)
        np.testing.assert_array_equal(torchcodec_audio, ffmpeg_audio)

    def test_auto_falls_back_per_file_when_torchcodec_decode_fails(self) -> None:
        expected = np.arange(16, dtype=np.float32)
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="torchcodec"
            ),
            mock.patch.object(
                decoding,
                "load_audio_torchcodec",
                side_effect=RuntimeError("unsupported input"),
            ),
            mock.patch.object(decoding.shutil, "which", return_value="/usr/bin/ffmpeg"),
            mock.patch.object(
                decoding, "load_audio_ffmpeg", return_value=expected
            ) as ffmpeg,
        ):
            decoded, backend, reason = decoding.decode_audio_resolved(
                Path("broken.wav"), "auto"
            )

        self.assertEqual(backend, "ffmpeg")
        self.assertEqual(reason, "RuntimeError: unsupported input")
        np.testing.assert_array_equal(decoded, expected)
        ffmpeg.assert_called_once_with(Path("broken.wav"))

    def test_explicit_torchcodec_does_not_hide_decode_failure(self) -> None:
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="torchcodec"
            ),
            mock.patch.object(
                decoding,
                "load_audio_torchcodec",
                side_effect=RuntimeError("unsupported input"),
            ),
            mock.patch.object(decoding, "load_audio_ffmpeg") as ffmpeg,
            self.assertRaisesRegex(RuntimeError, "unsupported input"),
        ):
            decoding.decode_audio_resolved(Path("broken.wav"), "torchcodec")

        ffmpeg.assert_not_called()

    def test_explicit_librosa_remains_available(self) -> None:
        expected = np.arange(8, dtype=np.float32)
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="librosa"
            ),
            mock.patch(
                "transformers.audio_utils.load_audio", return_value=expected
            ) as load_audio,
            mock.patch.object(decoding, "load_audio_ffmpeg") as ffmpeg,
        ):
            decoded, backend, reason = decoding.decode_audio_resolved(
                Path("clip.wav"), "librosa"
            )

        self.assertEqual(backend, "librosa")
        self.assertIsNone(reason)
        np.testing.assert_array_equal(decoded, expected)
        load_audio.assert_called_once_with(
            "clip.wav", sampling_rate=16_000, backend="librosa"
        )
        ffmpeg.assert_not_called()

    def test_explicit_librosa_does_not_hide_decode_failure(self) -> None:
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="librosa"
            ),
            mock.patch(
                "transformers.audio_utils.load_audio",
                side_effect=RuntimeError("Librosa rejected input"),
            ),
            mock.patch.object(decoding, "load_audio_ffmpeg") as ffmpeg,
            self.assertRaisesRegex(RuntimeError, "Librosa rejected input"),
        ):
            decoding.decode_audio_resolved(Path("broken.wav"), "librosa")

        ffmpeg.assert_not_called()

    def test_auto_torchcodec_success_does_not_call_ffmpeg(self) -> None:
        expected = np.arange(8, dtype=np.float32)
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="torchcodec"
            ),
            mock.patch.object(
                decoding, "load_audio_torchcodec", return_value=expected
            ) as load_audio,
            mock.patch.object(decoding, "load_audio_ffmpeg") as ffmpeg,
        ):
            decoded, backend, reason = decoding.decode_audio_resolved(
                Path("clip.wav"), "auto"
            )

        self.assertEqual(backend, "torchcodec")
        self.assertIsNone(reason)
        np.testing.assert_array_equal(decoded, expected)
        load_audio.assert_called_once_with(Path("clip.wav"))
        ffmpeg.assert_not_called()

    def test_auto_without_torchcodec_decodes_directly_with_ffmpeg(self) -> None:
        expected = np.arange(8, dtype=np.float32)
        with (
            mock.patch.object(decoding, "resolve_audio_backend", return_value="ffmpeg"),
            mock.patch("transformers.audio_utils.load_audio") as transformers_decoder,
            mock.patch.object(
                decoding, "load_audio_ffmpeg", return_value=expected
            ) as ffmpeg,
        ):
            decoded, backend, reason = decoding.decode_audio_resolved(
                Path("clip.wav"), "auto"
            )

        self.assertEqual(backend, "ffmpeg")
        self.assertIsNone(reason)
        np.testing.assert_array_equal(decoded, expected)
        transformers_decoder.assert_not_called()
        ffmpeg.assert_called_once_with(Path("clip.wav"))

    def test_explicit_ffmpeg_does_not_import_transformers_decoder(self) -> None:
        expected = np.arange(8, dtype=np.float32)
        with (
            mock.patch.object(decoding, "resolve_audio_backend", return_value="ffmpeg"),
            mock.patch("transformers.audio_utils.load_audio") as transformers_decoder,
            mock.patch.object(
                decoding, "load_audio_ffmpeg", return_value=expected
            ) as ffmpeg,
        ):
            decoded, backend, reason = decoding.decode_audio_resolved(
                Path("clip.wav"), "ffmpeg"
            )

        self.assertEqual(backend, "ffmpeg")
        self.assertIsNone(reason)
        np.testing.assert_array_equal(decoded, expected)
        transformers_decoder.assert_not_called()
        ffmpeg.assert_called_once_with(Path("clip.wav"))

    def test_one_file_fallback_does_not_demote_torchcodec(self) -> None:
        first = np.arange(8, dtype=np.float32)
        second = np.arange(12, dtype=np.float32)
        with (
            mock.patch.object(
                audio_backends, "torchcodec_is_usable", return_value=True
            ),
            mock.patch.object(
                decoding,
                "load_audio_torchcodec",
                side_effect=[RuntimeError("unsupported first file"), second],
            ) as load_audio,
            mock.patch.object(decoding.shutil, "which", return_value="/usr/bin/ffmpeg"),
            mock.patch.object(
                decoding, "load_audio_ffmpeg", return_value=first
            ) as ffmpeg,
        ):
            first_audio, first_backend, first_reason = decoding.decode_audio_resolved(
                Path("first.wav"), "auto"
            )
            second_audio, second_backend, second_reason = (
                decoding.decode_audio_resolved(Path("second.wav"), "auto")
            )

        self.assertEqual(first_backend, "ffmpeg")
        self.assertEqual(first_reason, "RuntimeError: unsupported first file")
        np.testing.assert_array_equal(first_audio, first)
        self.assertEqual(second_backend, "torchcodec")
        self.assertIsNone(second_reason)
        np.testing.assert_array_equal(second_audio, second)
        self.assertEqual(load_audio.call_count, 2)
        ffmpeg.assert_called_once_with(Path("first.wav"))

    def test_unexpected_torchcodec_error_is_not_hidden(self) -> None:
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="torchcodec"
            ),
            mock.patch.object(
                decoding,
                "load_audio_torchcodec",
                side_effect=AssertionError("implementation defect"),
            ),
            mock.patch.object(decoding, "load_audio_ffmpeg") as ffmpeg,
            self.assertRaisesRegex(AssertionError, "implementation defect"),
        ):
            decoding.decode_audio_resolved(Path("clip.wav"), "auto")

        ffmpeg.assert_not_called()

    def test_failed_ffmpeg_fallback_reports_both_decoder_errors(self) -> None:
        with (
            mock.patch.object(
                decoding, "resolve_audio_backend", return_value="torchcodec"
            ),
            mock.patch.object(
                decoding,
                "load_audio_torchcodec",
                side_effect=RuntimeError("TorchCodec rejected input"),
            ),
            mock.patch.object(decoding.shutil, "which", return_value="/usr/bin/ffmpeg"),
            mock.patch.object(
                decoding,
                "load_audio_ffmpeg",
                side_effect=RuntimeError("FFmpeg rejected input"),
            ),
            self.assertRaisesRegex(
                RuntimeError, "TorchCodec rejected input.*FFmpeg rejected input"
            ),
        ):
            decoding.decode_audio_resolved(Path("clip.wav"), "auto")

    def test_real_librosa_backend_decodes_pcm_wav(self) -> None:
        samples = (-32_768, -16_384, -1, 0, 1, 16_384, 32_767) * 200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "librosa-smoke.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            decoded, backend, reason = decoding.decode_audio_resolved(path, "librosa")

        expected = np.asarray(samples, dtype=np.float32) / np.float32(32_768)
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(backend, "librosa")
        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
