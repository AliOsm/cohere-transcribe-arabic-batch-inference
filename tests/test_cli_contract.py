"""Executable contract for the two public command-line interfaces."""

from __future__ import annotations

import re
from dataclasses import asdict

import pytest

from cohere_transcribe import doctor
from cohere_transcribe._version import __version__
from cohere_transcribe.cli import _main
from cohere_transcribe.config import parse_args, validate_args
from cohere_transcribe.models import (
    DEFAULT_TORCH_VAD_BATCH_SIZE,
    DEFAULT_TORCH_VAD_BLOCK_FRAMES,
    MAX_TORCH_VAD_PADDED_FRAMES,
)

AUDIO = "audio.wav"

MAIN_OPTIONS = {
    "--adaptive-batch",
    "--align-batch-size",
    "--align-dtype",
    "--alignment",
    "--audio-backend",
    "--audio-memory-gb",
    "--batch-audio-seconds",
    "--batch-max-size",
    "--batch-size",
    "--batch-vram-target",
    "--device",
    "--dtype",
    "--energy-threshold",
    "--existing",
    "--formats",
    "--help",
    "--language",
    "--max-chars",
    "--max-cue-dur",
    "--max-dur",
    "--max-gap",
    "--max-new-tokens",
    "--max-retry-tokens",
    "--max-silence",
    "--min-dur",
    "--min-silence-ms",
    "--no-adaptive-batch",
    "--no-pin-memory",
    "--no-pipeline-preparation",
    "--no-recursive",
    "--no-stop-repetition-loops",
    "--no-vad-merge",
    "--output-dir",
    "--pin-memory",
    "--pipeline-preparation",
    "--preprocess-workers",
    "--profile-json",
    "--recursive",
    "--speech-pad-ms",
    "--stop-repetition-loops",
    "--text-only",
    "--truncation-policy",
    "--vad",
    "--vad-batch-size",
    "--vad-block-frames",
    "--vad-engine",
    "--vad-merge",
    "--vad-threads",
    "--vad-threshold",
    "--version",
}

DOCTOR_OPTIONS = {
    "--audio-backend",
    "--help",
    "--mode",
    "--model-access",
}


def validated(*options: str):
    args = parse_args([AUDIO, *options])
    validate_args(args)
    return args


def long_options(help_text: str) -> set[str]:
    # Descriptive prose can wrap references such as ``--batch-size`` after the
    # hyphen. The usage paragraph is generated from actions and contains only
    # complete option strings, so it is the stable source for this inventory.
    usage = help_text.split("\n\n", maxsplit=1)[0]
    options = set(re.findall(r"--[a-z](?:[a-z-]*[a-z])?", usage))
    if "[-h]" in usage:
        options.add("--help")
    return options


def test_main_help_is_the_complete_option_inventory(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        parse_args(["--help"])

    assert raised.value.code == 0
    output = capsys.readouterr().out
    assert "audio [audio ...]" in output
    assert long_options(output) == MAIN_OPTIONS


@pytest.mark.parametrize(
    "parser",
    [parse_args, doctor.parse_args],
    ids=["main", "doctor"],
)
def test_short_help_alias_exits_successfully(parser, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        parser(["-h"])

    assert raised.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_main_version_is_available_without_an_audio_argument(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        parse_args(["--version"])

    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"cohere-transcribe {__version__}"


def test_main_defaults_are_an_explicit_contract() -> None:
    args = parse_args([AUDIO])
    assert args.formats is None

    validate_args(args)

    assert asdict(args) == {
        "audio": [AUDIO],
        "language": "ar",
        "formats": ["txt", "srt", "vtt"],
        "text_only": False,
        "output_dir": None,
        "recursive": True,
        "existing": "error",
        "device": "auto",
        "dtype": "auto",
        "audio_backend": "auto",
        "audio_memory_gb": 4.0,
        "preprocess_workers": None,
        "pipeline_preparation": True,
        "vad": "silero",
        "vad_engine": "auto",
        "vad_batch_size": DEFAULT_TORCH_VAD_BATCH_SIZE,
        "vad_block_frames": DEFAULT_TORCH_VAD_BLOCK_FRAMES,
        "vad_threads": None,
        "vad_merge": False,
        "min_dur": 0.5,
        "max_dur": 30.0,
        "max_silence": 0.6,
        "energy_threshold": 50.0,
        "vad_threshold": 0.5,
        "min_silence_ms": 300,
        "speech_pad_ms": 60,
        "batch_size": None,
        "batch_max_size": None,
        "batch_audio_seconds": None,
        "batch_vram_target": 0.9,
        "adaptive_batch": False,
        "pin_memory": False,
        "max_new_tokens": 445,
        "max_retry_tokens": 896,
        "truncation_policy": "retry",
        "stop_repetition_loops": True,
        "alignment": "segment",
        "align_batch_size": 4,
        "align_dtype": "fp32",
        "max_chars": 80,
        "max_cue_dur": 6.0,
        "max_gap": 0.6,
        "profile_json": None,
    }


def test_main_accepts_multiple_inputs_and_freeform_output_paths() -> None:
    args = validated(
        "second.flac",
        "--output-dir",
        "nested outputs",
        "--profile-json",
        "profiles/run.json",
    )
    assert args.audio == [AUDIO, "second.flac"]
    assert args.output_dir == "nested outputs"
    assert args.profile_json == "profiles/run.json"


def test_main_requires_input_and_rejects_unknown_options() -> None:
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args([AUDIO, "--not-an-option"])


@pytest.mark.parametrize(
    ("option", "field", "values"),
    [
        ("--language", "language", ("ar", "en")),
        ("--existing", "existing", ("error", "overwrite", "skip")),
        ("--device", "device", ("auto", "mps", "cuda", "cpu")),
        ("--dtype", "dtype", ("auto", "bf16", "fp16", "fp32")),
        (
            "--audio-backend",
            "audio_backend",
            ("auto", "torchcodec", "ffmpeg", "librosa"),
        ),
        ("--vad", "vad", ("silero", "auditok", "none")),
        ("--vad-engine", "vad_engine", ("auto", "torch", "onnx", "jit")),
        ("--truncation-policy", "truncation_policy", ("retry", "warn")),
        ("--alignment", "alignment", ("word", "segment", "none")),
        ("--align-dtype", "align_dtype", ("fp32", "fp16")),
    ],
)
def test_every_main_choice_is_accepted(
    option: str, field: str, values: tuple[str, ...]
) -> None:
    for value in values:
        args = validated(option, value)
        assert getattr(args, field) == value


@pytest.mark.parametrize(
    "option",
    [
        "--language",
        "--formats",
        "--existing",
        "--device",
        "--dtype",
        "--audio-backend",
        "--vad",
        "--vad-engine",
        "--truncation-policy",
        "--alignment",
        "--align-dtype",
    ],
)
def test_every_main_choice_rejects_unknown_values(option: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([AUDIO, option, "not-a-choice"])


@pytest.mark.parametrize(
    ("option", "field", "default"),
    [
        ("--recursive", "recursive", True),
        ("--pipeline-preparation", "pipeline_preparation", True),
        ("--vad-merge", "vad_merge", False),
        ("--adaptive-batch", "adaptive_batch", False),
        ("--pin-memory", "pin_memory", False),
        ("--stop-repetition-loops", "stop_repetition_loops", True),
    ],
)
def test_every_boolean_optional_action_accepts_both_forms(
    option: str, field: str, default: bool
) -> None:
    assert getattr(validated(), field) is default
    assert getattr(validated(option), field) is True
    assert getattr(validated(f"--no-{option[2:]}"), field) is False


def test_repeated_boolean_options_use_the_last_form() -> None:
    assert validated("--recursive", "--no-recursive").recursive is False
    assert validated("--no-recursive", "--recursive").recursive is True


def test_boolean_options_do_not_accept_assigned_values() -> None:
    with pytest.raises(SystemExit):
        parse_args([AUDIO, "--recursive=true"])


def test_every_numeric_option_parses_its_declared_type() -> None:
    args = validated(
        "--audio-memory-gb",
        "2.5",
        "--preprocess-workers",
        "3",
        "--vad-engine",
        "torch",
        "--vad-batch-size",
        "4",
        "--vad-block-frames",
        "1024",
        "--vad-threads",
        "2",
        "--min-dur",
        "0.25",
        "--max-dur",
        "20.5",
        "--max-silence",
        "0.3",
        "--energy-threshold",
        "42.5",
        "--vad-threshold",
        "0.6",
        "--min-silence-ms",
        "200",
        "--speech-pad-ms",
        "40",
        "--batch-size",
        "8",
        "--adaptive-batch",
        "--batch-max-size",
        "16",
        "--batch-audio-seconds",
        "160.5",
        "--batch-vram-target",
        "0.85",
        "--max-new-tokens",
        "400",
        "--max-retry-tokens",
        "800",
        "--alignment",
        "word",
        "--align-batch-size",
        "3",
        "--max-chars",
        "70",
        "--max-cue-dur",
        "5.5",
        "--max-gap",
        "0.4",
    )

    assert {
        field: getattr(args, field)
        for field in (
            "audio_memory_gb",
            "preprocess_workers",
            "vad_batch_size",
            "vad_block_frames",
            "vad_threads",
            "min_dur",
            "max_dur",
            "max_silence",
            "energy_threshold",
            "vad_threshold",
            "min_silence_ms",
            "speech_pad_ms",
            "batch_size",
            "batch_max_size",
            "batch_audio_seconds",
            "batch_vram_target",
            "max_new_tokens",
            "max_retry_tokens",
            "align_batch_size",
            "max_chars",
            "max_cue_dur",
            "max_gap",
        )
    } == {
        "audio_memory_gb": 2.5,
        "preprocess_workers": 3,
        "vad_batch_size": 4,
        "vad_block_frames": 1024,
        "vad_threads": 2,
        "min_dur": 0.25,
        "max_dur": 20.5,
        "max_silence": 0.3,
        "energy_threshold": 42.5,
        "vad_threshold": 0.6,
        "min_silence_ms": 200,
        "speech_pad_ms": 40,
        "batch_size": 8,
        "batch_max_size": 16,
        "batch_audio_seconds": 160.5,
        "batch_vram_target": 0.85,
        "max_new_tokens": 400,
        "max_retry_tokens": 800,
        "align_batch_size": 3,
        "max_chars": 70,
        "max_cue_dur": 5.5,
        "max_gap": 0.4,
    }


@pytest.mark.parametrize(
    "option",
    [
        "--audio-memory-gb",
        "--preprocess-workers",
        "--vad-batch-size",
        "--vad-block-frames",
        "--vad-threads",
        "--min-dur",
        "--max-dur",
        "--max-silence",
        "--energy-threshold",
        "--vad-threshold",
        "--min-silence-ms",
        "--speech-pad-ms",
        "--batch-size",
        "--batch-max-size",
        "--batch-audio-seconds",
        "--batch-vram-target",
        "--max-new-tokens",
        "--max-retry-tokens",
        "--align-batch-size",
        "--max-chars",
        "--max-cue-dur",
        "--max-gap",
    ],
)
def test_every_numeric_option_rejects_a_non_numeric_value(option: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([AUDIO, option, "not-a-number"])


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--audio-memory-gb", "0"), "audio-memory-gb"),
        (("--audio-memory-gb", "nan"), "audio-memory-gb"),
        (("--preprocess-workers", "0"), "preprocess-workers"),
        (("--vad-batch-size", "0"), "vad-batch-size"),
        (("--vad-block-frames", "0"), "vad-block-frames"),
        (
            ("--vad-batch-size", "65", "--vad-block-frames", "512"),
            "must not exceed",
        ),
        (("--vad-threads", "0"), "vad-threads"),
        (("--vad-engine", "onnx", "--vad-threads", "1"), "packed Torch"),
        (("--vad", "auditok", "--vad-threads", "1"), "packed Torch"),
        (("--batch-size", "0"), "batch-size"),
        (
            ("--adaptive-batch", "--batch-max-size", "0"),
            "batch-max-size",
        ),
        (
            (
                "--adaptive-batch",
                "--batch-size",
                "8",
                "--batch-max-size",
                "7",
            ),
            "at least --batch-size",
        ),
        (("--batch-max-size", "8"), "requires --adaptive-batch"),
        (("--batch-audio-seconds", "0"), "batch-audio-seconds"),
        (("--batch-audio-seconds", "nan"), "batch-audio-seconds"),
        (("--batch-vram-target", "0.49"), "between 0.50 and 0.98"),
        (("--batch-vram-target", "0.99"), "between 0.50 and 0.98"),
        (("--batch-vram-target", "nan"), "between 0.50 and 0.98"),
        (("--alignment", "word", "--align-batch-size", "0"), "align-batch-size"),
        (("--max-dur", "0"), "max-dur"),
        (("--max-dur", "nan"), "max-dur"),
        (("--min-dur", "nan"), "min-dur"),
        (("--min-dur", "-0.1"), "0 <= --min-dur"),
        (("--min-dur", "31"), "0 <= --min-dur"),
        (("--vad", "none", "--max-dur", "0.99"), "max-dur >= 1"),
        (("--vad-threshold", "-0.01"), "vad-threshold"),
        (("--vad-threshold", "1.01"), "vad-threshold"),
        (("--min-silence-ms", "-1"), "must be non-negative"),
        (("--speech-pad-ms", "-1"), "must be non-negative"),
        (("--vad", "auditok", "--max-silence", "-0.1"), "Auditok"),
        (("--vad", "auditok", "--energy-threshold", "-1"), "Auditok"),
        (("--vad", "auditok", "--min-dur", "0"), "min-dur > 0"),
        (
            ("--vad", "auditok", "--max-dur", "2", "--max-silence", "2"),
            "max-silence < --max-dur",
        ),
        (("--vad", "auditok", "--vad-merge"), "only with --vad silero"),
        (("--vad", "none", "--vad-merge"), "only with --vad silero"),
        (("--max-new-tokens", "0"), "max-new-tokens"),
        (
            ("--max-new-tokens", "10", "--max-retry-tokens", "9"),
            "at least --max-new-tokens",
        ),
        (
            (
                "--truncation-policy",
                "warn",
                "--max-new-tokens",
                "10",
                "--max-retry-tokens",
                "9",
            ),
            "at least --max-new-tokens",
        ),
        (("--max-chars", "0"), "Subtitle cue limits"),
        (("--max-cue-dur", "0"), "Subtitle cue limits"),
        (("--max-gap", "-0.01"), "Subtitle cue limits"),
    ],
)
def test_semantically_invalid_configurations_are_rejected(
    options: tuple[str, ...], message: str
) -> None:
    args = parse_args([AUDIO, *options])
    with pytest.raises(SystemExit, match=message):
        validate_args(args)


@pytest.mark.parametrize(
    "value",
    ["nan", "inf", "-inf"],
)
@pytest.mark.parametrize(
    "option",
    [
        "--vad-threshold",
        "--max-silence",
        "--energy-threshold",
        "--max-cue-dur",
        "--max-gap",
    ],
)
def test_thresholds_and_cue_limits_must_be_finite_even_when_inactive(
    option: str, value: str
) -> None:
    args = parse_args(
        [AUDIO, "--vad", "none", "--alignment", "none", f"{option}={value}"]
    )
    with pytest.raises(SystemExit):
        validate_args(args)


def test_inclusive_numeric_boundaries_are_accepted() -> None:
    args = validated(
        "--vad-batch-size",
        "64",
        "--vad-block-frames",
        "512",
        "--vad-threshold",
        "0",
        "--adaptive-batch",
        "--batch-size",
        "1",
        "--batch-max-size",
        "1",
        "--batch-vram-target",
        "0.50",
        "--max-new-tokens",
        "1",
        "--max-retry-tokens",
        "1",
        "--alignment",
        "word",
        "--align-batch-size",
        "1",
        "--max-chars",
        "1",
        "--max-cue-dur",
        "0.001",
        "--max-gap",
        "0",
    )
    assert args.vad_batch_size * args.vad_block_frames == MAX_TORCH_VAD_PADDED_FRAMES
    assert args.batch_vram_target == 0.50
    assert args.vad_threshold == 0

    assert validated("--vad-threshold", "1").vad_threshold == 1
    assert validated("--batch-vram-target", "0.98").batch_vram_target == 0.98
    assert validated("--vad", "none", "--max-dur", "1").max_dur == 1
    auditok = validated(
        "--vad",
        "auditok",
        "--min-dur",
        "0.001",
        "--max-dur",
        "1",
        "--max-silence",
        "0",
        "--energy-threshold",
        "0",
    )
    assert (auditok.max_silence, auditok.energy_threshold) == (0, 0)


def test_formats_default_from_alignment_mode() -> None:
    assert validated().formats == ["txt", "srt", "vtt"]
    assert validated("--alignment", "word").formats == ["txt", "srt", "vtt"]
    assert validated("--alignment", "none").formats == ["txt"]


def test_explicit_formats_are_deduplicated_in_first_seen_order() -> None:
    args = validated(
        "--formats",
        "json",
        "txt",
        "json",
        "srt",
        "txt",
        "vtt",
        "srt",
    )
    assert args.formats == ["json", "txt", "srt", "vtt"]


def test_plain_text_mode_accepts_only_deduplicated_txt() -> None:
    assert validated("--alignment", "none", "--formats", "txt", "txt").formats == [
        "txt"
    ]

    for formats in (("srt",), ("txt", "srt"), ("json",)):
        args = parse_args([AUDIO, "--alignment", "none", "--formats", *formats])
        with pytest.raises(SystemExit, match="supports only --formats txt"):
            validate_args(args)


def test_text_only_is_a_mutually_exclusive_alias() -> None:
    args = validated("--text-only")
    assert args.text_only is True
    assert args.alignment == "none"
    assert args.formats == ["txt"]

    with pytest.raises(SystemExit):
        parse_args([AUDIO, "--text-only", "--alignment", "none"])


def test_finite_inactive_mode_values_do_not_reject_a_configuration() -> None:
    args = validated(
        "--vad",
        "none",
        "--vad-engine",
        "onnx",
        "--vad-batch-size",
        "-1",
        "--vad-block-frames",
        "-1",
        "--min-dur",
        "-100",
        "--max-silence",
        "-1",
        "--energy-threshold",
        "-1",
        "--vad-threshold",
        "-1",
        "--min-silence-ms",
        "-1",
        "--speech-pad-ms",
        "-1",
        "--alignment",
        "none",
        "--align-batch-size",
        "0",
        "--max-chars",
        "0",
        "--max-cue-dur",
        "-1",
        "--max-gap",
        "-1",
    )
    assert args.vad == "none"
    assert args.alignment == "none"


@pytest.mark.parametrize("engine", ["onnx", "jit"])
def test_nonpacked_silero_engines_ignore_packed_batch_limits(engine: str) -> None:
    args = validated(
        "--vad-engine",
        engine,
        "--vad-batch-size",
        "-1",
        "--vad-block-frames",
        "-1",
    )
    assert (args.vad_batch_size, args.vad_block_frames) == (-1, -1)


def test_word_alignment_fp16_is_rejected_before_input_or_model_work_on_cpu(
    monkeypatch,
) -> None:
    for variable in (
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "TOKENIZERS_PARALLELISM",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        monkeypatch.delenv(variable, raising=False)

    with pytest.raises(SystemExit, match="supported only with CUDA"):
        _main(
            [
                AUDIO,
                "--device",
                "cpu",
                "--alignment",
                "word",
                "--align-dtype",
                "fp16",
            ]
        )


def test_doctor_help_is_the_complete_option_inventory(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        doctor.parse_args(["--help"])

    assert raised.value.code == 0
    assert long_options(capsys.readouterr().out) == DOCTOR_OPTIONS


def test_doctor_defaults_are_an_explicit_contract() -> None:
    assert vars(doctor.parse_args([])) == {
        "mode": "segment",
        "model_access": False,
        "audio_backend": "auto",
    }


@pytest.mark.parametrize("mode", ["segment", "word"])
@pytest.mark.parametrize("backend", ["auto", "torchcodec", "ffmpeg", "librosa"])
@pytest.mark.parametrize("model_access", [False, True])
def test_every_doctor_choice_and_boolean_form_is_accepted(
    mode: str, backend: str, model_access: bool
) -> None:
    argv = ["--mode", mode, "--audio-backend", backend]
    if model_access:
        argv.append("--model-access")

    args = doctor.parse_args(argv)
    assert (args.mode, args.audio_backend, args.model_access) == (
        mode,
        backend,
        model_access,
    )


@pytest.mark.parametrize("option", ["--mode", "--audio-backend"])
def test_doctor_choices_reject_unknown_values(option: str) -> None:
    with pytest.raises(SystemExit):
        doctor.parse_args([option, "not-a-choice"])


def test_doctor_model_access_does_not_accept_a_value() -> None:
    with pytest.raises(SystemExit):
        doctor.parse_args(["--model-access=true"])


def test_doctor_rejects_unknown_options_and_positional_arguments() -> None:
    with pytest.raises(SystemExit):
        doctor.parse_args(["--not-an-option"])
    with pytest.raises(SystemExit):
        doctor.parse_args(["unexpected-positional"])


def test_doctor_main_routes_selected_features(monkeypatch, capsys) -> None:
    calls: list[object] = []
    sentinel_torch = object()

    monkeypatch.setattr(doctor, "validate_files", lambda results: calls.append("files"))
    monkeypatch.setattr(
        doctor,
        "validate_common_runtime",
        lambda results: calls.append("common") or sentinel_torch,
    )
    monkeypatch.setattr(
        doctor,
        "validate_silero",
        lambda results, torch: calls.append(("silero", torch)),
    )
    monkeypatch.setattr(
        doctor,
        "validate_word_alignment",
        lambda results, torch: calls.append(("word", torch)),
    )
    monkeypatch.setattr(
        doctor,
        "report_optional_runtime",
        lambda results, backend: calls.append(("backend", backend)),
    )
    monkeypatch.setattr(
        doctor,
        "validate_model_access",
        lambda results, include_aligner: calls.append(("access", include_aligner)),
    )

    assert (
        doctor.main(
            [
                "--mode",
                "word",
                "--audio-backend",
                "librosa",
                "--model-access",
            ]
        )
        == 0
    )
    assert calls == [
        "files",
        "common",
        ("silero", sentinel_torch),
        ("word", sentinel_torch),
        ("backend", "librosa"),
        ("access", True),
    ]
    assert "Validation passed for word mode" in capsys.readouterr().out
