from pathlib import Path

import pytest

from cohere_transcribe import (
    PublicationOptions,
    TranscriptionConfigurationError,
    TranscriptionInputError,
    TranscriptionOptions,
    transcribe,
)

from ._support import patch_cpu_runtime


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("audio", "Invalid input path"),
        ("output_dir", "Invalid output directory path"),
        ("profile_json", "Invalid profile path"),
    ],
)
def test_invalid_nul_paths_are_typed_api_input_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, message: str
) -> None:
    patch_cpu_runtime(monkeypatch)
    source = tmp_path / "clip.wav"
    source.write_bytes(b"fixture")
    invalid_path = "invalid\0path"
    audio = invalid_path if field == "audio" else source
    publication = PublicationOptions(
        formats=("txt",),
        output_dir=invalid_path if field == "output_dir" else tmp_path / "out",
        existing="overwrite",
        profile_json=invalid_path if field == "profile_json" else None,
    )

    with pytest.raises(TranscriptionInputError, match=message) as captured:
        transcribe(
            audio,
            options=TranscriptionOptions(device="cpu", publication=publication),
        )

    assert "\0" not in str(captured.value)


def test_text_only_and_word_alignment_conflict_is_rejected() -> None:
    with pytest.raises(
        TranscriptionConfigurationError,
        match=r"text-only.*word|word.*text-only",
    ):
        transcribe(
            "unused.wav",
            options=TranscriptionOptions(text_only=True, alignment="word"),
        )


_INTEGER_OPTIONS = (
    "preprocess_workers",
    "vad_batch_size",
    "vad_block_frames",
    "vad_threads",
    "min_silence_ms",
    "speech_pad_ms",
    "batch_size",
    "batch_max_size",
    "max_new_tokens",
    "max_retry_tokens",
    "align_batch_size",
    "max_chars",
)


@pytest.mark.parametrize("field_name", _INTEGER_OPTIONS)
@pytest.mark.parametrize("invalid_value", [True, 1.5])
def test_integer_options_reject_bools_and_floats(
    field_name: str, invalid_value: object
) -> None:
    values: dict[str, object] = {field_name: invalid_value}
    if field_name == "batch_max_size":
        values["adaptive_batch"] = True
    if field_name == "align_batch_size":
        values["alignment"] = "word"
    options = TranscriptionOptions(**values)  # type: ignore[arg-type]
    option = field_name.replace("_", "-")

    with pytest.raises(
        TranscriptionConfigurationError,
        match=rf"--{option}.*integer",
    ):
        transcribe("unused.wav", options=options)
