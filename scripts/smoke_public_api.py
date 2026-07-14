"""Validate the installed wheel's dependency-light public Python surface."""

from __future__ import annotations

import sys
from importlib.resources import files

from cohere_transcribe import (
    ProgressCallbackError,
    PublicationOptions,
    Transcriber,
    TranscriptionInputError,
    TranscriptionOptions,
    TranscriptionRun,
    transcribe,
)

HEAVY_MODULES = {
    "librosa",
    "numpy",
    "torch",
    "torchaudio",
    "torchcodec",
    "transformers",
}

assert not HEAVY_MODULES.intersection(sys.modules)
assert files("cohere_transcribe").joinpath("py.typed").is_file()
assert (
    Transcriber
    and TranscriptionOptions
    and PublicationOptions
    and TranscriptionRun
    and ProgressCallbackError
)

try:
    transcribe("")
except TranscriptionInputError:
    pass
else:
    raise AssertionError("empty API input did not raise TranscriptionInputError")

assert not HEAVY_MODULES.intersection(sys.modules)
