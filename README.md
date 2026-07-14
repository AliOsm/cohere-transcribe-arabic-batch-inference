# Cohere Arabic/English Batch Transcription

`cohere-transcribe-arabic` is an unofficial Python package for high-throughput offline transcription with Cohere's 2B Arabic/English ASR model. Its CLI and Python API process individual files, multiple paths, and nested directories with bounded-memory batching. Results can be returned or published as plain text, approximate segment-timed subtitles, or optional word-timed subtitles.

The Cohere ASR weights are downloaded from a pinned Hugging Face revision after you accept the model terms. The package includes the validated Silero VAD weights but does not redistribute the Cohere model.

## Requirements

- Linux with Python 3.10 through 3.13.
- Access to [CohereLabs/cohere-transcribe-arabic-07-2026](https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026).
- System FFmpeg libraries for TorchCodec. Installing the `ffmpeg` OS package also provides the command-line fallback used when TorchCodec is unavailable or rejects a file.
- A CUDA GPU is strongly recommended for the 2B model. A CPU code path exists, but full-model CPU inference was not validated for this release.

On Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Install

Create a virtual environment and install the package from PyPI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install cohere-transcribe-arabic
```

On a GPU host, the PyTorch wheel selected by the public package index must match the installed driver and accelerator. If it does not, install the appropriate Torch 2.11 build first, then install this package; see [Device-Specific PyTorch](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/usage.md#device-specific-pytorch).

The base installation includes TorchCodec, Librosa, and the dependencies required for the default segment-timestamp pipeline. Optional extras add Auditok segmentation, ONNX Runtime, or word alignment:

```bash
python -m pip install "cohere-transcribe-arabic[auditok]"
python -m pip install "cohere-transcribe-arabic[onnx]"
python -m pip install "cohere-transcribe-arabic[word]"
```

Extras can be combined, for example `cohere-transcribe-arabic[auditok,onnx,word]`.

## Model Access

Accept the model terms on Hugging Face, create a read token, and authenticate the same account:

```bash
hf auth login
```

For non-interactive systems, set `HF_TOKEN`. Use `HF_HOME` when the model cache should live on a larger disk.

Arabic is the default language. Use `--language en` for English audio.

## Quick Start

The default path uses Silero speech boundaries and creates approximate segment-timed subtitles. For continuous long-form recordings, the measured configuration below also combines consecutive spans when their complete interval fits the duration limit:

```bash
cohere-transcribe input.wav \
  --language ar \
  --vad-merge
```

This writes `input.txt`, `input.srt`, and `input.vtt`. Add `--formats txt srt vtt json` for provenance-rich JSON.

The default `--existing error` protects existing outputs. Use `--existing overwrite` to replace them or `--existing skip` to reuse only a complete manifest-verified generation.

Plain text with Silero speech selection:

```bash
cohere-transcribe input.wav --language ar --vad-merge --text-only
```

After installing the `word` extra, request word-level timestamps with:

```bash
cohere-transcribe input.wav --language ar --vad-merge --alignment word
```

## Batch Transcription

Pass any combination of files and directories. Directory traversal is recursive by default, and the model is loaded at most once when inference is needed:

```bash
cohere-transcribe a.wav b.mp3 recordings/ \
  --language ar \
  --vad-merge \
  --output-dir transcripts/ \
  --existing skip
```

Directory inputs preserve their relative subtree under the output directory; explicitly supplied files use their basename. Audio decoding, VAD, preparation, and ASR batching operate across files while each recording keeps independent segmentation state and output files. Successful files are published even when another file fails, and the command exits nonzero when any file fails.

## Python API

`transcribe()` accepts one string or path-like object, or an ordered list or tuple containing files and directories. It returns results in memory and creates no transcript files by default:

```python
from pathlib import Path

from cohere_transcribe import TranscriptionOptions, transcribe

run = transcribe(
    [Path("interview.wav"), "recordings/"],
    options=TranscriptionOptions(language="ar", vad_merge=True),
)

for result in run:
    print(result.path, result.status)
    if result.text is not None:
        print(result.text)
```

For one expanded audio file, `run.single` returns its `TranscriptionResult`. To write durable outputs, checkpoints, manifests, and an optional profile, add `PublicationOptions`:

```python
from cohere_transcribe import PublicationOptions, TranscriptionOptions, transcribe

options = TranscriptionOptions(
    language="ar",
    vad_merge=True,
    publication=PublicationOptions(
        formats=("txt", "srt", "vtt", "json"),
        output_dir="transcripts/",
        existing="skip",
        profile_json="transcripts/run.profile.json",
    ),
)
run = transcribe("recordings/", options=options)
```

Use `Transcriber` as a context manager for repeated calls. It loads models lazily and can retain a compatible ASR model between text-only or segment-timed calls:

```python
from cohere_transcribe import Transcriber, TranscriptionOptions

with Transcriber(TranscriptionOptions(vad_merge=True)) as transcriber:
    first = transcriber.transcribe("first.wav").single
    second = transcriber.transcribe(["second.wav", "third.wav"])
```

See the [Python API guide](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/usage.md#python-api) for result fields, partial failures, progress callbacks, resource lifetime, and concurrency behavior.

## Output Modes

| Mode | Command | Output |
|---|---|---|
| Segment timestamps | Default or `--alignment segment` | TXT, SRT, and VTT with fast approximate timing |
| Plain text | `--text-only` | TXT only, without alignment work |
| Word timestamps | `--alignment word` | TXT, SRT, and VTT using MMS CTC forced alignment |

Segment timing is the default because it keeps the fast ASR path and uses retained detected speech spans for approximate cue timing. Word alignment provides per-word CTC boundaries but loads another model and takes additional time; segments that cannot be aligned use an explicit approximate fallback. Fixed-window text mode with `--vad none --text-only` is faster on the measured clean continuous speech, but it can split words at window boundaries and transcribe silence.

## Validate the Installation

The doctor checks package data, dependency compatibility, the selected decoder, VAD, and accelerator availability without loading the 2B model:

```bash
cohere-transcribe-doctor
cohere-transcribe-doctor --model-access
```

For the complete word-alignment dependency and model-access check:

```bash
cohere-transcribe-doctor --mode word --model-access
```

## Performance

On the validated RTX 3060 12 GB system, the installed package transcribed a 69-minute Arabic grammar lecture in a 32.27-second external median with approximate segment timing. A 500-file, 83.9-minute batch completed in a 39.27-second external median. Measured transcripts and subtitle files were byte-identical to their stored validation baselines; this is an implementation-stability check, not a human-reference WER claim.

See [Performance](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/performance.md) for configurations, methodology, resource measurements, and the reasons behind the default runtime choices. See [Accuracy Benchmarks](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/benchmarks.md) for the human-reference WER/CER evaluation and quality safeguards.

## Documentation

- [Usage guide](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/usage.md): CLI and Python API usage, modes, batching, recovery, tuning, and troubleshooting.
- [Architecture](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/architecture.md): runtime stages, module ownership, packaged assets, and design decisions.
- [Performance](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/performance.md): installed-wheel baselines, configuration studies, alternate engines, and reproducible timing guidance.
- [Accuracy benchmarks](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/benchmarks.md): datasets, normalization, WER/CER, confidence intervals, and official-result comparisons.
- [Development](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/docs/development.md): uv environment, tests, package builds, and releases.
- [Release reports](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/tree/main/reports): versioned CLI and runtime validation evidence.
- [Changelog](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/main/CHANGELOG.md): release-level user and developer changes.

Run `cohere-transcribe --help` for the complete CLI reference.
