# Architecture

The package is an offline batch-inference library and CLI around Cohere's Arabic/English ASR model. Its architecture prioritizes throughput across files, bounded host and device memory, failure isolation, reproducible model behavior, immutable Python results, resumable work, and optional transactional outputs.

## Runtime Invariants

- One ASR model instance serves every compatible file in a CLI command and is retained across compatible calls in a reusable API session until lifecycle or word-alignment rules require eviction.
- TorchCodec/FFmpeg decoded PCM, VAD packs, processor rows, and generation batches have explicit limits; prepared-group memory is a scheduling target, while decoder transients and explicit Librosa materialization can temporarily exceed it.
- A failed file or segment does not invalidate completed work for other files.
- Model, segmentation, generation, and render settings participate in durable state contracts.
- In-process publication failures trigger rollback; if rollback is incomplete, that condition is reported. A manifest committed last detects incomplete generations after abrupt process termination or machine failure.
- CLI-handled SIGINT and SIGTERM cancellation releases locks, workers, and child processes before returning the documented exit status; the API cleans up and propagates interruption without replacing application signal handlers.
- Model revisions and implementation fingerprints are bound into state contracts and written explicitly in output JSON and profile provenance.
- Public package import and API object construction do not import the ML runtime or load a model.
- One process executes one heavy API run at a time so cancellation state, PyTorch thread settings, CUDA telemetry, and retained resources cannot interfere.

## Runtime Flow

```text
CLI arguments or typed Python options
        |
        v
input discovery and optional publication planning
        |
        v
selected-path dependency preflight
        |
        v
bounded decode -> segmentation/VAD -> prepared audio groups
        |                                  |
        |                         next-group prefetch
        v                                  |
group-local duration ordering -> frame-aware ASR batches
        |
        v
generation safety checks and optional resumable ASR checkpoints
        |
        +-----------------------------+
        |                             |
        v                             v
segment/text result data      optional MMS word alignment
        |                             |
        +--------------+--------------+
                       v
optional transactional outputs and manifest
                       |
                       v
optional separately atomic run profile
                       |
                       v
immutable API results or CLI summary
```

CLI argument parsing, `--help`, `--version`, and basic semantic validation complete before PyTorch or pipeline modules are imported. Importing `cohere_transcribe`, creating `TranscriptionOptions`, and constructing `Transcriber` are also dependency-light; the private runtime is imported on the first transcription call. Device selection, input discovery, optional filesystem collision checks, and selected-path dependency preflight finish before any model weights are loaded.

## Module Ownership

| Area | Modules | Responsibility |
|---|---|---|
| Public API | `api/`, `py.typed` | Dependency-light entry points, immutable public contracts, path normalization, exceptions, and typing marker |
| Shared runtime | `runtime/`, `progress.py` | Execute typed or CLI configuration, own reusable models, build immutable results, render CLI telemetry, translate API errors, serialize process-wide runtime use, and route reporting |
| Core contracts and entry points | `models.py`, `_version.py`, `__init__.py`, `__main__.py` | Internal domain types/constants, schema identifiers, lightweight public exports, and module entry points |
| CLI application | `cli.py`, `config.py`, `preflight.py`, `device.py` | Parse and validate configuration, select the compute device and precision, and coordinate command exit behavior |
| Inputs | `inputs.py` | Expand files and directories, probe durations, preserve relative paths, reject collisions, and construct jobs |
| Audio | `audio/` | Select decoders, enforce bounded PCM loading, construct fixed or Auditok spans, and prepare bounded source groups |
| VAD | `vad/` | Run packed CPU PyTorch, sequence-based ONNX, or packaged TorchScript Silero and preserve engine provenance |
| ASR | `asr/` | Load and patch the model, prepare processor rows, batch by frames, generate, retry safely, and isolate failures |
| Alignment | `alignment/` | Generate MMS emissions, normalize Arabic/English targets, run TorchAudio CTC forced alignment, and produce word intervals |
| Output | `output/` | Render TXT/SRT/VTT/JSON, reload alignment audio when necessary, and publish output generations transactionally |
| State | `state/` | Define ASR/render contracts, checkpoints, manifests, integrity checks, and per-stem advisory locks |
| Pipeline | `pipeline/` | Coordinate prepared groups, model lifetime, cross-file batching, resource release, and progressive checkpoints |
| Operations | `profiling.py`, `cancellation.py`, `doctor.py`, `doctor_support.py` | Record telemetry, propagate signals, validate installed assets, and diagnose runtime compatibility |

Dependencies point toward small data and policy modules. `cohere_transcribe.__init__` eagerly exposes the dependency-light public API and immutable result types without importing PyTorch or Transformers. Model and pipeline imports remain behind the first `transcribe()` call. Modules and names not exported from `cohere_transcribe` are internal implementation details for the 0.1 series.

## Input and Preparation Pipeline

`inputs.py` resolves every source to a canonical path, removes duplicates, and preserves directory-relative result paths. When publication is enabled, it also maps output paths, rejects ambiguous or unsafe plans, and claims output locks before model work begins. In-memory API calls bypass output directories, checkpoints, manifests, and locks entirely.

Preparation operates in decoded-audio groups controlled by `--audio-memory-gb`. Each file also has a hard decoded float32 PCM limit. Duration hints guide scheduling, while actual decoded sizes can repartition a group before inference.

First-group preparation starts before model loading, so initial decode and VAD can overlap ASR startup. For multi-file work, bounded preparation of the next group can overlap ASR for the current group when the memory budget permits. Decode workers are deliberately limited because more workers increase host memory, decoder contention, and concurrent native-library state.

## Audio Decoding

Automatic decoding selects usable TorchCodec first, otherwise selects an available OS `ffmpeg` executable. When TorchCodec was selected successfully but rejects an individual input, that file is retried through FFmpeg when available. Explicit TorchCodec, FFmpeg, and Librosa modes are strict.

TorchCodec and FFmpeg enforce the decoded-PCM cap while reading. Librosa is a core model dependency because the Cohere feature extractor constructs mel filters with it, but its decoder is explicit-only and validates size after materialization.

Active decoder processes are registered with the cancellation controller. FFmpeg and FFprobe are terminated and reaped after timeout, failure, SIGINT, or SIGTERM. TorchCodec runs in-process and can observe cancellation only before and after a native call.

## VAD and Segmentation

Packed CPU PyTorch Silero is the automatic default. It batches frames from independent recordings while maintaining separate recurrent state and waveform context for every file. The packed call is limited by file count, frames per file, total padded frames, and learned OOM caps.

If packed CPU PyTorch cannot initialize, automatic selection tries sequence-based ONNX when its optional dependency is installed, then the packaged TorchScript implementation. Explicit engine selection never switches implementations, and per-file inference failures remain isolated to the affected file.

Silero merge greedily combines consecutive speech spans when their complete start-to-end interval fits within the configured maximum duration, while retaining the raw subspans. It can reduce processor-row count and provide more recognition context, but fewer rows do not guarantee lower wall time; the measured tradeoff is documented in [Performance](performance.md#retained-vad-policy-research). ASR receives the merged audio interval, while approximate cue timing distributes words across retained speech subspans rather than across intervening silence.

Auditok and fixed windows are independent segmentation modes. Auditok uses an optional lightweight energy detector. Fixed windows retain all audio and avoid VAD, which is useful for clean pre-clipped speech but can split words and expose the model to silence.

## ASR Execution

The model loader applies two Cohere-specific hot-path optimizations: the encoder-to-decoder projection is memoized for an autoregressive decode, and the encoder attention mask is converted once before repeated decoder steps. Exact Transformers and model revisions protect these private integration points.

Segments available within each prepared group are ordered by duration, then packed with both row and padded-audio constraints. CPU feature preparation for the next batch overlaps model generation for the current batch. Host pinning and adaptive growth remain optional; static defaults are selected by device.

OOM recovery splits the active batch, lowers a persistent effective cap, and preserves completed siblings. High-token retries have a separate cap so an exceptional long decode does not reduce normal batch throughput.

Generation detects rows that consume the token limit without EOS. The default policy retries only those rows up to the configured ceiling. A conservative periodic-loop detector stops repeated 8-to-32-token cycles, and every event is recorded in API result provenance and, when requested, JSON output and profile provenance.

## Python API and Model Lifetime

`transcribe()` creates a one-shot `Transcriber`, performs one run, and closes it in all exit paths. `Transcriber` separates its dependency-light public façade from a private runtime session. The session loads ASR lazily at the same point where first-group audio preparation is already in flight, preserving startup overlap.

Text-only and segment-timed calls can reuse a compatible ASR processor and model across calls. A process-wide ownership lease permits only one retained 2B ASR model; when another session needs ASR, it evicts the previous owner's model before loading or reusing its own, and the previous session reloads if called again. A word-alignment call evicts the current ASR owner before the aligner is loaded, including checkpoint-only calls that do not acquire ASR themselves, because the runtime does not assume both models fit on the accelerator together. The aligner remains one-shot; a later transcription reloads ASR. Closing a session releases any model state it still owns and is idempotent.

The public result layer snapshots mutable internal jobs into frozen `TranscriptionRun`, `TranscriptionResult`, segment, word, cue, provenance, option, and statistics objects. A failed file can therefore coexist with successful siblings without exposing internal lifecycle state. Verified skipped publications are represented explicitly, but existing transcript contents are not reloaded into the result.

The package emits no API-mode console progress unless a callback is supplied; third-party logging follows those libraries' settings. Reporting routes messages and bounded progress through one serialized callback. A callback exception becomes `ProgressCallbackError` without relabeling completed or published files as transcription failures. CLI reporting keeps its console summaries and progress bars. The API does not install signal handlers; interruption propagates through the embedding application after runtime cleanup.

One process-wide gate serializes heavy API calls across threads and `Transcriber` instances. It protects cancellation state, temporary PyTorch thread-count changes, CUDA peak accounting, and model ownership. Reentrant use is rejected. Separate processes remain independent; publication locks coordinate only processes targeting the same output stem.

## Word Alignment

Word alignment is lazy and optional. The `word` extra supplies TorchAudio and Uroman; non-word modes do not import or load the alignment runtime.

The aligner creates bounded overlapping emission windows, crops normalized FP32 log probabilities, and calls `torchaudio.functional.forced_align`. Emission OOM recovery lowers the active batch without discarding completed windows. Invalid or unalignable segments fall back to explicit approximate timing rather than failing the complete file.

Only four pure-Python normalization and span helpers are required from the maintained forced-aligner project. Their evaluated source is included under `alignment/`, while TorchAudio supplies the alignment kernel. This keeps the wheel self-contained and avoids compiling an unused native extension and CLI stack. `alignment/UPSTREAM.md` records the source repository, revision, and local scope.

## Durable State and Publication

The ASR contract covers source identity, language, resolved device precision, model/runtime revisions, segmentation, and generation settings. The render contract separately covers formats, timing mode, and cue settings. A render-only change can therefore reuse compatible ASR text.

Filesystem state is opt-in for the Python API through `PublicationOptions` and is always enabled by the CLI. Each output stem has an ASR checkpoint and a manifest. The checkpoint preserves transcript and segmentation state after inference. The manifest is written last and records the complete output generation and file hashes. `--existing skip` and `PublicationOptions(existing="skip")` trust outputs only when the source, contracts, requested formats, manifest, and hashes all agree.

Output files are rendered into the destination directory, flushed, synchronized, and replaced individually before the manifest is committed last. Existing outputs are backed up during publication, and an in-process replacement failure triggers rollback; incomplete rollback is reported and preserves available backups. An abrupt process or machine failure can interrupt the replacement sequence, but the missing or mismatched manifest prevents that partial generation from being trusted by `--existing skip`. The source is checked again before commit so a file modified during transcription cannot publish stale output.

One private per-user registry file uses deterministic advisory byte ranges for individual stems. Every planned stem is claimed before model loading, while a process holds only one registry descriptor regardless of the number of output directories. The first process to claim a stem proceeds, while concurrent contenders sharing the same registry namespace fail before model loading; different stems remain independent. Multi-host publication and isolated `/tmp` namespaces require external coordination.

## Package Data and Provenance

The wheel includes:

- The Silero 6.2.1 TorchScript weight used by automatic and explicit JIT VAD.
- The sequence-based ONNX Silero model used by the optional ONNX engine.
- Required Silero and faster-whisper notices.
- Forced-aligner text/span helpers, punctuation data, and upstream provenance.

Cohere ASR and MMS alignment weights are not redistributed. Their repository revisions are pinned in the runtime and downloaded from Hugging Face on first use.

Output JSON and profile JSON have independent schema versions. A package release may change without changing a schema, and a schema change remains explicit in every artifact.

## Why These Defaults

- **TorchCodec plus FFmpeg recovery:** TorchCodec is fastest for the validated concurrent decoder workload, while FFmpeg recovers formats or files rejected by TorchCodec.
- **Packed CPU PyTorch Silero:** it batches independent files efficiently and does not require ONNX Runtime; ONNX remains a validated fallback with matching timestamps on the validation corpus.
- **Segment timestamps:** they preserve the fast ASR path and use retained VAD subspans for approximate cue timing; word alignment is available when the added model and runtime cost is justified.
- **Static GPU batching:** batch 24 is the measured RTX 3060 baseline. Adaptive growth is functional but opt-in because it did not establish a better default on the reference device.
- **Exact implementation-sensitive pins:** Cohere model patches, Torch/TorchAudio ABI coupling, packaged VAD behavior, and resumable-state contracts require deterministic versions.

The runtime measurements and default-choice methodology are in [Performance](performance.md). Human-reference WER/CER and implementation-agreement evidence are in [Accuracy Benchmarks](benchmarks.md).

## Compatibility Surface

Supported executable interfaces:

```text
cohere-transcribe
python -m cohere_transcribe
cohere-transcribe-doctor
python -m cohere_transcribe.doctor
```

Supported Python entry points and contracts are exported directly from `cohere_transcribe`:

```text
transcribe
Transcriber
TranscriptionOptions
PublicationOptions
TranscriptionRun and TranscriptionResult
segment, word, cue, provenance, statistics, and progress types
documented TranscriptionError subclasses
__version__
```

The wheel includes `py.typed`, so type checkers can consume these annotations from an installed distribution. Internal modules are not part of this compatibility surface.

Linux and Python 3.10 through 3.13 are release-tested. MPS has an explicit but unvalidated device path. ROCm has no dedicated CLI mode and is experimental through PyTorch's CUDA-compatible interface. Multi-GPU scheduling is not provided; one command selects one compute device.
