# Development

This guide covers source setup, tests, dependency rules, public API maintenance, package builds, and releases.

## Prerequisites

- Linux with Python 3.10 through 3.13.
- [uv](https://docs.astral.sh/uv/).
- System FFmpeg libraries and executables.
- A CUDA GPU and gated model access only for manual inference and performance validation.

Install FFmpeg on Ubuntu or Debian:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Source Environment

Create an environment with every optional feature and the development group:

```bash
uv venv --python 3.12
uv pip install --editable ".[auditok,onnx,word]" --group dev --torch-backend=auto
```

`--torch-backend=auto` selects a device-appropriate PyTorch index. Run project commands with `uv run --no-sync` afterward so synchronization does not replace that device-specific Torch build.

For the standard PyPI source-checkout development resolution:

```bash
uv sync --locked --all-extras --group dev
```

`uv.lock` belongs to the source checkout and is not published in the wheel or sdist. CI checks that it is consistent, while CI installation uses public dependency metadata with an explicit CPU Torch backend. The lock does not define the correct accelerator wheel for every machine; device-specific Torch selection remains explicit.

## Repository Layout

```text
src/cohere_transcribe/            package entry points and shared contracts
src/cohere_transcribe/api/        dependency-light façade, input normalization, and public contracts
src/cohere_transcribe/runtime/     shared engine, result construction, CLI presentation, and reusable model ownership
src/cohere_transcribe/asr/        model loading, generation, batching, and retries
src/cohere_transcribe/audio/      decoding, segmentation, and preparation
src/cohere_transcribe/output/     rendering and transactional publication
src/cohere_transcribe/pipeline/   bounded cross-file orchestration
src/cohere_transcribe/state/      checkpoints, manifests, contracts, and locks
src/cohere_transcribe/alignment/  word-alignment runtime and source provenance
src/cohere_transcribe/vad/        Silero runtimes, weights, and notices
tests/                            unit, contract, integration, and failure-path tests
scripts/                          typing, release, and clean-install validation helpers
docs/                             current user and developer documentation
reports/                          versioned validation evidence
.github/workflows/                CI, TestPyPI, and PyPI publishing
```

The package uses a `src/` layout and `uv_build`. The façade, public contracts, and path normalization live under `cohere_transcribe/api/`; orchestration lives under `runtime/` and `pipeline/`; and cross-cutting configuration, cancellation, device, input, model, preflight, profiling, and CLI modules remain at the package root. Runtime assets live beside their owning code under `cohere_transcribe/vad/` and `cohere_transcribe/alignment/`. Domain `__init__.py` files stay lightweight, and root API exports must not import PyTorch, Transformers, or pipeline modules until the first transcription call.

## Quality Gate

Run the normal source checks:

```bash
uv lock --check
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync mypy --strict scripts/typecheck_public_api.py
uv run --no-sync pytest
```

CI validates the locked all-extras environment and runs the test suite on Python 3.10, 3.11, 3.12, and 3.13 with CPU PyTorch. The lock validation catches releases that declare Python compatibility without publishing an installable artifact for that interpreter. The CPU suite does not download gated model weights. Real model, CUDA, long-form, and corpus benchmarks are separate manual validation because they require access credentials, representative audio, and stable hardware.

### Test Scope

Tests should cover behavior that affects users or protects inference correctness: CLI and Python API contracts, input and output planning, dependency preflight, decoding, segmentation, VAD, batching, OOM recovery, alignment, checkpoint recovery, and interruption handling. Check asset integrity at its owning package path, and use installed-wheel smoke tests to verify distribution contents.

Any new CLI option must have parser, accepted-value, rejected-value, inactive-mode, and help-inventory coverage. Changes to output or state behavior should include interruption or partial-generation cases where relevant.

Every CLI transcription option must map to `TranscriptionOptions`, with filesystem settings grouped under `PublicationOptions`. API coverage should include one path, ordered path sequences, recursive directory expansion, duplicate removal, in-memory execution, publication, verified skips, partial failures, `raise_on_error`, immutable result snapshots, progress callbacks, process serialization, reentrant-use rejection, lazy model loading, reuse, eviction, and close behavior. Test the one-shot helper and reusable context manager independently.

Public exports must remain statically visible from `cohere_transcribe`, and the wheel must include `py.typed`. Strictly type-check `scripts/typecheck_public_api.py` in the source environment. Test public imports and the typing marker from an installed wheel outside the source checkout so an undeclared file or accidental checkout import cannot hide a distribution problem. A clean import should expose API types without importing the ML runtime.

## Dependency Rules

Core dependencies support the default segment-timestamp path. Optional dependencies remain isolated by feature:

| Extra | Purpose |
|---|---|
| `auditok` | Energy-based segmentation |
| `onnx` | Sequence-based ONNX Silero engine |
| `word` | TorchAudio forced alignment and Uroman normalization |

Keep implementation-sensitive versions exact when behavior or ABI requires it:

- Transformers is exact because the Cohere hot-path patches use internal model structure.
- Torch is exact, and TorchAudio must match its major/minor release.
- TorchCodec is exact because decoder behavior participates in the validated runtime.
- Packaged Silero models and forced-aligner helpers carry explicit upstream versions or revisions.
- ONNX Runtime stays below 1.24 only on Python 3.10 because newer releases do not publish CPython 3.10 artifacts.

General-purpose dependencies should remain unpinned unless the package has a demonstrated compatibility boundary. `uv.lock` captures a repeatable development resolution without turning every transitive version into public package metadata.

Do not add direct Git dependencies to published wheel metadata. Any included upstream source must be limited to the required code, preserve its notices, record repository and revision provenance, and have parity tests against the evaluated upstream behavior.

## Build Distributions

Build from declared sources only:

```bash
uv build --clear --no-sources
uvx twine check --strict dist/*
```

`--no-sources` verifies that the published dependency metadata is sufficient without local uv source overrides. By default, `uv build` creates the sdist and builds the wheel from that sdist. Twine checks the generated package metadata and long description. The clean-wheel check below verifies package contents, dependencies, entry points, and runtime assets through the installed package rather than by duplicating the build backend's archive rules.

## Clean-Wheel Validation

Validate the built artifact rather than importing the checkout:

```bash
WHEEL_TEST="$(mktemp -d)"
python -m venv "$WHEEL_TEST/venv"
"$WHEEL_TEST/venv/bin/python" -m pip install --upgrade pip
"$WHEEL_TEST/venv/bin/python" -m pip install dist/*.whl
"$WHEEL_TEST/venv/bin/cohere-transcribe" --help
"$WHEEL_TEST/venv/bin/python" -m cohere_transcribe --help
"$WHEEL_TEST/venv/bin/cohere-transcribe-doctor"
"$WHEEL_TEST/venv/bin/python" scripts/smoke_public_api.py
"$WHEEL_TEST/venv/bin/python" scripts/smoke_clean_audio.py
```

Then validate every extra and dependency contract:

```bash
WHEEL="$(realpath dist/*.whl)"
"$WHEEL_TEST/venv/bin/python" -m pip install "${WHEEL}[auditok,onnx,word]"
"$WHEEL_TEST/venv/bin/python" -m pip check
"$WHEEL_TEST/venv/bin/python" -c "import auditok.core"
"$WHEEL_TEST/venv/bin/cohere-transcribe-doctor" --mode word --audio-backend librosa
rm -rf "$WHEEL_TEST"
```

## Runtime and Performance Validation

Run real-model validation when changing model integration, processor behavior, decoding, VAD, batching, alignment, output timing, or dependencies used in those paths.

At minimum, capture and compare:

- Transcript and output hashes for the reference long-form workload.
- All reference transcript comparisons for the balanced 500-file corpus.
- Segment, generation-row, generated-token, decoder, VAD, OOM, and repetition provenance.
- External wall time, profile time, allocated/reserved CUDA peaks, and process RSS.
- Word/alignment parity for alignment changes.
- CLI and Python API transcript, timing, and provenance parity for equivalent options.
- One-shot cleanup, reusable segment/text ASR retention, word-mode ASR eviction, and post-alignment reload behavior.
- Installed-wheel execution rather than source-only execution.

Use repeated alternating runs for performance comparisons and report medians. A single run can prove completion but should not support a speed claim. Transcript hashes prove implementation stability, not absolute ASR accuracy; timestamp implementation agreement is not ground-truth boundary accuracy.

Intentional transcript, segmentation, or timing changes should replace hash parity with an appropriate reference-accuracy evaluation and a documented explanation of the changed behavior.

Update [Performance](performance.md) when a recommended configuration or supported runtime statement changes. Update [Accuracy Benchmarks](benchmarks.md) only when human-reference WER/CER, transcript-quality, normalization, or timestamp-agreement evidence changes. Store detailed release evidence under `reports/` with a versioned filename.

## Versioning

The release version appears in:

```text
pyproject.toml
src/cohere_transcribe/_version.py
```

Keep these values synchronized. The source tests and installed-package doctor both validate the version agreement. Output and profile JSON schemas have independent version numbers and should change only when their serialized contracts change.

For a release:

1. Update both package version locations.
2. Update `CHANGELOG.md` with current behavior and user-visible changes.
3. Run the source, build, Twine, clean-wheel, and applicable model validation gates.
4. Commit and review the exact release tree.
5. Ensure CI passes on `main` for that commit.
6. Draft a GitHub Release with tag `v<package-version>`, for example `v0.1.0`, targeting that commit.
7. Publish the GitHub Release.

Publishing the GitHub Release triggers `release.yml`. Its unprivileged build job checks that the tag matches the package version, builds the wheel and source distribution once, validates their metadata, and transfers those exact artifacts to a minimal PyPI Trusted Publishing job. Quality checks and tests stay in CI rather than being duplicated during publication.

## TestPyPI

Use the manual `Publish to TestPyPI` workflow after changing package metadata, dependency metadata, build configuration, or publishing infrastructure.

Run it from a commit that has passed CI. The workflow builds and validates the distributions, publishes them with TestPyPI Trusted Publishing, then retries the registry download and verifies the published wheel's version, public import, and typing marker without installing the large runtime dependency set.

TestPyPI does not allow replacing a distribution file for an existing version. Increment the package version before rerunning a publication that already uploaded the same filenames.

For a complete manual installation check, download the exact project wheel from TestPyPI without resolving dependencies, then install that local artifact with normal PyPI dependency resolution:

```bash
TESTPYPI_ROOT="$(mktemp -d)"
PACKAGE_VERSION="$(python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open("pyproject.toml", "rb") as handle:
    print(tomllib.load(handle)["project"]["version"])
PY
)"
python -m pip download --no-deps \
  --index-url https://test.pypi.org/simple/ \
  "cohere-transcribe-arabic==$PACKAGE_VERSION" \
  --dest "$TESTPYPI_ROOT"
python -m venv "$TESTPYPI_ROOT/venv"
"$TESTPYPI_ROOT/venv/bin/python" -m pip install --upgrade pip
"$TESTPYPI_ROOT/venv/bin/python" -m pip install "$TESTPYPI_ROOT"/*.whl
"$TESTPYPI_ROOT/venv/bin/cohere-transcribe" --help
"$TESTPYPI_ROOT/venv/bin/cohere-transcribe-doctor"
rm -rf "$TESTPYPI_ROOT"
```

## Trusted Publishing

Configure TestPyPI with this trusted publisher:

```text
Repository: AliOsm/cohere-transcribe-arabic-batch-inference
Workflow: testpypi.yml
Environment: testpypi
```

Configure PyPI with this trusted publisher:

```text
Repository: AliOsm/cohere-transcribe-arabic-batch-inference
Workflow: release.yml
Environment: pypi
```

The `pypi` GitHub environment should require manual approval. No long-lived PyPI token is stored in repository secrets; the publish job requests a short-lived identity token.
