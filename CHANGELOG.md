# Changelog

## 0.1.3 - 2026-07-15

- Continue publishing completed transcripts when resumable ASR checkpoint persistence fails with an I/O error, while reporting the checkpoint warning and retaining fatal handling for invalid checkpoint data.

## 0.1.2 - 2026-07-15

- Use one directory-sync policy for outputs, profiles, and resumable state, preserving best-effort behavior on unsupported filesystems while propagating genuine I/O failures.
- Prevent target-mode and commit-boundary interruption races from leaking resources or escaping rollback, preserve primary transaction diagnostics when cleanup also fails, and retain backups when rollback cannot restore an output.
- Add exact state-envelope, SRT/VTT rendering, adaptive batch growth, weighted OOM splitting, incomplete rollback, and atomic cleanup regression coverage.
- Extend strict type checking to the CLI and installation doctor, and reject non-finite values in JSON transcript output.

## 0.1.1 - 2026-07-15

- Add custom-model selection with immutable Hub revisions and canonical local identities for native dense Cohere ASR checkpoints, saved bitsandbytes INT8/INT4 checkpoints through the `quantized` extra, and safely merged PEFT LoRA adapters through the `adapters` extra.
- Load compatible model and adapter directories directly from canonical local paths without Hub resolution, revisions, artifact hashing, or another cache; accept path-like references in the Python API and retain canonical/null provenance in outputs and state contracts.
- Validate native model, processor, quantization, adapter, and weight-artifact contracts before inference; reject unsupported remote-code, quantizer, ONNX-only, GGUF, MLX, device, and adapter/base combinations explicitly.
- Bind resolved model and adapter identity plus detected model format into reusable-resource keys. Retain complete quantization metadata in resumable state, JSON output, and profile provenance; report detected format and runtime readiness through the doctor. Advance output schema 8, profile schema 9, and state contract schema 3.
- Preserve all 500 retained default-model transcripts and established work counts while documenting independent performance and accuracy results for every evaluated alternate checkpoint.

## 0.1.0 - 2026-07-14

- Package the validated batch inference runtime under `cohere_transcribe` with `cohere-transcribe` and doctor console commands.
- Add a typed Python API with one-shot `transcribe()` and reusable `Transcriber`, accepting one file or directory or an ordered path sequence and exposing the complete CLI transcription option surface.
- Return immutable run, per-file, segment, word, cue, provenance, and statistics objects in memory by default; make transactional files, checkpoints, manifests, verified skips, and profiles explicit through `PublicationOptions`.
- Preserve partial batch results through status fields and `BatchTranscriptionError.run`, provide serialized progress callbacks and typed setup/runtime errors, and ship a `py.typed` marker.
- Load reusable API models lazily, retain compatible ASR state for text and segment calls, evict ASR before one-shot word alignment, and serialize heavy calls within a process.
- Organize the public API, audio, ASR, output, pipeline, state, VAD, and alignment code into shallow domain packages while keeping cross-cutting configuration and entry-point modules at the package root, without changing inference logic.
- Remove redundant facades, duplicate helpers, unreachable aligner modes, unused confidence processing, and internal API inventories while preserving exact long-form, 500-file, and word-timestamp parity.
- Make standard `pip install cohere-transcribe` the primary user path while retaining uv for development and advanced device-aware installation.
- Add wheel builds, clean-install validation, Linux CPU CI, TestPyPI verification, and PyPI Trusted Publishing workflows.
- Package the maintained forced-aligner normalization and span utilities at the evaluated `main` revision so PyPI users do not resolve the unrelated project that owns the distribution name.
- Preserve non-word startup performance with lazy alignment imports and keep package imports free of CLI environment side effects.
- Move ONNX Runtime behind an optional extra and update Auditok to its lightweight 0.4 API after exact 500-file segmentation parity validation.
- Keep Python 3.10 on the latest compatible ONNX Runtime 1.23 release because later releases do not publish CPython 3.10 artifacts.
- Make TorchCodec 0.14 a core dependency and the default decoder, while using an available OS `ffmpeg` executable for per-file recovery without bundling FFmpeg in the package.
- Make fast approximate segment timestamps the default output and move TorchAudio/Uroman word alignment into the optional `word` extra.
- Package the exact Silero 6.2.1 TorchScript weight directly so its otherwise unused Python package and TorchAudio dependency do not leak into the default environment.
- Provide actionable gated-model errors that link to the access terms and name the Hugging Face authentication command.
- Keep Librosa in the core runtime because Cohere's feature extractor uses it to construct mel filters; its compatibility decoder remains explicit and automatic decoding never selects it.
- Avoid the unused Transformers `torch` extra and its Accelerate dependency; direct Torch requirements are sufficient for the validated offline loading path.
- Add progressive per-group publication, durable ASR checkpoints, manifest-last output generations, and verified `--existing skip` semantics.
- Add scalable per-stem advisory locks for same-user processes sharing one registry namespace, backed by one private descriptor with bounded use across deeply nested batches.
- Add cooperative SIGINT/SIGTERM cancellation across probes, decoder workers, preparation prefetch, ASR, and alignment, with explicit 130/143 exit statuses.
- Keep packed Silero cancellation responsive between bounded inference groups and during long sample validation and timestamp postprocessing.
- Enforce `--audio-memory-gb` as a decoded float32 PCM cap per file as well as a preparation-group target, including bounded TorchCodec range decoding and alignment re-decode.
- Reject invalid user paths with concise typed errors, report symlink loops consistently across Python 3.10-3.13, and prevent nested output-directory symlinks from escaping the canonical output root during planning.
- Reject same-thread and worker-progress-callback reentry before session locks so cross-session `transcribe()` and `close()` calls cannot deadlock.
- Add a fatal-runtime circuit breaker, allocation-aware OOM recovery, and recursive data-failure isolation without allowing malformed segments to poison healthy files.
- Extend allocation-aware alignment retries to host batch construction and backend-reported allocator failures, and bind preparation grouping settings into state contract schema 2.
- Keep base ASR and high-token retry OOM caps independent, persist learned caps across recursive siblings, and report final output-producing token counts separately from total generation work.
- Detect nonzero media timestamps consistently in probes and bounded TorchCodec decoding, and kill/reap failed or timed-out FFmpeg and FFprobe children.
- Publish output schema 7 and profile schema 8 provenance with separate inferred-segment duration statistics, API serialization wait, and exact retry telemetry.
- Resolve automatic precision and device-specific batch defaults consistently, expose `--version`, and declare the release-tested Linux/Python 3.10-3.13 support surface.
- Upgrade the exact Transformers compatibility release to 5.13.1 after byte-level processor comparison, patched CUDA generation parity, retained long-form and 500-file output parity, and performance validation.
- Include a universal `uv.lock` for repeatable source development and CI consistency checks while keeping device-specific PyTorch wheel selection as an explicit advanced installation step.
- Add broad CLI contract, filesystem-planning, preflight, installed-wheel, signal, locked-environment, and real CUDA validation across public options and supported Python versions.
- Organize CI into focused quality and Python-version test jobs, use maintained GitHub Action release tags, and publish GitHub Releases through separate build and Trusted Publishing jobs without repeating CI.
- Organize current user, architecture, development, performance, and accuracy benchmark guidance under `docs/`, with versioned release evidence under `reports/`.
