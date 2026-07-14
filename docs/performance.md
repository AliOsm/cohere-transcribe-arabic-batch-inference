# Performance

This page records runtime evidence for the v0.1.0 `cohere-transcribe` CLI and the research that determined its defaults. It separates installed-wheel measurements from retained experiments produced by earlier single-file and evaluation harnesses. Retained results remain useful for decisions, but they are not v0.1.0 package baselines.

See [Accuracy Benchmarks](benchmarks.md) for human-reference WER/CER, configuration sensitivity, timestamp limitations, and comparison with the official model-card results. Transcript or subtitle hash equality demonstrates implementation stability, not ASR accuracy.

## Measurement Scope

The authoritative release baselines are repeated fresh-process runs of the installed wheel. External wall time includes process startup, imports, model loading, audio decoding, VAD, ASR, rendering, checkpoint and output publication, and process shutdown. Profile elapsed time starts inside the application and is therefore lower than external wall time. RTFx is decoded source duration divided by wall time, so higher values are faster.

Unless a row explicitly says otherwise, the validated CUDA path used Arabic decoding, BF16 ASR, static length-sorted batch 24, adaptive growth disabled, pinned transfers disabled, and warm local model downloads.

Retained research is labeled because it used an earlier entry point, a different dependency set, a component-only harness, a single observation, or a different segmentation policy. Values with different timing boundaries or ASR inputs must not be compared as a pure implementation speedup.

The long-form workload is a 4,160.679-second Arabic grammar lecture sourced from this [public video](https://www.youtube.com/watch?v=Cdle09QLPLI). It is called the long-form lecture throughout this page.

## Validated Host

```text
OS: Ubuntu 24.04.4 LTS
Kernel: 6.17.0-35-generic
CPU: AMD Ryzen 5 5600X
RAM: 78 GiB
GPU: NVIDIA GeForce RTX 3060 12 GB
NVIDIA driver: 535.309.01
Python: 3.12.12
PyTorch: 2.11.0+cu128
CUDA runtime: 12.8
Transformers: 5.13.1
TorchCodec: 0.14.0+cpu
FFmpeg: 6.1.1
ASR model: CohereLabs/cohere-transcribe-arabic-07-2026
```

CPU selection and dependency handling are tested, but full-model CPU inference and throughput are not validated here. The throughput numbers on this page are not portable performance claims for CPU, Apple MPS, ROCm, Windows, other CUDA cards, or multiple GPUs.

## v0.1.0 Installed-Wheel Baselines

| Workload | Configuration | External runs | Median | External RTFx | Work performed |
|---|---|---|---:|---:|---|
| Long-form Arabic grammar lecture, 4,160.679 seconds | TorchCodec, packed CPU PyTorch Silero with merge, segment timing, batch 24 | 32.27s, 32.27s, 32.20s | 32.27s | 128.9x | 182 processor rows, 8 generation batches, 28,277 generated tokens |
| Balanced 500 files, 5,035.715 seconds | Automatic decoding, packed CPU PyTorch Silero without merge, text only, batch 24 | 39.27s, 39.37s, 38.89s | 39.27s | 128.2x | 729 processor rows, 33 generation batches, 21,587 generated tokens |

All three long-form runs produced the same TXT, SRT, and VTT hashes. Every balanced-corpus run matched all 500 retained transcripts. Automatic decoding used TorchCodec for 499 files and recovered one WAV that TorchCodec could not decode through the available FFmpeg executable.

The long-form run reached 6.06 GiB peak CUDA allocation, 6.10 GiB peak CUDA reservation, and approximately 5.5 GiB maximum process RSS. These are process measurements on the validated host, not minimum system requirements.

The balanced corpus contains 100 files each from Casablanca, Common Voice 18 Arabic, FLEURS `ar_eg`, a Quran Classical Arabic proxy, and SADA22. The files total 1.399 hours and exercise short heterogeneous batch input rather than one continuous recording.

A rebuilt v0.1.0 wheel completed later regression smokes in 32.42 seconds for the long-form workload and 38.52 seconds for the balanced 500. Output hashes, all 500 transcripts, processor-row counts, generation-batch counts, and generated-token counts matched the retained baselines. These are single observations; the repeated medians above remain the performance baselines.

## Output and Alignment Modes

### v0.1.0 Package Evidence

| Mode | Workload | Measurement | Evidence and limit |
|---|---|---:|---|
| Segment timing | Long-form lecture, Silero merge | 32.27s external median | Three installed-wheel runs; v0.1.0 release baseline |
| Text only | Balanced 500, Silero without merge | 39.27s external median | Three installed-wheel runs; v0.1.0 batch baseline |
| Text only, fixed windows | Long-form lecture, no VAD | 29.75s external median | Package runs were 29.46s, 29.75s, and 29.83s |

The fixed-window run used 139 rows across six generation batches and produced a different transcript from Silero segmentation. It retains silence and may split speech at a 30-second boundary, so the 2.52-second difference from the current segment baseline is not an accuracy-preserving output-mode speedup.

There is no repeated v0.1.0 long-form comparison that changes only segment rendering to text-only while preserving the same Silero spans. Segment timing avoids the CTC aligner and remains on the fast ASR path; rendering itself is not established as a material cost by the package measurements.

### Alignment Precision

The following retained measurements exercise the maintained MMS/TorchAudio alignment implementation now packaged by the project. They cover one alignment-compute observation per precision, not complete fresh-process transcription distributions.

| Frozen input | FP16 alignment compute | FP32 alignment compute | FP16 speedup | FP16/FP32 boundary agreement |
|---|---:|---:|---:|---:|
| Long-form lecture | 19.58s | 46.24s | 2.36x | 99.94% within 20 ms |
| Balanced 500 alignment input | 67.22s | 173.03s | 2.57x | 99.77% within 20 ms |

The corpora do not have human word-boundary labels. Agreement with FP32 measures numerical sensitivity, not absolute timestamp accuracy. Rare FP16 path changes can be much larger than the median, particularly in repeated Classical-Arabic text, so FP32 remains the reference and FP16 is the measured speed option.

A retained end-to-end smoke run of the merge and alignment logic later packaged in v0.1.0 transcribed and FP16-aligned the long-form lecture in 57.75 seconds, with 182 segments, 9,852 words, 1,226 cues, and no alignment fallbacks. This is a single observation and is not an installed-wheel baseline.

The only complete-process four-mode campaign used an earlier alignment implementation and sequence-based ONNX Silero spans. It covered the same 500 files in every mode:

| Retained balanced-500 mode | External wall | RTFx | Relative to segment |
|---|---:|---:|---:|
| Plain transcript | 48.70s | 103.40x | 1.00x |
| Segment interpolation | 48.52s | 103.79x | 1.00x |
| FP16 word alignment | 119.07s | 42.29x | 2.45x |
| FP32 word alignment | 233.26s | 21.59x | 4.81x |

All four modes produced identical ASR text because timing is downstream of recognition. These values estimate the historical end-to-end cost shape, not v0.1.0 installed-wheel wall time; the aligner compute measurements above are the newer implementation evidence.

## Voice Activity Detection and Segmentation

### Current Policy

`--vad silero --vad-engine auto` selects the packed CPU PyTorch runtime. Sequence-based ONNX is an optional explicit or recovery engine when the `onnx` extra is installed. The packaged TorchScript runtime is the last automatic option and is also selectable explicitly. Auditok is an optional energy-based policy from the `auditok` extra. `--vad none` constructs fixed windows and does not run a VAD model.

Use Silero for arbitrary continuous or noisy recordings. Use `--vad-merge` when the additional ASR context and potentially lower row count are worth combining raw speech spans into longer requests; approximate segment timing is still mapped over the retained raw speech spans. Leave merge off when each detected speech span should remain an independent ASR request. Use fixed windows only when the input is clean continuous speech or already clipped and its boundary tradeoffs are acceptable.

### Packed PyTorch Versus ONNX

The retained paired batch benchmark used the same Silero weights and timestamp logic now packaged by the project. Package integration preserved exact output parity, but the five-pair timing campaign was not repeated from the installed wheel.

| Balanced 500 measurement | Packed CPU PyTorch | Sequence-based ONNX | Difference |
|---|---:|---:|---:|
| Isolated VAD compute | 1.676s | 5.938s | Torch 3.54x faster |
| Complete-process external median, five paired runs | 40.11s | 40.33s | Torch 0.22s faster |
| VAD worker median inside full ASR | 2.262s | 7.153s | Torch 4.891s lower accumulated worker time |
| Exposed preparation wait median | 0.046s | 0.440s | Torch 0.394s lower |

Both engines produced identical timestamp lists for all 500 files, their maximum probability difference was 0.0000273, and all 2,500 paired transcript comparisons matched. The much smaller end-to-end difference is expected because next-group VAD overlaps GPU ASR. Packed CPU PyTorch used 5,104 MiB median peak RSS versus 5,081 MiB for ONNX; in this campaign its measured advantage was throughput, with slightly higher RSS.

### Engine and Policy Coverage

| Policy or engine | v0.1.0 status | Measured performance status |
|---|---|---|
| Packed CPU PyTorch Silero (`torch`) | Automatic default; repeatedly validated | Direct installed-wheel release baselines plus retained paired engine study |
| Sequence-based ONNX Silero (`onnx`) | Optional extra and explicit/recovery engine; real CLI path validated | Retained five-pair study; no repeated installed-wheel campaign |
| Packaged TorchScript Silero (`jit`) | Explicit engine and last automatic option; real CLI path and ONNX/TorchScript span parity validated | No comparable throughput campaign |
| Auditok 0.4.2 | Optional segmentation policy; real CLI path validated | No v0.1.0 installed-wheel ASR throughput campaign |
| No VAD | Fixed 30-second windows; real CLI path validated | Three installed-package long-form runs, 29.75s median |

Auditok 0.4.2 reproduced the retained 500-file span lists; no comparable throughput campaign is available.

### Retained VAD-Policy Research

The following values came from the retained pre-package script and are single complete-process observations. The Silero variant used sequence-based ONNX; Auditok and no-VAD did not. Their absolute times are superseded by v0.1.0, TorchCodec-first decoding, and packed PyTorch Silero.

| Workload | Sequence-based ONNX Silero | Auditok threshold 50 | No VAD, fixed 30s |
|---|---:|---:|---:|
| Long-form lecture | 50.61s | 36.45s | 34.97s |
| Balanced 500 | 48.70s | 49.75s | 49.58s |

On the already segmented 500-clip evaluation corpus, the policies also produced different WER. That corpus structurally favors keeping each complete clip and does not establish segmentation accuracy on continuous audio; the detailed values and caveats are in [Accuracy Benchmarks](benchmarks.md#vad-and-segment-construction). The long-form lecture has no reference transcript, so its approximately nine-percent transcript disagreements between policies are not WER.

A separate pre-package sample-accurate Silero study measured 40.13 seconds without merge and 41.50 seconds with merge on the balanced 500 files. Merge reduced 729 ASR rows to 508 and improved WER on those presegmented clips, but was 1.37 seconds slower in this single observation; the cause was not isolated because segment layout and ASR inputs changed. Merge is therefore an explicit context/quality option, not a universal speed optimization.

## Audio Decoding

Decoder-only timings exclude VAD, ASR, alignment, output publication, and process-level model startup. Each value is the median of fresh runs after backend import, with every decoded array consumed and exact sample comparisons performed.

| Workload | Workers | TorchCodec | FFmpeg | PyAV 18.0 |
|---|---:|---:|---:|---:|
| Long-form lecture | 1 | 2.963s | 3.824s | 2.519s |
| Balanced 500 | 1 | 2.891s | 19.916s | 4.876s |
| Balanced 500 | 2 | 1.889s | 10.301s | 4.842s |

TorchCodec is 2.56x faster than PyAV in the package's two-worker batch configuration. PyAV is fastest for the single long file but materially slower for the concurrent batch and adds another decoder stack. On this corpus, PyAV recovered the same one TorchCodec-rejected file as FFmpeg. It is therefore not a package dependency or automatic backend.

The automatic decoder uses TorchCodec when its runtime probe succeeds. If TorchCodec is unavailable or unusable at startup, automatic mode selects the OS `ffmpeg` executable; when TorchCodec starts successfully but rejects an individual file, that file is retried through FFmpeg when available. The installed-wheel balanced baseline demonstrated the per-file recovery path on 499 direct TorchCodec files and one WAV recovered by FFmpeg. Explicit `torchcodec`, `ffmpeg`, and `librosa` selection is strict and does not cross-fallback.

Librosa is already required by the Cohere feature extractor and remains available as an explicit compatibility decoder. Its CLI path and exact fixtures are validated, but there is no retained decoder-throughput campaign suitable for comparison with the table above. Librosa materializes the waveform before the package can enforce its decoded-audio limit, so automatic decoding avoids it for large inputs or inputs with unreliable duration metadata.

## Batching, Pipeline, and Memory

### v0.1.0 Package Behavior

Static batch 24 is the validated RTX 3060 setting. The current long-form baseline required eight generation batches and the balanced 500-file baseline required 33. Segments are duration-sorted within each bounded prepared-audio group and constrained by row count and padded-audio budget; the next feature batch is prepared on CPU while the GPU generates the current batch. The full-suite research table below used one global length sort, so its gain is not directly attributable to directory execution across multiple package groups.

Persistent OOM learning remains active under static batching. An OOM lowers the effective cap for later work and repartitions prefetched work rather than repeatedly retrying a known-unsafe size. Upward adaptive growth and pinned host transfers are opt-in because v0.1.0 has no paired benchmark demonstrating an improvement on the validated host.

Multi-file input uses bounded decoded-audio groups and overlaps preparation of the next group with GPU ASR for the current group. The 39.27-second balanced baseline exercises the default pipeline. There is no v0.1.0 installed-wheel paired run with `--no-pipeline-preparation`, so no percentage improvement should be claimed for the pipeline itself.

The long-form baseline's 6.06 GiB allocated and 6.10 GiB reserved CUDA peaks leave apparent VRAM headroom, but generated sequence length, padded frames, allocator state, and the display workload make peak allocation alone insufficient evidence that a larger batch is faster or safe.

### Retained Batch-Size and Pipeline Research

| Full-suite configuration | Wall | Generation | Peak allocated | Lexical WER |
|---|---:|---:|---:|---:|
| Ordered batch 24 | 2,933.72s | 1,723.70s | 9.75 GiB | 32.3006% |
| Length-sorted batch 16 | 2,051.25s | 942.06s | 9.85 GiB | 32.2841% |
| Length-sorted batch 24 | 2,005.64s | 891.27s | 10.18 GiB | 32.2576% |
| Length-sorted batch 32 | 2,028.05s | 911.79s | 11.03 GiB | 32.4129% |

Length sorting was the large gain and batch 24 was the fastest tested size. Batch 32 was slower, consumed another 0.85 GiB, and left little device headroom. BF16 batch composition can change a small number of tokens, so the WER comparison is retained alongside performance rather than assuming byte identity.

The retained pre-package implementation measured the long-form lecture twice per option at 36.49 seconds median for static/pageable batch 24, 36.61 seconds for adaptive growth from 24 to 30, and 36.45 seconds for static batch 24 with pinned transfers. Neither option produced a material wall-time gain. Adaptive batching also changed 12 output tokens because batch membership changed. These values support keeping both options opt-in, but they are not v0.1.0 package timings.

The retained 500-file implementation completed in 61.25 seconds with parallel duration probes and serialized preparation, versus 47.90 seconds with bounded next-group preparation, a 21.8% wall reduction. Batch membership changed 31 of 500 hypotheses, and the measured WER delta was -0.388 percentage points with a paired 95% interval of `[-0.860, 0.000]`. The package retained the pipeline and reproduced the retained outputs for its release workload, but the 21.8% figure remains historical until the installed package receives a paired pipeline-on/off campaign.

## Model Hot-Path Optimizations

The full 24,414-clip engine harness measured the combined projection-cache and repetition-stop candidate at 1,705.49 seconds versus 2,005.64 seconds for the length-sorted batch-24 baseline. Generation fell from 891.27 to 681.48 seconds. The 14.97% wall reduction includes a 95.31-second decode difference outside the model hot path, so the 23.54% generation reduction is the directly attributable model-path measurement.

On the balanced 500-clip attribution probe, caching the invariant encoder projection changed no transcripts and reduced generation from 38.677 to 37.034 seconds. Adding the repetition guard reduced generation to 28.533 seconds by stopping five periodic decoder loops. Preparing the invariant encoder attention mask once reduced generation to 28.171 seconds with no further output changes. The repetition behavior and WER effect are documented in [Accuracy Benchmarks](benchmarks.md#repetition-guard).

The guard waits for at least 96 generated tokens and requires four consecutive copies of an 8-to-32-token period. It targets periodic decoder loops and worst-case latency rather than acting as a generic early-exit heuristic. Token-ceiling detection and affected-row-only retry separately protect genuine long outputs from silent truncation.

An all-segment execution probe on the long-form lecture measured 64.64 seconds with chronological full padding and 26.61 seconds after duration sorting; generation fell from 52.20 to 22.55 seconds. CPU feature lookahead then overlaps feature construction with GPU generation in the package. GPU feature extraction, regional `torch.compile`, and a custom graph path did not provide a stable output-preserving end-to-end gain on the dynamic workload.

## Alternate Inference Engines

These experiments explain why the package remains on the optimized Transformers offline path. They are not selectable package backends.

### vLLM

| Engine | Full 36.393-hour suite wall | RTFx | Lexical WER |
|---|---:|---:|---:|
| Transformers length-sorted batch 24 baseline | 2,005.64s | 65.32x | 32.2576% |
| vLLM 0.19.1, sequence concurrency 8 | 2,185.45s | 59.95x | 32.2018% |
| Optimized Transformers harness | 1,705.49s | 76.82x | 31.3205% |

vLLM was 8.97% slower than its comparable Transformers baseline. Its WER delta was not statistically distinguishable from zero, with paired 95% CI `[-0.5849, +0.4572]` percentage points. On the 12 GB card, vLLM sustained eight active sequences while the offline Transformers path encoder-batched 24 length-sorted clips. The optimized Transformers harness was 1.28x as fast as the frozen vLLM run. vLLM remains relevant for online dynamic arrivals or a larger GPU, but it did not improve this static folder workload.

### Native GGUF Engine

| Balanced 500 engine | External wall | RTFx | Peak GPU | Lexical WER |
|---|---:|---:|---:|---:|
| Python BF16, fixed 30s, no VAD | 49.58s | 101.57x | 6.07 GiB | 23.266% |
| Native F16 | 110.20s | 45.67x | 7.11 GiB | 22.629% |
| Native Q8_0 | 77.33s | 65.08x | 4.94 GiB | 22.784% |
| Native Q4_K with importance matrix | 63.41s | 79.36x | 4.09 GiB | 23.157% |

The custom C++/GGML path reduced memory and accelerated decoding as quantization increased, but its FastConformer encoder remained slower and the best native variant was still 28% slower than the Python BF16 comparison. Its quiet-boundary planner also differs from the Python fixed-window planner, so WER is not a pure quantization comparison. The package does not ship this research engine.

### Precision and Compilation

BF16 is the only ASR precision with a repeated v0.1.0 release baseline on the validated RTX 3060. In the duration-sorted probe described above, FP16 took 34.26 seconds total and 30.24 seconds generation, versus 26.61 and 22.55 seconds for BF16; FP16 also changed 24 of 365 segment outputs.

Aligner `torch.compile` required 27.325 seconds cold and then measured 1.255 seconds versus 1.310 seconds eager. Its 4.35% steady-state gain cannot amortize on this offline process boundary. Full ASR compilation encountered dynamic-shape recompilation or slower one-shot execution, and a regional static probe changed generated positions. Compilation therefore remains outside the package runtime.

## External Wit.ai/Tafrigh Comparison

These are system comparisons, not controlled local engine comparisons. Wit.ai includes Tafrigh preprocessing, Auditok segmentation, MP3 encoding, padding, network latency, cloud scheduling, and service behavior.

| Workload | Cohere boundary | Cohere wall | Wit/Tafrigh boundary | Wit wall | Relative result |
|---|---|---:|---|---:|---:|
| Long-form Arabic grammar lecture, 4,161s | v0.1.0 package, Silero merge, segment timing, three-run median | 32.27s | Stock Tafrigh 1.7.8, eight independent Arabic apps, one observation | 131.79s | Cohere 4.08x faster |
| 24,414 presegmented clips, 36.393h | Optimized direct Cohere evaluation harness | 1,705.49s | Tafrigh-compatible full scheduler, eight apps | 4,397.25s | Cohere 2.58x faster |

The full-suite RTFx values were 76.82x for Cohere and 29.79x for Wit/Tafrigh. The Cohere row is the direct accuracy harness rather than the package CLI, while the long-form row combines a repeated Cohere result with a single Wit observation. Neither comparison establishes how Wit throughput scales with additional independent quotas.

See [Accuracy Benchmarks](benchmarks.md#cohere-and-witai-through-tafrigh) for paired WER/CER and the important domain differences between the systems.

## Checkpoints, Resume, and Startup

| v0.1.0 installed-wheel operation | External wall | Profile evidence |
|---|---:|---|
| Verified skip of 500 complete files | 1.35s | Manifest-verified existing outputs; no inference |
| Render-only resume of the long-form lecture | 3.01s | 2.459s profile elapsed; model load, decode, VAD, and generation all 0.0s |
| Five-file word-alignment smoke | 14.11s | No failures or alignment fallbacks; input duration was not recorded, so this is not throughput evidence |

The v0.1.0 500-file baseline recorded a median 0.677 seconds of checkpoint work and 1.035 seconds of progressive output work per run. These stages enable crash recovery and completed-file publication while the batch is still running.

Lazy imports reduced fresh `cohere-transcribe --help` startup from retained pre-hardening observations of 2.98 and 3.01 seconds to 0.09 and 0.10 seconds, with approximately 37 MiB peak RSS. This startup result concerns CLI responsiveness, not transcription throughput.

For clean performance measurements, use a fresh output directory for every run; derived state files follow the outputs. Reusing a compatible checkpoint can intentionally turn an apparent transcription run into render-only resume, and reusing a complete manifest under skip policy can avoid inference entirely.

## Recommendations

| Goal | Configuration | Evidence-based reason |
|---|---|---|
| Fast approximate subtitles | `--vad silero --vad-engine auto --vad-merge --alignment segment` | Maps approximate timings over retained speech spans without loading the aligner; the tested RTX 3060 configuration used batch 24 and measured a 32.27-second median |
| Plain text with VAD | `--vad silero --vad-engine auto --vad-merge --text-only` | Preserves the same speech-selection policy and skips word alignment; rendering alone is not established as a material ASR cost |
| Fast text for clean continuous speech | `--vad none --text-only` | Fixed windows measured a 29.75-second median on the tested RTX 3060, with known silence and hard-boundary tradeoffs |
| FP32 word timestamps | `--vad-merge --alignment word --align-dtype fp32` | Full-precision MMS emissions are the numerical reference; no human boundary labels were available |
| Faster word timestamps | `--vad-merge --alignment word --align-dtype fp16` | Alignment compute was 2.36x to 2.57x faster in retained measurements |
| Heterogeneous folder batch | Automatic decoding, automatic Silero engine, default preparation pipeline | The tested RTX 3060 configuration used static batch 24 and measured a 39.27-second median for 500 files; one file recovered through FFmpeg |

Adaptive growth, pinned memory, ONNX VAD, Auditok, no VAD, and FP16 word alignment are workload-dependent options. Select them when their documented tradeoff matches the input, then verify output parity or reference accuracy on the target corpus.

## Reproduce

Complete the operating-system FFmpeg and device-specific PyTorch steps in [Install](usage.md#install), then complete [Model Access](usage.md#model-access) before timing. The reference host used Torch 2.11.0 with CUDA 12.8; install Torch and TorchAudio builds compatible with the target GPU and driver. To reproduce the reference runtime:

```bash
python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install the package extras needed by the modes under test:

```bash
python -m pip install "cohere-transcribe-arabic[onnx,word,auditok]"
```

Run the validated long-form configuration into a fresh directory:

```bash
LONG_FORM_AUDIO=/path/to/arabic-grammar-lecture.wav

/usr/bin/time -v cohere-transcribe "$LONG_FORM_AUDIO" \
  --language ar \
  --audio-backend torchcodec \
  --vad silero \
  --vad-engine torch \
  --vad-merge \
  --alignment segment \
  --batch-size 24 \
  --output-dir bench/long-form-segment-r1 \
  --profile-json bench/long-form-segment-r1.profile.json
```

Run the balanced directory configuration:

```bash
/usr/bin/time -v cohere-transcribe balanced-500/ \
  --language ar \
  --vad silero \
  --vad-engine torch \
  --text-only \
  --batch-size 24 \
  --output-dir bench/balanced-batch-r1 \
  --profile-json bench/balanced-batch-r1.profile.json
```

Run fixed-window text for a clean continuous recording:

```bash
LONG_FORM_AUDIO=/path/to/arabic-grammar-lecture.wav

/usr/bin/time -v cohere-transcribe "$LONG_FORM_AUDIO" \
  --language ar \
  --vad none \
  --text-only \
  --batch-size 24 \
  --output-dir bench/long-form-fixed-r1 \
  --profile-json bench/long-form-fixed-r1.profile.json
```

Run word alignment by changing `fp32` to `fp16` only after preserving the same ASR configuration and input:

```bash
LONG_FORM_AUDIO=/path/to/arabic-grammar-lecture.wav

/usr/bin/time -v cohere-transcribe "$LONG_FORM_AUDIO" \
  --language ar \
  --vad silero \
  --vad-engine torch \
  --vad-merge \
  --alignment word \
  --align-dtype fp32 \
  --batch-size 24 \
  --output-dir bench/long-form-word-fp32-r1 \
  --profile-json bench/long-form-word-fp32-r1.profile.json
```

For an engine comparison, keep every other argument fixed and change only `--vad-engine torch`, `--vad-engine onnx`, or `--vad-engine jit`. For a pipeline comparison, keep the same files, group limits, workers, decoder, VAD, batch size, and output mode, and change only `--pipeline-preparation` or `--no-pipeline-preparation`.

Run every performance configuration at least three times as a fresh process after model downloads are warm. Use a new output directory per repetition, report the median and all observations, retain `/usr/bin/time -v` and profile JSON, verify output hashes or human-reference metrics, and compare only configurations with the same timing boundary and audio segmentation when claiming an implementation speedup.

## Known Measurement Gaps

- No repeated v0.1.0 installed-wheel comparison of Silero text-only versus segment rendering on the same spans.
- No repeated v0.1.0 installed-wheel long-form end-to-end FP16 versus FP32 word-alignment campaign.
- No v0.1.0 installed-wheel paired packed-PyTorch versus ONNX versus TorchScript VAD campaign; the available five-pair result predates packaging.
- No v0.1.0 installed-wheel Auditok throughput campaign.
- No v0.1.0 installed-wheel paired merge versus no-merge campaign on continuous audio with human references.
- No v0.1.0 installed-wheel adaptive-growth, pinned-memory, batch-size sweep, or pipeline-on/off campaign.
- No v0.1.0 Librosa decoder throughput distribution.
- No human word-boundary corpus for absolute timestamp accuracy.
- No performance validation outside the Linux RTX 3060 BF16 environment.

## Evidence Sources

- [`reports/0.1.0-release-validation.json`](../reports/0.1.0-release-validation.json) contains installed-wheel run arrays, configurations, hashes, packaged execution and structural parity checks, decoder measurements, Silero validation, alignment measurements, and resume evidence.
- [Accuracy Benchmarks](benchmarks.md) records the frozen evaluation suite, scoring and normalization methods, Cohere/Wit comparisons, confidence intervals, configuration sensitivity, and source-artifact hashes.
- Repository-level research artifacts used to derive the retained sections are not part of the published wheel; package claims are grounded in the versioned runtime report above.
