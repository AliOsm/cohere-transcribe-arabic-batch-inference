# Production setup

This guide installs the production snapshot of `transcribe.py` on a new machine. Python 3.10 or newer is required; Python 3.12.12 was used for the final RTX 3060 validation.

## What runs where

| Device | Status | Recommended starting configuration |
|---|---|---|
| NVIDIA CUDA | Production-validated on RTX 3060 12 GB | BF16, ASR batch 24 |
| NVIDIA 6-8 GB | Supported, not validated by the final run | BF16/FP16, batch 8-12 |
| CPU | Supported but much slower; ASR uses FP32 | batch 1-4 |
| Apple MPS | Implemented but not benchmarked in this project | FP16, batch 4-8 |
| AMD ROCm | Uses PyTorch's CUDA-compatible API, but unvalidated | start conservatively |

The final 69m21s test file completed in 36.49 seconds median with Silero merge, approximate segment timestamps, and static batch 24 on the RTX 3060. The fastest measured text-only/no-VAD run was 31.30 seconds, but fixed windows can split words and include silence; it is not the reliability default.

Expect roughly 5.3 GB of model downloads for ASR plus word alignment. Keep at least 12 GB of free disk space for model cache, package wheels, and temporary files. CPU inference of this 2B model needs substantial system memory; 16 GB or more is recommended.

## 1. Copy the complete directory

Keep `transcribe.py` and `transcribe_assets/` together. The local Silero ONNX runtime is imported relative to the script directory.

## 2. Install system prerequisites

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev build-essential git ffmpeg libsndfile1
```

`git`, `python3-dev`, and `build-essential` are used to fetch and compile the official `ctc-forced-aligner` revision pinned in `requirements.txt`. FFmpeg is the robust fallback for compressed and video containers. A standalone CUDA toolkit is not required when using official PyTorch wheels; the NVIDIA driver is required.

On Windows, WSL2 with an NVIDIA-enabled Ubuntu distribution is the closest equivalent to the validated environment. Native Windows requires Git, FFmpeg, and Microsoft C++ Build Tools with the Desktop development with C++ workload. macOS requires Git, FFmpeg, libsndfile, and the Xcode command-line tools (`xcode-select --install`).

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## 3. Install PyTorch for the device

Torch and TorchAudio must be installed together from the same source and must have matching major/minor versions. Do this before installing the remaining requirements.

Use the current command from <https://pytorch.org/get-started/locally/>. The following commands reproduce the validated 2.11 environment where those wheels are available.

NVIDIA CUDA 12.8 wheels:

```bash
pip install torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

CPU-only Linux/Windows wheels:

```bash
pip install torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Apple Silicon/macOS:

```bash
pip install torch==2.11.0 torchaudio==2.11.0
```

For ROCm, select the matching ROCm wheel index from the official PyTorch installer. This path was not part of the final validation.

Confirm the installation:

```bash
python - <<'PY'
import torch, torchaudio
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("CUDA/ROCm visible:", torch.cuda.is_available())
print("MPS visible:", torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("accelerator:", torch.cuda.get_device_name(0))
PY
```

## 4. Install the runtime

For the supported version range:

```bash
pip install -r requirements.txt
```

This command clones the maintained forced-aligner package from its pinned GitHub commit, builds its small pybind11 extension, and installs the pinned Uroman runtime automatically. Install the device-specific Torch/TorchAudio pair first so the aligner's broad dependency declaration reuses that environment.

Install optional modes only when needed:

```bash
pip install -r requirements-optional.txt
```

- TorchCodec is installed with the official aligner dependency and enables `--audio-backend torchcodec`; `auto` can still fall back to librosa or FFmpeg when decoding fails.
- Auditok enables `--vad auditok`; Silero is the production default. Auditok may require PortAudio development packages because of its optional PyAudio dependency.

## 5. Accept and authenticate for the Cohere model

The ASR model is gated.

1. Open <https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026>.
2. Sign in and accept the model's access terms.
3. Create a Hugging Face read token and authenticate:

```bash
hf auth login
```

Alternatively export the token for the current process:

```bash
export HF_TOKEN=hf_your_read_token
```

The script pins these exact revisions:

```text
CohereLabs/cohere-transcribe-arabic-07-2026
  0a8193caa4f3f92131471ab08824e488141cb392
MahmoudAshraf/mms-300m-1130-forced-aligner
  49402e9577b1158620820667c218cd494cc44486
MahmoudAshraf97/ctc-forced-aligner
  c344f5bc900323aa434a7cb200b7c629d463bd02
```

Use `HF_HOME` to move the model cache to a larger disk:

```bash
export HF_HOME=/large-disk/huggingface
```

After both models are cached, `HF_HUB_OFFLINE=1` prevents network checks.

## 6. Validate before loading model weights

The validator checks hashes, imports, the CLI, the bundled ONNX VAD, the exact official aligner Git provenance, Uroman, matching Torch/TorchAudio versions, and TorchAudio's forced-align operation. It does not load the 2B model.

```bash
python validate_install.py
```

Also verify access to the pinned Hugging Face processor, aligner tokenizer, and configuration files:

```bash
python validate_install.py --model-access
```

For an installation intentionally limited to approximate timestamps or plain text, use `--mode segment` or `--mode text` to omit the forced-alignment checks.

## 7. Run transcription

### Reliable fast output with segment timestamps

This is the final RTX 3060 production command. Segment timestamps are approximate, but the script preserves detected internal pauses when distributing words and building subtitle cues.

```bash
python transcribe.py input.wav \
  --language ar \
  --vad silero \
  --vad-merge \
  --alignment segment \
  --existing overwrite \
  --profile-json input.profile.json
```

### Word-level CTC timestamps

This loads the 300M MMS aligner after freeing the ASR model:

```bash
python transcribe.py input.wav \
  --language ar \
  --vad silero \
  --vad-merge \
  --alignment word \
  --align-dtype fp32 \
  --profile-json input.profile.json
```

`--align-dtype fp16` is much faster on CUDA and usually produces nearly identical boundaries, but FP32 remains the strict timestamp reference.

The maintained-package migration was checked against frozen ASR output for 500 balanced clips and `1.wav`: all 15,874 word texts and keys were preserved, no invalid intervals were produced, and `1.wav` retained zero alignment fallbacks. The 500-clip set changed from two to three fallbacks because one pathological 0.7-second segment contains a single word with roughly 100 repeated alefs; Uroman preserves those letters while the former Unidecode path discarded them, so the complete target cannot fit the available CTC frames.

There are no human word-boundary labels in these retained sets. Old-to-current drift therefore measures the intentional implementation change, not absolute timestamp accuracy: the FP32 median was 20 ms for both corpora, with p95 140 ms on the 500 clips and 100 ms on `1.wav`. Current FP16 and FP32 boundaries matched within 20 ms for 99.77% of the 500-set boundaries and 99.94% of the `1.wav` boundaries. Exact provenance and timing measurements are recorded in `VERSION.json`.

### Plain text

Reliable text-only transcription that still uses Silero boundaries:

```bash
python transcribe.py input.wav \
  --language ar --vad silero --vad-merge --text-only
```

Maximum measured throughput, with fixed-window tradeoffs:

```bash
python transcribe.py input.wav \
  --language ar --vad none --text-only
```

No-VAD mode retains silence and uses contiguous windows. It can split a word at a window edge and may hallucinate in silence. Use it for clean, already clipped speech or when those risks are acceptable.

### Multiple files and folders

```bash
python transcribe.py a.wav b.mp3 recordings/ \
  --language ar \
  --vad silero \
  --vad-merge \
  --alignment segment \
  --output-dir transcripts/ \
  --existing skip \
  --profile-json batch.profile.json
```

Directory traversal, bounded decode/VAD workers, next-group preparation, and cross-file ASR batching are enabled automatically. The output directory preserves relative input paths. `--existing skip` resumes complete output sets and rebuilds partial sets.

English uses the same model:

```bash
python transcribe.py english.wav --language en --alignment segment
```

## Outputs

Timestamped modes write:

```text
input.txt
input.srt
input.vtt
```

Add `--formats txt srt vtt json` for a provenance-rich JSON result. Plain-text mode writes only `.txt`. Outputs are written transactionally and source changes during processing are rejected before publication.

Profile JSON records exact stage times, package/device versions, resolved audio and VAD backends, segment statistics, generated tokens, batch/OOM history, and CUDA allocated/reserved peaks.

## Tuning without destabilizing the machine

- RTX 3060 12 GB: keep the static default batch 24. Adaptive batching was not faster and changed 12 tokens in the final long-file comparison.
- CUDA 6-8 GB: start with `--batch-size 8`; increase gradually to 12 while watching device memory.
- CPU: start with `--device cpu --dtype fp32 --batch-size 1` or 2.
- MPS: start with `--device mps --dtype fp16 --batch-size 4`.
- If alignment runs out of memory, lower `--align-batch-size` from 4 to 2 or 1. The script also halves it automatically after an OOM.
- Bound decoded host audio with `--audio-memory-gb`; this is a group target, so one very large decoded file can still exceed it.
- Leave `--adaptive-batch` and `--pin-memory` off unless profiling a different platform. Both were throughput ties on the RTX 3060.

## Troubleshooting

### Torch/TorchAudio CUDA mismatch

Errors such as `libcudart.so` missing or undefined TorchAudio symbols usually mean the wheel builds do not match:

```bash
pip uninstall -y torch torchaudio
pip install torch torchaudio --index-url YOUR_PYTORCH_WHEEL_INDEX
```

Install both in the same command, then rerun `validate_install.py`.

### CUDA device does not support BF16

```bash
python transcribe.py input.wav --dtype fp16
```

### Cohere model access denied

Accept the terms in the browser for the same Hugging Face account used by `hf auth login`. Confirm with `python validate_install.py --model-access`.

### Uroman or forced alignment missing

```bash
pip uninstall -y ctc-forced-aligner
pip install -r requirements.txt
```

This replaces any package installed from the unrelated PyPI project with the maintained GitHub revision. Also confirm that Git and a C++ compiler are available and that TorchAudio matches Torch exactly at the major/minor release.

### ONNX Runtime DRM warning

A warning about `/sys/class/drm/card0/device/vendor` does not mean VAD failed. The bundled Silero session explicitly uses `CPUExecutionProvider`; the profile records the actual provider. Run the validator's ONNX smoke test if uncertain.

### Unsupported audio container

Install FFmpeg and use:

```bash
python transcribe.py input.media --audio-backend ffmpeg
```

### Silero asset integrity failure

Restore the ONNX file from the production bundle. Do not download it from an unverified mirror; its expected hash is recorded in `VERSION.json` and checked by `validate_install.py`.
