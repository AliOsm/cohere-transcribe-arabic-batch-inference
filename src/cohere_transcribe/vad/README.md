# Transcription runtime assets

`torch_silero.py` reconstructs the public Silero v6.2.1 network from the packaged TorchScript model's exact STFT basis and weights. It batches the convolutional encoder across frames and files, packs variable-length LSTM sequences with independent recurrent state, carries state and waveform context across bounded blocks, validates outputs before publication, and isolates failed packs recursively in the caller.

The network reconstruction and weight mapping follow Mahmoud Ashraf's current Silero V5/V6 reference: <https://gist.github.com/MahmoudAshraf97/29f73de73beb8e4549dedb8b5eac9702>. Independent-file packed sequence batching descends from his earlier batched V5 prototype: <https://gist.github.com/MahmoudAshraf97/7ed36a87c874a8354cea36670feb3a0d>. Production additions include strict version/weight validation, official STFT basis loading, duration bucketing, independent valid/padded-frame limits, long-file blocks, empty and partial input handling, finite/range validation, deterministic caller-order restoration, and telemetry.

`silero_vad.jit` is the exact TorchScript model distributed by `silero-vad==6.2.1`. Packaging the weight directly avoids installing the otherwise unused Silero Python runtime and its TorchAudio dependency in the default segment-timestamp environment.

- Asset source: `silero_vad/data/silero_vad.jit` from the `silero-vad==6.2.1` wheel
- SHA-256: `e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720`
- Upstream project: [Silero VAD](https://github.com/snakers4/silero-vad)
- License: MIT; see `LICENSE.silero-vad`

`silero_vad_v6.onnx` is the sequence-form Silero VAD v6 export distributed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper). It allows the CPU fallback to evaluate many 512-sample frames in one ONNX call while preserving Silero's recurrent state and timestamp rules.

- Source revision: `SYSTRAN/faster-whisper@ed9a06cd89a93e47838f564998a6c09b655d7f43`
- Asset source: `faster_whisper/assets/silero_vad_v6.onnx`
- Runtime reference: `faster_whisper/vad.py`
- SHA-256: `914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9`
- Upstream project: [Silero VAD](https://github.com/snakers4/silero-vad)
- License: MIT; see `LICENSE.silero-vad` for the model and `LICENSE.faster-whisper` for the repository that distributes this exact sequence export. Both notices apply to the bundled asset.

The local ONNX runtime fixes two boundary-only edge cases in the reference runner: an exactly divisible waveform does not receive an extra frame, and creating the first frame's zero context does not mutate the last audio frame. Input assembly is bounded to 256 frames per call based on the retained block sweep, so multi-hour recordings do not create a whole-recording `(frames, 576)` intermediate array.

`vectorized_silero.py` owns the shared Silero 6.2.1 timestamp state machine. Both packed Torch and sequence ONNX feed probabilities into that implementation, which validates the exact frame count, finite values, and probability range before producing sample-accurate spans.
