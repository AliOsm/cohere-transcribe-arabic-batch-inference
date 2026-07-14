from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest
import torch

from cohere_transcribe import cli, device, inputs
from cohere_transcribe.cancellation import (
    TerminationRequested,
    cancellation_requested,
    request_cancellation,
)
from cohere_transcribe.output import pipeline as output_pipeline
from cohere_transcribe.pipeline import transcription as transcription_pipeline


def test_cli_allocator_default_preserves_user_configuration() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        cli.configure_cli_environment()
        assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
        assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ

    with mock.patch.dict(
        os.environ, {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}, clear=True
    ):
        cli.configure_cli_environment()
        assert os.environ["PYTORCH_ALLOC_CONF"] == "backend:cudaMallocAsync"
        assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ

    with mock.patch.dict(
        os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"}, clear=True
    ):
        cli.configure_cli_environment()
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
        assert "PYTORCH_ALLOC_CONF" not in os.environ


def test_multi_file_cli_publishes_complete_outputs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    sources = [tmp_path / "a.wav", tmp_path / "b.wav"]
    for source in sources:
        source.write_bytes(b"test audio placeholder")
    output_dir = tmp_path / "outputs"
    seen_batches: list[list[str]] = []
    seen_dtypes: list[tuple[str, str]] = []
    output_calls = 0
    write_outputs = output_pipeline.write_segment_timed_outputs

    def track_outputs(jobs, args) -> None:
        nonlocal output_calls
        output_calls += 1
        write_outputs(jobs, args)

    def fake_transcribe_all(jobs, args, device, dtype, stats) -> None:
        seen_batches.append([job.path.name for job in jobs])
        seen_dtypes.append((args.dtype, str(dtype)))
        for index, job in enumerate(jobs):
            job.duration = 1.0
            job.segment_times = [(0.0, 1.0)]
            job.speech_spans = [(0.0, 1.0)]
            job.segment_texts = [f"transcript {index}"]
            job.decode_backend = "test"
            job.vad_engine_actual = "none"
        output_pipeline.write_segment_timed_outputs(jobs, args)

    monkeypatch.setattr(cli, "preflight_runtime", lambda _args: None)
    monkeypatch.setattr(device, "pick_device", lambda _requested: "cpu")
    monkeypatch.setattr(output_pipeline, "write_segment_timed_outputs", track_outputs)
    monkeypatch.setattr(transcription_pipeline, "transcribe_all", fake_transcribe_all)

    result = cli.main(
        [
            *(str(source) for source in sources),
            "--vad",
            "none",
            "--alignment",
            "segment",
            "--output-dir",
            str(output_dir),
            "--existing",
            "overwrite",
        ]
    )

    assert result == 0
    assert seen_batches == [["a.wav", "b.wav"]]
    assert seen_dtypes == [("fp32", "torch.float32")]
    assert output_calls == 1
    for index, source in enumerate(sources):
        stem = output_dir / source.stem
        assert stem.with_suffix(".txt").read_text(encoding="utf-8") == (
            f"transcript {index}\n"
        )
        assert stem.with_suffix(".srt").is_file()
        assert stem.with_suffix(".vtt").is_file()
    capsys.readouterr()


def test_cli_builds_checkpoint_contract_from_runtime_and_requested_vad_policy(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "clip.wav"
    seen: dict[str, tuple[str, str, str]] = {}

    def fake_build_jobs(args, *, contract_args=None):
        assert contract_args is not None
        seen["requested"] = (args.device, args.dtype, args.vad_engine)
        seen["contract"] = (
            contract_args.device,
            contract_args.dtype,
            contract_args.vad_engine,
        )
        return []

    monkeypatch.setattr(device, "pick_device", lambda _requested: "cpu")
    monkeypatch.setattr(inputs, "build_jobs", fake_build_jobs)

    assert cli.main([str(source)]) == 0
    assert seen == {
        "requested": ("auto", "auto", "auto"),
        "contract": ("cpu", "fp32", "auto"),
    }


@pytest.mark.parametrize("requested", ["bf16", "fp16", "fp32"])
def test_cli_normalizes_every_cpu_precision_to_fp32(
    tmp_path: Path, monkeypatch, requested: str
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"unused")
    resolved: list[str] = []

    def fake_build_jobs(_args, *, contract_args=None):
        assert contract_args is not None
        resolved.append(contract_args.dtype)
        return []

    monkeypatch.setattr(device, "pick_device", lambda _requested: "cpu")
    monkeypatch.setattr(inputs, "build_jobs", fake_build_jobs)

    assert cli.main([str(source), "--device", "cpu", "--dtype", requested]) == 0
    assert resolved == ["fp32"]


@pytest.mark.parametrize(
    ("bf16_supported", "expected"),
    [(True, "bf16"), (False, "fp16")],
)
def test_cli_resolves_cuda_auto_precision(
    tmp_path: Path, monkeypatch, bf16_supported: bool, expected: str
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"unused")
    resolved: list[str] = []

    def fake_build_jobs(_args, *, contract_args=None):
        assert contract_args is not None
        resolved.append(contract_args.dtype)
        return []

    monkeypatch.setattr(device, "pick_device", lambda _requested: "cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)
    monkeypatch.setattr(inputs, "build_jobs", fake_build_jobs)

    assert cli.main([str(source), "--device", "cuda", "--dtype", "auto"]) == 0
    assert resolved == [expected]


def test_cli_resolves_mps_auto_precision_to_fp16(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"unused")
    resolved: list[str] = []

    def fake_build_jobs(_args, *, contract_args=None):
        assert contract_args is not None
        resolved.append(contract_args.dtype)
        return []

    monkeypatch.setattr(device, "pick_device", lambda _requested: "mps")
    monkeypatch.setattr(inputs, "build_jobs", fake_build_jobs)

    assert cli.main([str(source), "--device", "mps", "--dtype", "auto"]) == 0
    assert resolved == ["fp16"]


def test_cli_rejects_bf16_on_unsupported_cuda(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"unused")
    monkeypatch.setattr(device, "pick_device", lambda _requested: "cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(SystemExit, match="does not support BF16"):
        cli.main([str(source), "--device", "cuda", "--dtype", "bf16"])


def test_cli_rejects_bf16_when_mps_probe_fails(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"unused")
    monkeypatch.setattr(device, "pick_device", lambda _requested: "mps")

    def fail_probe(*_args, **_kwargs):
        raise RuntimeError("unsupported")

    monkeypatch.setattr(torch, "zeros", fail_probe)

    with pytest.raises(SystemExit, match="MPS device/runtime does not support BF16"):
        cli.main([str(source), "--device", "mps", "--dtype", "bf16"])


def test_cli_reports_an_invalid_nul_input_without_a_raw_path_error() -> None:
    with pytest.raises(SystemExit, match="Invalid input path") as captured:
        cli.main(["invalid\0path", "--device", "cpu"])

    assert "\0" not in str(captured.value)


def test_main_resets_cancellation_for_a_new_in_process_run(monkeypatch) -> None:
    request_cancellation()
    monkeypatch.setattr(cli, "_main", lambda _argv=None: 0)

    assert cli.main([]) == 0
    assert not cancellation_requested()


def test_console_entry_point_maps_sigterm_cleanup_to_143(capsys) -> None:
    with mock.patch.object(cli, "main", side_effect=TerminationRequested):
        assert cli.cli() == 143
    assert "Termination requested" in capsys.readouterr().out


def test_console_entry_point_maps_keyboard_interrupt_to_130(capsys) -> None:
    with mock.patch.object(cli, "main", side_effect=KeyboardInterrupt):
        assert cli.cli() == 130
    assert "Interrupted" in capsys.readouterr().out
