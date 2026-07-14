"""Data and small helpers shared by installation doctor checks."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ONNX_ASSET = PACKAGE_ROOT / "vad" / "silero_vad_v6.onnx"
EXPECTED_ONNX_SHA256 = (
    "914fd98ac0a73d69ba1e70c9b1d66acb740eff90500dfde08b89a961b168a6a9"
)
JIT_ASSET = PACKAGE_ROOT / "vad" / "silero_vad.jit"
EXPECTED_JIT_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
ALIGN_VOCABULARY = (
    "<blank>",
    "<pad>",
    "</s>",
    "<unk>",
    "a",
    "i",
    "e",
    "n",
    "o",
    "u",
    "t",
    "s",
    "r",
    "m",
    "k",
    "l",
    "d",
    "g",
    "h",
    "y",
    "b",
    "p",
    "w",
    "c",
    "v",
    "j",
    "z",
    "f",
    "'",
    "q",
    "x",
)


class Results:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")
