"""Cohere ASR model loading and model-specific inference optimizations."""

from __future__ import annotations

import torch

from ..models import (
    ASR_MODEL_REVISION,
    MODEL_ID,
    is_model_access_error,
    model_access_message,
)


class MemoizedEncoderProjection(torch.nn.Module):
    """Project each encoder output once across its autoregressive decode."""

    def __init__(self, projection: torch.nn.Module) -> None:
        super().__init__()
        self.projection = projection
        self._source: torch.Tensor | None = None
        self._projected: torch.Tensor | None = None

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source is not self._source:
            self._source = source
            self._projected = self.projection(source)
        assert self._projected is not None
        return self._projected

    def clear(self) -> None:
        self._source = None
        self._projected = None


def clear_encoder_projection_cache(model) -> None:
    projection = model.model.decoder.proj
    if isinstance(projection, MemoizedEncoderProjection):
        projection.clear()


def prepare_encoder_attention_mask_once(_module, _inputs, output):
    """Convert the encoder padding mask once instead of once per decoder token."""
    mask = getattr(output, "attention_mask", None)
    if mask is None or mask.ndim != 2:
        return output
    # The stock decoder checks this on CUDA for every token. One check here also
    # preserves its mask-free SDPA path for batches without encoder padding.
    if bool(mask.all()):
        output.attention_mask = None
    else:
        output.attention_mask = mask.to(dtype=torch.bool)[:, None, None, :]
    return output


def load_asr(
    device: str,
    dtype: torch.dtype,
    revision: str | None = ASR_MODEL_REVISION,
    projection_cache: bool = True,
    encoder_attention_mask_cache: bool = True,
):
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=dtype,
            attn_implementation="sdpa",
            revision=revision,
        )
    except Exception as exc:
        if is_model_access_error(exc):
            raise SystemExit(model_access_message(exc)) from exc
        raise
    if projection_cache:
        try:
            projection = model.model.decoder.proj
        except AttributeError as exc:
            raise RuntimeError(
                f"{MODEL_ID}@{revision} is incompatible with the encoder-projection cache"
            ) from exc
        model.model.decoder.proj = MemoizedEncoderProjection(projection)
    if encoder_attention_mask_cache:
        try:
            encoder = model.model.encoder
        except AttributeError as exc:
            raise RuntimeError(
                f"{MODEL_ID}@{revision} is incompatible with the encoder-mask cache"
            ) from exc
        encoder.register_forward_hook(prepare_encoder_attention_mask_once)
    model.to(device)
    model.eval()
    return processor, model
