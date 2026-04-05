"""Lightweight Transformer-compatible text encoder for stock-news classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput


@dataclass(frozen=True)
class TextEncoderConfig:
    """Configuration for :class:`TextEncoder`."""

    pretrained_model_name: str = "distilbert-base-uncased"
    max_length: int = 192
    dropout: float = 0.1
    use_mean_pooling: bool = True


class TextEncoder(nn.Module):
    """Pretrained text encoder with binary classification head.

    Inputs to ``forward`` are raw strings for simple training/inference wiring.
    ``encode_texts`` exposes per-sample embeddings so future fusion modules can
    consume text features without changing contracts.
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        if config.max_length <= 0:
            raise ValueError("max_length must be positive")

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
        self.backbone = AutoModel.from_pretrained(config.pretrained_model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)

    def _tokenize(
        self, texts: Sequence[str], *, device: torch.device
    ) -> dict[str, Tensor]:
        if len(texts) == 0:
            raise ValueError("texts must not be empty")

        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoded.items()}

    def _pool_sequence(
        self, outputs: BaseModelOutput, attention_mask: Tensor
    ) -> Tensor:
        token_embeddings = outputs.last_hidden_state
        if token_embeddings.ndim != 3:
            raise ValueError("Unexpected hidden-state shape from text backbone")

        if not self.config.use_mean_pooling:
            return token_embeddings[:, 0, :]

        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        """Encode raw text into one embedding per sample."""
        device = next(self.parameters()).device
        encoded = self._tokenize(texts, device=device)
        outputs = self.backbone(**encoded)
        return self._pool_sequence(outputs, attention_mask=encoded["attention_mask"])

    def forward(self, texts: Sequence[str]) -> Tensor:
        """Return binary logits with shape ``[batch]`` from raw text inputs."""
        embeddings = self.encode_texts(texts)
        logits = self.classifier(self.dropout(embeddings)).squeeze(-1)
        return logits
