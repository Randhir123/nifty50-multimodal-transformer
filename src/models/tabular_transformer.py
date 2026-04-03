"""Lightweight Transformer baseline for tabular rolling-window inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal positional encodings to token embeddings."""

    def __init__(self, model_dim: int, max_len: int = 2048) -> None:
        super().__init__()
        if model_dim <= 0:
            raise ValueError("model_dim must be a positive integer")
        if max_len <= 0:
            raise ValueError("max_len must be a positive integer")

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / model_dim)
        )

        pe = torch.zeros(max_len, model_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply positional encoding to input ``[batch, seq_len, model_dim]``."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max positional length {self.pe.size(1)}"
            )
        return x + self.pe[:, :seq_len]


@dataclass(frozen=True)
class TabularTransformerConfig:
    """Configuration for :class:`TabularTransformer`."""

    feature_dim: int
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    max_len: int = 2048
    pooling: str = "mean"


class TabularTransformer(nn.Module):
    """Transformer encoder for tabular rolling windows.

    Input shape:
        ``[batch, window_len, feature_dim]``

    Output shape:
        ``[batch]`` (single binary logit per sample)
    """

    def __init__(self, config: TabularTransformerConfig) -> None:
        super().__init__()
        if config.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if config.num_layers < 1 or config.num_layers > 4:
            raise ValueError("num_layers must be in [1, 4]")
        if config.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be either 'mean' or 'cls'")

        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=config.model_dim,
            max_len=config.max_len + (1 if config.pooling == "cls" else 0),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.model_dim),
        )

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.model_dim, 1)

        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        else:
            self.register_parameter("cls_token", None)

    def forward(self, x: Tensor) -> Tensor:
        """Run forward pass and return logits of shape ``[batch]``."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected x with shape [batch, window_len, feature_dim], got {tuple(x.shape)}"
            )
        tokens = self.input_projection(x)

        if self.config.pooling == "cls":
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)

        if self.config.pooling == "cls":
            pooled = encoded[:, 0, :]
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(self.dropout(pooled)).squeeze(-1)
        return logits
