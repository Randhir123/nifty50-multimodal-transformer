"""Lightweight image Transformer for candlestick chart classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.tabular_transformer import SinusoidalPositionalEncoding


@dataclass(frozen=True)
class ImageTransformerConfig:
    """Configuration for :class:`ImageTransformer`."""

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1


class ImageTransformer(nn.Module):
    """Patch-based Transformer encoder for binary chart-image classification.

    Input shape:
        ``[batch, channels, height, width]``

    Output:
        ``forward`` returns logits with shape ``[batch]``.
        ``encode_images`` returns pooled embeddings with shape ``[batch, model_dim]``.
    """

    def __init__(self, config: ImageTransformerConfig) -> None:
        super().__init__()
        if config.image_size <= 0:
            raise ValueError("image_size must be positive")
        if config.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if config.num_layers < 1 or config.num_layers > 6:
            raise ValueError("num_layers must be in [1, 6]")

        self.config = config
        self.patch_projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.model_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

        num_patches_per_side = config.image_size // config.patch_size
        self.num_patches = num_patches_per_side * num_patches_per_side

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=config.model_dim,
            max_len=self.num_patches + 1,
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

    def encode_images(self, x: Tensor) -> Tensor:
        """Encode image tensor and return one embedding per sample."""
        if x.ndim != 4:
            raise ValueError(f"Expected x as [batch, channels, height, width], got {tuple(x.shape)}")
        if x.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError(
                "Input image size mismatch. "
                f"Expected {(self.config.image_size, self.config.image_size)}, got {tuple(x.shape[-2:])}"
            )

        tokens = self.patch_projection(x)
        tokens = tokens.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.positional_encoding(tokens)

        encoded = self.encoder(tokens)
        return encoded[:, 0, :]

    def forward(self, x: Tensor) -> Tensor:
        """Return binary logits with shape ``[batch]``."""
        embeddings = self.encode_images(x)
        logits = self.classifier(self.dropout(embeddings)).squeeze(-1)
        return logits
