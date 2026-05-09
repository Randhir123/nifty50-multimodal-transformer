"""Central multimodal fusion Transformer for tabular/image/text/KG inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.tabular_transformer import SinusoidalPositionalEncoding


@dataclass(frozen=True)
class FusionTransformerConfig:
    """Configuration for :class:`FusionTransformer`."""

    tabular_dim: int
    image_dim: int | None = None
    text_dim: int | None = None
    kg_dim: int | None = None
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    pooling: str = "mean"
    max_tokens: int = 4096


class FusionTransformer(nn.Module):
    """Fuse available modalities and emit one binary logit per sample.

    The module keeps a stable training/inference contract while allowing each
    modality to be enabled or disabled:
    - tabular tokens (required): ``[batch, seq, tabular_dim]``
    - image embeddings/tokens (optional): ``[batch, image_dim]`` or ``[batch, seq, image_dim]``
    - text embeddings/tokens (optional): ``[batch, text_dim]`` or ``[batch, seq, text_dim]``
    - KG features/tokens (optional): ``[batch, kg_dim]`` or ``[batch, seq, kg_dim]``
    """

    def __init__(self, config: FusionTransformerConfig) -> None:
        super().__init__()
        if config.tabular_dim <= 0:
            raise ValueError("tabular_dim must be positive")
        if config.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if config.num_layers < 1 or config.num_layers > 6:
            raise ValueError("num_layers must be in [1, 6]")
        if config.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.config = config
        self.tabular_projection = nn.Linear(config.tabular_dim, config.model_dim)
        self.image_projection = (
            nn.Linear(config.image_dim, config.model_dim)
            if config.image_dim is not None
            else None
        )
        self.text_projection = (
            nn.Linear(config.text_dim, config.model_dim)
            if config.text_dim is not None
            else None
        )
        self.kg_projection = (
            nn.Linear(config.kg_dim, config.model_dim)
            if config.kg_dim is not None
            else None
        )

        self.modality_embedding = nn.Embedding(4, config.model_dim)

        cls_len = 1 if config.pooling == "cls" else 0
        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=config.model_dim,
            max_len=config.max_tokens + cls_len,
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

        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        else:
            self.register_parameter("cls_token", None)

        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.model_dim, 1)

    def _ensure_token_tensor(self, x: Tensor, *, name: str) -> Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim == 3:
            return x
        raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(x.shape)}")

    def _encode_modality(
        self,
        x: Tensor | None,
        *,
        projection: nn.Linear | None,
        modality_id: int,
        name: str,
    ) -> Tensor | None:
        if x is None:
            return None
        if projection is None:
            raise ValueError(f"{name} was provided but {name}_dim is not configured")

        tokens = self._ensure_token_tensor(x, name=name)
        if tokens.size(-1) != projection.in_features:
            raise ValueError(
                f"{name} last dimension mismatch. Expected {projection.in_features}, got {tokens.size(-1)}"
            )

        projected = projection(tokens)
        modality_ids = torch.full(
            (projected.size(0), projected.size(1)),
            fill_value=modality_id,
            dtype=torch.long,
            device=projected.device,
        )
        return projected + self.modality_embedding(modality_ids)

    def forward(
        self,
        *,
        tabular_tokens: Tensor,
        image_tokens: Tensor | None = None,
        text_tokens: Tensor | None = None,
        kg_tokens: Tensor | None = None,
    ) -> Tensor:
        """Fuse available modalities and return logits of shape ``[batch]``."""
        if tabular_tokens.ndim != 3:
            raise ValueError("tabular_tokens must be 3D [batch, seq, tabular_dim]")

        tabular_encoded = self._encode_modality(
            tabular_tokens,
            projection=self.tabular_projection,
            modality_id=0,
            name="tabular_tokens",
        )
        image_encoded = self._encode_modality(
            image_tokens,
            projection=self.image_projection,
            modality_id=1,
            name="image_tokens",
        )
        text_encoded = self._encode_modality(
            text_tokens,
            projection=self.text_projection,
            modality_id=2,
            name="text_tokens",
        )
        kg_encoded = self._encode_modality(
            kg_tokens,
            projection=self.kg_projection,
            modality_id=3,
            name="kg_tokens",
        )

        token_blocks = [tabular_encoded]
        token_blocks.extend(
            block
            for block in (image_encoded, text_encoded, kg_encoded)
            if block is not None
        )
        tokens = torch.cat(token_blocks, dim=1)

        if self.config.pooling == "cls":
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        if tokens.size(1) > self.positional_encoding.pe.size(1):
            raise ValueError(
                f"Combined token count {tokens.size(1)} exceeds max_tokens={self.config.max_tokens}"
            )

        tokens = self.positional_encoding(tokens)
        encoded = self.encoder(tokens)

        pooled = (
            encoded[:, 0, :] if self.config.pooling == "cls" else encoded.mean(dim=1)
        )
        return self.classifier(self.dropout(pooled)).squeeze(-1)
