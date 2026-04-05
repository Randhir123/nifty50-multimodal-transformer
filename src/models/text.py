"""Lightweight text branch for multi-source company text inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.tabular_transformer import SinusoidalPositionalEncoding

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class CompanyTextTransformerConfig:
    """Configuration for :class:`CompanyTextTransformer`."""

    vocab_size: int = 30_000
    max_length: int = 512
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    pooling: str = "cls"


class HashTextTokenizer:
    """Simple hashing tokenizer to keep the text branch dependency-light.

    The tokenizer intentionally ignores source origin assumptions; it tokenizes
    any normalized company-text string.
    """

    def __init__(self, *, vocab_size: int, pad_id: int = 0, unk_id: int = 1) -> None:
        if vocab_size < 32:
            raise ValueError("vocab_size must be >= 32")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.unk_id = unk_id

    def encode(self, text: str, *, max_length: int) -> list[int]:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        ids = [self._token_to_id(tok) for tok in tokens[:max_length]]
        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def batch_encode(self, texts: list[str], *, max_length: int) -> tuple[Tensor, Tensor]:
        if not texts:
            raise ValueError("texts must not be empty")
        all_ids = [self.encode(t or "", max_length=max_length) for t in texts]
        input_ids = torch.tensor(all_ids, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return input_ids, attention_mask

    def _token_to_id(self, token: str) -> int:
        bucket_count = self.vocab_size - 2
        if bucket_count <= 0:
            return self.unk_id
        return 2 + (hash(token) % bucket_count)


class CompanyTextTransformer(nn.Module):
    """Transformer encoder for normalized company-text inputs.

    Inputs are tokenized strings that can mix headlines, filings, guidance,
    investor-presentation text, and PDF-derived text records.
    """

    def __init__(self, config: CompanyTextTransformerConfig) -> None:
        super().__init__()
        if config.max_length <= 0:
            raise ValueError("max_length must be positive")
        if config.pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")

        self.config = config
        self.tokenizer = HashTextTokenizer(vocab_size=config.vocab_size)

        self.embedding = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.model_dim))

        self.positional_encoding = SinusoidalPositionalEncoding(
            model_dim=config.model_dim,
            max_len=config.max_length + 1,
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

    def tokenize_texts(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenize normalized per-sample company text strings."""
        return self.tokenizer.batch_encode(texts, max_length=self.config.max_length)

    def encode_tokens(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Encode token IDs and return one embedding per sample."""
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape [batch, seq], got {tuple(input_ids.shape)}")

        token_embeddings = self.embedding(input_ids)

        cls = self.cls_token.expand(token_embeddings.size(0), -1, -1)
        tokens = torch.cat([cls, token_embeddings], dim=1)
        tokens = self.positional_encoding(tokens)

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask shape must match input_ids")
            cls_mask = torch.ones((attention_mask.size(0), 1), device=attention_mask.device)
            key_padding_mask = torch.cat([cls_mask, attention_mask], dim=1) == 0
        else:
            key_padding_mask = None

        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)

        if self.config.pooling == "cls":
            return encoded[:, 0, :]

        token_states = encoded[:, 1:, :]
        if attention_mask is None:
            return token_states.mean(dim=1)

        weights = attention_mask.unsqueeze(-1).to(token_states.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (token_states * weights).sum(dim=1) / denom

    def encode_texts(self, texts: list[str]) -> Tensor:
        """Tokenize then encode normalized company-text samples."""
        input_ids, attention_mask = self.tokenize_texts(texts)
        return self.encode_tokens(input_ids=input_ids, attention_mask=attention_mask)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return one binary logit per sample."""
        embeddings = self.encode_tokens(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(embeddings)).squeeze(-1)
        return logits
