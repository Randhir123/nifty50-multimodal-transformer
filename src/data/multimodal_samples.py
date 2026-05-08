"""Aligned multimodal sample artifact helpers.

The functions in this module define the project-level bridge between modality
branches and the fusion Transformer.  The first implementation slice is small
and deterministic: it can build toy aligned arrays, validate the artifact
schema, and convert leakage-safe text/KG context into numeric tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib

import numpy as np
import pandas as pd

from src.data.text import build_company_text_input, normalize_company_text_records


@dataclass(frozen=True)
class MultimodalSampleArrays:
    """Aligned arrays for one multimodal fusion dataset artifact.

    Required arrays use the same first dimension. Optional modality arrays are
    included only when that modality is available for the same sample rows.
    """

    tabular_tokens: np.ndarray
    y: np.ndarray
    end_dates: np.ndarray
    stock_ids: np.ndarray
    image_tokens: np.ndarray | None = None
    text_tokens: np.ndarray | None = None
    kg_tokens: np.ndarray | None = None

    def validate(self) -> None:
        """Validate basic shape and sample-count invariants."""
        if self.tabular_tokens.ndim != 3:
            raise ValueError("tabular_tokens must be 3D [samples, window, features]")
        sample_count = self.tabular_tokens.shape[0]
        if sample_count == 0:
            raise ValueError("multimodal sample artifact must not be empty")

        required_lengths = {
            "y": self.y.shape[0],
            "end_dates": self.end_dates.shape[0],
            "stock_ids": self.stock_ids.shape[0],
        }
        for name, length in required_lengths.items():
            if length != sample_count:
                raise ValueError(
                    f"Sample count mismatch for {name}: expected {sample_count}, got {length}"
                )

        for name, values in (
            ("image_tokens", self.image_tokens),
            ("text_tokens", self.text_tokens),
            ("kg_tokens", self.kg_tokens),
        ):
            if values is None:
                continue
            if values.ndim not in (2, 3):
                raise ValueError(f"{name} must be 2D or 3D, got {values.ndim}D")
            if values.shape[0] != sample_count:
                raise ValueError(
                    f"Sample count mismatch for {name}: expected {sample_count}, got {values.shape[0]}"
                )

    def to_npz_kwargs(self) -> dict[str, np.ndarray]:
        """Return keyword arguments suitable for ``numpy.savez_compressed``."""
        self.validate()
        payload: dict[str, np.ndarray] = {
            "tabular_tokens": self.tabular_tokens.astype(np.float32, copy=False),
            "y": self.y.astype(np.int64, copy=False),
            "end_dates": self.end_dates,
            "stock_ids": self.stock_ids.astype(str),
        }
        if self.image_tokens is not None:
            payload["image_tokens"] = self.image_tokens.astype(np.float32, copy=False)
        if self.text_tokens is not None:
            payload["text_tokens"] = self.text_tokens.astype(np.float32, copy=False)
        if self.kg_tokens is not None:
            payload["kg_tokens"] = self.kg_tokens.astype(np.float32, copy=False)
        return payload


def save_multimodal_samples(arrays: MultimodalSampleArrays, path: str | Path) -> Path:
    """Save aligned multimodal arrays to a compressed `.npz` artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays.to_npz_kwargs())
    return output_path


def stable_text_token(text: str, *, dim: int = 16) -> np.ndarray:
    """Create a deterministic lightweight token vector for text.

    This is not a semantic embedding.  It is a CPU-only placeholder that makes
    the aligned artifact and fusion path testable until a real text encoder is
    wired into the same schema.
    """
    if dim <= 0:
        raise ValueError("dim must be positive")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    values = np.resize(raw, dim) / 255.0
    return values.astype(np.float32)


def stable_image_token(identifier: str, *, dim: int = 12) -> np.ndarray:
    """Create a deterministic placeholder token for a chart/image identifier."""
    return stable_text_token(f"image::{identifier}", dim=dim)


def build_text_tokens_for_samples(
    samples: pd.DataFrame,
    text_records: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    top_k: int = 5,
    dim: int = 16,
) -> np.ndarray:
    """Build deterministic text tokens with an as-of-date cutoff.

    Only records with ``event_date <= sample[date_col]`` are visible to a sample.
    """
    if stock_col not in samples.columns or date_col not in samples.columns:
        raise ValueError(f"samples must include '{stock_col}' and '{date_col}'")
    normalized_records = normalize_company_text_records(text_records)
    tokens: list[np.ndarray] = []
    for row in samples.itertuples(index=False):
        stock_id = getattr(row, stock_col)
        as_of_date = getattr(row, date_col)
        text = build_company_text_input(
            normalized_records,
            stock_id=stock_id,
            as_of_date=as_of_date,
            top_k=top_k,
        )
        tokens.append(stable_text_token(text, dim=dim))
    return np.stack(tokens).astype(np.float32)


def kg_context_to_token(context: dict[str, Any], *, event_types: list[str] | None = None) -> np.ndarray:
    """Flatten normalized KG context to a stable numeric token.

    The token intentionally uses simple scalar features so it can feed the
    existing fusion Transformer before a graph embedding model is introduced.
    """
    flags = context.get("event_flags", {}) or {}
    ordered_event_types = event_types or sorted(str(k) for k in flags)
    values = [
        float(context.get("peer_count") or 0.0),
        float(context.get("peer_avg_recent_return") or 0.0),
        float(context.get("sector_avg_recent_return") or 0.0),
    ]
    values.extend(float(flags.get(event_type, 0)) for event_type in ordered_event_types)
    return np.asarray(values, dtype=np.float32)


def build_kg_tokens_from_contexts(contexts: list[dict[str, Any]]) -> np.ndarray:
    """Convert a list of normalized KG context dictionaries into a token matrix."""
    event_types = sorted(
        {
            str(event_type)
            for context in contexts
            for event_type in (context.get("event_flags", {}) or {})
        }
    )
    return np.stack(
        [kg_context_to_token(context, event_types=event_types) for context in contexts]
    ).astype(np.float32)


def build_toy_multimodal_samples(
    *,
    num_samples: int = 12,
    window_size: int = 5,
    tabular_dim: int = 4,
    image_dim: int = 12,
    text_dim: int = 16,
) -> MultimodalSampleArrays:
    """Build a deterministic toy multimodal artifact for tests and smoke runs."""
    if num_samples < 4:
        raise ValueError("num_samples must be at least 4 for chronological splits")
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=num_samples, freq="D")
    stock_ids = np.array(["RELIANCE.NS" if i % 2 == 0 else "TCS.NS" for i in range(num_samples)])

    tabular_tokens = rng.normal(
        loc=0.0, scale=1.0, size=(num_samples, window_size, tabular_dim)
    ).astype(np.float32)
    y = (tabular_tokens[:, -1, 0] > 0.0).astype(np.int64)

    sample_frame = pd.DataFrame(
        {"stock_id": stock_ids, "date": dates, "chart_id": [f"chart-{i}" for i in range(num_samples)]}
    )
    image_tokens = np.stack(
        [stable_image_token(chart_id, dim=image_dim) for chart_id in sample_frame["chart_id"]]
    ).astype(np.float32)

    text_records = pd.DataFrame(
        [
            {
                "stock_id": stock_id,
                "event_date": date - pd.Timedelta(days=1),
                "source_type": "news",
                "title": f"{stock_id} update {i}",
                "body_text": f"Market update available before sample {i}.",
            }
            for i, (stock_id, date) in enumerate(zip(stock_ids, dates, strict=True))
        ]
    )
    text_tokens = build_text_tokens_for_samples(
        sample_frame, text_records, dim=text_dim
    )

    contexts = [
        {
            "peer_count": 1,
            "peer_avg_recent_return": float(i) / 100.0,
            "sector_avg_recent_return": float(i + 1) / 100.0,
            "event_flags": {"earnings": int(i % 2 == 0), "guidance": int(i % 3 == 0)},
        }
        for i in range(num_samples)
    ]
    kg_tokens = build_kg_tokens_from_contexts(contexts)

    arrays = MultimodalSampleArrays(
        tabular_tokens=tabular_tokens,
        y=y,
        end_dates=dates.to_numpy(),
        stock_ids=stock_ids,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        kg_tokens=kg_tokens,
    )
    arrays.validate()
    return arrays
