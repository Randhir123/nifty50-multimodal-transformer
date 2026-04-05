"""Embedding projection helpers for reusable visualization data products."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ProjectionMethod = Literal["pca", "tsne"]


REQUIRED_EMBEDDING_COLUMNS: tuple[str, ...] = ("sample_id",)


def _ensure_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("embeddings must have shape [num_samples, embedding_dim]")
    if arr.shape[0] < 2:
        raise ValueError("embeddings must contain at least 2 samples")
    if arr.shape[1] < 2:
        raise ValueError("embedding_dim must be at least 2")
    return arr


def _normalize_metadata(metadata: pd.DataFrame | None, *, n_samples: int) -> pd.DataFrame:
    if metadata is None:
        return pd.DataFrame({"sample_id": np.arange(n_samples, dtype=np.int64)})

    meta = metadata.copy().reset_index(drop=True)
    missing = sorted(set(REQUIRED_EMBEDDING_COLUMNS) - set(meta.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")
    if len(meta) != n_samples:
        raise ValueError("metadata length must match number of embedding rows")
    return meta


def project_embeddings_pca(
    embeddings: np.ndarray,
    *,
    metadata: pd.DataFrame | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Project embeddings to 2D with PCA and return a plotting-friendly dataframe."""
    matrix = _ensure_2d_embeddings(embeddings)
    meta = _normalize_metadata(metadata, n_samples=matrix.shape[0])

    projector = PCA(n_components=2, random_state=random_state)
    coords = projector.fit_transform(matrix)

    out = meta.copy()
    out["proj_x"] = coords[:, 0]
    out["proj_y"] = coords[:, 1]
    out["method"] = "pca"
    out["explained_variance_ratio_x"] = float(projector.explained_variance_ratio_[0])
    out["explained_variance_ratio_y"] = float(projector.explained_variance_ratio_[1])
    return out


def project_embeddings_tsne(
    embeddings: np.ndarray,
    *,
    metadata: pd.DataFrame | None = None,
    random_state: int = 42,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Project embeddings to 2D with t-SNE and return plotting-friendly dataframe."""
    matrix = _ensure_2d_embeddings(embeddings)
    meta = _normalize_metadata(metadata, n_samples=matrix.shape[0])

    max_perplexity = max(1.0, float(matrix.shape[0] - 1))
    capped_perplexity = min(perplexity, max_perplexity)

    projector = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate=learning_rate,
        perplexity=capped_perplexity,
        max_iter=max_iter,
    )
    coords = projector.fit_transform(matrix)

    out = meta.copy()
    out["proj_x"] = coords[:, 0]
    out["proj_y"] = coords[:, 1]
    out["method"] = "tsne"
    out["perplexity"] = float(capped_perplexity)
    return out


def project_embeddings(
    embeddings: np.ndarray,
    *,
    method: ProjectionMethod = "pca",
    metadata: pd.DataFrame | None = None,
    random_state: int = 42,
    tsne_perplexity: float = 30.0,
) -> pd.DataFrame:
    """Dispatch to supported embedding projection methods."""
    if method == "pca":
        return project_embeddings_pca(
            embeddings,
            metadata=metadata,
            random_state=random_state,
        )
    if method == "tsne":
        return project_embeddings_tsne(
            embeddings,
            metadata=metadata,
            random_state=random_state,
            perplexity=tsne_perplexity,
        )
    raise ValueError(f"Unsupported projection method: {method}")
