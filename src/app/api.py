"""Minimal API-like invocation surface for operational workflows.

This module intentionally keeps plain Python call signatures so the same
functions can be wired into HTTP, CLI, or OpenClaw adapters later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from torch import Tensor, nn

from src.app.workflows import (
    ProjectionMethod,
    EmbeddingMapResult,
    PeerGraphResult,
    RankedStocksResult,
    StockAnalysisResult,
    StockComparisonResult,
    analyze_stock,
    compare_stocks,
    rank_stocks,
    show_embedding_map,
    show_peer_graph,
)


def rank_stocks_endpoint(
    *,
    samples: pd.DataFrame,
    probabilities: np.ndarray | None = None,
    threshold: float = 0.5,
    model: nn.Module | None = None,
    tabular_tokens: Tensor | None = None,
    image_tokens: Tensor | None = None,
    text_tokens: Tensor | None = None,
    kg_tokens: Tensor | None = None,
) -> dict[str, Any]:
    """Endpoint-style wrapper for stock ranking workflow."""
    result: RankedStocksResult = rank_stocks(
        samples,
        probabilities=probabilities,
        threshold=threshold,
        model=model,
        tabular_tokens=tabular_tokens,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        kg_tokens=kg_tokens,
    )
    return {"ranking": result.ranking}


def analyze_stock_endpoint(
    *,
    stock_id: str,
    as_of_date: str,
    ranked_predictions: pd.DataFrame,
    graph: nx.Graph | None = None,
    returns: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Endpoint-style wrapper for single-stock analysis workflow."""
    result: StockAnalysisResult = analyze_stock(
        stock_id,
        as_of_date=as_of_date,
        ranked_predictions=ranked_predictions,
        graph=graph,
        returns=returns,
    )
    return {
        "stock_id": result.stock_id,
        "as_of_date": result.as_of_date,
        "ranking_row": result.ranking_row,
        "kg_context": result.kg_context,
    }


def compare_stocks_endpoint(
    *,
    stock_ids: list[str],
    as_of_date: str,
    ranked_predictions: pd.DataFrame,
    graph: nx.Graph | None = None,
    returns: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Endpoint-style wrapper for stock comparison workflow."""
    result: StockComparisonResult = compare_stocks(
        stock_ids,
        as_of_date=as_of_date,
        ranked_predictions=ranked_predictions,
        graph=graph,
        returns=returns,
    )
    return {
        "as_of_date": result.as_of_date,
        "comparison": result.comparison,
    }


def show_peer_graph_endpoint(
    *,
    graph: nx.Graph,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Endpoint-style wrapper for peer graph visualization workflow."""
    result: PeerGraphResult = show_peer_graph(graph, output_path=output_path)
    return {
        "payload": result.payload,
        "image_path": result.image_path,
    }


def show_embedding_map_endpoint(
    *,
    embeddings: np.ndarray,
    method: str = "pca",
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Endpoint-style wrapper for embedding map workflow."""
    projection_method: ProjectionMethod = "tsne" if method == "tsne" else "pca"
    result: EmbeddingMapResult = show_embedding_map(
        embeddings,
        method=projection_method,
        metadata=metadata,
    )
    return {"projection": result.projection}
