"""Operationalization workflows wrapping model, KG, and visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.kg.query_graph import retrieve_kg_context
from src.viz.embeddings import ProjectionMethod, project_embeddings
from src.viz.peer_graph import build_peer_graph_payload, plot_peer_graph
from src.viz.ranking import build_ranked_predictions


@dataclass(frozen=True)
class RankedStocksResult:
    """Typed output for ranking workflow."""

    ranking: pd.DataFrame


@dataclass(frozen=True)
class StockAnalysisResult:
    """Typed output for single-stock analysis workflow."""

    stock_id: str
    as_of_date: str
    ranking_row: dict[str, Any]
    kg_context: dict[str, Any] | None


@dataclass(frozen=True)
class StockComparisonResult:
    """Typed output for multi-stock comparison workflow."""

    as_of_date: str
    comparison: pd.DataFrame


@dataclass(frozen=True)
class PeerGraphResult:
    """Typed output for peer graph visualization workflow."""

    payload: dict[str, list[dict[str, Any]]]
    image_path: str | None


@dataclass(frozen=True)
class EmbeddingMapResult:
    """Typed output for embedding map workflow."""

    projection: pd.DataFrame


def predict_fusion_probabilities(
    model: nn.Module,
    *,
    tabular_tokens: Tensor,
    image_tokens: Tensor | None = None,
    text_tokens: Tensor | None = None,
    kg_tokens: Tensor | None = None,
) -> np.ndarray:
    """Run deterministic inference via the fusion contract and return probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(
            tabular_tokens=tabular_tokens,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            kg_tokens=kg_tokens,
        )
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy().astype(np.float64)


def rank_stocks(
    samples: pd.DataFrame,
    *,
    probabilities: np.ndarray | None = None,
    threshold: float = 0.5,
    model: nn.Module | None = None,
    tabular_tokens: Tensor | None = None,
    image_tokens: Tensor | None = None,
    text_tokens: Tensor | None = None,
    kg_tokens: Tensor | None = None,
) -> RankedStocksResult:
    """Rank stocks per day from either supplied probabilities or model inference tensors."""
    if probabilities is None:
        if model is None or tabular_tokens is None:
            raise ValueError(
                "Provide probabilities directly or pass model + tabular_tokens for inference"
            )
        probabilities = predict_fusion_probabilities(
            model,
            tabular_tokens=tabular_tokens,
            image_tokens=image_tokens,
            text_tokens=text_tokens,
            kg_tokens=kg_tokens,
        )

    ranking = build_ranked_predictions(samples=samples, probabilities=probabilities, threshold=threshold)
    return RankedStocksResult(ranking=ranking)


def analyze_stock(
    stock_id: str,
    *,
    as_of_date: str | pd.Timestamp,
    ranked_predictions: pd.DataFrame,
    graph: nx.Graph | None = None,
    returns: pd.DataFrame | None = None,
    lookback_periods: int = 5,
    event_lookback_days: int = 7,
    index_id: str = "NIFTY50",
) -> StockAnalysisResult:
    """Return ranking row and optional KG context for one stock and date."""
    date = pd.to_datetime(as_of_date).normalize()
    rank_scope = ranked_predictions.copy()
    rank_scope["date"] = pd.to_datetime(rank_scope["date"]).dt.normalize()

    mask = (rank_scope["stock_id"].astype(str) == str(stock_id)) & (rank_scope["date"] == date)
    if not mask.any():
        raise KeyError(f"No ranked prediction found for stock_id={stock_id} on date={date.date().isoformat()}")

    row = rank_scope.loc[mask].sort_values("rank").iloc[0].to_dict()
    kg_context = None
    if graph is not None:
        kg_context = retrieve_kg_context(
            graph,
            stock_id=str(stock_id),
            as_of_date=date,
            returns=returns,
            lookback_periods=lookback_periods,
            event_lookback_days=event_lookback_days,
            index_id=index_id,
        )

    return StockAnalysisResult(
        stock_id=str(stock_id),
        as_of_date=date.date().isoformat(),
        ranking_row=row,
        kg_context=kg_context,
    )


def compare_stocks(
    stock_ids: list[str],
    *,
    as_of_date: str | pd.Timestamp,
    ranked_predictions: pd.DataFrame,
    graph: nx.Graph | None = None,
    returns: pd.DataFrame | None = None,
    lookback_periods: int = 5,
    event_lookback_days: int = 7,
    index_id: str = "NIFTY50",
) -> StockComparisonResult:
    """Compare a set of stocks on a given date using ranking and optional KG context."""
    if not stock_ids:
        raise ValueError("stock_ids must not be empty")

    date = pd.to_datetime(as_of_date).normalize()
    rows: list[dict[str, Any]] = []

    for stock_id in stock_ids:
        analysis = analyze_stock(
            stock_id,
            as_of_date=date,
            ranked_predictions=ranked_predictions,
            graph=graph,
            returns=returns,
            lookback_periods=lookback_periods,
            event_lookback_days=event_lookback_days,
            index_id=index_id,
        )
        row = {
            "stock_id": analysis.stock_id,
            "date": analysis.as_of_date,
            "probability": analysis.ranking_row["probability"],
            "predicted_label": analysis.ranking_row["predicted_label"],
            "rank": analysis.ranking_row["rank"],
        }
        if analysis.kg_context is not None:
            row["sector_id"] = analysis.kg_context["sector_id"]
            row["peer_count"] = analysis.kg_context["peer_count"]
            row["peer_avg_recent_return"] = analysis.kg_context["peer_avg_recent_return"]
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(["rank", "stock_id"]).reset_index(drop=True)
    return StockComparisonResult(as_of_date=date.date().isoformat(), comparison=comparison)


def show_peer_graph(
    graph: nx.Graph,
    *,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> PeerGraphResult:
    """Return graph payload and optional rendered file path for peer graph visualization."""
    payload = build_peer_graph_payload(graph)

    image_path: str | None = None
    if output_path is not None:
        image_path = str(plot_peer_graph(graph, output_path=output_path, seed=seed))

    return PeerGraphResult(payload=payload, image_path=image_path)


def show_embedding_map(
    embeddings: np.ndarray,
    *,
    method: ProjectionMethod = "pca",
    metadata: pd.DataFrame | None = None,
    random_state: int = 42,
    tsne_perplexity: float = 30.0,
) -> EmbeddingMapResult:
    """Project model embeddings into 2D for downstream visualization consumers."""
    projection = project_embeddings(
        embeddings,
        method=method,
        metadata=metadata,
        random_state=random_state,
        tsne_perplexity=tsne_perplexity,
    )
    return EmbeddingMapResult(projection=projection)
