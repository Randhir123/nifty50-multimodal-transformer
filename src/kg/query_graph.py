"""Query helpers for retrieving normalized KG context per sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from src.kg.build_graph import (
    sector_node_id,
    stock_node_id,
)


@dataclass(frozen=True)
class KGContext:
    """Normalized context output for one ``(stock_id, date)`` sample."""

    stock_id: str
    as_of_date: str
    index_id: str
    sector_id: str
    peer_ids: tuple[str, ...]
    peer_count: int
    peer_avg_recent_return: float | None
    sector_avg_recent_return: float | None
    event_flags: dict[str, int]
    schema_version: str = "kg_context_v1"

    def to_normalized_dict(self) -> dict[str, Any]:
        """Return deployment-friendly primitive-only dictionary payload."""
        return {
            "schema_version": self.schema_version,
            "stock_id": self.stock_id,
            "as_of_date": self.as_of_date,
            "index_id": self.index_id,
            "sector_id": self.sector_id,
            "peer_ids": list(self.peer_ids),
            "peer_count": self.peer_count,
            "peer_avg_recent_return": self.peer_avg_recent_return,
            "sector_avg_recent_return": self.sector_avg_recent_return,
            "event_flags": dict(sorted(self.event_flags.items())),
        }


def _recent_stock_return(
    returns: pd.DataFrame,
    *,
    stock_id: str,
    as_of_date: pd.Timestamp,
    lookback_periods: int,
) -> float | None:
    scope = returns.loc[
        (returns["stock_id"] == stock_id) & (returns["date"] <= as_of_date),
        ["date", "recent_return"],
    ].sort_values("date")
    if scope.empty:
        return None
    values = scope["recent_return"].tail(lookback_periods)
    if values.empty:
        return None
    return float(values.mean())


def _normalize_returns(returns: pd.DataFrame | None) -> pd.DataFrame:
    if returns is None:
        return pd.DataFrame(columns=["stock_id", "date", "recent_return"])

    required = {"stock_id", "date", "recent_return"}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"returns missing required columns: {sorted(missing)}")

    out = returns.loc[:, ["stock_id", "date", "recent_return"]].copy()
    out["stock_id"] = out["stock_id"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.dropna(subset=["date", "recent_return"])
    out = out.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return out


def get_stock_sector_id(graph: nx.Graph, *, stock_id: str) -> str:
    """Return sector ID for a stock from graph topology."""
    stock_node = stock_node_id(stock_id)
    if stock_node not in graph:
        raise KeyError(f"Unknown stock_id: {stock_id}")

    sectors = [
        graph.nodes[n]["entity_id"]
        for n in graph.neighbors(stock_node)
        if graph.nodes[n].get("node_type") == "sector"
    ]
    if len(sectors) != 1:
        raise ValueError(
            f"Expected exactly one sector for stock_id={stock_id}, got {sectors}"
        )
    return sectors[0]


def get_peer_ids(graph: nx.Graph, *, stock_id: str) -> list[str]:
    """Return sorted stock peers from same sector (excluding stock_id itself)."""
    sector_id = get_stock_sector_id(graph, stock_id=stock_id)
    sector_node = sector_node_id(sector_id)
    peers = [
        graph.nodes[n]["entity_id"]
        for n in graph.neighbors(sector_node)
        if graph.nodes[n].get("node_type") == "stock"
        and graph.nodes[n].get("entity_id") != stock_id
    ]
    return sorted(peers)


def get_event_flags(
    graph: nx.Graph,
    *,
    stock_id: str,
    as_of_date: str | pd.Timestamp,
    lookback_days: int = 7,
) -> dict[str, int]:
    """Return binary flags per event type active in the lookback window."""
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")

    cutoff = pd.to_datetime(as_of_date).normalize()
    start = cutoff - pd.Timedelta(days=lookback_days - 1)

    all_event_types = sorted(
        graph.nodes[node]["entity_id"]
        for node in graph.nodes
        if graph.nodes[node].get("node_type") == "event_type"
    )
    flags = {event_type: 0 for event_type in all_event_types}

    stock_node = stock_node_id(stock_id)
    if stock_node not in graph:
        return flags

    for neighbor in graph.neighbors(stock_node):
        if graph.nodes[neighbor].get("node_type") != "event_type":
            continue
        event_type = graph.nodes[neighbor]["entity_id"]
        edge_dates = graph.edges[stock_node, neighbor].get("event_dates", ())
        for raw in edge_dates:
            event_date = pd.to_datetime(raw).normalize()
            if start <= event_date <= cutoff:
                flags[event_type] = 1
                break

    return dict(sorted(flags.items()))


def retrieve_kg_context(
    graph: nx.Graph,
    *,
    stock_id: str,
    as_of_date: str | pd.Timestamp,
    returns: pd.DataFrame | None = None,
    lookback_periods: int = 5,
    event_lookback_days: int = 7,
    index_id: str = "NIFTY50",
) -> dict[str, Any]:
    """Build one normalized KG context payload for a stock-date sample.

    ``returns`` is optional but, when provided, must include ``stock_id``, ``date``,
    and ``recent_return`` columns.
    """
    if lookback_periods <= 0:
        raise ValueError("lookback_periods must be > 0")

    as_of = pd.to_datetime(as_of_date).normalize()
    normalized_returns = _normalize_returns(returns)

    sector_id = get_stock_sector_id(graph, stock_id=stock_id)
    peer_ids = get_peer_ids(graph, stock_id=stock_id)

    peer_returns: list[float] = []
    sector_returns: list[float] = []

    sector_stocks = [stock_id, *peer_ids]
    for sid in sector_stocks:
        avg_recent = _recent_stock_return(
            normalized_returns,
            stock_id=sid,
            as_of_date=as_of,
            lookback_periods=lookback_periods,
        )
        if avg_recent is None:
            continue
        sector_returns.append(avg_recent)
        if sid != stock_id:
            peer_returns.append(avg_recent)

    peer_avg_recent_return = (
        float(sum(peer_returns) / len(peer_returns)) if peer_returns else None
    )
    sector_avg_recent_return = (
        float(sum(sector_returns) / len(sector_returns)) if sector_returns else None
    )

    context = KGContext(
        stock_id=str(stock_id),
        as_of_date=as_of.date().isoformat(),
        index_id=str(index_id),
        sector_id=sector_id,
        peer_ids=tuple(peer_ids),
        peer_count=len(peer_ids),
        peer_avg_recent_return=peer_avg_recent_return,
        sector_avg_recent_return=sector_avg_recent_return,
        event_flags=get_event_flags(
            graph,
            stock_id=stock_id,
            as_of_date=as_of,
            lookback_days=event_lookback_days,
        ),
    )
    return context.to_normalized_dict()


def retrieve_kg_context_for_samples(
    graph: nx.Graph,
    samples: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "date",
    returns: pd.DataFrame | None = None,
    lookback_periods: int = 5,
    event_lookback_days: int = 7,
    index_id: str = "NIFTY50",
    output_col: str = "kg_context",
) -> pd.DataFrame:
    """Attach normalized KG context dictionaries to a samples dataframe."""
    if stock_col not in samples.columns or date_col not in samples.columns:
        raise ValueError(f"samples must include columns '{stock_col}' and '{date_col}'")

    out = samples.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    out[output_col] = [
        retrieve_kg_context(
            graph,
            stock_id=row[stock_col],
            as_of_date=row[date_col],
            returns=returns,
            lookback_periods=lookback_periods,
            event_lookback_days=event_lookback_days,
            index_id=index_id,
        )
        for _, row in out.iterrows()
    ]
    return out


def kg_context_to_feature_dict(context: dict[str, Any]) -> dict[str, Any]:
    """Flatten normalized KG context into model-feature friendly key/value pairs."""
    feature_dict: dict[str, Any] = {
        "kg_sector_id": context["sector_id"],
        "kg_peer_count": context["peer_count"],
        "kg_peer_avg_recent_return": context["peer_avg_recent_return"],
        "kg_sector_avg_recent_return": context["sector_avg_recent_return"],
    }
    for event_type, flag in context.get("event_flags", {}).items():
        feature_dict[f"kg_event_{event_type}"] = int(flag)
    return feature_dict
