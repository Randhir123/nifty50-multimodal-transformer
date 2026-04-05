"""Build a lightweight, deterministic market knowledge graph.

Milestone 7 keeps the graph coursework-scale and explicit:

- nodes: index, sector, stock, event_type
- edges:
  - index -> sector
  - sector -> stock
  - stock <-> stock (peer link if same sector)
  - stock -> event_type (with dated event history)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import networkx as nx
import pandas as pd

NodeId = str


def index_node_id(index_id: str) -> NodeId:
    return f"index:{index_id}"


def sector_node_id(sector_id: str) -> NodeId:
    return f"sector:{sector_id}"


def stock_node_id(stock_id: str) -> NodeId:
    return f"stock:{stock_id}"


def event_type_node_id(event_type: str) -> NodeId:
    return f"event_type:{event_type}"


def _normalize_event_records(event_records: pd.DataFrame | None) -> pd.DataFrame:
    if event_records is None:
        return pd.DataFrame(columns=["stock_id", "event_date", "event_type"])

    required = {"stock_id", "event_date", "event_type"}
    missing = required - set(event_records.columns)
    if missing:
        raise ValueError(f"event_records missing required columns: {sorted(missing)}")

    normalized = event_records.loc[:, ["stock_id", "event_date", "event_type"]].copy()
    normalized["stock_id"] = normalized["stock_id"].astype(str).str.strip()
    normalized["event_type"] = normalized["event_type"].astype(str).str.strip().str.lower()
    normalized["event_date"] = pd.to_datetime(normalized["event_date"]).dt.normalize()

    normalized = normalized.dropna(subset=["event_date"]) \
        .query("stock_id != '' and event_type != ''") \
        .drop_duplicates(subset=["stock_id", "event_type", "event_date"]) \
        .sort_values(["stock_id", "event_type", "event_date"])

    return normalized.reset_index(drop=True)


def build_market_knowledge_graph(
    stock_to_sector: Mapping[str, str],
    *,
    index_id: str = "NIFTY50",
    event_types: Iterable[str] | None = None,
    event_records: pd.DataFrame | None = None,
) -> nx.Graph:
    """Build a deterministic undirected graph for market entities.

    Args:
        stock_to_sector: Mapping from stock ID to sector ID.
        index_id: Canonical index identifier.
        event_types: Optional universe of allowed event types.
        event_records: Optional dated events with columns
            ``stock_id``, ``event_date``, and ``event_type``.

    Returns:
        networkx.Graph instance with typed nodes/edges and minimal attributes.
    """
    if not stock_to_sector:
        raise ValueError("stock_to_sector must not be empty")

    graph = nx.Graph()

    normalized_pairs = sorted(
        (str(stock).strip(), str(sector).strip()) for stock, sector in stock_to_sector.items()
    )
    if any(not stock or not sector for stock, sector in normalized_pairs):
        raise ValueError("stock_to_sector cannot contain empty stock or sector IDs")

    index_node = index_node_id(index_id)
    graph.add_node(index_node, node_type="index", entity_id=str(index_id))

    sectors = sorted({sector for _, sector in normalized_pairs})
    for sector in sectors:
        sector_node = sector_node_id(sector)
        graph.add_node(sector_node, node_type="sector", entity_id=sector)
        graph.add_edge(index_node, sector_node, edge_type="index_contains_sector")

    sector_to_stocks: dict[str, list[str]] = defaultdict(list)
    for stock, sector in normalized_pairs:
        stock_node = stock_node_id(stock)
        sector_node = sector_node_id(sector)
        graph.add_node(stock_node, node_type="stock", entity_id=stock)
        graph.add_edge(sector_node, stock_node, edge_type="sector_contains_stock")
        sector_to_stocks[sector].append(stock)

    for stocks in sector_to_stocks.values():
        ordered = sorted(stocks)
        for i, left_stock in enumerate(ordered):
            left_node = stock_node_id(left_stock)
            for right_stock in ordered[i + 1 :]:
                right_node = stock_node_id(right_stock)
                graph.add_edge(left_node, right_node, edge_type="peer_in_sector")

    normalized_events = _normalize_event_records(event_records)
    event_type_universe = sorted(
        {
            *(str(event_type).strip().lower() for event_type in (event_types or [])),
            *normalized_events["event_type"].tolist(),
        }
    )
    event_type_universe = [event_type for event_type in event_type_universe if event_type]

    for event_type in event_type_universe:
        event_node = event_type_node_id(event_type)
        graph.add_node(event_node, node_type="event_type", entity_id=event_type)

    if not normalized_events.empty:
        grouped = (
            normalized_events.groupby(["stock_id", "event_type"], as_index=False)["event_date"]
            .apply(list)
            .reset_index(drop=True)
        )
        for row in grouped.itertuples(index=False):
            stock_node = stock_node_id(row.stock_id)
            if stock_node not in graph:
                continue
            event_node = event_type_node_id(row.event_type)
            graph.add_edge(
                stock_node,
                event_node,
                edge_type="stock_has_event_type",
                event_dates=tuple(d.date().isoformat() for d in row.event_date),
            )

    return graph
