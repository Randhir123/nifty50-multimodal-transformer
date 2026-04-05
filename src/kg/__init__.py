"""Knowledge-graph construction and retrieval interfaces."""

from src.kg.build_graph import build_market_knowledge_graph
from src.kg.query_graph import (
    get_event_flags,
    get_peer_ids,
    get_stock_sector_id,
    kg_context_to_feature_dict,
    retrieve_kg_context,
    retrieve_kg_context_for_samples,
)

__all__ = [
    "build_market_knowledge_graph",
    "get_stock_sector_id",
    "get_peer_ids",
    "get_event_flags",
    "retrieve_kg_context",
    "retrieve_kg_context_for_samples",
    "kg_context_to_feature_dict",
]
