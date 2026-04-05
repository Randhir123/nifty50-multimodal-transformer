"""Operationalization workflows and minimal API entry points."""

from src.app.api import (
    analyze_stock_endpoint,
    compare_stocks_endpoint,
    rank_stocks_endpoint,
    show_embedding_map_endpoint,
    show_peer_graph_endpoint,
)
from src.app.workflows import (
    EmbeddingMapResult,
    PeerGraphResult,
    RankedStocksResult,
    StockAnalysisResult,
    StockComparisonResult,
    analyze_stock,
    compare_stocks,
    predict_fusion_probabilities,
    rank_stocks,
    show_embedding_map,
    show_peer_graph,
)

__all__ = [
    "RankedStocksResult",
    "StockAnalysisResult",
    "StockComparisonResult",
    "PeerGraphResult",
    "EmbeddingMapResult",
    "predict_fusion_probabilities",
    "rank_stocks",
    "analyze_stock",
    "compare_stocks",
    "show_peer_graph",
    "show_embedding_map",
    "rank_stocks_endpoint",
    "analyze_stock_endpoint",
    "compare_stocks_endpoint",
    "show_peer_graph_endpoint",
    "show_embedding_map_endpoint",
]
