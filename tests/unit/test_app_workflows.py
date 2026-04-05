from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.app.api import (
    analyze_stock_endpoint,
    compare_stocks_endpoint,
    rank_stocks_endpoint,
    show_embedding_map_endpoint,
    show_peer_graph_endpoint,
)
from src.app.workflows import (
    analyze_stock,
    compare_stocks,
    rank_stocks,
    show_embedding_map,
    show_peer_graph,
)
from src.kg.build_graph import build_market_knowledge_graph


class DummyFusionModel(nn.Module):
    def forward(
        self,
        *,
        tabular_tokens: Tensor,
        image_tokens: Tensor | None = None,
        text_tokens: Tensor | None = None,
        kg_tokens: Tensor | None = None,
    ) -> Tensor:
        _ = image_tokens, text_tokens, kg_tokens
        # [batch, seq, feat] -> [batch]
        return tabular_tokens.mean(dim=(1, 2))


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stock_id": ["TCS", "INFY", "RELIANCE"],
            "date": ["2024-03-02", "2024-03-02", "2024-03-02"],
        }
    )


def test_rank_stocks_from_probabilities() -> None:
    ranked = rank_stocks(
        _sample_rows(),
        probabilities=np.array([0.4, 0.9, 0.7]),
    ).ranking

    assert ranked.loc[0, "stock_id"] == "INFY"
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_rank_stocks_from_model_inference() -> None:
    model = DummyFusionModel()
    tabular_tokens = torch.tensor(
        [
            [[0.2, 0.2], [0.2, 0.2]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=torch.float32,
    )

    ranked = rank_stocks(_sample_rows(), model=model, tabular_tokens=tabular_tokens).ranking

    assert ranked["stock_id"].tolist() == ["INFY", "RELIANCE", "TCS"]


def test_analyze_compare_peer_graph_and_embedding_workflows(toy_event_records: pd.DataFrame, tmp_path) -> None:
    ranked_predictions = rank_stocks(
        _sample_rows(),
        probabilities=np.array([0.4, 0.9, 0.7]),
    ).ranking
    graph = build_market_knowledge_graph(
        {"TCS": "IT", "INFY": "IT", "RELIANCE": "ENERGY"},
        event_records=toy_event_records,
        event_types=["earnings", "guidance"],
    )

    analysis = analyze_stock(
        "INFY",
        as_of_date="2024-03-02",
        ranked_predictions=ranked_predictions,
        graph=graph,
    )
    assert analysis.ranking_row["rank"] == 1
    assert analysis.kg_context is not None
    assert analysis.kg_context["sector_id"] == "IT"

    comparison = compare_stocks(
        ["TCS", "INFY"],
        as_of_date="2024-03-02",
        ranked_predictions=ranked_predictions,
        graph=graph,
    ).comparison
    assert comparison["stock_id"].tolist() == ["INFY", "TCS"]

    graph_result = show_peer_graph(graph, output_path=tmp_path / "peer_graph.png")
    assert graph_result.image_path is not None

    embeddings = np.array([[1.0, 0.0, 0.2], [0.8, 0.1, 0.4], [0.1, 0.9, 0.3]])
    map_result = show_embedding_map(
        embeddings,
        metadata=pd.DataFrame({"sample_id": [0, 1, 2], "stock_id": ["A", "B", "C"]}),
    )
    assert {"proj_x", "proj_y", "method"}.issubset(map_result.projection.columns)


def test_api_endpoint_surfaces(toy_event_records: pd.DataFrame, tmp_path) -> None:
    graph = build_market_knowledge_graph(
        {"TCS": "IT", "INFY": "IT", "RELIANCE": "ENERGY"},
        event_records=toy_event_records,
        event_types=["earnings", "guidance"],
    )
    rank_payload = rank_stocks_endpoint(
        samples=_sample_rows(),
        probabilities=np.array([0.4, 0.9, 0.7]),
    )
    assert "ranking" in rank_payload

    analysis_payload = analyze_stock_endpoint(
        stock_id="INFY",
        as_of_date="2024-03-02",
        ranked_predictions=rank_payload["ranking"],
        graph=graph,
    )
    assert analysis_payload["stock_id"] == "INFY"

    compare_payload = compare_stocks_endpoint(
        stock_ids=["TCS", "INFY"],
        as_of_date="2024-03-02",
        ranked_predictions=rank_payload["ranking"],
        graph=graph,
    )
    assert not compare_payload["comparison"].empty

    peer_graph_payload = show_peer_graph_endpoint(graph=graph, output_path=tmp_path / "peer_graph_api.png")
    assert peer_graph_payload["image_path"] is not None

    embedding_payload = show_embedding_map_endpoint(
        embeddings=np.array([[1.0, 0.0, 0.2], [0.8, 0.1, 0.4], [0.1, 0.9, 0.3]]),
        metadata=pd.DataFrame({"sample_id": [0, 1, 2]}),
    )
    assert "projection" in embedding_payload
