from __future__ import annotations

import numpy as np
import pandas as pd

from src.kg.build_graph import build_market_knowledge_graph
from src.viz.embeddings import project_embeddings, project_embeddings_pca, project_embeddings_tsne
from src.viz.peer_graph import build_peer_graph_payload
from src.viz.ranking import build_ranked_predictions


def test_ranked_predictions_unit() -> None:
    samples = pd.DataFrame(
        {
            "stock_id": ["TCS", "INFY", "RELIANCE", "TCS", "INFY", "RELIANCE"],
            "date": [
                "2024-03-01",
                "2024-03-01",
                "2024-03-01",
                "2024-03-02",
                "2024-03-02",
                "2024-03-02",
            ],
        }
    )
    probabilities = np.array([0.6, 0.9, 0.4, 0.51, 0.5, 0.7])

    ranked = build_ranked_predictions(samples, probabilities, threshold=0.5)

    assert list(ranked.columns) == ["stock_id", "date", "probability", "predicted_label", "rank"]
    assert ranked.loc[0, "stock_id"] == "INFY"
    assert ranked.loc[0, "rank"] == 1
    assert ranked[ranked["date"] == pd.Timestamp("2024-03-02")]["rank"].tolist() == [1, 2, 3]


def test_embedding_projection_unit() -> None:
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(30, 8))
    metadata = pd.DataFrame({"sample_id": np.arange(30), "stock_id": [f"S{i % 3}" for i in range(30)]})

    pca_df = project_embeddings_pca(embeddings, metadata=metadata)
    tsne_df = project_embeddings_tsne(embeddings, metadata=metadata, random_state=7, perplexity=10)
    generic_df = project_embeddings(embeddings, method="pca", metadata=metadata)

    assert {"proj_x", "proj_y", "method"}.issubset(pca_df.columns)
    assert (tsne_df["method"] == "tsne").all()
    assert (generic_df["method"] == "pca").all()


def test_peer_graph_payload_unit(toy_event_records: pd.DataFrame) -> None:
    graph = build_market_knowledge_graph(
        {"TCS": "IT", "INFY": "IT", "RELIANCE": "ENERGY"},
        event_records=toy_event_records,
        event_types=["earnings", "guidance"],
    )

    payload = build_peer_graph_payload(graph)

    assert "nodes" in payload and "edges" in payload
    assert len(payload["nodes"]) == graph.number_of_nodes()
    assert any(edge["edge_type"] == "peer_in_sector" for edge in payload["edges"])
