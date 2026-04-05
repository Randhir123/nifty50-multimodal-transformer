"""Run a tiny end-to-end demo using the src.app workflow/API surface.

This script intentionally uses a minimal synthetic inference input and the toy
knowledge-graph event records committed under ``data/toy`` so reviewers can run
it quickly without training checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.api import (
    rank_stocks_endpoint,
    show_embedding_map_endpoint,
    show_peer_graph_endpoint,
)
from src.kg.build_graph import build_market_knowledge_graph


def _build_tiny_samples() -> pd.DataFrame:
    """Return a tiny deterministic sample batch for ranking."""
    return pd.DataFrame(
        {
            "stock_id": ["TCS", "INFY", "RELIANCE"],
            "date": ["2024-03-02", "2024-03-02", "2024-03-02"],
        }
    )


def _build_tiny_embeddings() -> tuple[np.ndarray, pd.DataFrame]:
    """Return deterministic embeddings + metadata for projection."""
    embeddings = np.array(
        [
            [1.00, 0.05, 0.20, 0.40],
            [0.85, 0.15, 0.35, 0.30],
            [0.10, 0.95, 0.20, 0.50],
        ],
        dtype=np.float64,
    )
    metadata = pd.DataFrame(
        {
            "sample_id": [0, 1, 2],
            "stock_id": ["TCS", "INFY", "RELIANCE"],
            "as_of_date": ["2024-03-02", "2024-03-02", "2024-03-02"],
        }
    )
    return embeddings, metadata


def run_demo(output_dir: Path) -> None:
    """Run ranking + embedding map + peer graph demo and write artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _build_tiny_samples()
    ranking_payload = rank_stocks_endpoint(
        samples=samples,
        probabilities=np.array([0.62, 0.86, 0.73], dtype=np.float64),
        threshold=0.5,
    )
    ranking_df = ranking_payload["ranking"]
    ranking_path = output_dir / "ranked_stocks.csv"
    ranking_df.to_csv(ranking_path, index=False)

    stock_to_sector = {
        "TCS": "IT",
        "INFY": "IT",
        "RELIANCE": "ENERGY",
    }
    event_records = pd.read_csv("data/toy/event_records.csv")
    graph = build_market_knowledge_graph(
        stock_to_sector,
        event_records=event_records,
        event_types=["earnings", "guidance"],
    )
    peer_graph_image_path = output_dir / "peer_graph.png"
    peer_graph_payload = show_peer_graph_endpoint(
        graph=graph,
        output_path=peer_graph_image_path,
    )
    peer_graph_json_path = output_dir / "peer_graph_payload.json"
    peer_graph_json_path.write_text(
        json.dumps(peer_graph_payload["payload"], indent=2), encoding="utf-8"
    )

    embeddings, embedding_metadata = _build_tiny_embeddings()
    embedding_payload = show_embedding_map_endpoint(
        embeddings=embeddings,
        method="pca",
        metadata=embedding_metadata,
    )
    embedding_projection_path = output_dir / "embedding_projection.csv"
    embedding_payload["projection"].to_csv(embedding_projection_path, index=False)

    print("Demo completed successfully.")
    print(f"- Ranked output: {ranking_path}")
    print(f"- Embedding projection output: {embedding_projection_path}")
    print(f"- Peer graph payload: {peer_graph_json_path}")
    print(f"- Peer graph image: {peer_graph_payload['image_path']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny end-to-end demo workflow")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/demo"),
        help="Directory where demo artifacts will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.output_dir)
