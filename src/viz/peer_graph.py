"""Peer graph transformation and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx


def build_peer_graph_payload(graph: nx.Graph) -> dict[str, list[dict[str, Any]]]:
    """Convert KG graph into serializable node/edge payload for visualization layers."""
    nodes: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append(
            {
                "id": str(node_id),
                "node_type": str(attrs.get("node_type", "unknown")),
                "entity_id": str(attrs.get("entity_id", node_id)),
            }
        )

    edges: list[dict[str, Any]] = []
    for source, target, attrs in graph.edges(data=True):
        edges.append(
            {
                "source": str(source),
                "target": str(target),
                "edge_type": str(attrs.get("edge_type", "unknown")),
                "event_dates": list(attrs.get("event_dates", ())),
            }
        )

    nodes = sorted(nodes, key=lambda x: x["id"])
    edges = sorted(edges, key=lambda x: (x["source"], x["target"], x["edge_type"]))
    return {"nodes": nodes, "edges": edges}


def plot_peer_graph(
    graph: nx.Graph,
    *,
    output_path: str | Path,
    seed: int = 42,
) -> Path:
    """Create a lightweight peer graph plot artifact for reports or notebooks."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(graph, seed=seed)

    type_to_color = {
        "index": "#2E86AB",
        "sector": "#F18F01",
        "stock": "#28AFB0",
        "event_type": "#C73E1D",
    }
    node_colors = [type_to_color.get(graph.nodes[n].get("node_type"), "#7A7A7A") for n in graph.nodes]

    plt.figure(figsize=(8, 6))
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        labels={n: graph.nodes[n].get("entity_id", n) for n in graph.nodes},
        node_size=850,
        node_color=node_colors,
        font_size=8,
        edge_color="#808080",
    )
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()
    return out
