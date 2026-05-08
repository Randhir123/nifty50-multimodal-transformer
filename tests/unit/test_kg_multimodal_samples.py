from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.multimodal_samples import (
    attach_kg_tokens,
    build_kg_tokens_for_samples,
    build_tabular_multimodal_samples,
    save_multimodal_samples,
)
from src.kg.build_graph import build_market_knowledge_graph


def _tabular_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock_id, base in [("AAA.NS", 0), ("BBB.NS", 100)]:
        for i in range(4):
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": f"2024-01-0{i + 1}",
                    "feature_1": float(base + i),
                    "feature_2": float(base + i + 10),
                    "label": int(i % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def _graph():
    events = pd.DataFrame(
        [
            {"stock_id": "AAA.NS", "event_date": "2024-01-03", "event_type": "earnings"},
            {"stock_id": "AAA.NS", "event_date": "2024-01-04", "event_type": "guidance"},
            {"stock_id": "BBB.NS", "event_date": "2024-01-03", "event_type": "guidance"},
        ]
    )
    return build_market_knowledge_graph(
        {"AAA.NS": "IT", "BBB.NS": "IT"}, event_records=events
    )


def _returns() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stock_id": "AAA.NS", "date": "2024-01-01", "recent_return": 0.01},
            {"stock_id": "AAA.NS", "date": "2024-01-02", "recent_return": 0.02},
            {"stock_id": "AAA.NS", "date": "2024-01-03", "recent_return": 0.03},
            {"stock_id": "AAA.NS", "date": "2024-01-04", "recent_return": 0.04},
            {"stock_id": "BBB.NS", "date": "2024-01-01", "recent_return": 0.10},
            {"stock_id": "BBB.NS", "date": "2024-01-02", "recent_return": 0.20},
            {"stock_id": "BBB.NS", "date": "2024-01-03", "recent_return": 0.30},
            {"stock_id": "BBB.NS", "date": "2024-01-04", "recent_return": 0.40},
        ]
    )


def test_build_kg_tokens_for_samples_aligns_to_stock_and_date_rows() -> None:
    tokens = build_kg_tokens_for_samples(
        _graph(),
        stock_ids=["AAA.NS", "AAA.NS", "BBB.NS"],
        end_dates=pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-03"]),
        returns=_returns(),
        lookback_periods=2,
        event_lookback_days=1,
    )

    assert tokens.shape == (3, 5)
    # Token order is: peer_count, peer_avg_recent_return,
    # sector_avg_recent_return, earnings flag, guidance flag.
    np.testing.assert_allclose(tokens[0], np.array([1, 0.25, 0.14, 1, 0], dtype=np.float32))
    np.testing.assert_allclose(tokens[1], np.array([1, 0.35, 0.195, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(tokens[2], np.array([1, 0.025, 0.14, 0, 1], dtype=np.float32))


def test_attach_kg_tokens_preserves_existing_arrays_and_adds_aligned_tokens() -> None:
    arrays = build_tabular_multimodal_samples(
        _tabular_frame(), feature_cols=["feature_1", "feature_2"], window_size=3
    )

    enriched = attach_kg_tokens(
        arrays,
        _graph(),
        returns=_returns(),
        lookback_periods=2,
        event_lookback_days=1,
    )

    enriched.validate()
    np.testing.assert_allclose(enriched.tabular_tokens, arrays.tabular_tokens)
    np.testing.assert_array_equal(enriched.y, arrays.y)
    np.testing.assert_array_equal(enriched.stock_ids, arrays.stock_ids)
    assert enriched.kg_tokens is not None
    assert enriched.kg_tokens.shape[0] == arrays.tabular_tokens.shape[0]


def test_kg_tokens_ignore_future_events_for_earlier_samples() -> None:
    graph = build_market_knowledge_graph(
        {"AAA.NS": "IT", "BBB.NS": "IT"},
        event_records=pd.DataFrame(
            [
                {
                    "stock_id": "AAA.NS",
                    "event_date": "2024-01-04",
                    "event_type": "earnings",
                }
            ]
        ),
    )

    tokens = build_kg_tokens_for_samples(
        graph,
        stock_ids=["AAA.NS", "AAA.NS"],
        end_dates=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        returns=_returns(),
        event_lookback_days=1,
    )

    assert tokens[0, -1] == 0.0
    assert tokens[1, -1] == 1.0


def test_save_multimodal_samples_with_kg_schema(tmp_path) -> None:
    arrays = build_tabular_multimodal_samples(
        _tabular_frame(), feature_cols=["feature_1", "feature_2"], window_size=3
    )
    enriched = attach_kg_tokens(arrays, _graph(), returns=_returns())
    output_path = save_multimodal_samples(enriched, tmp_path / "tabular_kg.npz")

    loaded = np.load(output_path, allow_pickle=False)
    assert set(loaded.files) == {"tabular_tokens", "y", "end_dates", "stock_ids", "kg_tokens"}
    assert loaded["kg_tokens"].shape[0] == loaded["tabular_tokens"].shape[0]
