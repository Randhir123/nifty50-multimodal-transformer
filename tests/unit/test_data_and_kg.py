from __future__ import annotations

import pandas as pd

from src.data.dataset import create_rolling_transformer_dataset
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.kg.build_graph import build_market_knowledge_graph
from src.kg.query_graph import retrieve_kg_context


FEATURE_COLS = [
    "log_return_1d",
    "cum_return_3d",
    "cum_return_5d",
    "realized_vol_5d",
    "high_low_range_over_close",
    "stock_minus_index_return",
]


def test_feature_generation_unit(toy_ohlcv: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    stock_df, index_df = toy_ohlcv
    featured = compute_technical_features(stock_df, index_df)

    assert "log_return_1d" in featured.columns
    assert "stock_minus_index_return" in featured.columns
    assert featured["stock_minus_index_return"].notna().sum() > 0


def test_label_generation_unit(toy_ohlcv: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    stock_df, index_df = toy_ohlcv
    featured = compute_technical_features(stock_df, index_df)

    labeled = generate_outperformance_label(featured)
    assert "label" in labeled.columns
    assert labeled["label"].dropna().isin([0, 1]).all()


def test_rolling_window_dataset_creation_unit(toy_ohlcv: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    stock_df, index_df = toy_ohlcv
    featured = compute_technical_features(stock_df, index_df)
    labeled = generate_outperformance_label(featured)

    windows = create_rolling_transformer_dataset(
        labeled,
        feature_cols=FEATURE_COLS,
        window_size=20,
        dropna=True,
    )

    assert windows.X.ndim == 3
    assert windows.X.shape[-1] == len(FEATURE_COLS)
    assert windows.X.shape[0] == windows.y.shape[0] == windows.end_dates.shape[0]


def test_kg_context_retrieval_unit(toy_event_records: pd.DataFrame) -> None:
    returns = pd.DataFrame(
        {
            "stock_id": ["TCS", "TCS", "INFY", "INFY"],
            "date": ["2024-03-01", "2024-03-05", "2024-03-01", "2024-03-05"],
            "recent_return": [0.01, 0.015, 0.009, 0.011],
        }
    )
    returns["date"] = pd.to_datetime(returns["date"])

    graph = build_market_knowledge_graph(
        {"TCS": "IT", "INFY": "IT"},
        event_records=toy_event_records,
        event_types=["earnings", "guidance"],
    )

    context = retrieve_kg_context(graph, stock_id="TCS", as_of_date="2024-03-06", returns=returns)

    assert context["sector_id"] == "IT"
    assert context["peer_count"] == 1
    assert "event_flags" in context
