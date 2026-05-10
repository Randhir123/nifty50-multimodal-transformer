"""Integration tests for backtest mechanics.

These tests guard the corrected backtest against duplicate prediction rows,
lookahead selection, and sequential compounding of overlapping holding periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_backtest import (
    compute_daily_portfolio_returns,
    select_top_predictions,
    validate_no_duplicate_predictions,
)


def test_no_duplicate_predictions() -> None:
    preds = pd.DataFrame(
        {
            "stock_id": ["A", "B"],
            "end_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "y_prob": [0.6, 0.4],
        }
    )
    validate_no_duplicate_predictions(preds)

    duplicated = pd.concat([preds, preds.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate prediction rows"):
        validate_no_duplicate_predictions(duplicated)


def test_rebalance_dates_le_trading_days() -> None:
    selected, samples, index_returns = _concurrent_case()
    results, legs = compute_daily_portfolio_returns(
        selected,
        samples,
        horizon_days=3,
        index_daily_returns=index_returns,
    )

    assert legs["rebalance_date"].nunique() <= len(results)


def test_concurrent_positions_aggregated_correctly() -> None:
    """Two overlapping 3-day holds are aggregated as daily portfolio returns.

    Day 1: predict stock A as top, hold for 3 days. A returns +1%, +2%, -1%
    on days 2, 3, 4.

    Day 2: predict stock B as top, hold for 3 days. B returns +0.5%, -1%, +2%
    on days 3, 4, 5.

    Daily portfolio returns:
    day 2 = +1% (A only)
    day 3 = (2% + 0.5%) / 2 = +1.25%
    day 4 = (-1% + -1%) / 2 = -1%
    day 5 = +2% (B only)

    Cumulative return = 1.01 * 1.0125 * 0.99 * 1.02 - 1 = 3.2647%.
    """
    selected, samples, index_returns = _concurrent_case()
    results, _ = compute_daily_portfolio_returns(
        selected,
        samples,
        horizon_days=3,
        index_daily_returns=index_returns,
    )

    expected_daily = np.array([0.01, 0.0125, -0.01, 0.02])
    np.testing.assert_allclose(results["portfolio_return"].to_numpy(), expected_daily)

    cumulative = float((1.0 + results["portfolio_return"]).prod() - 1.0)
    assert cumulative == pytest.approx(0.032646725)


def test_no_lookahead_in_selection() -> None:
    merged = pd.DataFrame(
        {
            "end_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "stock_id": ["A", "B"],
            "y_prob": [0.9, 0.1],
            "stock_future_return": [-0.20, 1.50],
        }
    )

    selected = select_top_predictions(merged, top_k=1)

    assert selected.iloc[0]["stock_id"] == "A"


def _concurrent_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    selected = pd.DataFrame(
        {
            "end_date": [dates[0], dates[1]],
            "stock_id": ["A", "B"],
            "y_prob": [0.8, 0.7],
        }
    )
    samples = pd.DataFrame(
        {
            "stock_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "date": [
                dates[0],
                dates[1],
                dates[2],
                dates[3],
                dates[1],
                dates[2],
                dates[3],
                dates[4],
            ],
            "close": [
                100.0,
                101.0,
                103.02,
                101.9898,
                200.0,
                201.0,
                198.99,
                202.9698,
            ],
        }
    )
    index_returns = pd.DataFrame(
        {
            "date": dates,
            "benchmark_return": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return selected, samples, index_returns
