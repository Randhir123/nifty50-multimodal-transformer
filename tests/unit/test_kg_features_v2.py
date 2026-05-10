from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.kg_features_v2 import FEATURE_NAMES, build_kg_v2
from src.data.sector_mapping import SECTOR_NAMES


def test_output_shape_matches_expected_features() -> None:
    universe, benchmark, sectors = _synthetic_universe()
    result = build_kg_v2(
        universe_ohlcv=universe,
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS", "BBB.NS"],
        end_dates=[pd.Timestamp("2024-04-30"), pd.Timestamp("2024-04-30")],
    )

    assert result.values.shape == (2, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 37


def test_deterministic_output() -> None:
    universe, benchmark, sectors = _synthetic_universe()
    kwargs = dict(
        universe_ohlcv=universe,
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS", "CCC.NS"],
        end_dates=[pd.Timestamp("2024-04-30"), pd.Timestamp("2024-04-30")],
    )

    first = build_kg_v2(**kwargs).values
    second = build_kg_v2(**kwargs).values

    np.testing.assert_array_equal(first, second)


def test_one_hot_sector_sums_to_one() -> None:
    universe, benchmark, sectors = _synthetic_universe()
    result = build_kg_v2(
        universe_ohlcv=universe,
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS", "CCC.NS"],
        end_dates=[pd.Timestamp("2024-04-30"), pd.Timestamp("2024-04-30")],
    )

    one_hot = result.values[:, : len(SECTOR_NAMES)]
    np.testing.assert_array_equal(one_hot.sum(axis=1), np.ones(2))


def test_sector_index_return_matches_manual_compute() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    aaa_close = np.exp(np.arange(10) * 0.01) * 100.0
    bbb_close = np.exp(np.arange(10) * 0.03) * 100.0
    universe = {
        "AAA.NS": _frame(dates, aaa_close),
        "BBB.NS": _frame(dates, bbb_close),
    }
    benchmark = _frame(dates, np.exp(np.arange(10) * 0.02) * 100.0)
    sectors = {"AAA.NS": "it", "BBB.NS": "it"}

    result = build_kg_v2(
        universe_ohlcv=universe,
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[dates[-1]],
    )

    idx = result.feature_names.index("sector_return_5d")
    expected = 5 * ((0.01 + 0.03) / 2.0)
    assert result.values[0, idx] == pytest.approx(expected)


def test_peer_correlation_handles_lone_peer() -> None:
    universe, benchmark, sectors = _synthetic_universe()
    result = build_kg_v2(
        universe_ohlcv=universe,
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["CCC.NS"],
        end_dates=[pd.Timestamp("2024-04-30")],
    )

    peer_cols = [result.feature_names.index(f"peer_corr_top{i}") for i in range(1, 6)]
    np.testing.assert_array_equal(result.values[0, peer_cols], np.zeros(5))
    flag_idx = result.feature_names.index("n_peers_below_5")
    assert result.values[0, flag_idx] == 1.0


def test_lead_lag_features_are_finite() -> None:
    universe, benchmark, sectors = _synthetic_universe(constant_peer=True)
    result = build_kg_v2(
        universe_ohlcv=universe,
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[pd.Timestamp("2024-04-30")],
    )

    for name in ["lead_lag_peer_t_minus_1", "lead_lag_peer_t_minus_2"]:
        assert np.isfinite(result.values[0, result.feature_names.index(name)])


def test_features_use_only_past_data() -> None:
    universe, benchmark, sectors = _synthetic_universe()
    changed = {ticker: frame.copy() for ticker, frame in universe.items()}
    for frame in changed.values():
        future_mask = frame["date"] > pd.Timestamp("2024-04-30")
        frame.loc[future_mask, "close"] = 999_999.0
        frame.loc[future_mask, "volume"] = 999_999_999.0

    kwargs = dict(
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS", "BBB.NS", "CCC.NS"],
        end_dates=[pd.Timestamp("2024-04-30")] * 3,
    )
    clean = build_kg_v2(universe_ohlcv=universe, **kwargs).values
    poisoned = build_kg_v2(universe_ohlcv=changed, **kwargs).values

    np.testing.assert_allclose(clean, poisoned)


def test_macro_features_use_benchmark_only_past() -> None:
    universe, benchmark, sectors = _synthetic_universe()
    poisoned_benchmark = benchmark.copy()
    future_mask = poisoned_benchmark["date"] > pd.Timestamp("2024-04-30")
    poisoned_benchmark.loc[future_mask, "close"] = 999_999.0

    kwargs = dict(
        universe_ohlcv=universe,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[pd.Timestamp("2024-04-30")],
    )
    clean = build_kg_v2(benchmark_ohlcv=benchmark, **kwargs).values
    poisoned = build_kg_v2(benchmark_ohlcv=poisoned_benchmark, **kwargs).values

    macro_names = [
        "nifty_return_5d",
        "nifty_return_20d",
        "nifty_return_60d",
        "nifty_vol_zscore_20d",
        "nifty_vol_term_structure_5d_20d",
    ]
    cols = [FEATURE_NAMES.index(name) for name in macro_names]
    np.testing.assert_allclose(clean[:, cols], poisoned[:, cols])


def _synthetic_universe(*, constant_peer: bool = False):
    dates = pd.bdate_range("2024-01-01", periods=100)
    base = np.arange(100, dtype=float)
    universe = {
        "AAA.NS": _frame(dates, 100.0 + base * 0.4),
        "BBB.NS": _frame(dates, np.full(100, 100.0) if constant_peer else 100.0 + base * 0.3),
        "CCC.NS": _frame(dates, 200.0 + base * 0.2),
    }
    benchmark = _frame(dates, 1000.0 + base * 0.25)
    sectors = {"AAA.NS": "it", "BBB.NS": "it", "CCC.NS": "energy"}
    return universe, benchmark, sectors


def _frame(dates: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(close)),
        }
    )
