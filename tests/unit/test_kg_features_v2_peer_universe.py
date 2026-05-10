from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.kg_features_v2 import FEATURE_NAMES, build_kg_v2


def test_peer_universe_must_contain_training() -> None:
    training, benchmark, sectors = _peer_universe()
    peer = {k: v for k, v in training.items() if k != "AAA.NS"}

    with pytest.raises(ValueError, match="peer_ohlcv must contain all training tickers"):
        build_kg_v2(
            training_ohlcv={"AAA.NS": training["AAA.NS"]},
            peer_ohlcv=peer,
            benchmark_ohlcv=benchmark,
            sector_mapping=sectors,
            stock_ids=["AAA.NS"],
            end_dates=[pd.Timestamp("2024-04-30")],
        )


def test_peer_universe_strictly_larger_changes_features() -> None:
    universe, benchmark, sectors = _peer_universe()
    kwargs = dict(
        training_ohlcv={"AAA.NS": universe["AAA.NS"]},
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[pd.Timestamp("2024-04-30")],
    )

    lone = build_kg_v2(peer_ohlcv={"AAA.NS": universe["AAA.NS"]}, **kwargs).values
    expanded = build_kg_v2(peer_ohlcv=universe, **kwargs).values

    peer_cols = [FEATURE_NAMES.index(f"peer_corr_top{i}") for i in range(1, 6)]
    rank_col = FEATURE_NAMES.index("sector_return_rank_5d")
    assert not np.allclose(lone[:, peer_cols], expanded[:, peer_cols])
    assert lone[0, rank_col] != pytest.approx(expanded[0, rank_col])


def test_macro_features_invariant_to_peer_universe() -> None:
    universe, benchmark, sectors = _peer_universe()
    kwargs = dict(
        training_ohlcv={"AAA.NS": universe["AAA.NS"]},
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[pd.Timestamp("2024-04-30")],
    )

    lone = build_kg_v2(peer_ohlcv={"AAA.NS": universe["AAA.NS"]}, **kwargs).values
    expanded = build_kg_v2(peer_ohlcv=universe, **kwargs).values

    macro_cols = [
        FEATURE_NAMES.index(name)
        for name in [
            "nifty_return_5d",
            "nifty_return_20d",
            "nifty_return_60d",
            "nifty_vol_zscore_20d",
            "nifty_vol_term_structure_5d_20d",
        ]
    ]
    np.testing.assert_allclose(lone[:, macro_cols], expanded[:, macro_cols])


def test_peer_data_respects_date_cutoff() -> None:
    universe, benchmark, sectors = _peer_universe()
    poisoned = {ticker: frame.copy() for ticker, frame in universe.items()}
    cutoff = pd.Timestamp("2024-04-30")
    for ticker, frame in poisoned.items():
        if ticker == "AAA.NS":
            continue
        future_mask = frame["date"] > cutoff
        frame.loc[future_mask, "close"] = 999_999.0
        frame.loc[future_mask, "volume"] = 999_999_999.0

    kwargs = dict(
        training_ohlcv={"AAA.NS": universe["AAA.NS"]},
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[cutoff],
    )
    clean = build_kg_v2(peer_ohlcv=universe, **kwargs).values
    changed = build_kg_v2(peer_ohlcv=poisoned, **kwargs).values

    np.testing.assert_allclose(clean, changed)


def test_n_peers_below_5_flag_correct() -> None:
    universe, benchmark, sectors = _peer_universe(n_peers=5)
    kwargs = dict(
        training_ohlcv={"AAA.NS": universe["AAA.NS"]},
        benchmark_ohlcv=benchmark,
        sector_mapping=sectors,
        stock_ids=["AAA.NS"],
        end_dates=[pd.Timestamp("2024-04-30")],
    )
    flag_col = FEATURE_NAMES.index("n_peers_below_5")

    four_peers = {ticker: frame for ticker, frame in universe.items() if ticker != "FFF.NS"}
    five_peers = universe
    assert build_kg_v2(peer_ohlcv=four_peers, **kwargs).values[0, flag_col] == 1.0
    assert build_kg_v2(peer_ohlcv=five_peers, **kwargs).values[0, flag_col] == 0.0


def _peer_universe(*, n_peers: int = 3):
    dates = pd.bdate_range("2024-01-01", periods=100)
    tickers = ["AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS", "EEE.NS", "FFF.NS"][: n_peers + 1]
    slopes = [0.004, 0.001, 0.002, 0.006, 0.003, 0.005]
    x = np.arange(len(dates), dtype=float)
    universe = {
        ticker: _frame(
            dates,
            np.exp(np.cumsum(slopes[i] + 0.001 * np.sin(x / (i + 2)))) * 100.0,
        )
        for i, ticker in enumerate(tickers)
    }
    benchmark = _frame(dates, np.exp(np.cumsum(0.0025 + 0.001 * np.sin(x / 5))) * 1000.0)
    sectors = {ticker: "it" for ticker in tickers}
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
