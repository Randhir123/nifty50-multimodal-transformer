"""Leakage-safe relational KG v2 feature builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.data.sector_mapping import SECTOR_NAMES, sector_for_ticker

EPS = 1e-8
ROTATION_SECTORS = ("banking", "it", "energy", "fmcg")


FEATURE_NAMES: tuple[str, ...] = (
    *(f"sector_onehot_{sector}" for sector in SECTOR_NAMES),
    "sector_return_5d",
    "sector_return_20d",
    "sector_beta_60d",
    "sector_vol_ratio_20d",
    *(f"peer_corr_top{i}" for i in range(1, 6)),
    "sector_return_rank_5d",
    "stock_return_zscore_5d",
    "volume_zscore_5d",
    "stock_return_zscore_20d",
    "lead_lag_peer_t_minus_1",
    "lead_lag_peer_t_minus_2",
    "peer_dispersion_5d",
    "stock_peer_spread_5d",
    "stock_peer_spread_20d",
    "nifty_return_5d",
    "nifty_return_20d",
    "nifty_return_60d",
    "nifty_vol_zscore_20d",
    "nifty_vol_term_structure_5d_20d",
    *(f"sector_rotation_leader_{sector}" for sector in ROTATION_SECTORS),
    "sector_rotation_leader_other",
    "n_peers_below_5",
)


@dataclass(frozen=True)
class KGV2FeatureResult:
    """KG v2 feature matrix and column names."""

    values: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


def build_kg_v2(
    *,
    universe_ohlcv: dict[str, pd.DataFrame],
    benchmark_ohlcv: pd.DataFrame,
    sector_mapping: dict[str, str] | None,
    stock_ids: Sequence[str],
    end_dates: Sequence[object],
) -> KGV2FeatureResult:
    """Build leakage-safe sector, peer, and regime features per sample."""
    if len(stock_ids) != len(end_dates):
        raise ValueError("stock_ids and end_dates must have identical lengths")
    if not universe_ohlcv:
        raise ValueError("universe_ohlcv must not be empty")

    prepared = {
        str(ticker): _prepare_ohlcv(df)
        for ticker, df in universe_ohlcv.items()
    }
    benchmark = _prepare_ohlcv(benchmark_ohlcv)
    ticker_to_sector = {
        ticker: sector_for_ticker(ticker, sector_mapping)
        for ticker in prepared
    }

    rows = [
        _build_one_row(
            stock_id=str(stock_id),
            end_date=pd.Timestamp(end_date).normalize(),
            prepared=prepared,
            benchmark=benchmark,
            ticker_to_sector=ticker_to_sector,
        )
        for stock_id, end_date in zip(stock_ids, end_dates, strict=True)
    ]
    values = np.asarray(rows, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return KGV2FeatureResult(values=values, feature_names=FEATURE_NAMES)


def _prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "close", "volume"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"OHLCV frame missing required columns: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["log_return"] = np.log(out["close"].astype(float)).diff()
    out["volume"] = out["volume"].astype(float)
    return out


def _build_one_row(
    *,
    stock_id: str,
    end_date: pd.Timestamp,
    prepared: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    ticker_to_sector: dict[str, str],
) -> list[float]:
    if stock_id not in prepared:
        raise ValueError(f"stock_id {stock_id!r} not found in universe_ohlcv")
    sector = ticker_to_sector.get(stock_id, "infra_other")
    sector_members = [
        ticker for ticker, ticker_sector in ticker_to_sector.items() if ticker_sector == sector
    ]
    peers = [ticker for ticker in sector_members if ticker != stock_id]

    stock_hist = _history(prepared[stock_id], end_date)
    benchmark_hist = _history(benchmark, end_date)
    sector_returns = _sector_return_frame(prepared, sector_members, end_date)
    peer_returns = _sector_return_frame(prepared, peers, end_date)

    values: list[float] = [1.0 if sector == name else 0.0 for name in SECTOR_NAMES]
    values.extend(
        [
            _window_log_return(sector_returns["sector_return"], 5),
            _window_log_return(sector_returns["sector_return"], 20),
            _rolling_beta(sector_returns["sector_return"], benchmark_hist["log_return"], 60),
            _realized_vol(sector_returns["sector_return"], 20)
            / (_realized_vol(benchmark_hist["log_return"], 20) + EPS),
        ]
    )

    values.extend(_top_peer_correlations(prepared, stock_id, peers, end_date, window=60))
    values.extend(
        [
            _sector_rank_5d(prepared, stock_id, sector_members, end_date),
            _sector_zscore(prepared, stock_id, sector_members, end_date, window=5, field="return"),
            _sector_zscore(prepared, stock_id, sector_members, end_date, window=5, field="volume"),
            _sector_zscore(prepared, stock_id, sector_members, end_date, window=20, field="return"),
            _lead_lag(stock_hist["log_return"], peer_returns["sector_return"], lag=1, window=20),
            _lead_lag(stock_hist["log_return"], peer_returns["sector_return"], lag=2, window=20),
            _peer_dispersion_5d(prepared, peers, end_date),
            _stock_peer_spread(prepared, stock_id, peers, end_date, window=5),
            _stock_peer_spread(prepared, stock_id, peers, end_date, window=20),
        ]
    )

    values.extend(
        [
            _window_log_return(benchmark_hist["log_return"], 5),
            _window_log_return(benchmark_hist["log_return"], 20),
            _window_log_return(benchmark_hist["log_return"], 60),
            _vol_zscore(benchmark_hist["log_return"]),
            _realized_vol(benchmark_hist["log_return"], 5)
            / (_realized_vol(benchmark_hist["log_return"], 20) + EPS),
        ]
    )
    values.extend(_sector_rotation(prepared, ticker_to_sector, end_date))
    values.append(float(len(peers) < 5))
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError(f"KG v2 feature count mismatch: {len(values)} != {len(FEATURE_NAMES)}")
    return values


def _history(df: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    return df.loc[df["date"] <= end_date].copy()


def _window_log_return(returns: pd.Series, window: int) -> float:
    clean = returns.dropna().tail(window)
    if clean.empty:
        return 0.0
    return float(clean.sum())


def _realized_vol(returns: pd.Series, window: int) -> float:
    clean = returns.dropna().tail(window)
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=0))


def _sector_return_frame(
    prepared: dict[str, pd.DataFrame],
    members: Sequence[str],
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frames = []
    for ticker in members:
        hist = _history(prepared[ticker], end_date)
        frames.append(hist.loc[:, ["date", "log_return"]].rename(columns={"log_return": ticker}))
    if not frames:
        return pd.DataFrame({"date": [], "sector_return": []})
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date")
    value_cols = [c for c in merged.columns if c != "date"]
    merged["sector_return"] = merged[value_cols].mean(axis=1)
    return merged.loc[:, ["date", "sector_return"]]


def _aligned_tail(left: pd.Series, right: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna().tail(window)
    if len(frame) < 2:
        return np.array([]), np.array([])
    return frame["left"].to_numpy(dtype=float), frame["right"].to_numpy(dtype=float)


def _rolling_beta(sector_returns: pd.Series, benchmark_returns: pd.Series, window: int) -> float:
    x, y = _aligned_tail(benchmark_returns.reset_index(drop=True), sector_returns.reset_index(drop=True), window)
    if len(x) < 2 or np.var(x) <= EPS:
        return 0.0
    return float(np.cov(x, y, ddof=0)[0, 1] / (np.var(x) + EPS))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2 or np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _top_peer_correlations(
    prepared: dict[str, pd.DataFrame],
    stock_id: str,
    peers: Sequence[str],
    end_date: pd.Timestamp,
    *,
    window: int,
) -> list[float]:
    stock = _history(prepared[stock_id], end_date).loc[:, ["date", "log_return"]]
    values = []
    for peer in peers:
        peer_df = _history(prepared[peer], end_date).loc[:, ["date", "log_return"]]
        merged = stock.merge(peer_df, on="date", suffixes=("_stock", "_peer")).dropna().tail(window)
        values.append(_corr(merged["log_return_stock"].to_numpy(), merged["log_return_peer"].to_numpy()))
    values = sorted(values, key=abs, reverse=True)[:5]
    return values + [0.0] * (5 - len(values))


def _stock_window_return(prepared: dict[str, pd.DataFrame], ticker: str, end_date: pd.Timestamp, window: int) -> float:
    hist = _history(prepared[ticker], end_date)
    return _window_log_return(hist["log_return"], window)


def _sector_rank_5d(
    prepared: dict[str, pd.DataFrame],
    stock_id: str,
    members: Sequence[str],
    end_date: pd.Timestamp,
) -> float:
    if len(members) <= 1:
        return 0.5
    returns = pd.Series({ticker: _stock_window_return(prepared, ticker, end_date, 5) for ticker in members})
    ranks = returns.rank(method="average", ascending=True)
    return float((ranks.loc[stock_id] - 1.0) / (len(returns) - 1.0))


def _sector_zscore(
    prepared: dict[str, pd.DataFrame],
    stock_id: str,
    members: Sequence[str],
    end_date: pd.Timestamp,
    *,
    window: int,
    field: str,
) -> float:
    if field == "return":
        values = pd.Series({ticker: _stock_window_return(prepared, ticker, end_date, window) for ticker in members})
    elif field == "volume":
        values = pd.Series({
            ticker: float(_history(prepared[ticker], end_date)["volume"].tail(window).mean())
            for ticker in members
        })
    else:
        raise ValueError(f"unsupported zscore field: {field}")
    if len(values) <= 1:
        return 0.0
    return float((values.loc[stock_id] - values.mean()) / (values.std(ddof=0) + EPS))


def _lead_lag(stock_returns: pd.Series, peer_returns: pd.Series, *, lag: int, window: int) -> float:
    if peer_returns.empty:
        return 0.0
    stock = stock_returns.reset_index(drop=True).tail(window)
    peer = peer_returns.reset_index(drop=True).shift(lag).tail(window)
    frame = pd.concat([stock.rename("stock"), peer.rename("peer")], axis=1).dropna()
    return _corr(frame["stock"].to_numpy(), frame["peer"].to_numpy())


def _peer_dispersion_5d(prepared: dict[str, pd.DataFrame], peers: Sequence[str], end_date: pd.Timestamp) -> float:
    if not peers:
        return 0.0
    values = [_stock_window_return(prepared, peer, end_date, 5) for peer in peers]
    return float(np.std(values, ddof=0))


def _stock_peer_spread(
    prepared: dict[str, pd.DataFrame],
    stock_id: str,
    peers: Sequence[str],
    end_date: pd.Timestamp,
    *,
    window: int,
) -> float:
    stock_return = _stock_window_return(prepared, stock_id, end_date, window)
    if not peers:
        return 0.0
    peer_mean = float(np.mean([_stock_window_return(prepared, peer, end_date, window) for peer in peers]))
    return stock_return - peer_mean


def _vol_zscore(benchmark_returns: pd.Series) -> float:
    rolling = benchmark_returns.dropna().rolling(20).std(ddof=0).dropna().tail(126)
    if len(rolling) < 2:
        return 0.0
    return float((rolling.iloc[-1] - rolling.mean()) / (rolling.std(ddof=0) + EPS))


def _sector_rotation(
    prepared: dict[str, pd.DataFrame],
    ticker_to_sector: dict[str, str],
    end_date: pd.Timestamp,
) -> list[float]:
    sector_values = {}
    for sector in SECTOR_NAMES:
        members = [ticker for ticker, ticker_sector in ticker_to_sector.items() if ticker_sector == sector]
        if members:
            sector_returns = _sector_return_frame(prepared, members, end_date)
            sector_values[sector] = _window_log_return(sector_returns["sector_return"], 5)
    if not sector_values:
        leader = "other"
    else:
        leader = max(sector_values.items(), key=lambda item: item[1])[0]
    values = [1.0 if leader == sector else 0.0 for sector in ROTATION_SECTORS]
    values.append(0.0 if leader in ROTATION_SECTORS else 1.0)
    return values
