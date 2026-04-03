"""Feature engineering utilities for OHLCV time-series data.

This module provides a first, lightweight set of technical features for the
NIFTY 50 relative outperformance problem.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Validate that all required columns are present.

    Args:
        df: Input dataframe.
        required: Column names required for a computation.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_technical_features(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    *,
    date_col: str = "date",
    index_close_col: str = "close",
) -> pd.DataFrame:
    """Compute technical features from stock and index OHLCV data.

    Expected stock columns: ``open``, ``high``, ``low``, ``close``, ``volume`` and
    a date column (default ``date``). Expected index columns: a date column and
    a close column (default ``close``).

    Args:
        stock_df: Stock OHLCV dataframe.
        index_df: Index dataframe containing at least date and close.
        date_col: Name of the date column present in both dataframes.
        index_close_col: Name of the index close column in ``index_df``.

    Returns:
        Dataframe with original stock columns and engineered feature columns.

    Raises:
        ValueError: If required columns are missing.
    """
    _validate_required_columns(stock_df, REQUIRED_OHLCV_COLUMNS)
    _validate_required_columns(stock_df, [date_col])
    _validate_required_columns(index_df, [date_col, index_close_col])

    stock = stock_df.copy()
    index = index_df[[date_col, index_close_col]].copy().rename(
        columns={index_close_col: "index_close"}
    )

    stock[date_col] = pd.to_datetime(stock[date_col])
    index[date_col] = pd.to_datetime(index[date_col])

    stock = stock.sort_values(date_col).reset_index(drop=True)
    index = index.sort_values(date_col).reset_index(drop=True)

    df = stock.merge(index, on=date_col, how="left", validate="many_to_one")

    # Core return features.
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))
    df["cum_return_3d"] = df["close"] / df["close"].shift(3) - 1.0
    df["cum_return_5d"] = df["close"] / df["close"].shift(5) - 1.0
    df["cum_return_10d"] = df["close"] / df["close"].shift(10) - 1.0

    # Volatility from daily log returns, annualized with sqrt(window) scaling.
    df["realized_vol_5d"] = df["log_return_1d"].rolling(window=5, min_periods=5).std() * np.sqrt(5)
    df["realized_vol_10d"] = (
        df["log_return_1d"].rolling(window=10, min_periods=10).std() * np.sqrt(10)
    )

    # Price/volume structure features.
    df["high_low_range_over_close"] = (df["high"] - df["low"]) / df["close"]
    df["close_over_10dma_minus_1"] = df["close"] / df["close"].rolling(10, min_periods=10).mean() - 1.0
    df["close_over_20dma_minus_1"] = df["close"] / df["close"].rolling(20, min_periods=20).mean() - 1.0
    df["volume_over_20d_avg"] = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()

    # Relative return feature versus index.
    stock_return_1d = df["close"].pct_change(1)
    index_return_1d = df["index_close"].pct_change(1)
    df["stock_minus_index_return"] = stock_return_1d - index_return_1d

    return df
