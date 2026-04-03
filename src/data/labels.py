"""Label generation for relative outperformance targets."""

from __future__ import annotations

import pandas as pd



def generate_outperformance_label(
    df: pd.DataFrame,
    *,
    stock_close_col: str = "close",
    index_close_col: str = "index_close",
    horizon_days: int = 3,
    label_col: str = "label",
) -> pd.DataFrame:
    """Create binary labels for stock outperformance over a forward horizon.

    The label is defined as:
    ``1 if stock_return_next_3d > nifty_return_next_3d else 0``

    Args:
        df: Input dataframe containing stock and index close prices.
        stock_close_col: Column name of stock close prices.
        index_close_col: Column name of NIFTY/index close prices.
        horizon_days: Forward return horizon in trading days.
        label_col: Output column name for the label.

    Returns:
        A copy of the input dataframe with helper forward-return columns and
        the binary label column.

    Raises:
        ValueError: If required columns are missing or horizon is invalid.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be a positive integer")

    missing = [
        col
        for col in (stock_close_col, index_close_col)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["stock_return_next_3d"] = (
        out[stock_close_col].shift(-horizon_days) / out[stock_close_col] - 1.0
    )
    out["nifty_return_next_3d"] = (
        out[index_close_col].shift(-horizon_days) / out[index_close_col] - 1.0
    )
    out[label_col] = (
        out["stock_return_next_3d"] > out["nifty_return_next_3d"]
    ).astype("Int64")

    return out
