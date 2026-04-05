"""Ranking table helpers for model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_SAMPLE_COLUMNS: tuple[str, str] = ("stock_id", "date")


def build_ranked_predictions(
    samples: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    stock_col: str = "stock_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Build a deterministic ranked table per date from model output probabilities."""
    required = {stock_col, date_col}
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"samples missing required columns: {missing}")

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1:
        raise ValueError("probabilities must be a 1D array")
    if len(probs) != len(samples):
        raise ValueError("probabilities length must equal sample rows")

    out = samples.loc[:, [stock_col, date_col]].copy()
    out[stock_col] = out[stock_col].astype(str)
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()

    out["probability"] = probs
    out["predicted_label"] = (out["probability"] >= threshold).astype(np.int64)

    out = out.sort_values([date_col, "probability", stock_col], ascending=[True, False, True])
    out["rank"] = (
        out.groupby(date_col, sort=True)["probability"]
        .rank(method="first", ascending=False)
        .astype(np.int64)
    )

    return out.rename(columns={stock_col: "stock_id", date_col: "date"}).reset_index(drop=True)
