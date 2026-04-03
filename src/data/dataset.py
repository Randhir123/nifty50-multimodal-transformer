"""Dataset helpers for rolling-window Transformer inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RollingWindowDataset:
    """Container for rolling-window tabular sequences.

    Attributes:
        X: Feature tensor of shape ``[num_samples, window_size, num_features]``.
        y: Label vector of shape ``[num_samples]``.
        end_dates: Date values corresponding to each sample's prediction date.
    """

    X: np.ndarray
    y: np.ndarray
    end_dates: np.ndarray


def load_ohlcv_csv(path: str | Path, *, date_col: str = "date") -> pd.DataFrame:
    """Load OHLCV CSV input with basic validation.

    Args:
        path: Path to a CSV file.
        date_col: Name of the date column.

    Returns:
        Parsed dataframe sorted by date.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"open", "high", "low", "close", "volume", date_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def create_rolling_transformer_dataset(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    label_col: str = "label",
    date_col: str = "date",
    window_size: int = 60,
    dropna: bool = True,
) -> RollingWindowDataset:
    """Create rolling-window sequences for Transformer training/inference.

    For each index ``t`` starting from ``window_size - 1``, one sample is built
    from rows ``[t-window_size+1, ..., t]`` using ``feature_cols``. The target is
    the label at row ``t``.

    Args:
        df: Input dataframe containing feature and label columns.
        feature_cols: Ordered feature columns to include in each window.
        label_col: Label column name.
        date_col: Date column name for sample traceability.
        window_size: Number of timesteps per sample.
        dropna: If True, rows with NaNs in required columns are removed first.

    Returns:
        ``RollingWindowDataset`` with numpy arrays.

    Raises:
        ValueError: If inputs are invalid or there are insufficient rows.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    required_cols = list(feature_cols) + [label_col, date_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy().sort_values(date_col).reset_index(drop=True)
    if dropna:
        work_df = work_df.dropna(subset=required_cols).reset_index(drop=True)

    if len(work_df) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows after preprocessing; got {len(work_df)}"
        )

    features = work_df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    labels = work_df.loc[:, label_col].to_numpy(dtype=np.int64)
    dates = work_df.loc[:, date_col].to_numpy()

    num_samples = len(work_df) - window_size + 1
    X = np.empty((num_samples, window_size, len(feature_cols)), dtype=np.float32)
    y = np.empty((num_samples,), dtype=np.int64)
    end_dates = np.empty((num_samples,), dtype=dates.dtype)

    for i in range(num_samples):
        end_idx = i + window_size - 1
        X[i] = features[i : end_idx + 1]
        y[i] = labels[end_idx]
        end_dates[i] = dates[end_idx]

    return RollingWindowDataset(X=X, y=y, end_dates=end_dates)
