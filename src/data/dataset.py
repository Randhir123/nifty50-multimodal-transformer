"""Dataset helpers for rolling-window, image-only, and text-only model inputs."""

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


@dataclass(frozen=True)
class ImagePathDataset:
    """Container for image-only samples tied to binary labels.

    Attributes:
        image_paths: Absolute or relative chart image paths.
        y: Label vector of shape ``[num_samples]`` with values in ``{0, 1}``.
        sample_dates: Date values used for chronological splitting.
    """

    image_paths: np.ndarray
    y: np.ndarray
    sample_dates: np.ndarray


@dataclass(frozen=True)
class TextSampleDataset:
    """Container for text-only samples tied to binary labels.

    Attributes:
        texts: Per-sample news headline strings.
        y: Label vector of shape ``[num_samples]`` with values in ``{0, 1}``.
        sample_dates: Date values used for chronological splitting.
    """

    texts: np.ndarray
    y: np.ndarray
    sample_dates: np.ndarray


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


def create_image_path_dataset(
    df: pd.DataFrame,
    *,
    image_path_col: str = "chart_path",
    label_col: str = "label",
    date_col: str = "date",
    dropna: bool = True,
    require_existing_files: bool = False,
) -> ImagePathDataset:
    """Create image-path dataset for image-only training.

    Expects one row per prediction sample with deterministic chart paths from
    ``src/viz/charts.py`` and the project's binary label.
    """
    required_cols = [image_path_col, label_col, date_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    if dropna:
        work_df = work_df.dropna(subset=required_cols)

    work_df = work_df.sort_values(date_col).reset_index(drop=True)

    image_paths = work_df[image_path_col].astype(str).to_numpy(dtype=object)
    labels = work_df[label_col].astype(np.int64).to_numpy()
    sample_dates = work_df[date_col].to_numpy()

    if require_existing_files:
        exists_mask = np.array([Path(p).exists() for p in image_paths], dtype=bool)
        if not np.any(exists_mask):
            raise ValueError("No image paths exist on disk after filtering")
        image_paths = image_paths[exists_mask]
        labels = labels[exists_mask]
        sample_dates = sample_dates[exists_mask]

    if len(image_paths) == 0:
        raise ValueError("Image dataset is empty")

    return ImagePathDataset(image_paths=image_paths, y=labels, sample_dates=sample_dates)


def create_text_sample_dataset(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    label_col: str = "label",
    date_col: str = "date",
    dropna: bool = True,
) -> TextSampleDataset:
    """Create text sample dataset for text-only training.

    Expects one row per prediction sample where ``text`` is a single string
    composed from the top 3-5 most recent headlines for that sample.
    """
    required_cols = [text_col, label_col, date_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    if dropna:
        work_df = work_df.dropna(subset=required_cols)

    work_df = work_df.sort_values(date_col).reset_index(drop=True)

    texts = work_df[text_col].astype(str).to_numpy(dtype=object)
    labels = work_df[label_col].astype(np.int64).to_numpy()
    sample_dates = work_df[date_col].to_numpy()

    if len(texts) == 0:
        raise ValueError("Text dataset is empty")

    return TextSampleDataset(texts=texts, y=labels, sample_dates=sample_dates)
