"""Integration test: walk-forward CV with the full training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from src.training.cv import PurgedWalkForwardSplit
from src.training.train_fusion import FusionArrays, slice_fusion_arrays, train_on_arrays


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SAMPLES = 60
_TABULAR_DIM = 4
_WINDOW_SIZE = 5


def _make_synthetic_arrays(rng: np.random.Generator) -> FusionArrays:
    """Return a small FusionArrays with tabular tokens, labels, and dates."""
    tabular = rng.random((_N_SAMPLES, _WINDOW_SIZE, _TABULAR_DIM)).astype(np.float32)
    y = rng.integers(0, 2, size=_N_SAMPLES).astype(np.int64)
    # Business-day dates, 60 consecutive days from 2024-01-02.
    import pandas as pd
    dates = np.array(
        pd.bdate_range(start="2024-01-02", periods=_N_SAMPLES),
        dtype="datetime64[ns]",
    )
    return FusionArrays(tabular_tokens=tabular, y=y, end_dates=dates)


def _make_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cpu",
        model_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        dropout=0.0,
        pooling="cls",
        max_tokens=64,
    )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cv_pipeline_returns_valid_metrics(tmp_path: Path) -> None:
    """PurgedWalkForwardSplit + train_on_arrays produces finite metrics for all folds."""
    rng = np.random.default_rng(42)
    arrays = _make_synthetic_arrays(rng)
    args = _make_args(tmp_path)

    splitter = PurgedWalkForwardSplit(n_splits=2, horizon_days=3, embargo_days=0)
    splits = list(splitter.split(arrays.end_dates))
    assert len(splits) == 2, "Expected exactly 2 CV folds"

    for cv_split in splits:
        train_arrays = slice_fusion_arrays(arrays, cv_split.train_idx)
        val_arrays = slice_fusion_arrays(arrays, cv_split.val_idx)

        assert len(train_arrays.y) > 0, f"Fold {cv_split.fold}: empty train set"
        assert len(val_arrays.y) > 0, f"Fold {cv_split.fold}: empty val set"

        checkpoint_path = tmp_path / f"fold_{cv_split.fold}.pt"
        metrics = train_on_arrays(
            train_arrays,
            val_arrays,
            args=args,
            checkpoint_path=checkpoint_path,
        )

        assert isinstance(metrics, dict), "train_on_arrays must return a dict"
        for key in ("accuracy", "f1", "roc_auc"):
            assert key in metrics, f"Missing metric '{key}' in fold {cv_split.fold}"
            assert np.isfinite(metrics[key]), (
                f"Metric '{key}' is not finite in fold {cv_split.fold}: {metrics[key]}"
            )
