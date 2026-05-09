"""Unit tests for PurgedWalkForwardSplit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.cv import CVSplit, PurgedWalkForwardSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dates(n: int, start: str = "2024-01-02") -> np.ndarray:
    """Return *n* business-day dates as datetime64[ns]."""
    dates = pd.bdate_range(start=start, periods=n)
    return np.array(dates, dtype="datetime64[ns]")


def _collect_splits(splitter: PurgedWalkForwardSplit, dates: np.ndarray) -> list[CVSplit]:
    return list(splitter.split(dates))


# ---------------------------------------------------------------------------
# Positive tests
# ---------------------------------------------------------------------------


def test_yields_correct_number_of_folds() -> None:
    """split() yields exactly n_splits CVSplit objects."""
    dates = _make_dates(40)
    splitter = PurgedWalkForwardSplit(n_splits=3, horizon_days=1)
    folds = _collect_splits(splitter, dates)
    assert len(folds) == 3


def test_train_and_val_are_disjoint() -> None:
    """No sample index appears in both train and val of the same fold."""
    dates = _make_dates(60)
    splitter = PurgedWalkForwardSplit(n_splits=3, horizon_days=1)
    for cv in splitter.split(dates):
        overlap = set(cv.train_idx.tolist()) & set(cv.val_idx.tolist())
        assert overlap == set(), f"Fold {cv.fold}: train/val overlap at indices {overlap}"


def test_val_dates_strictly_after_train_dates() -> None:
    """Every validation date is strictly greater than every training date."""
    dates = _make_dates(60)
    splitter = PurgedWalkForwardSplit(n_splits=3, horizon_days=1)
    for cv in splitter.split(dates):
        max_train = dates[cv.train_idx].max()
        min_val = dates[cv.val_idx].min()
        assert max_train < min_val, (
            f"Fold {cv.fold}: max train date {max_train} >= min val date {min_val}"
        )


def test_purging_removes_overlapping_train_samples() -> None:
    """No retained train sample has a label window that reaches into the val period."""
    horizon_days = 5
    dates = _make_dates(80)
    splitter = PurgedWalkForwardSplit(n_splits=3, horizon_days=horizon_days, embargo_days=0)
    for cv in splitter.split(dates):
        train_dates = pd.to_datetime(dates[cv.train_idx])
        val_start = pd.to_datetime(dates[cv.val_idx]).min()
        # After purge no train sample should have end_date + horizon >= val_start.
        label_ends = train_dates + pd.Timedelta(days=horizon_days)
        assert (label_ends < val_start).all(), (
            f"Fold {cv.fold}: some train label windows extend into val period"
        )


def test_embargo_excludes_samples_close_to_val_start() -> None:
    """With embargo_days > horizon_days, train samples within embargo window are dropped."""
    horizon_days = 2
    embargo_days = 10
    dates = _make_dates(100)
    splitter = PurgedWalkForwardSplit(
        n_splits=3, horizon_days=horizon_days, embargo_days=embargo_days
    )
    for cv in splitter.split(dates):
        train_dates = pd.to_datetime(dates[cv.train_idx])
        val_start = pd.to_datetime(dates[cv.val_idx]).min()
        cutoff = val_start - pd.Timedelta(days=embargo_days)
        assert (train_dates < cutoff).all(), (
            f"Fold {cv.fold}: some train samples fall within embargo window"
        )


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_too_few_samples_raises_value_error() -> None:
    """Fewer samples than n_splits + 1 raises ValueError."""
    dates = _make_dates(3)  # only 3 samples
    splitter = PurgedWalkForwardSplit(n_splits=3, horizon_days=1)
    with pytest.raises(ValueError, match="Need at least"):
        list(splitter.split(dates))


def test_invalid_n_splits_raises() -> None:
    with pytest.raises(ValueError, match="n_splits"):
        PurgedWalkForwardSplit(n_splits=0, horizon_days=1)


def test_invalid_horizon_days_raises() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        PurgedWalkForwardSplit(n_splits=2, horizon_days=0)


def test_empty_train_after_purge_raises() -> None:
    """A configuration that purges every candidate training sample raises ValueError."""
    # 4 samples, n_splits=3 → chunk size ≈ 1 sample each.
    # With horizon_days=30 the first chunk's single sample has label window
    # extending 30 days forward, guaranteed to overlap any fold start.
    dates = _make_dates(4)
    splitter = PurgedWalkForwardSplit(n_splits=3, horizon_days=30)
    with pytest.raises(ValueError, match="no training samples"):
        list(splitter.split(dates))
