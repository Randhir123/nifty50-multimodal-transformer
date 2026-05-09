"""Walk-forward cross-validation with purging and embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CVSplit:
    """One walk-forward fold: original-index arrays for train and validation."""

    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray


class PurgedWalkForwardSplit:
    """Expanding-window walk-forward CV with purging and embargo.

    Divides *n* samples chronologically into ``n_splits + 1`` equal-size chunks.
    Fold *k* (0-indexed) uses chunks 0..k as candidate training data and chunk
    *k+1* as validation.  Two groups are removed from the candidate training set:

    * **Purged** – samples whose label window (end_date + horizon_days calendar
      days) reaches into the validation period.
    * **Embargoed** – samples within ``embargo_days`` calendar days before the
      first validation date.

    The combined cutoff is ``fold_start − max(horizon_days, embargo_days)`` in
    calendar days; any training sample at or after that date is excluded.
    """

    def __init__(
        self,
        *,
        n_splits: int,
        horizon_days: int,
        embargo_days: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if horizon_days < 1:
            raise ValueError(f"horizon_days must be >= 1, got {horizon_days}")
        if embargo_days < 0:
            raise ValueError(f"embargo_days must be >= 0, got {embargo_days}")
        self.n_splits = n_splits
        self.horizon_days = horizon_days
        self.embargo_days = embargo_days

    def split(self, end_dates: np.ndarray) -> Iterator[CVSplit]:
        """Yield one :class:`CVSplit` per fold.

        Parameters
        ----------
        end_dates:
            1-D array of sample end dates.  Accepts ``datetime64``, ISO-string,
            or any type accepted by :func:`pandas.to_datetime`.

        Raises
        ------
        ValueError
            If fewer than ``n_splits + 1`` samples are provided, or if purging
            and embargo together empty the training set for any fold.
        """
        n = len(end_dates)
        min_required = self.n_splits + 1
        if n < min_required:
            raise ValueError(
                f"Need at least {min_required} samples for {self.n_splits} CV "
                f"splits, got {n}."
            )

        order = np.argsort(end_dates, kind="stable")
        dates = pd.to_datetime(end_dates)

        # Divide sorted positions into n_splits + 1 roughly equal chunks.
        # boundaries[k] = start position (in order) of chunk k.
        boundaries = [
            int(round(k * n / (self.n_splits + 1)))
            for k in range(self.n_splits + 2)
        ]
        boundaries[-1] = n  # ensure all samples are covered

        gap = pd.Timedelta(days=max(self.horizon_days, self.embargo_days))

        for fold_k in range(self.n_splits):
            val_start = boundaries[fold_k + 1]
            val_end = boundaries[fold_k + 2]

            val_idx = order[val_start:val_end]

            # First date in the validation chunk.
            fold_start_date: pd.Timestamp = dates[order[val_start]]
            cutoff: pd.Timestamp = fold_start_date - gap

            # Candidate training indices (everything before the val chunk).
            candidate_idx = order[:val_start]
            keep: np.ndarray = np.asarray(dates[candidate_idx] < cutoff)
            train_idx = candidate_idx[keep]

            if len(train_idx) == 0:
                raise ValueError(
                    f"Fold {fold_k}: no training samples remain after "
                    f"purge/embargo (horizon_days={self.horizon_days}, "
                    f"embargo_days={self.embargo_days}).  Reduce n_splits or "
                    f"embargo_days."
                )
            if len(val_idx) == 0:
                raise ValueError(f"Fold {fold_k}: validation set is empty.")

            yield CVSplit(fold=fold_k, train_idx=train_idx, val_idx=val_idx)
