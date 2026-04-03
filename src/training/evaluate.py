"""Evaluation helpers for binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels as ``[num_samples]`` in ``{0, 1}``.
        y_prob: Predicted probabilities for class 1 in ``[0, 1]``.
        threshold: Decision threshold for computing class predictions.

    Returns:
        Dict with ``accuracy``, ``precision``, ``recall``, ``f1``, ``roc_auc``.
    """
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D arrays")
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must have equal length")

    y_pred = (y_prob >= threshold).astype(np.int64)

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return {k: float(v) for k, v in metrics.items()}
