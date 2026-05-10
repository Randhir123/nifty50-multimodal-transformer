"""Integration test: Training produces non-degenerate predictions."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.train_fusion import FusionArrays, train_on_arrays


@pytest.mark.integration
def test_training_does_not_collapse():
    """Assert that a 5-epoch training run escapes constant-prediction collapse."""
    # Create 100 synthetic samples with a strong signal in feature 0
    n_samples = 100
    seq_len = 5
    dim = 8
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    tabular = rng.normal(0, 1, (n_samples, seq_len, dim)).astype(np.float32)
    # Label is heavily correlated with a persistent signal in feature 0.
    signal = rng.normal(0, 1, n_samples).astype(np.float32)
    y = (signal > 0).astype(np.int64)
    tabular[:, :, 0] += signal[:, None] * 3.0
    end_dates = np.arange(n_samples)

    train_arrays = FusionArrays(tabular[:80], y[:80], end_dates[:80])
    val_arrays = FusionArrays(tabular[80:], y[80:], end_dates[80:])

    args = argparse.Namespace(
        batch_size=16,
        device="cpu",
        model_dim=16,
        num_heads=2,
        num_layers=1,
        ff_dim=32,
        dropout=0.0,
        pooling="mean",
        max_tokens=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=5,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "ckpt.pt"
        metrics = train_on_arrays(train_arrays, val_arrays, args=args, checkpoint_path=ckpt)
        
        assert metrics.get("roc_auc", 0.0) > 0.55, f"Expected ROC-AUC > 0.55, got {metrics.get('roc_auc')}"
        
        ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
        val_prob = ckpt_data["val_y_prob"]
        
        prob_range = float(val_prob.max() - val_prob.min())
        pred_pos_rate = float((val_prob >= 0.5).mean())
        
        assert prob_range > 0.10, f"Probabilities collapsed, range is only {prob_range:.4f}"
        assert 0.10 < pred_pos_rate < 0.90, f"Predictions collapsed to single class, rate is {pred_pos_rate:.4f}"
