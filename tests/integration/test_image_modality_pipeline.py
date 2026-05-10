"""Integration tests for the GAF/MTF image modality pipeline."""

import csv
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.multimodal_samples import (
    attach_gaf_mtf_image_tokens,
    build_gaf_mtf_image_tokens,
    build_tabular_multimodal_samples,
    save_multimodal_samples,
)
from src.models.image_cnn import ImageCNN, ImageCNNConfig


def _write_close_csv(path, prices, dates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for date, price in zip(dates, prices):
            writer.writerow([date, price, price, price, price, 1000])


@pytest.mark.integration
def test_end_to_end_artifact_has_correct_image_shape():
    model = ImageCNN(ImageCNNConfig(output_dim=16))
    out = model.encode_images(torch.randn(2, 2, 32, 32))
    assert out.shape == (2, 16)


@pytest.mark.integration
def test_gaf_mtf_tokens_differ_across_stocks(tmp_path):
    """Two stocks with unrelated price series should produce different image tokens."""
    dates = pd.date_range("2024-01-01", periods=40)
    prices_a = np.cumsum(np.random.default_rng(0).standard_normal(40)) + 100.0
    prices_b = np.cumsum(np.random.default_rng(1).standard_normal(40)) + 200.0

    _write_close_csv(tmp_path / "AAA_NS.csv", prices_a, [d.date() for d in dates])
    _write_close_csv(tmp_path / "BBB_NS.csv", prices_b, [d.date() for d in dates])

    end_date = dates[-1]
    tokens = build_gaf_mtf_image_tokens(
        stock_ids=["AAA.NS", "BBB.NS"],
        end_dates=[end_date, end_date],
        raw_dir=tmp_path,
        image_size=32,
        window_size=20,
        output_dim=16,
    )

    assert tokens.shape == (2, 16)
    assert not np.allclose(tokens[0], tokens[1])


@pytest.mark.integration
def test_gaf_mtf_artifact_roundtrip(tmp_path):
    """Build tabular samples, attach GAF/MTF tokens, save NPZ, reload; image_tokens preserved."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=30)
    prices = np.cumsum(np.random.default_rng(42).standard_normal(30)) + 100.0
    for i, d in enumerate(dates):
        rows.append({
            "stock_id": "AAA.NS",
            "date": d,
            "f1": float(prices[i]),
            "f2": float(prices[i] * 0.5),
            "label": int(i % 2),
        })
    df = pd.DataFrame(rows)

    _write_close_csv(tmp_path / "AAA_NS.csv", prices, [d.date() for d in dates])

    arrays = build_tabular_multimodal_samples(df, feature_cols=["f1", "f2"], window_size=20)
    enriched = attach_gaf_mtf_image_tokens(arrays, raw_dir=tmp_path, image_size=16, output_dim=8)

    npz_path = save_multimodal_samples(enriched, tmp_path / "artifact.npz")
    loaded = np.load(npz_path, allow_pickle=False)

    assert "image_tokens" in loaded.files
    assert loaded["image_tokens"].shape == (enriched.tabular_tokens.shape[0], 8)
    assert np.isfinite(loaded["image_tokens"]).all()
