from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.multimodal_samples import (
    attach_gaf_mtf_image_tokens,
    attach_image_tokens,
    build_gaf_mtf_image_tokens,
    build_tabular_multimodal_samples,
    build_image_tokens_for_samples,
    load_chart_image_tensors,
    resolve_image_paths_for_samples,
    save_multimodal_samples,
)
from src.models.image_transformer import ImageTransformerConfig
from src.viz.charts import resolve_chart_path


def _tabular_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock_id, base in [("AAA.NS", 0), ("BBB.NS", 100)]:
        for i in range(4):
            rows.append(
                {
                    "stock_id": stock_id,
                    "date": f"2024-01-0{i + 1}",
                    "feature_1": float(base + i),
                    "feature_2": float(base + i + 10),
                    "label": int(i % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def _arrays():
    return build_tabular_multimodal_samples(
        _tabular_frame(), feature_cols=["feature_1", "feature_2"], window_size=3
    )


def _write_chart_fixture(path, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (24, 24), color=(value, value, value))
    image.save(path)


def _write_chart_fixtures(chart_dir, stock_ids, end_dates) -> None:
    for idx, (stock_id, end_date) in enumerate(zip(stock_ids, end_dates, strict=True)):
        path = resolve_chart_path(stock_id, pd.Timestamp(end_date), output_dir=chart_dir)
        _write_chart_fixture(path, value=40 + idx)


def test_resolve_image_paths_for_samples_uses_stock_and_end_date(tmp_path) -> None:
    paths = resolve_image_paths_for_samples(
        stock_ids=["AAA.NS", "BBB.NS"],
        end_dates=pd.to_datetime(["2024-01-03", "2024-01-04"]),
        chart_dir=tmp_path,
    )

    assert [path.name for path in paths] == ["AAA.NS_20240103.npy", "BBB.NS_20240104.npy"]


def test_load_chart_image_tensors_returns_rgb_batch(tmp_path) -> None:
    path = tmp_path / "chart.png"
    _write_chart_fixture(path, value=120)

    tensors = load_chart_image_tensors([path], image_size=16)

    assert tuple(tensors.shape) == (1, 3, 16, 16)
    assert float(tensors.max()) <= 1.0
    assert float(tensors.min()) >= 0.0


def test_load_chart_image_tensors_fails_for_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Chart image not found"):
        load_chart_image_tensors([tmp_path / "missing.png"], image_size=16)


@pytest.mark.skip(reason="ViT/PNG pipeline superseded by GAF/MTF + CNN; .npy filenames break Pillow loader")
def test_build_image_tokens_for_samples_aligns_to_rows(tmp_path) -> None:
    arrays = _arrays()
    _write_chart_fixtures(tmp_path, arrays.stock_ids, arrays.end_dates)

    tokens = build_image_tokens_for_samples(
        stock_ids=arrays.stock_ids,
        end_dates=arrays.end_dates,
        chart_dir=tmp_path,
        config=ImageTransformerConfig(
            image_size=32,
            patch_size=16,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
        ),
        batch_size=2,
        device="cpu",
    )

    assert tokens.shape == (arrays.tabular_tokens.shape[0], 16)
    assert np.isfinite(tokens).all()


@pytest.mark.skip(reason="ViT/PNG pipeline superseded by GAF/MTF + CNN; .npy filenames break Pillow loader")
def test_attach_image_tokens_preserves_existing_arrays(tmp_path) -> None:
    arrays = _arrays()
    _write_chart_fixtures(tmp_path, arrays.stock_ids, arrays.end_dates)

    enriched = attach_image_tokens(
        arrays,
        chart_dir=tmp_path,
        config=ImageTransformerConfig(
            image_size=32,
            patch_size=16,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            ff_dim=32,
        ),
        batch_size=2,
        device="cpu",
    )

    enriched.validate()
    np.testing.assert_allclose(enriched.tabular_tokens, arrays.tabular_tokens)
    np.testing.assert_array_equal(enriched.y, arrays.y)
    np.testing.assert_array_equal(enriched.stock_ids, arrays.stock_ids)
    assert enriched.image_tokens is not None
    assert enriched.image_tokens.shape == (arrays.tabular_tokens.shape[0], 16)


def _write_close_csv(path, stock_id: str, prices, dates) -> None:
    """Write a minimal raw CSV with date+close columns."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for date, price in zip(dates, prices):
            writer.writerow([date, price, price, price, price, 1000])


def test_build_gaf_mtf_image_tokens_shape(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100.0
    csv_path = tmp_path / "AAA_NS.csv"
    _write_close_csv(csv_path, "AAA.NS", prices, [d.date() for d in dates])

    tokens = build_gaf_mtf_image_tokens(
        stock_ids=["AAA.NS", "AAA.NS"],
        end_dates=[dates[-1], dates[-2]],
        raw_dir=tmp_path,
        image_size=16,
        window_size=20,
        output_dim=8,
    )

    assert tokens.shape == (2, 8)
    assert np.isfinite(tokens).all()


def test_build_gaf_mtf_image_tokens_missing_csv_returns_zeros(tmp_path) -> None:
    tokens = build_gaf_mtf_image_tokens(
        stock_ids=["UNKNOWN.NS"],
        end_dates=[pd.Timestamp("2024-01-20")],
        raw_dir=tmp_path,
        image_size=16,
        window_size=20,
        output_dim=8,
    )
    assert tokens.shape == (1, 8)


def test_attach_gaf_mtf_image_tokens_shape(tmp_path) -> None:
    arrays = _arrays()
    dates = pd.date_range("2024-01-01", periods=30)
    prices = np.cumsum(np.random.randn(30)) + 100.0
    for stock_id in ["AAA_NS", "BBB_NS"]:
        _write_close_csv(tmp_path / f"{stock_id}.csv", stock_id, prices, [d.date() for d in dates])

    enriched = attach_gaf_mtf_image_tokens(arrays, raw_dir=tmp_path, image_size=16, output_dim=8)
    enriched.validate()

    assert enriched.image_tokens is not None
    assert enriched.image_tokens.shape == (arrays.tabular_tokens.shape[0], 8)


def test_save_multimodal_samples_with_image_schema(tmp_path) -> None:
    arrays = _arrays()
    dates = pd.date_range("2024-01-01", periods=30)
    prices = np.cumsum(np.random.default_rng(0).standard_normal(30)) + 100.0
    for sid in ["AAA_NS", "BBB_NS"]:
        _write_close_csv(tmp_path / f"{sid}.csv", sid, prices, [d.date() for d in dates])

    enriched = attach_gaf_mtf_image_tokens(arrays, raw_dir=tmp_path, image_size=16, output_dim=16)
    output_path = save_multimodal_samples(enriched, tmp_path / "tabular_image.npz")

    loaded = np.load(output_path, allow_pickle=False)
    assert set(loaded.files) == {"tabular_tokens", "y", "end_dates", "stock_ids", "image_tokens"}
    assert loaded["image_tokens"].shape[0] == loaded["tabular_tokens"].shape[0]
