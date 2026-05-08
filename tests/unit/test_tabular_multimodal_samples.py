from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.multimodal_samples import (
    build_tabular_multimodal_samples,
    infer_numeric_feature_columns,
    save_multimodal_samples,
)


def _sample_frame() -> pd.DataFrame:
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
    # Intentionally interleave rows to prove the builder sorts/groups correctly.
    return pd.DataFrame([rows[0], rows[4], rows[1], rows[5], rows[2], rows[6], rows[3], rows[7]])


def test_build_tabular_multimodal_samples_groups_windows_per_stock() -> None:
    arrays = build_tabular_multimodal_samples(
        _sample_frame(), feature_cols=["feature_1", "feature_2"], window_size=3
    )

    arrays.validate()
    assert arrays.tabular_tokens.shape == (4, 3, 2)
    assert arrays.y.tolist() == [1, 0, 1, 0]
    assert arrays.stock_ids.tolist() == ["AAA.NS", "AAA.NS", "BBB.NS", "BBB.NS"]

    np.testing.assert_allclose(
        arrays.tabular_tokens[0],
        np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        arrays.tabular_tokens[1],
        np.array([[1.0, 11.0], [2.0, 12.0], [3.0, 13.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        arrays.tabular_tokens[2],
        np.array([[100.0, 110.0], [101.0, 111.0], [102.0, 112.0]], dtype=np.float32),
    )


def test_build_tabular_multimodal_samples_preserves_end_dates() -> None:
    arrays = build_tabular_multimodal_samples(
        _sample_frame(), feature_cols=["feature_1", "feature_2"], window_size=3
    )

    assert arrays.end_dates.astype("datetime64[D]").astype(str).tolist() == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-03",
        "2024-01-04",
    ]


def test_build_tabular_multimodal_samples_rejects_missing_columns() -> None:
    df = _sample_frame().drop(columns=["label"])

    with pytest.raises(ValueError, match="Missing required columns"):
        build_tabular_multimodal_samples(
            df, feature_cols=["feature_1", "feature_2"], window_size=3
        )


def test_build_tabular_multimodal_samples_rejects_too_large_window() -> None:
    with pytest.raises(ValueError, match="No tabular samples could be built"):
        build_tabular_multimodal_samples(
            _sample_frame(), feature_cols=["feature_1", "feature_2"], window_size=5
        )


def test_infer_numeric_feature_columns_excludes_label_and_metadata() -> None:
    df = _sample_frame()

    assert infer_numeric_feature_columns(df) == ["feature_1", "feature_2"]


def test_tabular_multimodal_samples_save_npz_schema(tmp_path) -> None:
    arrays = build_tabular_multimodal_samples(
        _sample_frame(), feature_cols=["feature_1", "feature_2"], window_size=3
    )
    output_path = save_multimodal_samples(arrays, tmp_path / "tabular_multimodal.npz")

    loaded = np.load(output_path, allow_pickle=False)
    assert set(loaded.files) == {"tabular_tokens", "y", "end_dates", "stock_ids"}
    assert loaded["tabular_tokens"].shape == (4, 3, 2)
    assert loaded["y"].shape == (4,)
    assert loaded["stock_ids"].tolist() == ["AAA.NS", "AAA.NS", "BBB.NS", "BBB.NS"]
