from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.multimodal_samples import (
    attach_text_tokens,
    build_tabular_multimodal_samples,
    build_text_tokens_for_sample_arrays,
    save_multimodal_samples,
)


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


def _text_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock_id": "AAA.NS",
                "event_date": "2024-01-02",
                "source_type": "news",
                "title": "AAA early update",
                "body_text": "AAA early information.",
            },
            {
                "stock_id": "AAA.NS",
                "event_date": "2024-01-04",
                "source_type": "filing",
                "title": "AAA later update",
                "body_text": "AAA later information.",
            },
            {
                "stock_id": "BBB.NS",
                "event_date": "2024-01-03",
                "source_type": "guidance",
                "title": "BBB update",
                "body_text": "BBB information.",
            },
        ]
    )


def test_build_text_tokens_for_sample_arrays_aligns_to_rows() -> None:
    arrays = _arrays()
    tokens = build_text_tokens_for_sample_arrays(
        arrays, _text_records(), top_k=3, dim=10
    )

    assert tokens.shape == (arrays.tabular_tokens.shape[0], 10)
    assert np.isfinite(tokens).all()
    assert not np.allclose(tokens[0], tokens[1])
    assert not np.allclose(tokens[0], tokens[2])


def test_text_tokens_ignore_future_records_for_earlier_samples() -> None:
    arrays = _arrays()
    base_records = pd.DataFrame(
        [
            {
                "stock_id": "AAA.NS",
                "event_date": "2024-01-02",
                "source_type": "news",
                "title": "Visible early",
                "body_text": "This is visible to both AAA samples.",
            }
        ]
    )
    records_with_future = pd.concat(
        [
            base_records,
            pd.DataFrame(
                [
                    {
                        "stock_id": "AAA.NS",
                        "event_date": "2024-01-04",
                        "source_type": "filing",
                        "title": "Future for first AAA sample",
                        "body_text": "This should affect only the later AAA sample.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    base_tokens = build_text_tokens_for_sample_arrays(arrays, base_records, dim=12)
    future_tokens = build_text_tokens_for_sample_arrays(arrays, records_with_future, dim=12)

    # First AAA sample ends on 2024-01-03, so the 2024-01-04 record is future.
    np.testing.assert_allclose(base_tokens[0], future_tokens[0])
    # Second AAA sample ends on 2024-01-04, so the new record is visible.
    assert not np.allclose(base_tokens[1], future_tokens[1])


def test_attach_text_tokens_preserves_existing_arrays() -> None:
    arrays = _arrays()
    enriched = attach_text_tokens(arrays, _text_records(), top_k=3, dim=14)

    enriched.validate()
    np.testing.assert_allclose(enriched.tabular_tokens, arrays.tabular_tokens)
    np.testing.assert_array_equal(enriched.y, arrays.y)
    np.testing.assert_array_equal(enriched.stock_ids, arrays.stock_ids)
    assert enriched.text_tokens is not None
    assert enriched.text_tokens.shape == (arrays.tabular_tokens.shape[0], 14)


def test_missing_text_records_still_produce_deterministic_tokens() -> None:
    arrays = _arrays()
    records = pd.DataFrame(
        [
            {
                "stock_id": "ZZZ.NS",
                "event_date": "2024-01-02",
                "source_type": "news",
                "title": "Other stock",
                "body_text": "Not relevant to sample stocks.",
            }
        ]
    )

    tokens = build_text_tokens_for_sample_arrays(arrays, records, dim=8)

    assert tokens.shape == (arrays.tabular_tokens.shape[0], 8)
    np.testing.assert_allclose(tokens[0], tokens[1])
    np.testing.assert_allclose(tokens[0], tokens[-1])


def test_save_multimodal_samples_with_text_schema(tmp_path) -> None:
    arrays = _arrays()
    enriched = attach_text_tokens(arrays, _text_records(), dim=16)
    output_path = save_multimodal_samples(enriched, tmp_path / "tabular_text.npz")

    loaded = np.load(output_path, allow_pickle=False)
    assert set(loaded.files) == {"tabular_tokens", "y", "end_dates", "stock_ids", "text_tokens"}
    assert loaded["text_tokens"].shape[0] == loaded["tabular_tokens"].shape[0]
