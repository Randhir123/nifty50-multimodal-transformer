from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.multimodal_samples import (
    MultimodalSampleArrays,
    build_kg_tokens_from_contexts,
    build_text_tokens_for_samples,
    build_toy_multimodal_samples,
    save_multimodal_samples,
)


def test_toy_multimodal_samples_have_expected_schema() -> None:
    arrays = build_toy_multimodal_samples(num_samples=8, window_size=3, tabular_dim=4)

    arrays.validate()
    assert arrays.tabular_tokens.shape == (8, 3, 4)
    assert arrays.image_tokens is not None
    assert arrays.image_tokens.shape[0] == 8
    assert arrays.text_tokens is not None
    assert arrays.text_tokens.shape[0] == 8
    assert arrays.kg_tokens is not None
    assert arrays.kg_tokens.shape[0] == 8
    assert arrays.y.shape == (8,)
    assert arrays.end_dates.shape == (8,)
    assert arrays.stock_ids.shape == (8,)


def test_save_multimodal_samples_round_trips_required_and_optional_keys(tmp_path) -> None:
    arrays = build_toy_multimodal_samples(num_samples=8, window_size=3, tabular_dim=4)
    output_path = save_multimodal_samples(arrays, tmp_path / "multimodal_samples.npz")

    loaded = np.load(output_path, allow_pickle=False)
    assert set(loaded.files) == {
        "tabular_tokens",
        "y",
        "end_dates",
        "stock_ids",
        "image_tokens",
        "text_tokens",
        "kg_tokens",
    }
    assert loaded["tabular_tokens"].shape == arrays.tabular_tokens.shape
    assert loaded["image_tokens"].shape == arrays.image_tokens.shape
    assert loaded["text_tokens"].shape == arrays.text_tokens.shape
    assert loaded["kg_tokens"].shape == arrays.kg_tokens.shape


def test_validation_rejects_misaligned_optional_modality() -> None:
    arrays = MultimodalSampleArrays(
        tabular_tokens=np.zeros((3, 2, 4), dtype=np.float32),
        y=np.zeros((3,), dtype=np.int64),
        end_dates=np.array(["2024-01-01", "2024-01-02", "2024-01-03"]),
        stock_ids=np.array(["A", "B", "C"]),
        image_tokens=np.zeros((2, 8), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Sample count mismatch for image_tokens"):
        arrays.validate()


def test_text_tokens_respect_as_of_date_cutoff() -> None:
    samples = pd.DataFrame(
        {
            "stock_id": ["ABC.NS", "ABC.NS"],
            "date": pd.to_datetime(["2024-01-05", "2024-01-10"]),
        }
    )
    records_without_future = pd.DataFrame(
        [
            {
                "stock_id": "ABC.NS",
                "event_date": "2024-01-04",
                "source_type": "news",
                "title": "Known before first sample",
                "body_text": "This information is visible to both samples.",
            }
        ]
    )
    records_with_future = pd.concat(
        [
            records_without_future,
            pd.DataFrame(
                [
                    {
                        "stock_id": "ABC.NS",
                        "event_date": "2024-01-08",
                        "source_type": "filing",
                        "title": "Future for first sample",
                        "body_text": "This must not affect the first sample token.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    base_tokens = build_text_tokens_for_samples(samples, records_without_future, dim=10)
    future_tokens = build_text_tokens_for_samples(samples, records_with_future, dim=10)

    np.testing.assert_allclose(base_tokens[0], future_tokens[0])
    assert not np.allclose(base_tokens[1], future_tokens[1])


def test_kg_context_tokens_have_stable_event_order() -> None:
    contexts = [
        {
            "peer_count": 2,
            "peer_avg_recent_return": 0.1,
            "sector_avg_recent_return": 0.2,
            "event_flags": {"guidance": 1, "earnings": 0},
        },
        {
            "peer_count": 3,
            "peer_avg_recent_return": 0.3,
            "sector_avg_recent_return": 0.4,
            "event_flags": {"earnings": 1, "guidance": 0},
        },
    ]

    tokens = build_kg_tokens_from_contexts(contexts)

    assert tokens.shape == (2, 5)
    np.testing.assert_allclose(tokens[0], np.array([2, 0.1, 0.2, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(tokens[1], np.array([3, 0.3, 0.4, 1, 0], dtype=np.float32))
