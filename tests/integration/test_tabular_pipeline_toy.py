from __future__ import annotations

import pandas as pd
import torch

from src.data.dataset import create_rolling_transformer_dataset
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.models.tabular_transformer import TabularTransformer, TabularTransformerConfig


FEATURE_COLS = [
    "log_return_1d",
    "cum_return_3d",
    "cum_return_5d",
    "realized_vol_5d",
    "high_low_range_over_close",
    "stock_minus_index_return",
]


def test_tabular_pipeline_integration_with_toy_data(
    toy_ohlcv: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    stock_df, index_df = toy_ohlcv

    featured = compute_technical_features(stock_df, index_df)
    labeled = generate_outperformance_label(featured)
    windows = create_rolling_transformer_dataset(
        labeled,
        feature_cols=FEATURE_COLS,
        window_size=20,
        dropna=True,
    )

    model = TabularTransformer(
        TabularTransformerConfig(feature_dim=len(FEATURE_COLS), model_dim=32, num_heads=4, num_layers=2)
    )
    model.eval()

    x = torch.from_numpy(windows.X[:2])
    logits = model(x)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
