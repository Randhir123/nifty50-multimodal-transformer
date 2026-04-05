from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.models.image_transformer import ImageTransformer, ImageTransformerConfig
from src.models.tabular_transformer import TabularTransformer, TabularTransformerConfig
from src.models.text import CompanyTextTransformer, CompanyTextTransformerConfig
from src.viz.charts import generate_or_resolve_sample_chart


def test_candlestick_chart_generation_smoke(
    tmp_path: Path,
    toy_ohlcv: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    stock_df, _ = toy_ohlcv
    prediction_date = stock_df["date"].iloc[-1]

    chart_path = generate_or_resolve_sample_chart(
        stock_df,
        symbol="TCS",
        prediction_date=prediction_date,
        output_dir=tmp_path,
        lookback_days=60,
    )

    assert chart_path.exists()
    assert chart_path.suffix == ".png"


def test_tabular_transformer_forward_pass_smoke() -> None:
    model = TabularTransformer(
        TabularTransformerConfig(feature_dim=6, model_dim=32, num_heads=4, num_layers=2)
    )
    x = torch.randn(4, 20, 6)
    logits = model(x)

    assert logits.shape == (4,)


def test_image_branch_forward_pass_smoke() -> None:
    model = ImageTransformer(
        ImageTransformerConfig(image_size=64, patch_size=16, model_dim=32, num_heads=4, num_layers=2)
    )
    x = torch.randn(2, 3, 64, 64)

    logits = model(x)
    embeddings = model.encode_images(x)

    assert logits.shape == (2,)
    assert embeddings.shape == (2, 32)


def test_text_branch_forward_pass_smoke() -> None:
    model = CompanyTextTransformer(
        CompanyTextTransformerConfig(vocab_size=512, max_length=32, model_dim=32, num_heads=4, num_layers=2)
    )
    input_ids, attention_mask = model.tokenize_texts(
        ["earnings beat and guidance raised", "margin pressure from costs"]
    )
    logits = model(input_ids, attention_mask)

    assert logits.shape == (2,)
