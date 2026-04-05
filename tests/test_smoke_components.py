from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import create_rolling_transformer_dataset
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.data.text import build_company_text_input, normalize_company_text_records
from src.kg.build_graph import build_market_knowledge_graph
from src.kg.query_graph import retrieve_kg_context
from src.models.image_transformer import ImageTransformer, ImageTransformerConfig
from src.models.tabular_transformer import TabularTransformer, TabularTransformerConfig
from src.models.text import CompanyTextTransformer, CompanyTextTransformerConfig
from src.viz.charts import generate_or_resolve_sample_chart


TOY_DIR = Path("data/toy")


def _load_toy_ohlcv() -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_df = pd.read_csv(TOY_DIR / "stock_ohlcv.csv")
    index_df = pd.read_csv(TOY_DIR / "index_ohlcv.csv")
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    index_df["date"] = pd.to_datetime(index_df["date"])
    return stock_df, index_df


def test_feature_generation_smoke() -> None:
    stock_df, index_df = _load_toy_ohlcv()
    featured = compute_technical_features(stock_df, index_df)
    assert "log_return_1d" in featured.columns
    assert featured["stock_minus_index_return"].notna().sum() > 0


def test_label_generation_smoke() -> None:
    stock_df, index_df = _load_toy_ohlcv()
    featured = compute_technical_features(stock_df, index_df)
    labeled = generate_outperformance_label(featured)
    assert "label" in labeled.columns
    assert labeled["label"].dropna().isin([0, 1]).all()


def test_rolling_window_dataset_smoke() -> None:
    stock_df, index_df = _load_toy_ohlcv()
    featured = compute_technical_features(stock_df, index_df)
    labeled = generate_outperformance_label(featured)

    feature_cols = [
        "log_return_1d",
        "cum_return_3d",
        "cum_return_5d",
        "realized_vol_5d",
        "high_low_range_over_close",
        "stock_minus_index_return",
    ]

    windows = create_rolling_transformer_dataset(
        labeled,
        feature_cols=feature_cols,
        window_size=20,
        dropna=True,
    )
    assert windows.X.ndim == 3
    assert windows.X.shape[-1] == len(feature_cols)
    assert windows.X.shape[0] == windows.y.shape[0] == windows.end_dates.shape[0]


def test_candlestick_chart_generation_smoke(tmp_path: Path) -> None:
    stock_df, _ = _load_toy_ohlcv()
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


def test_tabular_transformer_forward_smoke() -> None:
    model = TabularTransformer(TabularTransformerConfig(feature_dim=6, model_dim=32, num_heads=4, num_layers=2))
    x = torch.randn(4, 20, 6)
    logits = model(x)
    assert logits.shape == (4,)


def test_image_branch_forward_smoke() -> None:
    model = ImageTransformer(ImageTransformerConfig(image_size=64, patch_size=16, model_dim=32, num_heads=4, num_layers=2))
    x = torch.randn(2, 3, 64, 64)
    logits = model(x)
    embeddings = model.encode_images(x)
    assert logits.shape == (2,)
    assert embeddings.shape == (2, 32)


def test_text_branch_forward_smoke() -> None:
    model = CompanyTextTransformer(
        CompanyTextTransformerConfig(vocab_size=512, max_length=32, model_dim=32, num_heads=4, num_layers=2)
    )
    input_ids, attention_mask = model.tokenize_texts(["earnings beat and guidance raised", "margin pressure from costs"])
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2,)


def test_kg_context_retrieval_smoke() -> None:
    records = pd.read_csv(TOY_DIR / "event_records.csv")
    returns = pd.DataFrame(
        {
            "stock_id": ["TCS", "TCS", "INFY", "INFY"],
            "date": ["2024-03-01", "2024-03-05", "2024-03-01", "2024-03-05"],
            "recent_return": [0.01, 0.015, 0.009, 0.011],
        }
    )
    returns["date"] = pd.to_datetime(returns["date"])

    graph = build_market_knowledge_graph(
        {"TCS": "IT", "INFY": "IT"},
        event_records=records,
        event_types=["earnings", "guidance"],
    )

    context = retrieve_kg_context(graph, stock_id="TCS", as_of_date="2024-03-06", returns=returns)
    assert context["sector_id"] == "IT"
    assert context["peer_count"] == 1
    assert "event_flags" in context


def test_text_input_assembly_smoke() -> None:
    records = pd.read_csv(TOY_DIR / "text_records.csv")
    normalized = normalize_company_text_records(records)
    text = build_company_text_input(normalized, stock_id="TCS", as_of_date="2024-03-06", top_k=2)
    assert "Order win" in text or "Quarterly update" in text
