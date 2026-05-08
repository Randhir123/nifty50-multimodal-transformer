"""Build aligned multimodal fusion sample artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.multimodal_samples import (
    attach_image_tokens,
    attach_kg_tokens,
    attach_text_tokens,
    build_tabular_multimodal_samples,
    build_toy_multimodal_samples,
    infer_numeric_feature_columns,
    save_multimodal_samples,
)
from src.kg.build_graph import build_market_knowledge_graph
from src.models.image_transformer import ImageTransformerConfig


def _parse_feature_cols(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    cols = [col.strip() for col in raw.split(",") if col.strip()]
    if not cols:
        raise ValueError("--feature-cols was provided but no columns were parsed")
    return cols


def _load_stock_sector_mapping(path: str, *, stock_col: str, sector_col: str) -> dict[str, str]:
    df = pd.read_csv(path)
    missing = [col for col in (stock_col, sector_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Stock-sector CSV missing required columns: {missing}")
    return dict(zip(df[stock_col].astype(str).str.strip(), df[sector_col].astype(str).str.strip(), strict=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aligned multimodal samples")
    parser.add_argument("--toy-output", type=str, default="data/processed/multimodal_samples.npz")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tabular-csv", type=str, default=None)
    parser.add_argument("--feature-cols", type=str, default=None)
    parser.add_argument("--stock-col", type=str, default="stock_id")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--tabular-dim", type=int, default=4)
    parser.add_argument("--image-dim", type=int, default=12)
    parser.add_argument("--text-dim", type=int, default=16)
    parser.add_argument("--text-records-csv", type=str, default=None)
    parser.add_argument("--text-top-k", type=int, default=5)
    parser.add_argument("--kg-stock-sector-csv", type=str, default=None)
    parser.add_argument("--kg-sector-col", type=str, default="sector_id")
    parser.add_argument("--kg-returns-csv", type=str, default=None)
    parser.add_argument("--kg-events-csv", type=str, default=None)
    parser.add_argument("--kg-lookback-periods", type=int, default=5)
    parser.add_argument("--kg-event-lookback-days", type=int, default=7)
    parser.add_argument("--kg-index-id", type=str, default="NIFTY50")
    parser.add_argument("--image-chart-dir", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--image-model-dim", type=int, default=16)
    parser.add_argument("--image-num-heads", type=int, default=4)
    parser.add_argument("--image-num-layers", type=int, default=1)
    parser.add_argument("--image-ff-dim", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--image-device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.tabular_csv:
        df = pd.read_csv(args.tabular_csv)
        feature_cols = _parse_feature_cols(args.feature_cols)
        if feature_cols is None:
            feature_cols = infer_numeric_feature_columns(df, stock_col=args.stock_col, date_col=args.date_col, label_col=args.label_col)
        arrays = build_tabular_multimodal_samples(
            df,
            feature_cols=feature_cols,
            stock_col=args.stock_col,
            date_col=args.date_col,
            label_col=args.label_col,
            window_size=args.window_size,
        )
        if args.text_records_csv:
            arrays = attach_text_tokens(
                arrays,
                pd.read_csv(args.text_records_csv),
                top_k=args.text_top_k,
                dim=args.text_dim,
            )
        if args.kg_stock_sector_csv:
            graph = build_market_knowledge_graph(
                _load_stock_sector_mapping(args.kg_stock_sector_csv, stock_col=args.stock_col, sector_col=args.kg_sector_col),
                event_records=pd.read_csv(args.kg_events_csv) if args.kg_events_csv else None,
                index_id=args.kg_index_id,
            )
            arrays = attach_kg_tokens(
                arrays,
                graph,
                returns=pd.read_csv(args.kg_returns_csv) if args.kg_returns_csv else None,
                lookback_periods=args.kg_lookback_periods,
                event_lookback_days=args.kg_event_lookback_days,
                index_id=args.kg_index_id,
            )
        if args.image_chart_dir:
            arrays = attach_image_tokens(
                arrays,
                chart_dir=args.image_chart_dir,
                config=ImageTransformerConfig(
                    image_size=args.image_size,
                    patch_size=args.image_patch_size,
                    model_dim=args.image_model_dim,
                    num_heads=args.image_num_heads,
                    num_layers=args.image_num_layers,
                    ff_dim=args.image_ff_dim,
                ),
                batch_size=args.image_batch_size,
                device=args.image_device,
            )
        output_path = save_multimodal_samples(arrays, Path(args.output or args.toy_output))
        print(f"Saved tabular multimodal sample artifact to: {output_path}")
        print(
            "Shapes: "
            f"tabular={arrays.tabular_tokens.shape}, "
            f"image={arrays.image_tokens.shape if arrays.image_tokens is not None else None}, "
            f"text={arrays.text_tokens.shape if arrays.text_tokens is not None else None}, "
            f"kg={arrays.kg_tokens.shape if arrays.kg_tokens is not None else None}, "
            f"y={arrays.y.shape}"
        )
        print(f"Feature columns: {feature_cols}")
        return

    arrays = build_toy_multimodal_samples(
        num_samples=args.num_samples,
        window_size=args.window_size,
        tabular_dim=args.tabular_dim,
        image_dim=args.image_dim,
        text_dim=args.text_dim,
    )
    output_path = save_multimodal_samples(arrays, Path(args.toy_output))
    print(f"Saved toy multimodal sample artifact to: {output_path}")
    print(
        "Shapes: "
        f"tabular={arrays.tabular_tokens.shape}, "
        f"image={arrays.image_tokens.shape if arrays.image_tokens is not None else None}, "
        f"text={arrays.text_tokens.shape if arrays.text_tokens is not None else None}, "
        f"kg={arrays.kg_tokens.shape if arrays.kg_tokens is not None else None}, "
        f"y={arrays.y.shape}"
    )


if __name__ == "__main__":
    main()
