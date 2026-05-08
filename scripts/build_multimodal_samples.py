"""Build aligned multimodal fusion sample artifacts.

The CLI supports three modes:

1. deterministic toy multimodal samples for smoke tests;
2. tabular CSV input converted into real per-stock rolling-window samples;
3. optional KG tokens attached to tabular samples from small CSV inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.multimodal_samples import (
    attach_kg_tokens,
    build_tabular_multimodal_samples,
    build_toy_multimodal_samples,
    infer_numeric_feature_columns,
    save_multimodal_samples,
)
from src.kg.build_graph import build_market_knowledge_graph


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
    return dict(
        zip(
            df[stock_col].astype(str).str.strip(),
            df[sector_col].astype(str).str.strip(),
            strict=True,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aligned multimodal samples")
    parser.add_argument(
        "--toy-output",
        type=str,
        default="data/processed/multimodal_samples.npz",
        help="Path to write a deterministic toy multimodal NPZ artifact.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write real tabular multimodal samples. Defaults to --toy-output.",
    )
    parser.add_argument(
        "--tabular-csv",
        type=str,
        default=None,
        help="Optional CSV with stock_id, date, label, and feature columns.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        default=None,
        help="Comma-separated feature columns for --tabular-csv. If omitted, numeric columns except label are inferred.",
    )
    parser.add_argument("--stock-col", type=str, default="stock_id")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--tabular-dim", type=int, default=4)
    parser.add_argument("--image-dim", type=int, default=12)
    parser.add_argument("--text-dim", type=int, default=16)

    parser.add_argument(
        "--kg-stock-sector-csv",
        type=str,
        default=None,
        help="Optional CSV with stock-to-sector mapping for KG token construction.",
    )
    parser.add_argument("--kg-sector-col", type=str, default="sector_id")
    parser.add_argument(
        "--kg-returns-csv",
        type=str,
        default=None,
        help="Optional CSV with stock_id,date,recent_return for KG aggregate features.",
    )
    parser.add_argument(
        "--kg-events-csv",
        type=str,
        default=None,
        help="Optional CSV with stock_id,event_date,event_type for KG event flags.",
    )
    parser.add_argument("--kg-lookback-periods", type=int, default=5)
    parser.add_argument("--kg-event-lookback-days", type=int, default=7)
    parser.add_argument("--kg-index-id", type=str, default="NIFTY50")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.tabular_csv:
        df = pd.read_csv(args.tabular_csv)
        feature_cols = _parse_feature_cols(args.feature_cols)
        if feature_cols is None:
            feature_cols = infer_numeric_feature_columns(
                df,
                stock_col=args.stock_col,
                date_col=args.date_col,
                label_col=args.label_col,
            )
        arrays = build_tabular_multimodal_samples(
            df,
            feature_cols=feature_cols,
            stock_col=args.stock_col,
            date_col=args.date_col,
            label_col=args.label_col,
            window_size=args.window_size,
        )

        if args.kg_stock_sector_csv:
            stock_to_sector = _load_stock_sector_mapping(
                args.kg_stock_sector_csv,
                stock_col=args.stock_col,
                sector_col=args.kg_sector_col,
            )
            event_records = (
                pd.read_csv(args.kg_events_csv) if args.kg_events_csv else None
            )
            returns = pd.read_csv(args.kg_returns_csv) if args.kg_returns_csv else None
            graph = build_market_knowledge_graph(
                stock_to_sector,
                event_records=event_records,
                index_id=args.kg_index_id,
            )
            arrays = attach_kg_tokens(
                arrays,
                graph,
                returns=returns,
                lookback_periods=args.kg_lookback_periods,
                event_lookback_days=args.kg_event_lookback_days,
                index_id=args.kg_index_id,
            )

        output_path = save_multimodal_samples(arrays, Path(args.output or args.toy_output))
        print(f"Saved tabular multimodal sample artifact to: {output_path}")
        print(
            "Shapes: "
            f"tabular={arrays.tabular_tokens.shape}, "
            f"kg={arrays.kg_tokens.shape if arrays.kg_tokens is not None else None}, "
            f"y={arrays.y.shape}, "
            f"stocks={arrays.stock_ids.shape}, "
            f"end_dates={arrays.end_dates.shape}"
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
