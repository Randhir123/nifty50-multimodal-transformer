"""Build aligned multimodal fusion sample artifacts.

The CLI supports two modes:

1. deterministic toy multimodal samples for smoke tests;
2. tabular CSV input converted into real per-stock rolling-window samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.multimodal_samples import (
    build_tabular_multimodal_samples,
    build_toy_multimodal_samples,
    infer_numeric_feature_columns,
    save_multimodal_samples,
)


def _parse_feature_cols(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    cols = [col.strip() for col in raw.split(",") if col.strip()]
    if not cols:
        raise ValueError("--feature-cols was provided but no columns were parsed")
    return cols


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
        output_path = save_multimodal_samples(arrays, Path(args.output or args.toy_output))
        print(f"Saved tabular multimodal sample artifact to: {output_path}")
        print(
            "Shapes: "
            f"tabular={arrays.tabular_tokens.shape}, "
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
