"""End-to-end verification for the tabular baseline using toy data."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.dataset import create_rolling_transformer_dataset
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.training.train_tabular import train_tabular_transformer


def run_verification(output_dir: Path) -> None:
    toy_dir = Path("data/toy")
    stock_df = pd.read_csv(toy_dir / "stock_ohlcv.csv")
    index_df = pd.read_csv(toy_dir / "index_ohlcv.csv")

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    index_df["date"] = pd.to_datetime(index_df["date"])

    featured = compute_technical_features(stock_df, index_df)
    labeled = generate_outperformance_label(featured)

    feature_cols = [
        "log_return_1d",
        "cum_return_3d",
        "cum_return_5d",
        "cum_return_10d",
        "realized_vol_5d",
        "realized_vol_10d",
        "high_low_range_over_close",
        "close_over_10dma_minus_1",
        "close_over_20dma_minus_1",
        "volume_over_20d_avg",
        "stock_minus_index_return",
    ]

    windows = create_rolling_transformer_dataset(
        labeled,
        feature_cols=feature_cols,
        window_size=20,
        dropna=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "toy_rolling_windows.npz"
    checkpoint_path = output_dir / "toy_tabular_checkpoint.pt"

    np.savez(
        dataset_path,
        X=windows.X,
        y=windows.y,
        end_dates=windows.end_dates,
    )

    args = SimpleNamespace(
        dataset=str(dataset_path),
        checkpoint_path=str(checkpoint_path),
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        val_fraction=0.2,
        device="cpu",
        model_dim=32,
        num_heads=4,
        num_layers=1,
        ff_dim=64,
        dropout=0.1,
        pooling="mean",
    )
    train_tabular_transformer(args)

    print(f"Verification complete. Dataset: {dataset_path}")
    print(f"Verification complete. Checkpoint: {checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify tabular baseline with toy data")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/verification"))
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    run_verification(cli_args.output_dir)
