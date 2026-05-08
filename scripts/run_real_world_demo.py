"""Build a real-world aligned multimodal artifact and optional ablations.

This script is intentionally manual because it uses live yfinance downloads.
It creates a small, reproducible local snapshot under the requested output
folder and then builds the same aligned multimodal artifact used by fusion
training and ablations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download_yfinance import (
    deterministic_csv_path_for_ticker,
    download_benchmark_data,
    download_multiple_tickers,
    save_ticker_csv,
)
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.data.multimodal_samples import (
    attach_image_tokens,
    attach_kg_tokens,
    attach_text_tokens,
    build_tabular_multimodal_samples,
    save_multimodal_samples,
)
from src.kg.build_graph import build_market_knowledge_graph
from src.models.image_transformer import ImageTransformerConfig
from src.viz.charts import generate_or_resolve_sample_chart

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
DEFAULT_SECTORS = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
}
FEATURE_COLUMNS = [
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


def _resolve_start_date(period: str, end: str) -> str:
    end_date = pd.Timestamp(end).date()
    if period.endswith("mo"):
        months = int(period[:-2])
        return (end_date - timedelta(days=months * 31)).isoformat()
    if period.endswith("y"):
        years = int(period[:-1])
        return (end_date - timedelta(days=years * 365)).isoformat()
    raise ValueError("period must use a suffix like 9mo or 2y")


def _load_or_download(
    *,
    tickers: list[str],
    benchmark: str,
    start: str,
    end: str,
    raw_dir: Path,
    force_refresh: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    provenance: dict[str, str] = {}
    stock_data: dict[str, pd.DataFrame] = {}

    missing = []
    for ticker in tickers:
        path = deterministic_csv_path_for_ticker(ticker, raw_dir)
        if path.exists() and not force_refresh:
            stock_data[ticker] = pd.read_csv(path, parse_dates=["date"])
            provenance[ticker] = f"cache:{path}"
        else:
            missing.append(ticker)

    if missing:
        downloaded = download_multiple_tickers(missing, start=start, end=end)
        for ticker, df in downloaded.items():
            save_path = save_ticker_csv(ticker, df, output_dir=raw_dir)
            stock_data[ticker] = df
            provenance[ticker] = f"download:{save_path}"

    benchmark_path = deterministic_csv_path_for_ticker(benchmark, raw_dir)
    if benchmark_path.exists() and not force_refresh:
        benchmark_df = pd.read_csv(benchmark_path, parse_dates=["date"])
        provenance[benchmark] = f"cache:{benchmark_path}"
    else:
        benchmark_df = download_benchmark_data(benchmark, start=start, end=end)
        save_path = save_ticker_csv(benchmark, benchmark_df, output_dir=raw_dir)
        provenance[benchmark] = f"download:{save_path}"

    return stock_data, benchmark_df, provenance


def _build_tabular_rows(
    *,
    stock_data: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame,
    horizon_days: int,
) -> pd.DataFrame:
    frames = []
    for ticker, stock_df in stock_data.items():
        features = compute_technical_features(stock_df, benchmark_df)
        labelled = generate_outperformance_label(features, horizon_days=horizon_days)
        labelled["stock_id"] = ticker
        frames.append(labelled)
    combined = pd.concat(frames, ignore_index=True)
    required = ["stock_id", "date", "label", *FEATURE_COLUMNS]
    return combined.dropna(subset=required).loc[:, required + ["close", "volume"]].reset_index(drop=True)


def _build_text_records(tabular_df: pd.DataFrame, *, text_every_n_days: int = 5) -> pd.DataFrame:
    records = []
    for ticker, frame in tabular_df.groupby("stock_id"):
        frame = frame.sort_values("date").reset_index(drop=True)
        for idx in range(0, len(frame), text_every_n_days):
            row = frame.iloc[idx]
            direction = "positive" if row["log_return_1d"] >= 0 else "negative"
            records.append(
                {
                    "stock_id": ticker,
                    "event_date": row["date"],
                    "source_type": "market_summary",
                    "title": f"{ticker} {direction} daily market summary",
                    "body_text": (
                        f"As of {pd.Timestamp(row['date']).date()}, {ticker} had a "
                        f"{direction} one-day return of {row['log_return_1d']:.4f}, "
                        f"relative return versus index of {row['stock_minus_index_return']:.4f}, "
                        f"and volume ratio {row['volume_over_20d_avg']:.4f}."
                    ),
                }
            )
    return pd.DataFrame(records)


def _build_kg_returns(tabular_df: pd.DataFrame) -> pd.DataFrame:
    return tabular_df.loc[:, ["stock_id", "date", "stock_minus_index_return"]].rename(
        columns={"stock_minus_index_return": "recent_return"}
    )


def _build_event_records(tabular_df: pd.DataFrame) -> pd.DataFrame:
    events = []
    for ticker, frame in tabular_df.groupby("stock_id"):
        top_volume = frame["volume_over_20d_avg"].quantile(0.9)
        for _, row in frame.loc[frame["volume_over_20d_avg"] >= top_volume].iterrows():
            events.append(
                {
                    "stock_id": ticker,
                    "event_date": row["date"],
                    "event_type": "high_volume",
                }
            )
    return pd.DataFrame(events) if events else pd.DataFrame(columns=["stock_id", "event_date", "event_type"])


def _generate_charts_for_samples(
    *,
    arrays,
    stock_data: dict[str, pd.DataFrame],
    chart_dir: Path,
    chart_lookback_days: int,
) -> int:
    count = 0
    for stock_id, end_date in zip(arrays.stock_ids, arrays.end_dates, strict=True):
        generate_or_resolve_sample_chart(
            stock_data[str(stock_id)],
            symbol=str(stock_id),
            prediction_date=pd.Timestamp(end_date),
            output_dir=chart_dir,
            lookback_days=chart_lookback_days,
        )
        count += 1
    return count


def _write_summary(path: Path, summary: dict[str, object]) -> None:
    lines = ["# Real-World Multimodal Demo Summary", ""]
    for key, value in summary.items():
        if isinstance(value, dict):
            lines.append(f"## {key}")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(value, indent=2, default=str))
            lines.append("```")
            lines.append("")
        else:
            lines.append(f"- **{key}**: {value}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_ablations(args: argparse.Namespace, dataset_path: Path, output_dir: Path) -> None:
    command = [
        sys.executable,
        "scripts/run_ablation_study.py",
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir / "ablations"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--model-dim",
        str(args.model_dim),
        "--num-heads",
        str(args.num_heads),
        "--num-layers",
        str(args.num_layers),
        "--ff-dim",
        str(args.ff_dim),
        "--val-fraction",
        str(args.val_fraction),
    ]
    subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-world multimodal demo")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--period", default="9mo")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output-dir", default="data/processed/real_world_demo")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--horizon-days", type=int, default=3)
    parser.add_argument("--chart-lookback-days", type=int, default=60)
    parser.add_argument("--text-dim", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-patch-size", type=int, default=16)
    parser.add_argument("--image-model-dim", type=int, default=16)
    parser.add_argument("--image-num-heads", type=int, default=4)
    parser.add_argument("--image-num-layers", type=int, default=1)
    parser.add_argument("--image-ff-dim", type=int, default=32)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-dim", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--ff-dim", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = args.start or _resolve_start_date(args.period, args.end)
    stock_data, benchmark_df, provenance = _load_or_download(
        tickers=args.tickers,
        benchmark=args.benchmark,
        start=start,
        end=args.end,
        raw_dir=raw_dir,
        force_refresh=args.force_refresh,
    )

    tabular_df = _build_tabular_rows(
        stock_data=stock_data,
        benchmark_df=benchmark_df,
        horizon_days=args.horizon_days,
    )
    tabular_csv = output_dir / "tabular_samples.csv"
    tabular_df.to_csv(tabular_csv, index=False)

    text_records = _build_text_records(tabular_df)
    text_records_csv = output_dir / "text_records.csv"
    text_records.to_csv(text_records_csv, index=False)

    sectors = {ticker: DEFAULT_SECTORS.get(ticker, "Unknown") for ticker in args.tickers}
    stock_sectors = pd.DataFrame(
        [{"stock_id": ticker, "sector_id": sector} for ticker, sector in sectors.items()]
    )
    stock_sectors_csv = output_dir / "stock_sectors.csv"
    stock_sectors.to_csv(stock_sectors_csv, index=False)

    kg_returns = _build_kg_returns(tabular_df)
    kg_returns_csv = output_dir / "kg_returns.csv"
    kg_returns.to_csv(kg_returns_csv, index=False)

    event_records = _build_event_records(tabular_df)
    event_records_csv = output_dir / "event_records.csv"
    event_records.to_csv(event_records_csv, index=False)

    arrays = build_tabular_multimodal_samples(
        tabular_df,
        feature_cols=FEATURE_COLUMNS,
        window_size=args.window_size,
    )
    _generate_charts_for_samples(
        arrays=arrays,
        stock_data=stock_data,
        chart_dir=chart_dir,
        chart_lookback_days=args.chart_lookback_days,
    )
    graph = build_market_knowledge_graph(sectors, event_records=event_records)
    arrays = attach_text_tokens(arrays, text_records, dim=args.text_dim)
    arrays = attach_kg_tokens(arrays, graph, returns=kg_returns)
    arrays = attach_image_tokens(
        arrays,
        chart_dir=chart_dir,
        config=ImageTransformerConfig(
            image_size=args.image_size,
            patch_size=args.image_patch_size,
            model_dim=args.image_model_dim,
            num_heads=args.image_num_heads,
            num_layers=args.image_num_layers,
            ff_dim=args.image_ff_dim,
        ),
        device=args.device,
    )

    dataset_path = save_multimodal_samples(
        arrays, output_dir / "real_world_multimodal_samples.npz"
    )

    if args.run_ablations:
        _run_ablations(args, dataset_path, output_dir)

    summary = {
        "tickers": args.tickers,
        "benchmark": args.benchmark,
        "start": start,
        "end": args.end,
        "window_size": args.window_size,
        "horizon_days": args.horizon_days,
        "samples": int(arrays.y.shape[0]),
        "positive_labels": int(np.asarray(arrays.y).sum()),
        "dataset_path": str(dataset_path),
        "tabular_shape": list(arrays.tabular_tokens.shape),
        "image_shape": list(arrays.image_tokens.shape) if arrays.image_tokens is not None else None,
        "text_shape": list(arrays.text_tokens.shape) if arrays.text_tokens is not None else None,
        "kg_shape": list(arrays.kg_tokens.shape) if arrays.kg_tokens is not None else None,
        "provenance": provenance,
    }
    _write_summary(output_dir / "DEMO_SUMMARY.md", summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
