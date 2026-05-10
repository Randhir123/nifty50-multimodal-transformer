"""Build a real-world aligned multimodal artifact from yfinance data and run optional ablations.

Downloads OHLCV data for a configurable stock universe (default: RELIANCE.NS, TCS.NS, INFY.NS),
computes tabular features and outperformance labels, builds GAF/MTF image tokens (ImageCNN),
fetches real news via yfinance for FinBERT text tokens, attaches KG context, and saves an
aligned multimodal NPZ artifact. With --run-ablations, also trains and evaluates modality
combinations using walk-forward cross-validation.

Outputs under --output-dir:
    raw/                                      Per-stock yfinance OHLCV CSVs
    tabular_samples.csv                       Feature and label rows
    text_records.csv                          FinBERT-encoded text records
    real_world_multimodal_samples_gaf.npz     Aligned multimodal artifact
    ablations/ablation_results.csv            Variant x metric table (if --run-ablations)
    ablations/ablation_results.json

Usage:
    python scripts/run_real_world_demo.py \
        --output-dir data/processed/real_world_demo \
        --period 9mo --window-size 20 --horizon-days 3 \
        --run-ablations --epochs 1 --batch-size 4 --device cpu

This script uses live yfinance downloads; results can differ across runs as news and prices
update. Keep the generated raw/ CSVs to reproduce a specific run's artifact exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from src.data.download_yfinance import (
    deterministic_csv_path_for_ticker,
    download_benchmark_data,
    download_multiple_tickers,
    save_ticker_csv,
)
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.data.multimodal_samples import (
    attach_gaf_mtf_image_tokens,
    attach_image_tokens,
    attach_kg_tokens,
    build_tabular_multimodal_samples,
    save_multimodal_samples,
)
from src.data.text import normalize_company_text_records
from src.kg.build_graph import build_market_knowledge_graph
from src.models.image_transformer import ImageTransformerConfig
from src.models.text_encoder import TextEncoder, TextEncoderConfig
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
    today = date.today()
    for ticker in tickers:
        path = deterministic_csv_path_for_ticker(ticker, raw_dir)
        if path.exists() and not force_refresh and date.fromtimestamp(path.stat().st_mtime) == today:
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
    if benchmark_path.exists() and not force_refresh and date.fromtimestamp(benchmark_path.stat().st_mtime) == today:
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
    
    # 1. Fallback Baseline: Deterministic market summaries (ensures historical coverage)
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
                        f"{direction} one-day return of {row['log_return_1d']:.4f}."
                    ),
                }
            )

    # 2. Add yfinance live news on top
    tickers = tabular_df["stock_id"].unique()
    print("Fetching real news from yfinance...")
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
        except Exception as e:
            print(f"Warning: Failed to fetch news for {ticker}: {e}")
            news = []
        
        for item in news:
            publish_time = item.get("providerPublishTime")
            if publish_time is None:
                continue
            
            event_date = pd.to_datetime(publish_time, unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None)
            title = item.get("title", "")
            
            records.append({
                "stock_id": ticker,
                "event_date": event_date,
                "source_type": "yfinance_news",
                "title": title,
                "body_text": title,
            })

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


def _attach_finbert_text_tokens(
    arrays, 
    text_records: pd.DataFrame, 
    device: str,
):
    print("Encoding text records using FinBERT...")
    config = TextEncoderConfig(
        pretrained_model_name="ProsusAI/finbert",
        max_length=192,
        use_mean_pooling=True,
    )
    hidden_size = 768
    
    if text_records.empty:
        text_tokens = np.zeros((len(arrays.stock_ids), hidden_size), dtype=np.float32)
        return dataclasses.replace(arrays, text_tokens=text_tokens)

    encoder = TextEncoder(config).to(device)
    encoder.eval()
    hidden_size = encoder.backbone.config.hidden_size
    text_tokens = np.zeros((len(arrays.stock_ids), hidden_size), dtype=np.float32)

    normalized = normalize_company_text_records(text_records)
    
    with torch.no_grad():
        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()
            
            visible = normalized.loc[
                (normalized["stock_id"] == stock_id)
                & (normalized["event_date"].dt.normalize() <= end_date)
            ]
            
            if visible.empty:
                continue
                
            visible = visible.sort_values("event_date", ascending=False)
            combined_text = " ".join(visible["title"].head(5).tolist())
            if not combined_text.strip():
                continue
                
            emb = encoder.encode_texts([combined_text])
            text_tokens[i] = emb[0].cpu().numpy()

    return dataclasses.replace(arrays, text_tokens=text_tokens)

def _attach_cnn_image_tokens(arrays, chart_dir: Path, device: str, output_dim: int):
    """Load .npy arrays and encode them with the ImageCNN."""
    from src.models.image_cnn import ImageCNN, ImageCNNConfig
    print("Encoding image arrays using CNN...")
    
    config = ImageCNNConfig(image_size=32, in_channels=2, output_dim=output_dim)
    encoder = ImageCNN(config).to(device)
    encoder.eval()
    
    image_tokens = np.zeros((len(arrays.stock_ids), output_dim), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()
            filename = f"{stock_id.upper().replace('.', '_')}_{end_date.strftime('%Y%m%d')}.npy"
            path = Path(chart_dir) / filename
            
            if path.exists():
                tensor = np.load(str(path))
                t_tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
                emb = encoder.encode_images(t_tensor)
                image_tokens[i] = emb[0].cpu().numpy()

    return dataclasses.replace(arrays, image_tokens=image_tokens)

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
    if not getattr(args, "single_split", False) and getattr(args, "cv_splits", 1) > 1:
        command += [
            "--cv-splits", str(args.cv_splits),
            "--horizon-days", str(args.horizon_days),
            "--embargo-days", str(args.embargo_days),
        ]
    elif getattr(args, "single_split", False):
        command.append("--single-split")
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
    parser.add_argument("--chart-lookback-days", type=int, default=20)
    parser.add_argument("--text-dim", type=int, default=768)
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
    parser.add_argument("--cv-splits", type=int, default=1,
                        help="Walk-forward CV folds (>1 activates CV mode).")
    parser.add_argument("--embargo-days", type=int, default=0,
                        help="Calendar-day embargo before each CV test fold.")
    parser.add_argument("--single-split", action="store_true",
                        help="Force single-split mode; overrides --cv-splits.")
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
    arrays = _attach_finbert_text_tokens(arrays, text_records, device=args.device)
    arrays = attach_kg_tokens(arrays, graph, returns=kg_returns)
    arrays = attach_gaf_mtf_image_tokens(
        arrays,
        raw_dir=raw_dir,
        image_size=32,
        output_dim=args.model_dim,
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
