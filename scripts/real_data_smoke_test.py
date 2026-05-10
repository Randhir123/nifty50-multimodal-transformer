"""Run a tiny real-data smoke test using cached/downloader yfinance OHLCV CSVs.

Workflow covered:
1) read cached CSVs when present (download only if missing)
2) compute technical features
3) generate outperformance labels versus benchmark
4) build rolling windows
5) generate candlestick charts for latest prediction sample
6) execute a fusion-ready ranking workflow via ``rank_stocks_endpoint``
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.api import rank_stocks_endpoint
from src.data.dataset import create_rolling_transformer_dataset
from src.data.download_yfinance import (
    deterministic_csv_path_for_ticker,
    download_single_ticker_ohlcv,
    save_ticker_csv,
)
from src.data.features import compute_technical_features
from src.data.labels import generate_outperformance_label
from src.viz.charts import generate_or_resolve_sample_chart

DEFAULT_TICKERS = ("RELIANCE.NS", "TCS.NS", "INFY.NS")
DEFAULT_BENCHMARK = "^NSEI"
DEFAULT_START = "2023-01-01"
DEFAULT_WINDOW_SIZE = 20
FEATURE_COLS = [
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


def _load_or_download_csv(
    ticker: str,
    *,
    raw_dir: Path,
    start: str,
    end: str,
    interval: str,
    force_refresh: bool,
) -> tuple[pd.DataFrame, Path, bool]:
    """Return OHLCV dataframe from cache, downloading when missing/forced.

    Returns:
        (df, csv_path, refreshed) where refreshed=True means a fresh download was used.
    """
    csv_path = deterministic_csv_path_for_ticker(ticker, output_dir=raw_dir)
    if csv_path.exists() and not force_refresh and date.fromtimestamp(csv_path.stat().st_mtime) == date.today():
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        return df, csv_path, False

    df = download_single_ticker_ohlcv(ticker, start=start, end=end, interval=interval)
    saved_path = save_ticker_csv(ticker, df, output_dir=raw_dir)
    return df, saved_path, True


def run_real_data_smoke_test(args: argparse.Namespace) -> None:
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    chart_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_end = args.end or date.today().isoformat()
    refresh_label = "enabled" if args.force_refresh else "disabled"
    print(f"End date used for downloads: {resolved_end} (force refresh: {refresh_label})")

    universe = list(args.tickers)
    benchmark_df, benchmark_path, benchmark_refreshed = _load_or_download_csv(
        args.benchmark,
        raw_dir=raw_dir,
        start=args.start,
        end=resolved_end,
        interval=args.interval,
        force_refresh=args.force_refresh,
    )

    print(
        f"Benchmark {args.benchmark} -> {benchmark_path} "
        f"({'refreshed' if benchmark_refreshed else 'cached'})"
    )

    ranking_rows: list[dict[str, str]] = []
    ranking_probs: list[float] = []
    stock_summaries: list[dict[str, object]] = []

    for ticker in universe:
        stock_df, stock_path, refreshed = _load_or_download_csv(
            ticker,
            raw_dir=raw_dir,
            start=args.start,
            end=resolved_end,
            interval=args.interval,
            force_refresh=args.force_refresh,
        )
        print(f"Stock {ticker} -> {stock_path} ({'refreshed' if refreshed else 'cached'})")

        featured = compute_technical_features(stock_df, benchmark_df)
        labeled = generate_outperformance_label(featured)
        windows = create_rolling_transformer_dataset(
            labeled,
            feature_cols=FEATURE_COLS,
            window_size=args.window_size,
            dropna=True,
        )

        latest_end_date = pd.Timestamp(windows.end_dates[-1])
        chart_path = generate_or_resolve_sample_chart(
            stock_df,
            symbol=ticker,
            prediction_date=latest_end_date,
            output_dir=chart_dir,
            lookback_days=args.window_size,
            regenerate=False,
        )

        latest_label = int(windows.y[-1])
        probability = 0.8 if latest_label == 1 else 0.2

        ranking_rows.append(
            {
                "stock_id": ticker,
                "date": latest_end_date.date().isoformat(),
            }
        )
        ranking_probs.append(probability)

        stock_summaries.append(
            {
                "stock": ticker,
                "source_csv": str(stock_path),
                "source_mode": "refreshed" if refreshed else "cached",
                "num_feature_rows": int(len(featured)),
                "num_labeled_rows": int(len(labeled)),
                "num_rolling_samples": int(windows.X.shape[0]),
                "window_size": int(args.window_size),
                "latest_end_date": latest_end_date.date().isoformat(),
                "latest_label": latest_label,
                "chart_path": str(chart_path),
            }
        )

    ranking_input = pd.DataFrame(ranking_rows)
    ranking_payload = rank_stocks_endpoint(
        samples=ranking_input,
        probabilities=np.array(ranking_probs, dtype=np.float64),
        threshold=0.5,
    )

    ranking_path = output_dir / "real_data_ranking.csv"
    summary_path = output_dir / "real_data_smoke_summary.json"
    ranking_payload["ranking"].to_csv(ranking_path, index=False)

    latest_sample_date = str(ranking_payload["ranking"]["date"].max())

    summary_payload = {
        "tickers": universe,
        "benchmark": args.benchmark,
        "benchmark_csv": str(benchmark_path),
        "benchmark_mode": "refreshed" if benchmark_refreshed else "cached",
        "start": args.start,
        "end": resolved_end,
        "interval": args.interval,
        "window_size": args.window_size,
        "feature_columns": FEATURE_COLS,
        "latest_ranking_sample_date": latest_sample_date,
        "stocks": stock_summaries,
        "ranking_csv": str(ranking_path),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Real-data smoke test completed successfully.")
    print(f"- Ranking output: {ranking_path}")
    print(f"- Summary output: {summary_path}")
    print(f"- Latest ranking sample date: {latest_sample_date}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny real-data smoke test with yfinance/cached CSV inputs"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=list(DEFAULT_TICKERS),
        help="Small stock ticker set (default: RELIANCE.NS TCS.NS INFY.NS)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=DEFAULT_BENCHMARK,
        help="Benchmark index ticker (default: ^NSEI)",
    )
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today's date)",
    )
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval")
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Rolling window length for dataset/chart generation",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for cached/downloaded OHLCV CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/real_data_smoke"),
        help="Directory where smoke-test artifacts are written",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cached CSVs and refresh OHLCV data from yfinance",
    )
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    run_real_data_smoke_test(cli_args)
