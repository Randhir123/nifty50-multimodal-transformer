"""Download Yahoo Finance OHLCV data and persist deterministic local CSV snapshots.

This module keeps the rest of the project file-based by using ``yfinance`` only for
acquisition. Downloaded data is normalized to the project's expected OHLCV schema
and cached immediately under ``data/raw/``.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


PROJECT_OHLCV_COLUMNS: tuple[str, ...] = ("date", "open", "high", "low", "close", "volume")
DEFAULT_TICKER_FILE = Path("config/nifty50_full.txt")
DEFAULT_OUTPUT_DIR = Path("data/raw")


def resolve_end_date(end: str | None) -> str:
    """Resolve optional end date to YYYY-MM-DD, defaulting to today's date."""
    return end or date.today().isoformat()


def read_tickers_from_file(path: str | Path) -> list[str]:
    """Read ticker symbols from a plain-text file.

    The file should contain one ticker per line. Empty lines and lines starting
    with ``#`` are ignored.

    Args:
        path: Path to a text file.

    Returns:
        Ordered ticker list in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ticker file not found: {file_path}")

    tickers: list[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tickers.append(line)
    return tickers


def resolve_ticker_universe(
    *,
    tickers: Iterable[str] | None,
    ticker_file: str | Path | None,
    default_ticker_file: str | Path = DEFAULT_TICKER_FILE,
) -> list[str]:
    """Resolve ticker universe from CLI tickers and/or a ticker file.

    If neither source is provided, this function falls back to
    ``default_ticker_file``.

    Args:
        tickers: Explicit tickers from CLI.
        ticker_file: Optional file path with one ticker per line.
        default_ticker_file: Fallback file when no explicit source is provided.

    Returns:
        De-duplicated list of tickers with deterministic order.

    Raises:
        FileNotFoundError: If fallback ticker file is needed but does not exist.
        ValueError: If no tickers remain after resolution.
    """
    ordered: list[str] = []

    file_to_use: str | Path | None = ticker_file
    if file_to_use is None and not tickers:
        file_to_use = default_ticker_file
        if not Path(file_to_use).exists():
            raise FileNotFoundError(
                "No tickers provided and default ticker universe file is missing: "
                f"{Path(file_to_use)}"
            )

    if file_to_use is not None:
        ordered.extend(read_tickers_from_file(file_to_use))

    if tickers:
        ordered.extend([ticker.strip() for ticker in tickers if ticker.strip()])

    deduplicated = list(dict.fromkeys(ordered))
    if not deduplicated:
        raise ValueError("Ticker universe is empty after parsing --ticker-file/--tickers")

    return deduplicated


def deterministic_csv_path_for_ticker(ticker: str, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Return deterministic CSV path for a ticker symbol.

    Symbols are sanitized into filename-safe identifiers while preserving
    deterministic mapping.
    """
    safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    if not safe:
        raise ValueError(f"Ticker produced empty filename after sanitization: {ticker!r}")
    return Path(output_dir) / f"{safe}.csv"


def normalize_ohlcv_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to project OHLCV schema.

    Expected output columns are exactly:
    ``date, open, high, low, close, volume``.

    Args:
        raw_df: Raw dataframe returned by ``yfinance.download``.

    Returns:
        Normalized OHLCV dataframe.

    Raises:
        ValueError: If required OHLCV columns cannot be found.
    """
    if raw_df.empty:
        raise ValueError("Download returned empty data")

    df = raw_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns.to_list()]

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }

    if "Date" not in df.columns and df.index.name in {"Date", "Datetime", None}:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=False)

    df = df.rename(columns=rename_map)

    required = set(PROJECT_OHLCV_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required OHLCV columns after normalization: {missing}")

    normalized = df.loc[:, PROJECT_OHLCV_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized.sort_values("date").reset_index(drop=True)

    return normalized


def download_single_ticker_ohlcv(
    ticker: str,
    *,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download and normalize OHLCV data for one ticker."""
    raw_df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if raw_df.empty:
        raise ValueError(
            f"Download returned empty data for ticker={ticker!r}, start={start}, end={end}"
        )

    return normalize_ohlcv_dataframe(raw_df)


def download_multiple_tickers(
    tickers: Iterable[str],
    *,
    start: str,
    end: str,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for multiple tickers."""
    return {
        ticker: download_single_ticker_ohlcv(ticker, start=start, end=end, interval=interval)
        for ticker in tickers
    }


def download_benchmark_data(
    benchmark: str,
    *,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download benchmark/index OHLCV data (for example ``^NSEI``)."""
    return download_single_ticker_ohlcv(benchmark, start=start, end=end, interval=interval)


def save_ticker_csv(
    ticker: str,
    df: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save a ticker dataframe to deterministic CSV path."""
    output_path = deterministic_csv_path_for_ticker(ticker, output_dir=output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Download and cache OHLCV CSV files from yfinance")
    parser.add_argument("--ticker-file", type=str, default=None, help="Path to ticker universe file")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Explicit ticker symbols (e.g. RELIANCE.NS TCS.NS INFY.NS)",
    )
    parser.add_argument("--benchmark", type=str, default=None, help="Optional benchmark ticker (e.g. ^NSEI)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, default: today's date)",
    )
    parser.add_argument("--interval", type=str, default="1d", help="yfinance interval (default: 1d)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="CSV output directory")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for yfinance ingestion."""
    args = parse_args()
    tickers = resolve_ticker_universe(
        tickers=args.tickers,
        ticker_file=args.ticker_file,
        default_ticker_file=DEFAULT_TICKER_FILE,
    )

    resolved_end = resolve_end_date(args.end)
    print(f"Using end date: {resolved_end}")

    ticker_data = download_multiple_tickers(
        tickers,
        start=args.start,
        end=resolved_end,
        interval=args.interval,
    )

    for ticker in tickers:
        output_path = save_ticker_csv(ticker, ticker_data[ticker], output_dir=args.output_dir)
        print(f"Saved {ticker} -> {output_path}")

    if args.benchmark:
        benchmark_df = download_benchmark_data(
            args.benchmark,
            start=args.start,
            end=resolved_end,
            interval=args.interval,
        )
        benchmark_path = save_ticker_csv(args.benchmark, benchmark_df, output_dir=args.output_dir)
        print(f"Saved benchmark {args.benchmark} -> {benchmark_path}")


if __name__ == "__main__":
    main()
