from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.download_yfinance import (
    deterministic_csv_path_for_ticker,
    normalize_ohlcv_dataframe,
    read_tickers_from_file,
    resolve_ticker_universe,
)


def test_ticker_file_parsing_and_dedup_order(tmp_path: Path) -> None:
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("RELIANCE.NS\nTCS.NS\n\n# comment\nRELIANCE.NS\n", encoding="utf-8")

    from_file = read_tickers_from_file(ticker_file)
    assert from_file == ["RELIANCE.NS", "TCS.NS", "RELIANCE.NS"]

    combined = resolve_ticker_universe(
        tickers=["INFY.NS", "TCS.NS"],
        ticker_file=ticker_file,
    )
    assert combined == ["RELIANCE.NS", "TCS.NS", "INFY.NS"]


def test_deterministic_csv_path_generation() -> None:
    assert deterministic_csv_path_for_ticker("RELIANCE.NS") == Path("data/raw/RELIANCE_NS.csv")
    assert deterministic_csv_path_for_ticker("^NSEI") == Path("data/raw/NSEI.csv")


def test_normalization_uses_project_schema() -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Adj Close": [10.4, 11.4],
            "Volume": [100, 120],
        }
    ).set_index("Date")

    normalized = normalize_ohlcv_dataframe(raw)
    assert normalized.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    assert len(normalized) == 2


def test_default_ticker_file_fallback_logic(tmp_path: Path) -> None:
    default_file = tmp_path / "nifty50_full.txt"
    default_file.write_text("RELIANCE.NS\nINFY.NS\n", encoding="utf-8")

    resolved = resolve_ticker_universe(
        tickers=None,
        ticker_file=None,
        default_ticker_file=default_file,
    )
    assert resolved == ["RELIANCE.NS", "INFY.NS"]

    with pytest.raises(FileNotFoundError):
        resolve_ticker_universe(
            tickers=None,
            ticker_file=None,
            default_ticker_file=tmp_path / "missing.txt",
        )
