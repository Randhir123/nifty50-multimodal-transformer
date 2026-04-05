"""Deterministic candlestick chart generation utilities.

This module is intentionally lightweight so it can be used both from local
batch scripts and future online entry points (for example ``src/app/api.py``
or a Cloud Run job).
"""

from __future__ import annotations

from pathlib import Path

import mplfinance as mpf
import pandas as pd


REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _validate_ohlcv_columns(df: pd.DataFrame, *, date_col: str) -> None:
    """Validate that a dataframe contains required OHLCV columns."""
    required = set(REQUIRED_OHLCV_COLUMNS) | {date_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_for_mplfinance(df: pd.DataFrame, *, date_col: str) -> pd.DataFrame:
    """Return a sorted OHLCV dataframe indexed by datetime for mplfinance."""
    _validate_ohlcv_columns(df, date_col=date_col)
    chart_df = df.copy()
    chart_df[date_col] = pd.to_datetime(chart_df[date_col])
    chart_df = chart_df.sort_values(date_col).set_index(date_col)
    return chart_df.loc[:, list(REQUIRED_OHLCV_COLUMNS)]


def build_chart_filename(symbol: str, prediction_date: pd.Timestamp) -> str:
    """Create a deterministic chart filename for one (stock, date) sample.

    Format: ``{symbol}_{YYYYMMDD}.png``.
    """
    normalized_symbol = symbol.strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")

    ts = pd.Timestamp(prediction_date)
    return f"{normalized_symbol}_{ts.strftime('%Y%m%d')}.png"


def resolve_chart_path(
    symbol: str,
    prediction_date: pd.Timestamp,
    *,
    output_dir: str | Path,
) -> Path:
    """Resolve the deterministic filesystem path for a chart image."""
    return Path(output_dir) / build_chart_filename(
        symbol=symbol, prediction_date=prediction_date
    )


def generate_candlestick_chart(
    ohlcv_window: pd.DataFrame,
    *,
    output_path: str | Path,
    date_col: str = "date",
) -> Path:
    """Render and save a candlestick chart with volume and MA(10, 20).

    Args:
        ohlcv_window: OHLCV rows for the chart window (typically 60 rows).
        output_path: Destination PNG file.
        date_col: Name of the date column in ``ohlcv_window``.

    Returns:
        The saved path.
    """
    chart_df = _normalize_for_mplfinance(ohlcv_window, date_col=date_col)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mpf.plot(
        chart_df,
        type="candle",
        volume=True,
        mav=(10, 20),
        style="classic",
        ylabel="Price",
        ylabel_lower="Volume",
        xrotation=0,
        datetime_format="%Y-%m-%d",
        savefig={
            "fname": str(out),
            "dpi": 120,
            "pad_inches": 0.05,
            "metadata": {"Software": "nifty50-multimodal-transformer"},
        },
    )
    return out


def generate_or_resolve_sample_chart(
    stock_ohlcv: pd.DataFrame,
    *,
    symbol: str,
    prediction_date: pd.Timestamp,
    output_dir: str | Path,
    lookback_days: int = 60,
    date_col: str = "date",
    regenerate: bool = False,
) -> Path:
    """Generate (or resolve) chart path for one ``(stock, date)`` sample.

    The chart uses the most recent ``lookback_days`` rows with ``date <=
    prediction_date``.
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be a positive integer")

    output_path = resolve_chart_path(symbol, prediction_date, output_dir=output_dir)
    if output_path.exists() and not regenerate:
        return output_path

    frame = stock_ohlcv.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])

    cutoff = pd.Timestamp(prediction_date)
    window = (
        frame.loc[frame[date_col] <= cutoff].sort_values(date_col).tail(lookback_days)
    )

    if len(window) < lookback_days:
        raise ValueError(
            f"Not enough rows to build chart for {symbol} at {cutoff.date()}: "
            f"need {lookback_days}, got {len(window)}"
        )

    return generate_candlestick_chart(
        window, output_path=output_path, date_col=date_col
    )


def attach_chart_paths(
    samples_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    symbol_col: str = "stock",
    date_col: str = "date",
    chart_col: str = "chart_path",
) -> pd.DataFrame:
    """Attach deterministic chart paths to dataset rows.

    This function only resolves paths; it does not generate files.
    """
    required = {symbol_col, date_col}
    missing = sorted(required - set(samples_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = samples_df.copy()
    result[date_col] = pd.to_datetime(result[date_col])
    result[chart_col] = [
        str(
            resolve_chart_path(
                symbol=row[symbol_col],
                prediction_date=row[date_col],
                output_dir=output_dir,
            )
        )
        for _, row in result.iterrows()
    ]
    return result
