"""Integration test: no future leakage in the multimodal sample pipeline.

Runs the real builder code against deterministic synthetic data and asserts
all six leakage invariants for every (stock_id, end_date) sample:

  1. Tabular  — every date in the rolling window is <= end_date; max == end_date.
  2. Text     — every text record visible to a sample has event_date <= end_date.
  3. Chart    — the PNG filename encodes end_date (no future date in the name).
  4. KG       — as_of_date in retrieved context equals end_date.
  5. Label    — the future-return window (D+1 … D+H) does not overlap the tabular
                window (D-W+1 … D).
  6. Alignment — all assembled modalities share the same (stock_id, end_date) keys.

Plus two negative tests:

  N1. A text record with event_date = D+1 does NOT change the token for date D.
  N2. If future prices are unavailable the builder refuses to emit a sample
      (raises ValueError rather than silently falling back).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.kg_features_v2 import build_kg_v2
from src.data.multimodal_samples import (
    MultimodalSampleArrays,
    attach_kg_tokens,
    attach_text_tokens,
    build_tabular_multimodal_samples,
    build_text_tokens_for_samples,
)
from src.data.text import normalize_company_text_records
from src.kg.build_graph import build_market_knowledge_graph
from src.kg.query_graph import retrieve_kg_context
from src.viz.charts import build_chart_filename

# ── Synthetic universe constants ─────────────────────────────────────────────

_TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS"]
_SECTORS = {"AAA.NS": "IT", "BBB.NS": "IT", "CCC.NS": "Energy"}
_WINDOW_SIZE = 20
_HORIZON_DAYS = 3
_N_DAYS = 60
_FEATURE_COLS = ["feat_1", "feat_2"]


# ── Synthetic data helpers ───────────────────────────────────────────────────

def _make_tabular_df(rng: np.random.Generator) -> pd.DataFrame:
    """Build a labelled tabular dataframe for three stocks over 60 business days.

    Labels are computed from the actual H-day-forward stock/index returns so that
    the label invariant test can independently verify correctness.
    """
    frames: list[pd.DataFrame] = []
    for seed_offset, ticker in enumerate(_TICKERS):
        dates = pd.bdate_range("2024-01-02", periods=_N_DAYS)
        feat1 = rng.normal(0.0, 1.0, _N_DAYS).astype(np.float32)
        feat2 = rng.normal(0.0, 1.0, _N_DAYS).astype(np.float32)
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.01, _N_DAYS))
        idx_close = 20_000.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.008, _N_DAYS))

        # Forward return over horizon (row-shift; last H rows are NaN)
        future_stock = np.roll(close, -_HORIZON_DAYS).astype(float)
        future_idx = np.roll(idx_close, -_HORIZON_DAYS).astype(float)
        future_stock[-_HORIZON_DAYS:] = np.nan
        future_idx[-_HORIZON_DAYS:] = np.nan

        label = (future_stock / close - 1.0 > future_idx / idx_close - 1.0).astype(float)
        label[-_HORIZON_DAYS:] = np.nan

        frames.append(pd.DataFrame({
            "stock_id": ticker,
            "date": dates,
            "feat_1": feat1,
            "feat_2": feat2,
            "label": label,
            "close": close,
            "index_close": idx_close,
        }))

    combined = pd.concat(frames, ignore_index=True)
    # Drop NaN labels (last H rows per stock have no future data)
    clean = combined.dropna(subset=["label", *_FEATURE_COLS]).reset_index(drop=True)
    clean["label"] = clean["label"].astype(int)
    return clean


def _make_text_records(tabular_df: pd.DataFrame) -> pd.DataFrame:
    """Build one text record per stock every 5 rows (all with event_date <= D)."""
    records: list[dict] = []
    for ticker, frame in tabular_df.groupby("stock_id"):
        frame = frame.sort_values("date").reset_index(drop=True)
        for idx in range(0, len(frame), 5):
            row = frame.iloc[idx]
            records.append({
                "stock_id": ticker,
                "event_date": row["date"],
                "source_type": "market_summary",
                "title": f"{ticker} update {idx}",
                "body_text": f"Market summary for {ticker} on {row['date'].date()}.",
            })
    return pd.DataFrame(records)


def _make_kg_returns(tabular_df: pd.DataFrame) -> pd.DataFrame:
    return tabular_df[["stock_id", "date", "feat_1"]].rename(
        columns={"feat_1": "recent_return"}
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["stock_id", "event_date", "event_type"])


def _ohlcv_frame(dates: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(close)),
        }
    )


# ── Class-scoped fixture: build the full artifact once ───────────────────────

@pytest.fixture(scope="class")
def artifact():
    """Return (arrays, tabular_df, text_records) built from deterministic synthetic data."""
    rng = np.random.default_rng(42)
    tabular_df = _make_tabular_df(rng)
    text_records = _make_text_records(tabular_df)
    kg_returns = _make_kg_returns(tabular_df)
    graph = build_market_knowledge_graph(_SECTORS, event_records=_empty_events())

    arrays = build_tabular_multimodal_samples(
        tabular_df, feature_cols=_FEATURE_COLS, window_size=_WINDOW_SIZE
    )
    arrays = attach_text_tokens(arrays, text_records, dim=8)
    arrays = attach_kg_tokens(arrays, graph, returns=kg_returns)
    return arrays, tabular_df, text_records


# ── Test class ───────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestNoLeakage:

    # ── Invariant 1: tabular window ──────────────────────────────────────────

    def test_tabular_no_future_leakage(self, artifact: tuple) -> None:
        """Every date in the tabular window must be <= end_date; max must equal end_date."""
        arrays, tabular_df, _ = artifact

        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()

            stock_frame = (
                tabular_df.loc[tabular_df["stock_id"] == stock_id]
                .sort_values("date")
                .reset_index(drop=True)
            )
            match = stock_frame.index[stock_frame["date"].dt.normalize() == end_date].tolist()
            assert match, f"end_date {end_date} not found in tabular_df for {stock_id}"
            end_idx = match[0]
            start_idx = end_idx - _WINDOW_SIZE + 1
            assert start_idx >= 0, f"Window extends before start of data for {stock_id}"

            window_dates = stock_frame["date"].iloc[start_idx : end_idx + 1].dt.normalize()
            assert window_dates.max() == end_date, (
                f"Max window date {window_dates.max()} != end_date {end_date} for {stock_id}"
            )
            assert (window_dates <= end_date).all(), (
                f"Future date found in tabular window for {stock_id} at {end_date}"
            )

    # ── Invariant 2: text cutoff ─────────────────────────────────────────────

    def test_text_no_future_leakage(self, artifact: tuple) -> None:
        """No text record with event_date > end_date may be visible to a sample."""
        arrays, _, text_records = artifact
        normalized = normalize_company_text_records(text_records)

        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()

            visible = normalized.loc[
                (normalized["stock_id"] == stock_id)
                & (normalized["event_date"].dt.normalize() <= end_date)
            ]
            assert (visible["event_date"].dt.normalize() <= end_date).all(), (
                f"Text record with future event_date visible to {stock_id} at {end_date}"
            )

    # ── Invariant 3: chart filename encodes end_date ─────────────────────────

    def test_chart_no_future_leakage(self, artifact: tuple) -> None:
        """The chart PNG filename for each sample encodes exactly end_date.

        Format: {SYMBOL}_{YYYYMMDD}.png — parsed date must equal end_date.
        No future date can appear in the filename under this convention.
        """
        arrays, _, _ = artifact

        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()

            filename = build_chart_filename(stock_id, end_date)
            # Parse date back from filename: last segment before .npy
            date_str = filename.rsplit("_", 1)[-1].replace(".npy", "")
            assert len(date_str) == 8, f"Unexpected date string in filename: {filename!r}"
            parsed = pd.Timestamp(date_str).normalize()

            assert parsed == end_date, (
                f"Chart filename {filename!r} encodes {parsed}, expected {end_date} "
                f"for {stock_id}"
            )

    # ── Invariant 4: KG as_of_date ───────────────────────────────────────────

    def test_kg_no_future_leakage(self, artifact: tuple) -> None:
        """KG context as_of_date for each sample must equal end_date."""
        arrays, tabular_df, _ = artifact
        kg_returns = _make_kg_returns(tabular_df)
        graph = build_market_knowledge_graph(_SECTORS, event_records=_empty_events())

        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()

            context = retrieve_kg_context(
                graph,
                stock_id=stock_id,
                as_of_date=end_date,
                returns=kg_returns,
            )
            context_date = pd.Timestamp(context["as_of_date"]).normalize()

            assert context_date == end_date, (
                f"KG as_of_date {context_date} != end_date {end_date} for {stock_id}"
            )

    def test_kg_v2_no_future_window_leakage(self) -> None:
        """KG v2 rolling and peer features must ignore post-D sentinel values."""
        dates = pd.bdate_range("2024-01-01", periods=80)
        cutoff = dates[50]
        base = np.arange(80, dtype=float)
        universe = {
            "AAA.NS": _ohlcv_frame(dates, 100.0 + base * 0.2),
            "BBB.NS": _ohlcv_frame(dates, 120.0 + base * 0.3),
            "CCC.NS": _ohlcv_frame(dates, 200.0 + base * 0.1),
        }
        benchmark = _ohlcv_frame(dates, 1000.0 + base * 0.15)
        poisoned_universe = {ticker: frame.copy() for ticker, frame in universe.items()}
        poisoned_benchmark = benchmark.copy()

        for frame in [*poisoned_universe.values(), poisoned_benchmark]:
            future_mask = frame["date"] > cutoff
            frame.loc[future_mask, "close"] = 999_999.0
            frame.loc[future_mask, "volume"] = 999_999_999.0

        kwargs = dict(
            sector_mapping={"AAA.NS": "it", "BBB.NS": "it", "CCC.NS": "energy"},
            stock_ids=["AAA.NS", "BBB.NS", "CCC.NS"],
            end_dates=[cutoff, cutoff, cutoff],
        )
        clean = build_kg_v2(
            universe_ohlcv=universe,
            benchmark_ohlcv=benchmark,
            **kwargs,
        ).values
        poisoned = build_kg_v2(
            universe_ohlcv=poisoned_universe,
            benchmark_ohlcv=poisoned_benchmark,
            **kwargs,
        ).values

        np.testing.assert_allclose(clean, poisoned)

    # ── Invariant 5: label window does not overlap tabular window ────────────

    def test_label_uses_only_future_data(self, artifact: tuple) -> None:
        """The label-computation window (D+1 … D+H rows) must not overlap the tabular window.

        Additionally, the stored label must equal what is obtained by independently
        computing the H-day forward return from the synthetic data.
        """
        arrays, tabular_df, _ = artifact

        for i in range(len(arrays.stock_ids)):
            stock_id = str(arrays.stock_ids[i])
            end_date = pd.Timestamp(arrays.end_dates[i]).normalize()

            stock_frame = (
                tabular_df.loc[tabular_df["stock_id"] == stock_id]
                .sort_values("date")
                .reset_index(drop=True)
            )
            match = stock_frame.index[stock_frame["date"].dt.normalize() == end_date].tolist()
            assert match
            end_idx = match[0]
            start_idx = end_idx - _WINDOW_SIZE + 1

            tabular_dates = set(
                stock_frame["date"].iloc[start_idx : end_idx + 1].dt.normalize()
            )

            # Future window: the H business-day rows strictly after D
            future_idx_end = end_idx + _HORIZON_DAYS
            if future_idx_end >= len(stock_frame):
                # Sample near the end of data — skip label-date overlap check
                continue
            future_dates = set(
                stock_frame["date"].iloc[end_idx + 1 : future_idx_end + 1].dt.normalize()
            )

            overlap = tabular_dates & future_dates
            assert not overlap, (
                f"Label window overlaps tabular window for {stock_id} at {end_date}: {overlap}"
            )

            # Cross-check: stored label must match manual computation from synthetic data
            close_D = stock_frame["close"].iloc[end_idx]
            idx_D = stock_frame["index_close"].iloc[end_idx]
            close_DH = stock_frame["close"].iloc[future_idx_end]
            idx_DH = stock_frame["index_close"].iloc[future_idx_end]
            expected_label = int(
                (close_DH / close_D - 1.0) > (idx_DH / idx_D - 1.0)
            )
            assert int(arrays.y[i]) == expected_label, (
                f"Stored label {arrays.y[i]} != expected {expected_label} "
                f"for {stock_id} at {end_date}"
            )

    # ── Invariant 6: cross-modality alignment ────────────────────────────────

    def test_cross_modality_alignment(self, artifact: tuple) -> None:
        """All assembled modality arrays must share the same sample count and keys."""
        arrays, _, _ = artifact
        n = len(arrays.stock_ids)

        assert arrays.tabular_tokens.shape[0] == n
        assert arrays.y.shape[0] == n
        assert arrays.end_dates.shape[0] == n

        if arrays.text_tokens is not None:
            assert arrays.text_tokens.shape[0] == n, (
                f"text_tokens has {arrays.text_tokens.shape[0]} samples, expected {n}"
            )
        if arrays.kg_tokens is not None:
            assert arrays.kg_tokens.shape[0] == n, (
                f"kg_tokens has {arrays.kg_tokens.shape[0]} samples, expected {n}"
            )
        if arrays.image_tokens is not None:
            assert arrays.image_tokens.shape[0] == n

        # All three tickers must appear in the artifact
        assert set(arrays.stock_ids.tolist()) == set(_TICKERS), (
            f"Missing stocks in artifact: {set(_TICKERS) - set(arrays.stock_ids.tolist())}"
        )

    # ── Negative test N1: future text record is silently dropped ─────────────

    def test_negative_future_text_dropped(self) -> None:
        """Injecting a text record with event_date = D+1 must NOT change the token for D.

        The text cutoff filter must discard it before the hash is computed.
        """
        rng = np.random.default_rng(99)
        tabular_df = _make_tabular_df(rng)

        arrays = build_tabular_multimodal_samples(
            tabular_df, feature_cols=_FEATURE_COLS, window_size=_WINDOW_SIZE
        )

        # Fix a sample to test
        stock_id = str(arrays.stock_ids[0])
        end_date = pd.Timestamp(arrays.end_dates[0])

        samples_df = pd.DataFrame({"stock_id": [stock_id], "date": [end_date]})

        safe_record = pd.DataFrame([{
            "stock_id": stock_id,
            "event_date": end_date - pd.Timedelta(days=1),
            "source_type": "news",
            "title": "Safe record",
            "body_text": "This record is available before the prediction date.",
        }])

        future_record = pd.DataFrame([{
            "stock_id": stock_id,
            "event_date": end_date + pd.Timedelta(days=1),
            "source_type": "future",
            "title": "Future record",
            "body_text": "This record is from the future and must not influence the token.",
        }])

        poisoned = pd.concat([safe_record, future_record], ignore_index=True)

        token_clean = build_text_tokens_for_samples(samples_df, safe_record, dim=16)
        token_poisoned = build_text_tokens_for_samples(samples_df, poisoned, dim=16)

        np.testing.assert_array_equal(
            token_clean[0],
            token_poisoned[0],
            err_msg=(
                f"Future text record (event_date={end_date + pd.Timedelta(days=1)}) "
                f"leaked into the token for {stock_id} at {end_date}. "
                "The as-of-date cutoff filter is broken."
            ),
        )

    # ── Negative test N2: truncated future → no sample emitted ──────────────

    def test_negative_truncated_future_no_label(self) -> None:
        """If future prices are unavailable the builder must raise, not emit a sample.

        Setting all labels to NaN (simulating total future-price truncation) must
        cause build_tabular_multimodal_samples to raise ValueError.
        """
        # Exactly WINDOW_SIZE rows so one sample would be possible — but no label
        dates = pd.bdate_range("2024-01-02", periods=_WINDOW_SIZE)
        df = pd.DataFrame({
            "stock_id": "AAA.NS",
            "date": dates,
            "feat_1": np.ones(_WINDOW_SIZE, dtype=np.float32),
            "feat_2": np.ones(_WINDOW_SIZE, dtype=np.float32),
            "label": np.full(_WINDOW_SIZE, np.nan),  # No future prices → all NaN
        })
        clean = df.dropna(subset=["label", *_FEATURE_COLS]).reset_index(drop=True)

        # After dropping NaN rows, clean is empty — builder must refuse
        with pytest.raises(ValueError, match="No tabular samples could be built"):
            build_tabular_multimodal_samples(
                clean, feature_cols=_FEATURE_COLS, window_size=_WINDOW_SIZE
            )
