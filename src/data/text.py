"""Utilities for multi-source company text records used by the text branch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class CompanyTextRecord:
    """Normalized company-text record.

    Attributes:
        stock_id: Identifier used across project datasets (e.g. ticker symbol).
        event_date: Date when the text became available to the market.
        source_type: Upstream source category (news, filing, guidance, investor_presentation, etc.).
        title: Short title or subject line for the record.
        body_text: Main free-form text content.
    """

    stock_id: str
    event_date: pd.Timestamp
    source_type: str
    title: str
    body_text: str


REQUIRED_TEXT_COLUMNS: tuple[str, ...] = (
    "stock_id",
    "event_date",
    "source_type",
    "title",
    "body_text",
)


def normalize_company_text_records(
    records: pd.DataFrame,
    *,
    stock_col: str = "stock_id",
    date_col: str = "event_date",
    source_col: str = "source_type",
    title_col: str = "title",
    body_col: str = "body_text",
    drop_empty_body: bool = True,
) -> pd.DataFrame:
    """Normalize heterogeneous text rows to one schema.

    The resulting frame always includes:
    ``stock_id``, ``event_date``, ``source_type``, ``title``, ``body_text``.
    """
    mapping = {
        stock_col: "stock_id",
        date_col: "event_date",
        source_col: "source_type",
        title_col: "title",
        body_col: "body_text",
    }

    missing = [src for src in mapping if src not in records.columns]
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")

    normalized = records.loc[:, list(mapping.keys())].rename(columns=mapping).copy()
    normalized["stock_id"] = normalized["stock_id"].astype(str).str.strip()
    normalized["source_type"] = normalized["source_type"].astype(str).str.strip().str.lower()
    normalized["title"] = normalized["title"].fillna("").astype(str).str.strip()
    normalized["body_text"] = normalized["body_text"].fillna("").astype(str).str.strip()
    normalized["event_date"] = pd.to_datetime(normalized["event_date"])

    normalized = normalized.dropna(subset=["event_date"])
    normalized = normalized[normalized["stock_id"] != ""]

    if drop_empty_body:
        normalized = normalized[normalized["body_text"] != ""]

    return normalized.sort_values(["stock_id", "event_date"], ascending=[True, False]).reset_index(
        drop=True
    )


def build_company_text_input(
    records: pd.DataFrame,
    *,
    stock_id: str,
    as_of_date: str | pd.Timestamp,
    top_k: int = 5,
    separator: str = "\n\n",
) -> str:
    """Build one model-ready text input for a ``(stock, date)`` sample.

    Rules:
      1. Keep only records for the stock where ``event_date <= as_of_date``.
      2. Sort by recency (descending ``event_date``).
      3. Concatenate the top-k rows into one string.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if any(col not in records.columns for col in REQUIRED_TEXT_COLUMNS):
        raise ValueError(f"records must include columns: {list(REQUIRED_TEXT_COLUMNS)}")

    cutoff = pd.to_datetime(as_of_date)
    scope = records.loc[
        (records["stock_id"].astype(str) == str(stock_id))
        & (pd.to_datetime(records["event_date"]) <= cutoff)
    ].copy()

    if scope.empty:
        return ""

    scope = scope.sort_values("event_date", ascending=False).head(top_k)

    chunks: list[str] = []
    for row in scope.itertuples(index=False):
        title = str(row.title).strip()
        if title:
            chunk = f"[{row.source_type}] {title}\n{row.body_text}"
        else:
            chunk = f"[{row.source_type}]\n{row.body_text}"
        chunks.append(chunk.strip())

    return separator.join(chunks)


def build_company_text_inputs_for_samples(
    samples: pd.DataFrame,
    records: pd.DataFrame,
    *,
    sample_stock_col: str = "stock_id",
    sample_date_col: str = "date",
    top_k: int = 5,
    output_col: str = "company_text_input",
) -> pd.DataFrame:
    """Attach normalized multi-source text inputs to sample rows."""
    if sample_stock_col not in samples.columns or sample_date_col not in samples.columns:
        raise ValueError(
            f"samples must include columns '{sample_stock_col}' and '{sample_date_col}'"
        )

    out = samples.copy()
    out[sample_date_col] = pd.to_datetime(out[sample_date_col])
    normalized = normalize_company_text_records(records)

    out[output_col] = [
        build_company_text_input(
            normalized,
            stock_id=row[sample_stock_col],
            as_of_date=row[sample_date_col],
            top_k=top_k,
        )
        for _, row in out.iterrows()
    ]
    return out


def extract_pdf_text_record(
    pdf_path: str | Path,
    *,
    stock_id: str,
    event_date: str | pd.Timestamp,
    source_type: str = "pdf_document",
    title: str | None = None,
    max_pages: int | None = 20,
) -> CompanyTextRecord:
    """Extract text from a text-based PDF into a normalized company-text record.

    This helper is intentionally lightweight: it uses direct text extraction only
    and does not attempt OCR or production-grade parsing.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import importlib

    if importlib.util.find_spec("pypdf") is None:
        raise ImportError("pypdf is required for PDF text extraction. Install with: pip install pypdf")

    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    page_text: list[str] = []
    pages = reader.pages if max_pages is None else reader.pages[: max_pages]
    for page in pages:
        extracted = (page.extract_text() or "").strip()
        if extracted:
            page_text.append(extracted)

    body_text = "\n\n".join(page_text).strip()
    return CompanyTextRecord(
        stock_id=str(stock_id),
        event_date=pd.to_datetime(event_date),
        source_type=source_type,
        title=title or pdf_path.stem,
        body_text=body_text,
    )


def records_to_dataframe(records: Iterable[CompanyTextRecord]) -> pd.DataFrame:
    """Convert a collection of ``CompanyTextRecord`` objects into normalized dataframe."""
    data = [
        {
            "stock_id": r.stock_id,
            "event_date": r.event_date,
            "source_type": r.source_type,
            "title": r.title,
            "body_text": r.body_text,
        }
        for r in records
    ]
    if not data:
        return pd.DataFrame(columns=list(REQUIRED_TEXT_COLUMNS))
    return normalize_company_text_records(pd.DataFrame(data))
