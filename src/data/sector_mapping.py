"""Static sector mapping for the Nifty50 experiment universe."""

from __future__ import annotations

SECTOR_NAMES: tuple[str, ...] = (
    "banking",
    "it",
    "energy",
    "fmcg",
    "auto",
    "pharma",
    "metals",
    "infra_other",
)

NIFTY50_SECTOR_MAPPING: dict[str, str] = {
    "HDFCBANK.NS": "banking",
    "ICICIBANK.NS": "banking",
    "SBIN.NS": "banking",
    "AXISBANK.NS": "banking",
    "KOTAKBANK.NS": "banking",
    "INDUSINDBK.NS": "banking",
    "TCS.NS": "it",
    "INFY.NS": "it",
    "WIPRO.NS": "it",
    "HCLTECH.NS": "it",
    "TECHM.NS": "it",
    "RELIANCE.NS": "energy",
    "ONGC.NS": "energy",
    "IOC.NS": "energy",
    "BPCL.NS": "energy",
    "NTPC.NS": "energy",
    "POWERGRID.NS": "energy",
    "COALINDIA.NS": "energy",
    "HINDUNILVR.NS": "fmcg",
    "ITC.NS": "fmcg",
    "NESTLEIND.NS": "fmcg",
    "BRITANNIA.NS": "fmcg",
    "TATACONSUM.NS": "fmcg",
    "MARUTI.NS": "auto",
    "M&M.NS": "auto",
    "TATAMOTORS.NS": "auto",
    "BAJAJ-AUTO.NS": "auto",
    "HEROMOTOCO.NS": "auto",
    "EICHERMOT.NS": "auto",
    "SUNPHARMA.NS": "pharma",
    "DRREDDY.NS": "pharma",
    "CIPLA.NS": "pharma",
    "DIVISLAB.NS": "pharma",
    "APOLLOHOSP.NS": "pharma",
    "TATASTEEL.NS": "metals",
    "JSWSTEEL.NS": "metals",
    "HINDALCO.NS": "metals",
    "ADANIENT.NS": "metals",
}


def normalize_ticker(ticker: str) -> str:
    """Return a normalized NSE ticker with .NS suffix when omitted."""
    value = ticker.strip().upper()
    if value and "." not in value:
        value = f"{value}.NS"
    return value


def sector_for_ticker(ticker: str, mapping: dict[str, str] | None = None) -> str:
    """Resolve a ticker to one of the canonical sector names."""
    sectors = mapping or NIFTY50_SECTOR_MAPPING
    return sectors.get(normalize_ticker(ticker), "infra_other")
