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
    "MAXHEALTH.NS": "pharma",
    "TATASTEEL.NS": "metals",
    "JSWSTEEL.NS": "metals",
    "HINDALCO.NS": "metals",
    "ADANIENT.NS": "metals",
    "ADANIPORTS.NS": "infra_other",
    "ASIANPAINT.NS": "infra_other",
    "BAJFINANCE.NS": "banking",
    "BAJAJFINSV.NS": "banking",
    "BEL.NS": "infra_other",
    "BHARTIARTL.NS": "infra_other",
    "ETERNAL.NS": "infra_other",
    "GRASIM.NS": "infra_other",
    "HDFCLIFE.NS": "banking",
    "INDIGO.NS": "infra_other",
    "JIOFIN.NS": "banking",
    "LT.NS": "infra_other",
    "SBILIFE.NS": "banking",
    "SHRIRAMFIN.NS": "banking",
    "TITAN.NS": "infra_other",
    "TRENT.NS": "infra_other",
    "ULTRACEMCO.NS": "infra_other",
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


def strict_sector_for_ticker(ticker: str, mapping: dict[str, str] | None = None) -> str:
    """Resolve a ticker to a canonical sector, raising when unmapped."""
    sectors = mapping or NIFTY50_SECTOR_MAPPING
    normalized = normalize_ticker(ticker)
    if normalized not in sectors:
        raise KeyError(f"Missing sector mapping for ticker: {normalized}")
    sector = sectors[normalized]
    if sector not in SECTOR_NAMES:
        raise KeyError(f"Ticker {normalized} maps to unknown sector: {sector}")
    return sector
