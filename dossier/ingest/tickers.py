"""Map FinanceBench company names to SEC CIKs.

FinanceBench company labels are colloquial ("3M", "Coca Cola", "Best Buy"); the SEC's
`company_tickers.json` uses registered legal titles ("3M CO", "COCA COLA CO"). Automatic
normalization gets most of them; the rest are handled by an explicit override table so a
mismatch is visible in the diff rather than silently dropping a company's financials.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from ..config import get_config

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Names that normalization cannot bridge. Values are tickers, resolved against the SEC file.
MANUAL_TICKER_OVERRIDES: dict[str, str] = {
    "3M": "MMM",
    "AES": "AES",
    "AMCOR": "AMCR",
    "AMD": "AMD",
    "AMERICANEXPRESS": "AXP",
    "AMEREN": "AEE",
    "APPLE": "AAPL",
    "BESTBUY": "BBY",
    "BLOCK": "XYZ",
    "BOEING": "BA",
    "COCACOLA": "KO",
    "CORNING": "GLW",
    "COSTCO": "COST",
    "CVSHEALTH": "CVS",
    "FOOTLOCKER": "FL",
    "GENERALMILLS": "GIS",
    "JOHNSONJOHNSON": "JNJ",
    "JOHNSONANDJOHNSON": "JNJ",
    "JPMORGAN": "JPM",
    "KRAFTHEINZ": "KHC",
    "LOCKHEEDMARTIN": "LMT",
    "MGMRESORTSINTERNATIONAL": "MGM",
    "MGMRESORTS": "MGM",
    "MICROSOFT": "MSFT",
    "NETFLIX": "NFLX",
    "NIKE": "NKE",
    "PAYPAL": "PYPL",
    "PEPSICO": "PEP",
    "PFIZER": "PFE",
    "PGE": "PCG",
    "PACIFICGASANDELECTRIC": "PCG",
    "ULTABEAUTY": "ULTA",
    "VERIZON": "VZ",
    "WALMART": "WMT",
    "ACTIVISIONBLIZZARD": "ATVI",
    "ADOBE": "ADBE",
    "AMAZON": "AMZN",
    "INTEL": "INTC",
    "ORACLE": "ORCL",
    "SALESFORCE": "CRM",
    "EBAY": "EBAY",
}

_NOISE = re.compile(r"\b(inc|corp|corporation|company|co|plc|ltd|holdings|group|the|and)\b", re.I)
_NONALNUM = re.compile(r"[^a-z0-9]")


def normalize_name(name: str) -> str:
    return _NONALNUM.sub("", _NOISE.sub("", (name or "").lower()))


def fetch_sec_tickers(cache: Path | None = None) -> list[dict]:
    cfg = get_config()
    cache = cache or (cfg.paths.raw_xbrl / "company_tickers.json")
    if cache.exists() and cache.stat().st_size > 0:
        return list(json.loads(cache.read_text()).values())
    resp = httpx.get(
        SEC_TICKERS_URL, headers={"User-Agent": cfg.sec_user_agent}, timeout=60.0, follow_redirects=True
    )
    resp.raise_for_status()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(resp.text)
    return list(resp.json().values())


def build_cik_map(companies: list[str], sec_rows: list[dict] | None = None) -> tuple[dict[str, str], list[str]]:
    """Return `({company: 'CIK##########'}, unmatched_companies)`."""
    rows = sec_rows if sec_rows is not None else fetch_sec_tickers()
    by_ticker = {str(r["ticker"]).upper(): r for r in rows}
    by_norm: dict[str, dict] = {}
    for r in rows:
        by_norm.setdefault(normalize_name(str(r["title"])), r)

    out: dict[str, str] = {}
    missing: list[str] = []
    for company in companies:
        norm = normalize_name(company)
        row = None
        ticker = MANUAL_TICKER_OVERRIDES.get(norm)
        if ticker:
            row = by_ticker.get(ticker)
        if row is None:
            row = by_norm.get(norm)
        if row is None:
            # last resort: unique prefix match on the normalized SEC title
            candidates = [r for k, r in by_norm.items() if norm and k.startswith(norm)]
            if len(candidates) == 1:
                row = candidates[0]
        if row is None:
            missing.append(company)
            continue
        out[company] = f"CIK{int(row['cik_str']):010d}"
    return out, missing
