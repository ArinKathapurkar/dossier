"""Structured financial facts from SEC XBRL `companyfacts`.

This is the second data store: the PDFs give the narrative, XBRL gives exact numbers with
a filing provenance. The agent uses it through `get_financials` / `compare_peers`, and the
output guard can check a stated number against a fact value rather than against prose.

Rate limit: SEC allows 10 requests/second with a descriptive User-Agent. One company per
request plus a small sleep keeps us far below that.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import httpx

from ..config import get_config

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/{cik}.json"

# A fixed concept set: enough to answer margin/leverage/liquidity/capex questions without
# pulling the whole taxonomy (companyfacts payloads are tens of MB each).
CONCEPTS: tuple[str, ...] = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "CashAndCashEquivalentsAtCarryingValue",
    "LongTermDebt",
    "NetCashProvidedByUsedInOperatingActivities",
    "PaymentsToAcquirePropertyPlantAndEquipment",
    "InventoryNet",
    "AccountsReceivableNetCurrent",
    "CostOfRevenue",
    "ResearchAndDevelopmentExpense",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    company       TEXT NOT NULL,
    cik           TEXT NOT NULL,
    concept       TEXT NOT NULL,
    unit          TEXT NOT NULL,
    fy            INTEGER,
    fp            TEXT,
    period_start  TEXT,
    period_end    TEXT,
    period_days   INTEGER,
    fiscal_year   INTEGER,
    value         REAL,
    form          TEXT,
    filed         TEXT,
    PRIMARY KEY (company, concept, unit, period_start, period_end, form, fy, fp)
);
CREATE INDEX IF NOT EXISTS idx_facts_company_concept ON facts(company, concept);
CREATE INDEX IF NOT EXISTS idx_facts_fiscal_year ON facts(fiscal_year);
"""

# A companyfacts entry's `fy`/`fp` describe the *filing* it appeared in, not the period the
# number covers: a FY2018 10-K reports 2016, 2017 and 2018 columns and tags all three
# fy=2018, fp=FY. Selecting on `fy` therefore returns whichever of the three rows sorts
# first, which is how a query for FY2018 capex silently returns the 2016 figure. The
# authoritative period is `end` (and `start` for duration concepts), so we derive a real
# fiscal year from `end` and query on that instead.
ANNUAL_MIN_DAYS = 300
ANNUAL_MAX_DAYS = 400


def _fiscal_year_from_end(end: str | None) -> int | None:
    """Fiscal year implied by a period end date.

    A period ending in January-May belongs to the prior fiscal year in most retail and
    tech calendars (Walmart's FY2019 ends 2019-01-31, so it stays 2019; Apple-style
    September year-ends stay in their own year). Using the end date's own year is right
    for the overwhelming majority and is at least *consistent*, which `fy` is not.
    """
    if not end:
        return None
    try:
        return int(str(end)[:4])
    except ValueError:
        return None


def _period_days(start: str | None, end: str | None) -> int | None:
    if not start or not end:
        return None
    from datetime import date

    try:
        s = date.fromisoformat(str(start)[:10])
        e = date.fromisoformat(str(end)[:10])
    except ValueError:
        return None
    return (e - s).days


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or get_config().paths.facts_sqlite
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def fetch_companyfacts(cik: str, cache_dir: Path | None = None) -> dict | None:
    cfg = get_config()
    cache_dir = cache_dir or cfg.paths.raw_xbrl
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{cik}.json"
    if cached.exists() and cached.stat().st_size > 0:
        return json.loads(cached.read_text())
    try:
        resp = httpx.get(
            COMPANYFACTS_URL.format(cik=cik),
            headers={"User-Agent": cfg.sec_user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=120.0,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    cached.write_text(resp.text)
    return resp.json()


def extract_rows(company: str, cik: str, payload: dict) -> list[tuple]:
    """Flatten the us-gaap section of a companyfacts payload into `facts` rows."""
    gaap = (payload.get("facts") or {}).get("us-gaap") or {}
    rows: list[tuple] = []
    seen: set[tuple] = set()
    for concept in CONCEPTS:
        node = gaap.get(concept)
        if not node:
            continue
        for unit, entries in (node.get("units") or {}).items():
            for e in entries:
                # `frames`-style duplicates are common; the primary key dedups but doing it
                # here keeps the insert count honest.
                key = (
                    company,
                    concept,
                    unit,
                    e.get("start"),
                    e.get("end"),
                    e.get("form"),
                    e.get("fy"),
                    e.get("fp"),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    (
                        company,
                        cik,
                        concept,
                        unit,
                        e.get("fy"),
                        e.get("fp"),
                        e.get("start"),
                        e.get("end"),
                        _period_days(e.get("start"), e.get("end")),
                        _fiscal_year_from_end(e.get("end")),
                        e.get("val"),
                        e.get("form"),
                        e.get("filed"),
                    )
                )
    return rows


def load_facts(cik_map: dict[str, str], db: Path | None = None, sleep_s: float = 0.12) -> dict:
    conn = connect(db)
    inserted = 0
    failures: list[str] = []
    for company, cik in sorted(cik_map.items()):
        payload = fetch_companyfacts(cik)
        if payload is None:
            failures.append(company)
            time.sleep(sleep_s)
            continue
        rows = extract_rows(company, cik, payload)
        conn.executemany(
            "INSERT OR REPLACE INTO facts "
            "(company, cik, concept, unit, fy, fp, period_start, period_end, period_days, "
            " fiscal_year, value, form, filed) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        inserted += len(rows)
        conn.commit()
        time.sleep(sleep_s)
    total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    companies = conn.execute("SELECT COUNT(DISTINCT company) FROM facts").fetchone()[0]
    conn.close()
    return {
        "rows_inserted": inserted,
        "rows_total": total,
        "companies_with_facts": companies,
        "fetch_failures": failures,
    }


def query_facts(
    companies: list[str],
    concepts: list[str],
    fiscal_years: list[int] | None = None,
    db: Path | None = None,
    forms: tuple[str, ...] = ("10-K", "10-Q", "20-F"),
    limit: int = 200,
    annual_only: bool = True,
) -> list[dict]:
    """Annual-first fact lookup used by the `get_financials` / `compare_peers` tools.

    Selection is on `fiscal_year` (derived from the period end date), never on the
    filing's `fy` tag -- see the note above ANNUAL_MIN_DAYS. Duration concepts are
    restricted to roughly-annual periods so a quarterly figure never answers an annual
    question; instant concepts (balance sheet items) carry no `start` and are kept.
    """
    conn = connect(db)
    q = (
        "SELECT company, cik, concept, unit, fy, fp, period_start, period_end, period_days, "
        "       fiscal_year, value, form, filed "
        "FROM facts WHERE company IN ({c}) AND concept IN ({k})"
    ).format(c=",".join("?" * len(companies)), k=",".join("?" * len(concepts)))
    params: list = [*companies, *concepts]
    if fiscal_years:
        q += " AND fiscal_year IN ({y})".format(y=",".join("?" * len(fiscal_years)))
        params += list(fiscal_years)
    if forms:
        q += " AND form IN ({f})".format(f=",".join("?" * len(forms)))
        params += list(forms)
    if annual_only:
        q += f" AND (period_days IS NULL OR period_days BETWEEN {ANNUAL_MIN_DAYS} AND {ANNUAL_MAX_DAYS})"
    # Most recent fiscal year first; within a year, the latest filing (restatements win).
    q += " ORDER BY fiscal_year DESC, filed DESC, period_end DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def available_companies(db: Path | None = None) -> list[str]:
    conn = connect(db)
    rows = [r[0] for r in conn.execute("SELECT DISTINCT company FROM facts ORDER BY company")]
    conn.close()
    return rows
