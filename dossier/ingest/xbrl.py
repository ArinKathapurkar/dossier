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
    company     TEXT NOT NULL,
    cik         TEXT NOT NULL,
    concept     TEXT NOT NULL,
    unit        TEXT NOT NULL,
    fy          INTEGER,
    fp          TEXT,
    period_end  TEXT,
    value       REAL,
    form        TEXT,
    filed       TEXT,
    PRIMARY KEY (company, concept, unit, fy, fp, period_end, form)
);
CREATE INDEX IF NOT EXISTS idx_facts_company_concept ON facts(company, concept);
CREATE INDEX IF NOT EXISTS idx_facts_fy ON facts(fy);
"""


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
                    e.get("fy"),
                    e.get("fp"),
                    e.get("end"),
                    e.get("form"),
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
                        e.get("end"),
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
            "(company, cik, concept, unit, fy, fp, period_end, value, form, filed) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
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
) -> list[dict]:
    """Annual-first fact lookup used by the `get_financials` / `compare_peers` tools."""
    conn = connect(db)
    q = (
        "SELECT company, cik, concept, unit, fy, fp, period_end, value, form, filed "
        "FROM facts WHERE company IN ({c}) AND concept IN ({k})"
    ).format(c=",".join("?" * len(companies)), k=",".join("?" * len(concepts)))
    params: list = [*companies, *concepts]
    if fiscal_years:
        q += " AND fy IN ({y})".format(y=",".join("?" * len(fiscal_years)))
        params += list(fiscal_years)
    if forms:
        q += " AND form IN ({f})".format(f=",".join("?" * len(forms)))
        params += list(forms)
    # FY annual figures first (fp='FY'), then most recent filing wins on ties.
    q += " ORDER BY (fp = 'FY') DESC, fy DESC, filed DESC LIMIT ?"
    params.append(limit)
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


def available_companies(db: Path | None = None) -> list[str]:
    conn = connect(db)
    rows = [r[0] for r in conn.execute("SELECT DISTINCT company FROM facts ORDER BY company")]
    conn.close()
    return rows
