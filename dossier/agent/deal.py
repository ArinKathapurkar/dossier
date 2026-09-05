"""Deals: the unit of cross-run memory.

A diligence engagement is many questions over days, not one query. A `Deal` holds the
target, the peer set, the thesis, and a list of `findings` -- short cited statements
accepted from earlier runs. Findings are injected into later runs' system prompts, so the
second question about a company starts from what the first one established rather than
re-deriving it.

Findings are deliberately *cited* statements only. Letting a run write an uncited
conclusion into deal memory would let one hallucination contaminate every later run, which
is the specific failure this design avoids: memory carries evidence ids, and the output
guard checks them just like any other claim.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from . import state as _state


def connect():
    """Call through the module, not a bound reference.

    `from .state import connect` would bind at import time, which silently defeats a
    test that redirects the run database -- and defeating it produced a real bug: cassette
    replay read a deal another scenario had mutated in the developer's live database.
    """
    return _state.connect()


@dataclass
class Deal:
    id: str
    target: str
    peers: list[str] = field(default_factory=list)
    thesis: str = ""
    findings: list[dict] = field(default_factory=list)

    def memory_block(self, limit: int = 12) -> str:
        if not self.findings:
            return "(no findings carried over from earlier runs)"
        rows = self.findings[-limit:]
        return "\n".join(f"- {f['text']}  [{', '.join(f.get('evidence', []))}]" for f in rows)

    def as_dict(self) -> dict:
        return {"id": self.id, "target": self.target, "peers": self.peers, "thesis": self.thesis, "findings": self.findings}


def _row_to_deal(row: sqlite3.Row) -> Deal:
    return Deal(
        id=row["deal_id"],
        target=row["target"],
        peers=json.loads(row["peers"] or "[]"),
        thesis=row["thesis"] or "",
        findings=json.loads(row["findings"] or "[]"),
    )


def create_deal(target: str, peers: list[str] | None = None, thesis: str = "", deal_id: str | None = None) -> Deal:
    conn = connect()
    d = Deal(id=deal_id or f"deal_{uuid.uuid4().hex[:10]}", target=target, peers=list(peers or []), thesis=thesis)
    conn.execute(
        "INSERT OR REPLACE INTO deals (deal_id, target, peers, thesis, findings, created_at) VALUES (?,?,?,?,?,?)",
        (d.id, d.target, json.dumps(d.peers), d.thesis, json.dumps(d.findings), time.time()),
    )
    conn.commit()
    return d


def get_deal(deal_id: str) -> Deal:
    conn = connect()
    row = conn.execute("SELECT * FROM deals WHERE deal_id = ?", (deal_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown deal {deal_id}")
    return _row_to_deal(row)


def list_deals() -> list[Deal]:
    conn = connect()
    return [_row_to_deal(r) for r in conn.execute("SELECT * FROM deals ORDER BY created_at DESC").fetchall()]


def add_findings(deal_id: str, findings: list[dict]) -> Deal:
    """Append cited findings. Uncited statements are rejected, not silently stored."""
    d = get_deal(deal_id)
    kept = [f for f in findings if f.get("text") and f.get("evidence")]
    d.findings.extend(kept)
    conn = connect()
    conn.execute("UPDATE deals SET findings = ? WHERE deal_id = ?", (json.dumps(d.findings), deal_id))
    conn.commit()
    return d
