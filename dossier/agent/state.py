"""Workflow state machine, persisted to SQLite.

    PLAN -> GATHER -> ANALYZE -> DRAFT -> REVIEW -> FINAL
                                   \\-> NEEDS_CLARIFICATION
                       (any state) -> FAILED

Two properties matter:

1. **Transitions are validated.** A run cannot go from PLAN to FINAL without gathering
   evidence, so "the model just answered from memory" is a state-machine error rather than
   a quality problem discovered later.
2. **Every transition is persisted.** A run that stops for human review is a row on disk,
   not an in-memory continuation. `dossier resume <run_id>` rebuilds the conversation from
   `messages`, re-injects the ledger and a state summary, and continues -- which is the only
   way a human-in-the-loop step can take hours without holding a process open.

Tables: runs, messages, ledger, sections, reviews (in hitl/queue.py), spans (in obs/tracer.py).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_config

PLAN = "PLAN"
GATHER = "GATHER"
ANALYZE = "ANALYZE"
DRAFT = "DRAFT"
REVIEW = "REVIEW"
FINAL = "FINAL"
FAILED = "FAILED"
NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

STATES = (PLAN, GATHER, ANALYZE, DRAFT, REVIEW, FINAL, FAILED, NEEDS_CLARIFICATION)
TERMINAL = (FINAL, FAILED, NEEDS_CLARIFICATION)

ALLOWED: dict[str, set[str]] = {
    PLAN: {GATHER, NEEDS_CLARIFICATION, FAILED},
    GATHER: {GATHER, ANALYZE, NEEDS_CLARIFICATION, FAILED},
    ANALYZE: {GATHER, DRAFT, REVIEW, FINAL, FAILED},
    DRAFT: {REVIEW, FINAL, ANALYZE, FAILED},
    REVIEW: {DRAFT, ANALYZE, FINAL, FAILED},
    FINAL: set(),
    FAILED: set(),
    NEEDS_CLARIFICATION: {PLAN, GATHER},
}


class InvalidTransition(ValueError):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    deal_id     TEXT,
    run_type    TEXT NOT NULL,
    state       TEXT NOT NULL,
    question    TEXT,
    answer      TEXT,
    ledger      TEXT NOT NULL DEFAULT '',
    guard       TEXT,
    meta        TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    run_id   TEXT NOT NULL,
    seq      INTEGER NOT NULL,
    role     TEXT NOT NULL,
    content  TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS sections (
    run_id       TEXT NOT NULL,
    section      TEXT NOT NULL,
    markdown     TEXT NOT NULL,
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    status       TEXT NOT NULL DEFAULT 'draft',
    PRIMARY KEY (run_id, section)
);
CREATE TABLE IF NOT EXISTS transitions (
    run_id    TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    from_state TEXT,
    to_state  TEXT NOT NULL,
    reason    TEXT,
    at        REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS deals (
    deal_id   TEXT PRIMARY KEY,
    target    TEXT NOT NULL,
    peers     TEXT NOT NULL DEFAULT '[]',
    thesis    TEXT NOT NULL DEFAULT '',
    findings  TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or get_config().paths.runs_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


@dataclass
class RunState:
    run_id: str
    deal_id: str | None
    run_type: str
    state: str = PLAN
    question: str = ""
    answer: str = ""
    guard: dict | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    _conn: sqlite3.Connection | None = None

    # -- lifecycle -----------------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = connect()
        return self._conn

    @classmethod
    def create(cls, run_type: str, deal_id: str | None = None, question: str = "", conn: sqlite3.Connection | None = None) -> RunState:
        rs = cls(run_id=new_run_id(), deal_id=deal_id, run_type=run_type, question=question)
        rs._conn = conn or connect()
        now = time.time()
        rs.conn.execute(
            "INSERT INTO runs (run_id, deal_id, run_type, state, question, answer, ledger, guard, meta, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (rs.run_id, deal_id, run_type, rs.state, question, "", "", None, "{}", now, now),
        )
        rs.conn.commit()
        rs._record_transition(None, rs.state, "created")
        return rs

    @classmethod
    def load(cls, run_id: str, conn: sqlite3.Connection | None = None) -> RunState:
        conn = conn or connect()
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run {run_id}")
        rs = cls(
            run_id=row["run_id"],
            deal_id=row["deal_id"],
            run_type=row["run_type"],
            state=row["state"],
            question=row["question"] or "",
            answer=row["answer"] or "",
            guard=json.loads(row["guard"]) if row["guard"] else None,
            meta=json.loads(row["meta"] or "{}"),
        )
        rs._conn = conn
        return rs

    # -- transitions ---------------------------------------------------------------
    def _record_transition(self, frm: str | None, to: str, reason: str) -> None:
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM transitions WHERE run_id = ?", (self.run_id,)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO transitions (run_id, seq, from_state, to_state, reason, at) VALUES (?,?,?,?,?,?)",
            (self.run_id, seq, frm, to, reason, time.time()),
        )
        self.conn.commit()

    def can_transition(self, to: str) -> bool:
        return to in ALLOWED.get(self.state, set())

    def transition(self, to: str, reason: str = "") -> None:
        if to not in STATES:
            raise InvalidTransition(f"{to!r} is not a state")
        if to == self.state:
            return
        if not self.can_transition(to):
            raise InvalidTransition(f"{self.state} -> {to} is not an allowed transition")
        frm, self.state = self.state, to
        self.conn.execute(
            "UPDATE runs SET state = ?, updated_at = ? WHERE run_id = ?", (to, time.time(), self.run_id)
        )
        self.conn.commit()
        self._record_transition(frm, to, reason)

    def fail(self, reason: str) -> None:
        if self.state in (FINAL, FAILED):
            return
        frm, self.state = self.state, FAILED
        self.meta["failure"] = reason
        self.conn.execute(
            "UPDATE runs SET state = ?, meta = ?, updated_at = ? WHERE run_id = ?",
            (FAILED, json.dumps(self.meta, default=str), time.time(), self.run_id),
        )
        self.conn.commit()
        self._record_transition(frm, FAILED, reason)

    # -- persistence ---------------------------------------------------------------
    def save_messages(self, messages: list[dict]) -> None:
        self.conn.execute("DELETE FROM messages WHERE run_id = ?", (self.run_id,))
        self.conn.executemany(
            "INSERT INTO messages (run_id, seq, role, content) VALUES (?,?,?,?)",
            [(self.run_id, i, m["role"], json.dumps(m["content"], default=str)) for i, m in enumerate(messages)],
        )
        self.conn.commit()

    def load_messages(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE run_id = ? ORDER BY seq", (self.run_id,)
        ).fetchall()
        return [{"role": r["role"], "content": json.loads(r["content"])} for r in rows]

    def save_ledger(self, ledger) -> None:
        self.conn.execute("UPDATE runs SET ledger = ?, updated_at = ? WHERE run_id = ?", (ledger.to_json(), time.time(), self.run_id))
        self.conn.commit()

    def load_ledger(self):
        from .ledger import Ledger

        row = self.conn.execute("SELECT ledger FROM runs WHERE run_id = ?", (self.run_id,)).fetchone()
        return Ledger.from_json(row["ledger"] if row and row["ledger"] else "")

    def save_answer(self, answer: str, guard: dict | None = None) -> None:
        self.answer = answer
        self.guard = guard
        self.conn.execute(
            "UPDATE runs SET answer = ?, guard = ?, updated_at = ? WHERE run_id = ?",
            (answer, json.dumps(guard, default=str) if guard else None, time.time(), self.run_id),
        )
        self.conn.commit()

    def save_meta(self, **kwargs) -> None:
        self.meta.update(kwargs)
        self.conn.execute(
            "UPDATE runs SET meta = ?, updated_at = ? WHERE run_id = ?",
            (json.dumps(self.meta, default=str), time.time(), self.run_id),
        )
        self.conn.commit()

    def save_section(self, section: str, markdown: str, evidence_ids: list[str], status: str = "draft") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sections (run_id, section, markdown, evidence_ids, status) VALUES (?,?,?,?,?)",
            (self.run_id, section, markdown, json.dumps(evidence_ids), status),
        )
        self.conn.commit()

    def sections(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM sections WHERE run_id = ?", (self.run_id,)).fetchall()
        return [{**dict(r), "evidence_ids": json.loads(r["evidence_ids"])} for r in rows]

    def transitions(self) -> list[dict]:
        return [
            dict(r)
            for r in self.conn.execute(
                "SELECT * FROM transitions WHERE run_id = ? ORDER BY seq", (self.run_id,)
            ).fetchall()
        ]

    def summary(self) -> str:
        """The state summary re-injected into the system prompt on every turn and on resume."""
        path = " -> ".join(t["to_state"] for t in self.transitions())
        bits = [f"Current workflow state: {self.state}.", f"Path so far: {path}."]
        if self.meta.get("review_notes"):
            bits.append(f"A human reviewer returned notes: {self.meta['review_notes']}")
        if self.meta.get("guard_violations"):
            bits.append(f"The output guard rejected a previous draft: {self.meta['guard_violations']}")
        return " ".join(bits)


def list_runs(limit: int = 50, conn: sqlite3.Connection | None = None) -> list[dict]:
    conn = conn or connect()
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
