"""Human-in-the-loop review queue.

The escalation path is the point of this module: a run that cannot ground a section does
not ship it and does not silently drop it. It enqueues a review, transitions to REVIEW, and
**stops** -- the process exits, the state is on disk, and the reviewer can take an hour or a
day. `decide()` writes the decision and resumes the run from persisted state.

Three decisions, three different resumptions:

    approved -> the draft is accepted as written
    edited   -> the reviewer's text replaces the draft
    rejected -> the reviewer's notes go back to the model as a redo instruction

Every decision is exportable as a regression case (`dossier eval export-reviews`), which
closes the loop: a human correction becomes a test that keeps the same mistake from
shipping again.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from ..agent import state as _state

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    section       TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    draft         TEXT NOT NULL DEFAULT '',
    guard_report  TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    reviewer_notes TEXT,
    edited_text   TEXT,
    created_at    REAL NOT NULL,
    decided_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON reviews(status);
CREATE INDEX IF NOT EXISTS idx_reviews_run ON reviews(run_id);
"""

VALID_DECISIONS = ("approved", "edited", "rejected")


def _conn():
    # Call through the module so tests can redirect the database -- see agent/deal.py.
    conn = _state.connect()
    conn.executescript(SCHEMA)
    return conn


def enqueue(run_id: str, section: str, reason: str, draft: str, guard_report: dict | None = None) -> str:
    conn = _conn()
    review_id = f"rev_{uuid.uuid4().hex[:10]}"
    conn.execute(
        "INSERT INTO reviews (id, run_id, section, reason, draft, guard_report, status, created_at) "
        "VALUES (?,?,?,?,?,?,'pending',?)",
        (review_id, run_id, section, reason, draft, json.dumps(guard_report, default=str) if guard_report else None, time.time()),
    )
    conn.commit()
    return review_id


def list_reviews(status: str | None = "pending") -> list[dict]:
    conn = _conn()
    if status:
        rows = conn.execute("SELECT * FROM reviews WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM reviews ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_review(review_id: str) -> dict:
    conn = _conn()
    row = conn.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
    if row is None:
        raise KeyError(f"unknown review {review_id}")
    return dict(row)


def reviews_for_run(run_id: str) -> list[dict]:
    conn = _conn()
    return [dict(r) for r in conn.execute("SELECT * FROM reviews WHERE run_id = ? ORDER BY created_at", (run_id,)).fetchall()]


def record_decision(review_id: str, decision: str, edited_text: str | None = None, notes: str = "") -> dict:
    """Write the decision without resuming -- used by the API, which resumes separately."""
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}")
    conn = _conn()
    conn.execute(
        "UPDATE reviews SET status = ?, edited_text = ?, reviewer_notes = ?, decided_at = ? WHERE id = ?",
        (decision, edited_text, notes, time.time(), review_id),
    )
    conn.commit()
    return get_review(review_id)


def decide(review_id: str, decision: str, edited_text: str | None = None, notes: str = "") -> dict[str, Any]:
    """Record the decision and resume the paused run."""
    review = record_decision(review_id, decision, edited_text=edited_text, notes=notes)
    from ..agent.loop import resume_run

    result = resume_run(review["run_id"], review=review)
    return {"review_id": review_id, "run_id": review["run_id"], "state": result.state, "answer": result.answer_markdown}
