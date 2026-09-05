"""Tier 2 -- grounding and guardrails. Free, deterministic, runs in CI.

Tier 1 asks "did we retrieve the right passage". Tier 2 asks "given what we retrieved, is
the answer actually supported by it" -- which is the question that matters for a system
whose whole claim is that every number is cited.

Sources of answers to check:

  (a) **Hand-written fixtures** in `tests/fixtures/answers/*.json`. Each carries a small
      ledger, an answer, and the verdict the guard should return. Half of them fail on
      purpose: an invented number, a citation to a nonexistent id, an unattributed
      projection, a silent no-answer. These are what make the guard itself testable --
      measuring only on real answers would let a guard that passes everything score 100%.
  (b) **Stored answers from previous runs** (Tier 3 or ad-hoc `ask` runs), pulled from
      `runs/runs.sqlite`. This is the live-traffic half.

Abstention accuracy is measured on `tests/fixtures/ooc_questions.json`: fifteen questions
about companies or fiscal years that are not in the corpus, where the only correct
behaviours are abstaining or asking for clarification.

Fallback spans per run come from the tracer, so silent degradation shows up as a number.
"""

from __future__ import annotations

import json

from ..config import REPO_ROOT
from ..guard import output_guard
from ..guard.output_guard import ABSTENTION

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "answers"
OOC_FILE = REPO_ROOT / "tests" / "fixtures" / "ooc_questions.json"


class _DictLedger:
    """Minimal ledger stand-in so fixtures can declare evidence inline as JSON."""

    def __init__(self, items: list[dict]):
        from ..agent.ledger import EvidenceItem

        self._items = {
            i["id"]: EvidenceItem(
                id=i["id"],
                kind=i.get("kind", "chunk"),
                citation=i.get("citation", ""),
                text=i.get("text", ""),
                meta=i.get("meta", {}),
            )
            for i in items
        }

    def get(self, eid: str):
        return self._items.get(eid)

    def items(self):
        return list(self._items.values())


def load_fixtures() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(FIXTURE_DIR.glob("*.json"))]


def _stored_answers(limit: int = 200) -> list[dict]:
    from ..agent.state import connect

    conn = connect()
    rows = conn.execute(
        "SELECT run_id, question, answer, ledger, guard FROM runs "
        "WHERE answer IS NOT NULL AND answer != '' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _fallback_spans_per_run() -> tuple[float, int, int]:
    from ..obs.tracer import get_tracer

    conn = get_tracer()._conn
    runs = conn.execute("SELECT COUNT(DISTINCT run_id) FROM spans WHERE kind = 'run'").fetchone()[0]
    fallbacks = conn.execute("SELECT COUNT(*) FROM spans WHERE kind = 'fallback'").fetchone()[0]
    return (fallbacks / runs if runs else 0.0), fallbacks, runs


def run_ooc_questions(deal_id: str = "deal_ooc_probe") -> dict:
    """Run every out-of-corpus question through `ask` so abstention can be scored.

    Costs API tokens (one short run per question). Separated from `run_tier2` because Tier 2
    itself must stay free and offline for CI; this populates the run history it reads.
    """
    from ..agent.deal import create_deal, get_deal
    from ..agent.loop import ask

    questions = json.loads(OOC_FILE.read_text())
    try:
        deal = get_deal(deal_id)
    except KeyError:
        deal = create_deal(target="3M", peers=["Costco", "Boeing"], deal_id=deal_id)
    rows = []
    for q in questions:
        res = ask(deal, q["question"], skip_input_guard=True)
        abstained = bool(ABSTENTION.search(res.answer_markdown or "")) or res.state == "NEEDS_CLARIFICATION"
        rows.append(
            {
                "question": q["question"],
                "why_out_of_corpus": q["why"],
                "run_id": res.run_id,
                "state": res.state,
                "abstained": abstained,
                "citations": len(res.evidence_ids),
                "cost_usd": round(res.cost_usd, 6),
                "answer": (res.answer_markdown or "")[:400],
            }
        )
    return {
        "questions": len(rows),
        "abstained": sum(1 for r in rows if r["abstained"]),
        "abstention_accuracy": round(sum(1 for r in rows if r["abstained"]) / len(rows), 4) if rows else 0.0,
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "rows": rows,
    }


def run_tier2(fixtures_only: bool = False) -> dict:
    fixtures = load_fixtures()
    checks = {"citation_validity": [], "numeric_grounding": [], "forward_looking": []}
    fixture_correct = 0
    fixture_detail: list[dict] = []

    for fx in fixtures:
        ledger = _DictLedger(fx.get("ledger", []))
        report = output_guard.check(fx["answer"], ledger, fx.get("evidence_ids"))
        expected = fx["expect_pass"]
        ok = report.passed == expected
        fixture_correct += int(ok)
        expected_kinds = set(fx.get("expect_violations", []))
        got_kinds = {v.kind for v in report.violations}
        fixture_detail.append(
            {
                "name": fx["name"],
                "expected_pass": expected,
                "actual_pass": report.passed,
                "ok": ok,
                "expected_violations": sorted(expected_kinds),
                "actual_violations": sorted(got_kinds),
                "violation_kinds_match": (not expected_kinds) or expected_kinds.issubset(got_kinds),
            }
        )
        for key in checks:
            checks[key].append(report.checks.get(key, True))

    stored: list[dict] = [] if fixtures_only else _stored_answers()
    stored_reports = []
    for row in stored:
        from ..agent.ledger import Ledger

        ledger = Ledger.from_json(row.get("ledger") or "")
        if not ledger.items():
            continue
        report = output_guard.check(row["answer"], ledger)
        stored_reports.append({"run_id": row["run_id"], **report.as_dict()})
        for key in checks:
            checks[key].append(report.checks.get(key, True))

    # ---- abstention accuracy on out-of-corpus questions ---------------------------
    ooc = json.loads(OOC_FILE.read_text()) if OOC_FILE.exists() else []
    ooc_answers = {}
    if not fixtures_only:
        from ..agent.state import connect

        conn = connect()
        for q in ooc:
            row = conn.execute(
                "SELECT answer, state FROM runs WHERE question = ? ORDER BY created_at DESC LIMIT 1", (q["question"],)
            ).fetchone()
            if row:
                ooc_answers[q["question"]] = dict(row)
    ooc_correct = 0
    ooc_scored = 0
    for q in ooc:
        row = ooc_answers.get(q["question"])
        if row is None:
            continue
        ooc_scored += 1
        abstained = bool(ABSTENTION.search(row["answer"] or "")) or row["state"] == "NEEDS_CLARIFICATION"
        ooc_correct += int(abstained)

    fb_per_run, fb_total, run_total = _fallback_spans_per_run()

    def rate(key: str) -> float:
        vals = checks[key]
        return round(sum(1 for v in vals if v) / len(vals), 4) if vals else 1.0

    return {
        "fixtures": len(fixtures),
        "answers_checked": len(fixtures) + len(stored_reports),
        "stored_answers": len(stored_reports),
        "metrics": {
            "citation_validity": rate("citation_validity"),
            "numeric_grounding": rate("numeric_grounding"),
            "forward_looking_attribution": rate("forward_looking"),
            # None, not 0.0: no out-of-corpus run has been scored, which is not the same as
            # scoring zero. `--mini` is fixtures-only and never has run history.
            "abstention_accuracy": round(ooc_correct / ooc_scored, 4) if ooc_scored else None,
            "fixture_accuracy": round(fixture_correct / len(fixtures), 4) if fixtures else 0.0,
            "fallback_spans_per_run": round(fb_per_run, 4),
        },
        "ooc_questions": len(ooc),
        "ooc_scored": ooc_scored,
        "fallback_spans_total": fb_total,
        "runs_total": run_total,
        "fixture_detail": fixture_detail,
        "stored_reports": stored_reports[:20],
    }


def assert_fixtures_pass() -> list[dict]:
    """Used by the unit tests: every fixture must produce its expected verdict."""
    rep = run_tier2(fixtures_only=True)
    return [d for d in rep["fixture_detail"] if not d["ok"] or not d["violation_kinds_match"]]


