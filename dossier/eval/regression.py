"""Regression cassettes and the reviewer-decision -> regression-case loop.

Two mechanisms live here.

**Cassettes.** `llm.py` records every (model, system, messages, tools) -> response exchange
to a JSON file keyed by a hash of the request. In replay mode a miss raises rather than
falling through to the API, so `tests/regression` runs in CI with no key and no spend, and
asserts the tool-call sequence, final state, guard verdict and span kinds against what was
recorded. Replaying twice must produce byte-identical sequences -- that is the property
that makes a change in agent behaviour visible instead of being lost in sampling noise.

**Reviewer decisions become tests.** `export_reviews` turns each decided review into a
regression case: what was asked, what the reviewer did, and the properties the answer must
satisfy afterwards (must cite, must not contain, must abstain). This is the loop that makes
human review pay for itself -- a correction made once becomes a check that runs forever.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

from ..config import REPO_ROOT, get_config

CASES_DIR = REPO_ROOT / "tests" / "regression" / "cases"


@contextmanager
def cassette(name: str, mode: str | None = None):
    """Activate a cassette for the duration of the block."""
    from ..agent.llm import Cassette, set_cassette

    cfg = get_config()
    path = cfg.cassette_dir / f"{name}.json"
    cas = Cassette(path)
    prev_mode = os.environ.get("DOSSIER_LLM_MODE")
    if mode:
        os.environ["DOSSIER_LLM_MODE"] = mode
    set_cassette(cas)
    try:
        yield cas
    finally:
        set_cassette(None)
        if mode:
            if prev_mode is None:
                os.environ.pop("DOSSIER_LLM_MODE", None)
            else:
                os.environ["DOSSIER_LLM_MODE"] = prev_mode


def list_cassettes() -> list[Path]:
    return sorted(get_config().cassette_dir.glob("*.json"))


def cassette_expectations(path: Path) -> dict:
    return json.loads(path.read_text()).get("meta", {}).get("expect", {})


# ---------------------------------------------------------------------------------
# reviewer decisions -> regression cases
# ---------------------------------------------------------------------------------

_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")


def _properties_from_decision(review: dict, run: dict) -> dict:
    """Derive the assertions a corrected answer must satisfy.

    Kept deliberately conservative. A rejected draft yields "must not contain" on the
    figures the reviewer objected to; an edited draft yields "must cite" on the ids the
    reviewer's own text keeps. Inferring more than that would produce brittle tests that
    fail on wording rather than on substance.
    """
    props: dict = {}
    guard = json.loads(review["guard_report"]) if review.get("guard_report") else None
    if guard:
        kinds = [v["kind"] for v in guard.get("violations", [])]
        props["guard_must_pass"] = True
        props["previously_violated"] = sorted(set(kinds))
        bad_numbers = sorted(
            {
                m.group(0)
                for v in guard.get("violations", [])
                if v["kind"] == "numeric_ungrounded"
                for m in [_NUM.search(v["detail"])]
                if m
            }
        )
        if bad_numbers:
            props["must_not_contain"] = bad_numbers
    if review["status"] == "edited" and review.get("edited_text"):
        ids = sorted(set(re.findall(r"\b([EFC]\d+)\b", review["edited_text"])))
        if ids:
            props["must_cite"] = ids
    if review["status"] == "rejected":
        props["must_address_notes"] = review.get("reviewer_notes") or ""
    if run.get("state") == "NEEDS_CLARIFICATION":
        props["must_abstain"] = True
    return props


def export_reviews(out_dir: Path | None = None) -> dict:
    from ..agent.state import connect
    from ..hitl.queue import list_reviews

    out_dir = out_dir or CASES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect()
    written = 0
    cases = []
    for review in list_reviews(status=None):
        if review["status"] == "pending":
            continue
        row = conn.execute("SELECT question, state, answer FROM runs WHERE run_id = ?", (review["run_id"],)).fetchone()
        run = dict(row) if row else {}
        case = {
            "case_id": review["id"],
            "source": "reviewer_decision",
            "question": run.get("question") or f"Draft the {review['section']} section.",
            "section": review["section"],
            "decision": review["status"],
            "reviewer_notes": review.get("reviewer_notes") or "",
            "reason_escalated": review["reason"],
            "expect": _properties_from_decision(review, run),
        }
        (out_dir / f"{review['id']}.json").write_text(json.dumps(case, indent=2))
        cases.append(case)
        written += 1
    return {"written": written, "dir": str(out_dir), "cases": cases}


def load_cases(directory: Path | None = None) -> list[dict]:
    directory = directory or CASES_DIR
    if not directory.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
