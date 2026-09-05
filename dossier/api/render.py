"""Jinja2 rendering for the reviewer page. Plain HTML, no build step."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

TEMPLATES = Path(__file__).parent / "templates"


@lru_cache(maxsize=1)
def _env():
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    return Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=select_autoescape(["html"]))


def render_reviews(rows: list[dict]) -> str:
    from ..agent.state import RunState

    enriched = []
    for r in rows:
        guard = json.loads(r["guard_report"]) if r.get("guard_report") else None
        evidence = []
        try:
            ledger = RunState.load(r["run_id"]).load_ledger()
            evidence = [
                {"id": i.id, "citation": i.citation, "excerpt": i.excerpt(400)} for i in ledger.items()[:25]
            ]
        except KeyError:
            pass
        enriched.append({**r, "guard": guard, "evidence": evidence})
    return _env().get_template("review.html").render(reviews=enriched, count=len(enriched))
