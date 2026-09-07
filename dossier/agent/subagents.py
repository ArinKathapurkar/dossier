"""Parallel section sub-agents and the synthesizer.

A five-section diligence memo written by one agent in one conversation would carry every
section's retrieved evidence in every section's context -- the Business Overview's forty
passages still sitting there while it drafts Key Risks. That is both expensive and worse:
context that is irrelevant to the current section measurably degrades what the model
attends to.

So each section is its **own conversation, its own ledger, and its own prompt file**, run
concurrently. They share the deal (target, peers, thesis, carried findings) and nothing
else. A synthesizer pass then merges the five drafts: it has no retrieval tools, so it
cannot introduce a fact that no section retrieved, and it must preserve evidence ids
verbatim.

Merging the ledgers is the fiddly part. Each sub-agent numbered its own evidence from E1,
so five sub-agents all have an E1 meaning five different passages. `Ledger.merge` returns
an id remap per section, and each section's markdown is rewritten through its remap before
the synthesizer ever sees it -- otherwise the memo's citations would silently point at the
wrong passages, which is exactly the class of error this whole system exists to prevent.

`--sequential` runs the same work one section at a time. It exists so the README can quote
a measured parallel-vs-sequential wall time rather than asserting that concurrency helped.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from ..config import get_config
from ..obs.prompts import registry
from ..obs.tracer import get_tracer
from . import state as S
from . import tools as T
from .ledger import Ledger
from .llm import complete
from .loop import run_loop

SECTIONS: list[tuple[str, str]] = [
    ("Business Overview", "section_business"),
    ("Financial Profile", "section_financial"),
    ("Competitive Position", "section_competitive"),
    ("Key Risks", "section_risks"),
    ("Open Diligence Questions", "section_questions"),
]

_ID_RE = re.compile(r"\b([EFC]\d+)\b")


def remap_ids(text: str, remap: dict[str, str]) -> str:
    """Rewrite evidence ids in a section's markdown through the merge remap.

    Longest-first is not needed because ids are matched on word boundaries, but the
    substitution must be single-pass: replacing E1->E7 and then E7->E12 in sequence would
    move the first citation twice.
    """
    return _ID_RE.sub(lambda m: remap.get(m.group(1), m.group(1)), text)


def _run_section(deal: Any, section: str, prompt_name: str, parent_run_id: str) -> dict:
    run_state = S.RunState.create(
        "memo_section",
        deal_id=getattr(deal, "id", None),
        question=f"Draft the {section} section for {deal.target}.",
    )
    run_state.save_meta(section=section, parent_run_id=parent_run_id, prompt=prompt_name)
    tracer = get_tracer()
    tracer.bind(run_state.run_id, parent_id=None)
    ledger = Ledger()
    result = run_loop(
        run_state,
        deal,
        f"Draft the {section} section of the diligence memo on {deal.target}. "
        f"Retrieve evidence first, then call draft_section.",
        T.SECTION_TOOLS,
        ledger=ledger,
        section=section,
        section_prompt=prompt_name,
    )
    return {
        "section": section,
        "prompt": prompt_name,
        "run_id": run_state.run_id,
        "state": result.state,
        "markdown": result.answer_markdown,
        "evidence_ids": result.evidence_ids,
        "ledger": ledger,
        "guard_pass": result.guard_pass,
        "guard_report": result.guard_report,
        "review_id": result.review_id,
        "cost_usd": result.cost_usd,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "turns": result.turns,
        "elapsed_s": result.elapsed_s,
    }


async def memo(deal: Any, sequential: bool = False) -> dict:
    cfg = get_config()
    started = time.time()
    parent = S.RunState.create("memo", deal_id=getattr(deal, "id", None), question=f"Diligence memo on {deal.target}")
    tracer = get_tracer()
    tracer.bind(parent.run_id)

    with tracer.span("run", "memo", attrs={"deal": getattr(deal, "id", None), "sequential": sequential}):
        if sequential:
            sections = []
            for name, prompt in SECTIONS:
                sections.append(await asyncio.to_thread(_run_section, deal, name, prompt, parent.run_id))
        else:
            sections = list(
                await asyncio.gather(
                    *(asyncio.to_thread(_run_section, deal, name, prompt, parent.run_id) for name, prompt in SECTIONS)
                )
            )

        # ---- merge ledgers, remapping each section's ids ---------------------------
        merged = Ledger()
        drafts: list[dict] = []
        for sec in sections:
            remap = merged.merge(sec["ledger"])
            drafts.append(
                {
                    "section": sec["section"],
                    "markdown": remap_ids(sec["markdown"] or "", remap),
                    "evidence_ids": [remap.get(i, i) for i in sec["evidence_ids"]],
                    "run_id": sec["run_id"],
                    "guard_pass": sec["guard_pass"],
                    "review_id": sec["review_id"],
                }
            )

        section_cost = sum(s["cost_usd"] for s in sections)
        tokens_in = sum(s["tokens_in"] for s in sections)
        tokens_out = sum(s["tokens_out"] for s in sections)

        # ---- synthesize -----------------------------------------------------------
        # No re-bind here. The sub-agents ran on worker threads via `asyncio.to_thread`,
        # and the tracer's run binding and parent stack are both thread-local, so this
        # thread is still bound to the parent run and still inside the `run` span opened
        # above. Re-binding would reset *this* thread's stack to empty while that span is
        # open, and the memo would fail on the way out of it.
        system, version = registry().get("synthesizer")
        body = "\n\n".join(f"## {d['section']}\n\n{d['markdown']}" for d in drafts)
        evidence_index = merged.render_index()
        resp = complete(
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Deal: {deal.target}. Peers: {', '.join(deal.peers) or 'none'}.\n\n"
                        f"## Evidence index (ids are authoritative; do not renumber)\n{evidence_index}\n\n"
                        f"## Section drafts\n\n{body}"
                    ),
                }
            ],
            model=cfg.primary_model,
            max_tokens=8000,
            prompt_name="synthesizer",
            prompt_version=version,
        )
        memo_markdown = resp.text()
        total_cost = section_cost + resp.cost_usd

        # ---- guard the synthesized memo -------------------------------------------
        from ..guard import output_guard

        with tracer.span("guard", "output_guard_memo") as gspan:
            all_ids = sorted({i for d in drafts for i in d["evidence_ids"]}, key=lambda s: (s[0], int(s[1:])))
            report = output_guard.check(memo_markdown, merged, all_ids)
            gspan.attrs.update({"pass": report.passed, "violations": len(report.violations)})

        parent.transition(S.GATHER, "sections dispatched")
        parent.transition(S.ANALYZE, "sections drafted")
        parent.transition(S.DRAFT, "memo synthesized")
        pending_reviews = [d["review_id"] for d in drafts if d["review_id"]]
        if pending_reviews:
            parent.transition(S.REVIEW, "one or more sections escalated")
        else:
            parent.transition(S.FINAL, "memo complete")

        for d in drafts:
            parent.save_section(d["section"], d["markdown"], d["evidence_ids"])
        parent.save_answer(memo_markdown, report.as_dict())
        parent.save_ledger(merged)
        parent.save_meta(
            sections=[d["section"] for d in drafts],
            section_runs=[d["run_id"] for d in drafts],
            sequential=sequential,
            cost_usd=round(total_cost, 6),
        )

    elapsed = round(time.time() - started, 2)
    return {
        "run_id": parent.run_id,
        "state": parent.state,
        "markdown": memo_markdown,
        "sections": drafts,
        "evidence_count": len(merged),
        "guard_pass": report.passed,
        "guard_report": report.as_dict(),
        "cost_usd": round(total_cost, 6),
        "section_cost_usd": round(section_cost, 6),
        "synthesizer_cost_usd": round(resp.cost_usd, 6),
        "tokens_in": tokens_in + resp.tokens_in,
        "tokens_out": tokens_out + resp.tokens_out,
        "pending_reviews": pending_reviews,
        "sequential": sequential,
        "elapsed_s": elapsed,
        "section_timings": {s["section"]: s["elapsed_s"] for s in sections},
    }
