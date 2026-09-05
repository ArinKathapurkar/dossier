"""Tier 3 -- answer correctness with an LLM judge. Costs API tokens; run on demand.

Tiers 1 and 2 are free and deterministic, which is why they gate CI. Tier 3 is neither, so
it is a deliberate, budgeted run rather than something that fires on every push.

Method: run the full `ask` path over N FinanceBench questions, one deal per company, then
grade each answer against the gold answer and justification with `prompts/judge.md`. The
judge returns correct / partially_correct / incorrect / **abstained**, kept as a separate
category on purpose -- for a system whose selling point is that it declines rather than
guesses, collapsing abstention into "incorrect" would penalise exactly the behaviour it is
built to produce, and collapsing it into "correct" would hide a recall failure. The report
splits abstentions by whether the source document was actually ingested, which is what
distinguishes correct caution from a retrieval miss.

Judge outputs are cached by (answer hash, judge prompt version) so re-running the report
after a prompt tweak costs only the changed rows.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from typing import Any

from ..config import get_config
from ..obs.prompts import registry

GRADE_TOOL = {
    "name": "grade",
    "description": "Grade the assistant's answer against the gold answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["correct", "partially_correct", "incorrect", "abstained"]},
            "rationale": {"type": "string"},
        },
        "required": ["category", "rationale"],
        "additionalProperties": False,
    },
}


def _cache_path():
    return get_config().paths.reports / "judge_cache.json"


def _load_cache() -> dict:
    p = _cache_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _cache_path().write_text(json.dumps(cache, indent=0, sort_keys=True))


def judge_answer(question: dict, answer: str, cache: dict, prompt_version: str) -> dict:
    key = hashlib.sha256(f"{question['financebench_id']}|{answer}|{prompt_version}".encode()).hexdigest()[:20]
    if key in cache:
        return {**cache[key], "cached": True}

    from ..agent.llm import complete
    from ..obs.tracer import get_tracer

    cfg = get_config()
    system, _ = registry().get("judge")
    payload = (
        f"## Question\n{question['question']}\n\n"
        f"## Gold answer\n{question['answer']}\n\n"
        f"## Gold justification\n{question.get('justification', '')}\n\n"
        f"## Assistant's answer\n{answer or '(empty)'}"
    )
    with get_tracer().span("judge", "tier3", attrs={"financebench_id": question["financebench_id"]}):
        resp = complete(
            system=system,
            messages=[{"role": "user", "content": payload}],
            tools=[GRADE_TOOL],
            tool_choice={"type": "tool", "name": "grade"},
            model=cfg.judge_model,
            max_tokens=512,
            prompt_name="judge",
            prompt_version=prompt_version,
            span_name="judge",
        )
    verdict = {"category": "incorrect", "rationale": "judge returned no grade"}
    for block in resp.tool_uses():
        if block.get("name") == "grade":
            verdict = block.get("input") or verdict
            break
    verdict["judge_cost_usd"] = resp.cost_usd
    cache[key] = verdict
    return {**verdict, "cached": False}


def run_tier3(limit: int = 50, questions: list[dict] | None = None) -> dict:
    from ..agent.deal import create_deal
    from ..agent.loop import ask
    from ..ingest.financebench import load_manifest, load_questions
    from ..obs.tracer import get_tracer

    cfg = get_config()
    all_questions = questions or load_questions()
    ingested = {d["doc_name"] for d in load_manifest()["documents"] if d.get("local_path")}
    scored = [q for q in all_questions if q["doc_name"] in ingested][:limit]

    _, judge_version = registry().get("judge")
    cache = _load_cache()

    # One deal per company; peers are the other companies in the same GICS sector, so the
    # deal memory and graph channel have something realistic to work with.
    by_sector: dict[str, list[str]] = defaultdict(list)
    for q in all_questions:
        if q["company"] not in by_sector[q.get("gics_sector") or ""]:
            by_sector[q.get("gics_sector") or ""].append(q["company"])
    deals: dict[str, Any] = {}

    rows: list[dict] = []
    started = time.time()
    total_cost = 0.0
    for i, q in enumerate(scored, start=1):
        company = q["company"]
        if company not in deals:
            peers = [c for c in by_sector.get(q.get("gics_sector") or "", []) if c != company][:3]
            deals[company] = create_deal(target=company, peers=peers, thesis="")
        t0 = time.time()
        try:
            res = ask(deals[company], q["question"], skip_input_guard=True)
            answer, run_id = res.answer_markdown, res.run_id
            cost, state = res.cost_usd, res.state
            revisions, review_id = res.revisions, res.review_id
        except Exception as exc:  # a failed run is a data point, not an abort
            answer, run_id, cost, state = f"(run failed: {type(exc).__name__}: {exc})", "", 0.0, "FAILED"
            revisions, review_id = 0, None
        latency = time.time() - t0
        verdict = judge_answer(q, answer, cache, judge_version)
        total_cost += cost + float(verdict.get("judge_cost_usd") or 0)
        rows.append(
            {
                "financebench_id": q["financebench_id"],
                "company": company,
                "question_type": q.get("question_type") or "unknown",
                "question": q["question"],
                "gold": q["answer"],
                "answer": answer,
                "run_id": run_id,
                "state": state,
                "category": verdict["category"],
                "rationale": verdict.get("rationale", ""),
                "cost_usd": round(cost, 6),
                "latency_s": round(latency, 2),
                "revisions": revisions,
                "review_id": review_id,
                "document_ingested": q["doc_name"] in ingested,
            }
        )
        if i % 5 == 0:
            _save_cache(cache)
    _save_cache(cache)

    def dist(subset: list[dict]) -> dict:
        n = len(subset) or 1
        out = {"n": len(subset)}
        for cat in ("correct", "partially_correct", "incorrect", "abstained"):
            out[cat] = round(sum(1 for r in subset if r["category"] == cat) / n, 4)
        return out

    by_type: dict[str, dict] = {}
    for qt in sorted({r["question_type"] for r in rows}):
        by_type[qt] = dist([r for r in rows if r["question_type"] == qt])

    tracer = get_tracer()
    run_ids = [r["run_id"] for r in rows if r["run_id"]]
    fallbacks = sum(tracer.count_kind(rid, "fallback") for rid in run_ids)

    n = len(rows) or 1
    return {
        "n": len(rows),
        "limit": limit,
        "model": cfg.primary_model,
        "judge_model": cfg.judge_model,
        "judge_prompt_version": judge_version,
        "overall": dist(rows),
        "by_question_type": by_type,
        "mean_cost_usd": round(sum(r["cost_usd"] for r in rows) / n, 6),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
        "revise_rate": round(sum(1 for r in rows if r["revisions"] > 0) / n, 4),
        "hitl_rate": round(sum(1 for r in rows if r["review_id"]) / n, 4),
        "fallback_spans": fallbacks,
        "fallback_per_100_runs": round(100 * fallbacks / max(len(run_ids), 1), 2),
        "total_cost_usd": round(total_cost, 4),
        "elapsed_s": round(time.time() - started, 1),
        "rows": rows,
    }
