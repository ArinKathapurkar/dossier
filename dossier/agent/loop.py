"""The agent loop.

Written directly against the Messages API on purpose. The loop is where the interesting
decisions live -- what goes into the system prompt each turn, when to compact, when to stop,
what to do when the guard rejects a draft -- and a framework would hide exactly those.

Per turn:

    build system prompt  (analyst prompt + diligence checklist + state summary
                          + ledger index + deal memory)
        -> call the model with the tool set for this run type
        -> dispatch tool calls, in parallel when they are independent
        -> append results, compact if over budget
        -> repeat until a terminal tool, max_turns, or the cost cap

Three controls that make it safe to run unattended:

  * `max_turns` (default 24) bounds the loop.
  * a per-run cost cap (default $1.50) ends the run with whatever it has and a `budget`
    flag rather than an unbounded bill.
  * the **retrieval-confidence guard**: if the first search comes back with a weak best
    reranker score and the question names no company in the corpus, the run transitions to
    NEEDS_CLARIFICATION and asks a question instead of inventing an answer. This is the
    single highest-value guard in the system, because the most damaging failure mode of a
    filings agent is confidently answering about a company it has never seen.

The revise loop: when the output guard rejects an answer, the report goes back to the model
as a normal turn (up to `max_revisions`). If it still fails, the run escalates to a human
automatically with the guard report attached, rather than shipping the answer.
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import get_config
from ..guard import output_guard
from ..obs.prompts import registry
from ..obs.tracer import get_tracer
from . import state as S
from . import tools as T
from .budget import BudgetManager, strip_internal_keys
from .ledger import Ledger
from .llm import complete

DILIGENCE_CHECKLIST = """\
Diligence checklist -- what a complete answer covers, where the question calls for it:
1. The figure or fact itself, with the fiscal period it belongs to.
2. Its source: which company, which filing, which page or which XBRL concept.
3. Any comparison the question implies (prior period, peer set) with the same rigour.
4. What the filings do *not* say, if the question asks for something they do not disclose."""


@dataclass
class RunResult:
    run_id: str
    state: str
    answer_markdown: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    guard_pass: bool = True
    guard_report: dict | None = None
    turns: int = 0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    revisions: int = 0
    review_id: str | None = None
    elapsed_s: float = 0.0
    events: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["events"] = self.events[-50:]
        return d


def build_system_prompt(run_state: S.RunState, ledger: Ledger, deal: Any, section_prompt: str | None = None, extra: str = "") -> tuple[str, str, str]:
    """Returns `(system_text, prompt_name, prompt_version)`.

    Order matters for prompt caching: the stable analyst prompt and checklist come first,
    the volatile ledger index and state summary come last.
    """
    reg = registry()
    base, version = reg.get("system_analyst")
    name = "system_analyst"
    parts = [base, "", DILIGENCE_CHECKLIST]
    if section_prompt:
        sec_text, sec_version = reg.render(
            section_prompt,
            target=getattr(deal, "target", ""),
            peers=", ".join(getattr(deal, "peers", []) or []) or "(none specified)",
        )
        parts += ["", "## Your assignment", sec_text]
        name, version = section_prompt, sec_version
    if deal is not None:
        parts += [
            "",
            "## Deal",
            f"Target: {deal.target}. Peers: {', '.join(deal.peers) or 'none'}."
            + (f" Thesis: {deal.thesis}" if deal.thesis else ""),
            "",
            "## Findings carried over from earlier runs on this deal",
            deal.memory_block(),
        ]
    parts += ["", "## Workflow state", run_state.summary()]
    parts += ["", "## Evidence so far", ledger.render_index(limit=60)]
    if extra:
        parts += ["", extra]
    return "\n".join(parts), name, version


def _dispatch_all(tool_uses: list[dict], ctx: T.ToolContext) -> list[dict]:
    """Run tool calls, concurrently when they are side-effect free.

    Ledger writes are serialized by doing the concurrent work only for the read-only tools
    and folding the results in deterministic call order afterwards, so the evidence ids a
    run assigns do not depend on thread scheduling -- which is what makes cassette replay
    byte-identical.
    """
    from .llm import active_cassette

    results: dict[str, str] = {}
    parallel = [tu for tu in tool_uses if tu.get("name") in T.PARALLEL_SAFE]
    serial = [tu for tu in tool_uses if tu.get("name") not in T.PARALLEL_SAFE]
    # The prefetch path calls the retriever directly, bypassing dispatch's record/replay.
    # While a cassette is active every call goes through dispatch instead, so recording and
    # replay stay complete; the concurrency is a latency optimisation, not a behaviour.
    if len(parallel) > 1 and active_cassette() is None:
        # Retrieval itself is the expensive part and is thread-safe; run those first, then
        # apply ledger writes in call order.
        prefetch: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(parallel))) as pool:
            futures = {pool.submit(_prefetch, tu, ctx): tu["id"] for tu in parallel}
            for fut in concurrent.futures.as_completed(futures):
                prefetch[futures[fut]] = fut.result()
        for tu in parallel:
            results[tu["id"]] = _apply_prefetched(tu, ctx, prefetch.get(tu["id"]))
    else:
        for tu in parallel:
            results[tu["id"]] = T.dispatch(tu["name"], tu.get("input") or {}, ctx)
    for tu in serial:
        results[tu["id"]] = T.dispatch(tu["name"], tu.get("input") or {}, ctx)
    return [
        {"type": "tool_result", "tool_use_id": tu["id"], "content": results.get(tu["id"], "")}
        for tu in tool_uses
    ]


def _prefetch(tool_use: dict, ctx: T.ToolContext) -> Any:
    """Warm caches for a read-only tool without touching the ledger."""
    if tool_use.get("name") != "search_filings":
        return None
    args = tool_use.get("input") or {}
    try:
        retriever = ctx.retriever
        if retriever is None:
            from ..retrieve.hybrid import get_retriever

            retriever = get_retriever()
        cfg = get_config()
        filters = {k: args[k] for k in ("company", "doc_type", "fiscal_period") if args.get(k)}
        k = min(int(args.get("k") or cfg.rerank_top_n), 15)
        return retriever.search(
            args.get("query", ""), deal=ctx.deal, filters=filters or None,
            channels=cfg.retrieval_channels, rerank=cfg.retrieval_rerank, top_n=k,
        )
    except Exception:
        return None


def _apply_prefetched(tool_use: dict, ctx: T.ToolContext, prefetched: Any) -> str:
    if tool_use.get("name") == "search_filings" and prefetched is not None:
        chunks = prefetched
        ctx.searches += 1
        if chunks and chunks[0].rerank_score is not None:
            ctx.last_rerank_top = chunks[0].rerank_score
        if not chunks:
            return "No passages matched. The corpus may not contain this company or period."
        added = [ctx.ledger.add_chunk(c) for c in chunks]
        return T._wrap_documents([{"id": a.id, "citation": a.citation, "text": a.text} for a in added])
    return T.dispatch(tool_use["name"], tool_use.get("input") or {}, ctx)


def _corpus_mentions(question: str) -> bool:
    """Does the question name a company that exists in the indexed corpus?"""
    try:
        from ..ingest.financebench import load_manifest

        companies = {d["company"].lower() for d in load_manifest()["documents"]}
    except Exception:
        return True  # cannot tell; do not block
    q = question.lower()
    return any(c and c in q for c in companies)


def run_loop(
    run_state: S.RunState,
    deal: Any,
    user_message: str,
    tool_set: list[dict],
    ledger: Ledger | None = None,
    section: str = "",
    section_prompt: str | None = None,
    messages: list[dict] | None = None,
    budget: BudgetManager | None = None,
    max_turns: int | None = None,
    check_output: bool = True,
) -> RunResult:
    cfg = get_config()
    tracer = get_tracer()
    tracer.bind(run_state.run_id)
    ledger = ledger if ledger is not None else Ledger()
    budget = budget or BudgetManager()
    max_turns = max_turns or cfg.max_turns
    started = time.time()
    result = RunResult(run_id=run_state.run_id, state=run_state.state)

    messages = list(messages or [{"role": "user", "content": user_message}])
    ctx = T.ToolContext(ledger=ledger, deal=deal, run_state=run_state, section=section)
    revisions = 0

    with tracer.span("run", run_state.run_type, attrs={"deal": getattr(deal, "id", None), "question": user_message[:200]}):
        for turn in range(max_turns):
            result.turns = turn + 1
            if result.cost_usd >= cfg.run_cost_cap_usd:
                run_state.save_meta(budget_capped=True)
                result.events.append({"type": "budget_cap", "cost_usd": result.cost_usd})
                break

            messages = budget.maybe_compact(messages, run_id=run_state.run_id)
            system, pname, pversion = build_system_prompt(run_state, ledger, deal, section_prompt)

            with tracer.span("turn", f"turn_{turn + 1}", attrs={"messages": len(messages)}):
                resp = complete(
                    system=system,
                    messages=strip_internal_keys(messages),
                    tools=tool_set,
                    prompt_name=pname,
                    prompt_version=pversion,
                )
            result.cost_usd += resp.cost_usd
            result.tokens_in += resp.tokens_in
            result.tokens_out += resp.tokens_out
            messages.append({"role": "assistant", "content": resp.content})

            tool_uses = resp.tool_uses()
            if not tool_uses:
                # The model answered in prose without calling `finish`. Treat the text as the
                # answer rather than looping: nudging costs a turn and rarely helps.
                result.answer_markdown = resp.text()
                run_state.transition(S.GATHER, "model answered in prose") if run_state.state == S.PLAN else None
                break

            for tu in tool_uses:
                result.events.append({"type": "tool_call", "tool": tu.get("name"), "input": tu.get("input")})

            if run_state.state == S.PLAN:
                run_state.transition(S.GATHER, "first tool call")

            tool_results = _dispatch_all(tool_uses, ctx)
            messages.append({"role": "user", "content": tool_results})

            # ---- retrieval-confidence guard -------------------------------------
            if (
                ctx.searches == 1
                and ctx.last_rerank_top is not None
                and ctx.last_rerank_top < cfg.retrieval_confidence_threshold
                and not _corpus_mentions(run_state.question or user_message)
            ):
                run_state.transition(S.NEEDS_CLARIFICATION, "low retrieval confidence and no corpus company named")
                result.state = run_state.state
                result.answer_markdown = (
                    "I could not find anything relevant in the indexed filings for that question, and it does not "
                    "name a company in this corpus. Which company and fiscal period should I look at? The corpus "
                    "covers the FinanceBench document set only."
                )
                result.events.append({"type": "clarification", "top_rerank_score": ctx.last_rerank_top})
                run_state.save_answer(result.answer_markdown)
                run_state.save_messages(messages)
                run_state.save_ledger(ledger)
                result.elapsed_s = round(time.time() - started, 2)
                return result

            # ---- human review escalation -----------------------------------------
            if ctx.review_request is not None:
                from ..hitl.queue import enqueue

                if run_state.state in (S.GATHER, S.ANALYZE):
                    run_state.transition(S.ANALYZE, "escalating") if run_state.state == S.GATHER else None
                    run_state.transition(S.REVIEW, "model requested human review")
                elif run_state.can_transition(S.REVIEW):
                    run_state.transition(S.REVIEW, "model requested human review")
                review_id = enqueue(
                    run_state.run_id,
                    ctx.review_request["section"],
                    ctx.review_request["reason"],
                    ctx.review_request["draft"],
                )
                with tracer.span("review_wait", "enqueued", attrs={"review_id": review_id, "section": ctx.review_request["section"]}):
                    pass
                result.review_id = review_id
                result.state = run_state.state
                result.answer_markdown = ctx.review_request["draft"]
                run_state.save_answer(result.answer_markdown)
                run_state.save_messages(messages)
                run_state.save_ledger(ledger)
                run_state.save_meta(pending_review=review_id, section=ctx.review_request["section"])
                result.events.append({"type": "review_required", "review_id": review_id})
                result.elapsed_s = round(time.time() - started, 2)
                return result

            # ---- terminal tools ---------------------------------------------------
            payload = ctx.finished or ctx.drafted
            if payload is None:
                continue

            answer = payload.get("answer_markdown") or payload.get("markdown") or ""
            evidence_ids = payload.get("evidence_ids") or []
            if run_state.state == S.GATHER:
                run_state.transition(S.ANALYZE, "terminal tool called")
            if run_state.state == S.ANALYZE and ctx.drafted:
                run_state.transition(S.DRAFT, "section drafted")

            if not check_output:
                result.answer_markdown = answer
                result.evidence_ids = evidence_ids
                break

            with tracer.span("guard", "output_guard") as gspan:
                report = output_guard.check(answer, ledger, evidence_ids)
                gspan.attrs.update({"pass": report.passed, "violations": len(report.violations), **report.checks})
            result.guard_report = report.as_dict()
            result.guard_pass = report.passed
            result.events.append({"type": "guard", "pass": report.passed, "violations": len(report.violations)})

            if report.passed:
                result.answer_markdown = answer
                result.evidence_ids = evidence_ids
                break

            if revisions < cfg.max_revisions:
                revisions += 1
                result.revisions = revisions
                run_state.save_meta(guard_violations=[v.kind for v in report.violations])
                ctx.finished = ctx.drafted = None
                messages.append({"role": "user", "content": report.to_prompt()})
                continue

            # Two revisions were not enough: escalate rather than ship it.
            from ..hitl.queue import enqueue

            if run_state.can_transition(S.REVIEW):
                run_state.transition(S.REVIEW, "output guard failed after revisions")
            review_id = enqueue(
                run_state.run_id,
                section or "answer",
                f"Output guard failed after {revisions} revision(s): "
                + "; ".join(f"{v.kind}" for v in report.violations[:4]),
                answer,
                guard_report=report.as_dict(),
            )
            result.review_id = review_id
            result.answer_markdown = answer
            result.evidence_ids = evidence_ids
            result.state = run_state.state
            result.events.append({"type": "review_required", "review_id": review_id, "cause": "guard"})
            run_state.save_answer(answer, report.as_dict())
            run_state.save_messages(messages)
            run_state.save_ledger(ledger)
            run_state.save_meta(pending_review=review_id, section=section or "answer")
            result.elapsed_s = round(time.time() - started, 2)
            return result

        # ---- normal termination -------------------------------------------------
        if run_state.state not in S.TERMINAL:
            if run_state.state == S.PLAN:
                run_state.transition(S.GATHER, "loop ended")
            if run_state.state == S.GATHER:
                run_state.transition(S.ANALYZE, "loop ended")
            run_state.transition(S.FINAL, "answer produced")
        result.state = run_state.state
        run_state.save_answer(result.answer_markdown, result.guard_report)
        run_state.save_messages(messages)
        run_state.save_ledger(ledger)
        if ctx.drafted:
            run_state.save_section(ctx.drafted["section"], ctx.drafted["markdown"], ctx.drafted["evidence_ids"])
        run_state.save_meta(
            turns=result.turns,
            cost_usd=round(result.cost_usd, 6),
            revisions=result.revisions,
            budget=budget.stats(),
        )
    result.elapsed_s = round(time.time() - started, 2)
    return result


def ask(deal: Any, question: str, skip_input_guard: bool = False) -> RunResult:
    """Single-question path: input guard -> loop -> output guard -> finish."""
    from ..guard.input_guard import classify

    run_state = S.RunState.create("ask", deal_id=getattr(deal, "id", None), question=question)
    get_tracer().bind(run_state.run_id)

    if not skip_input_guard:
        verdict = classify(question)
        if not verdict.allowed:
            run_state.save_meta(input_guard=verdict.as_dict())
            run_state.transition(S.NEEDS_CLARIFICATION, f"input guard: {verdict.category}")
            run_state.save_answer(verdict.message)
            return RunResult(
                run_id=run_state.run_id,
                state=run_state.state,
                answer_markdown=verdict.message,
                guard_pass=True,
                events=[{"type": "input_guard", **verdict.as_dict()}],
            )

    return run_loop(run_state, deal, question, T.ASK_TOOLS)


def resume_run(run_id: str, review: dict | None = None) -> RunResult:
    """Rebuild a paused run from disk and continue it.

    The conversation, the ledger and the state history all come back from SQLite. A reviewer
    decision enters as a normal user turn, which keeps the resumed run structurally
    identical to one that never paused.
    """
    from ..hitl.queue import reviews_for_run

    run_state = S.RunState.load(run_id)
    deal = None
    if run_state.deal_id:
        from .deal import get_deal

        try:
            deal = get_deal(run_state.deal_id)
        except KeyError:
            deal = None

    messages = run_state.load_messages()
    ledger = run_state.load_ledger()
    if review is None:
        pending = [r for r in reviews_for_run(run_id) if r["status"] != "pending"]
        review = pending[-1] if pending else None

    if review is not None:
        decision = review["status"]
        notes = review.get("reviewer_notes") or ""
        if decision == "approved":
            instruction = (
                f"A human reviewer approved the {review['section']} draft"
                + (f" with this note: {notes}" if notes else "")
                + ". Produce the final version incorporating any note, and call the terminal tool."
            )
            run_state.save_meta(review_notes=notes, review_decision="approved")
        elif decision == "edited":
            instruction = (
                f"A human reviewer edited the {review['section']} draft. The reviewer's text is authoritative; "
                f"use it verbatim as the section body and call the terminal tool with it.\n\n"
                f"<reviewer_text>\n{review.get('edited_text') or ''}\n</reviewer_text>"
            )
            run_state.save_meta(review_notes=notes, review_decision="edited")
        else:
            instruction = (
                f"A human reviewer rejected the {review['section']} draft with these notes: {notes or '(none given)'}. "
                f"Address every point, retrieving more evidence if you need it, then call the terminal tool again."
            )
            run_state.save_meta(review_notes=notes, review_decision="rejected")
        messages.append({"role": "user", "content": instruction})
        if run_state.state == S.REVIEW:
            run_state.transition(S.DRAFT if decision != "rejected" else S.ANALYZE, f"review {decision}")

    tool_set = T.SECTION_TOOLS if run_state.meta.get("section") and run_state.run_type == "memo_section" else T.ASK_TOOLS
    return run_loop(
        run_state,
        deal,
        run_state.question or "",
        tool_set,
        ledger=ledger,
        messages=messages,
        section=run_state.meta.get("section", ""),
    )
