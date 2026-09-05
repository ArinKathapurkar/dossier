"""FastAPI service.

Two things shape the API surface:

* **Agent runs are long.** `ask` takes tens of seconds and `memo` takes minutes, so those
  endpoints stream Server-Sent Events -- turn, tool_call, guard, review_required, final --
  rather than making a client hold a request open for an opaque three minutes.
* **A run that pauses for review is a resource, not a dropped connection.** The stream ends
  with `review_required` and the run's state lives in SQLite; the reviewer acts through
  `/reviews`, and `POST /reviews/{id}/decision` resumes it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..config import get_config

app = FastAPI(
    title="dossier",
    version="0.1.0",
    description=(
        "Agentic due-diligence copilot over public SEC filings from the FinanceBench "
        "benchmark. Not a client system and not investment advice."
    ),
)


# ---------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------


class DealIn(BaseModel):
    target: str
    peers: list[str] = Field(default_factory=list)
    thesis: str = ""


class AskIn(BaseModel):
    question: str


class DecisionIn(BaseModel):
    decision: str = Field(description="approved | edited | rejected")
    edited_text: str | None = None
    notes: str = ""


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream(fn, *args, **kwargs):
    """Run a blocking agent call in a thread and emit its events as SSE.

    The loop records events on the result rather than yielding them, so the stream sends a
    heartbeat while work is in flight and then replays the event list. That keeps the agent
    code synchronous and testable, which is worth more than true incremental streaming for
    a run that produces a handful of events.
    """
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def work():
        try:
            result = fn(*args, **kwargs)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", result))
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))

    task = loop.run_in_executor(None, work)
    yield _sse("start", {"status": "running"})
    while True:
        try:
            kind, payload = await asyncio.wait_for(queue.get(), timeout=10.0)
        except TimeoutError:
            yield _sse("heartbeat", {"status": "running"})
            continue
        if kind == "error":
            yield _sse("error", {"error": f"{type(payload).__name__}: {payload}"})
            break
        result = payload
        events = result.events if hasattr(result, "events") else result.get("events", [])
        for ev in events:
            yield _sse(ev.get("type", "event"), ev)
        final = result.as_dict() if hasattr(result, "as_dict") else result
        final.pop("ledger", None)
        if final.get("state") == "REVIEW" or final.get("review_id") or final.get("pending_reviews"):
            yield _sse("review_required", final)
        else:
            yield _sse("final", final)
        break
    await task


# ---------------------------------------------------------------------------------
# deals
# ---------------------------------------------------------------------------------


@app.post("/deals")
def create_deal_endpoint(body: DealIn) -> dict:
    from ..agent.deal import create_deal

    d = create_deal(target=body.target, peers=body.peers, thesis=body.thesis)
    return d.as_dict()


@app.get("/deals/{deal_id}")
def get_deal_endpoint(deal_id: str) -> dict:
    from ..agent.deal import get_deal

    try:
        return get_deal(deal_id).as_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/deals/{deal_id}/ask")
async def ask_endpoint(deal_id: str, body: AskIn):
    from ..agent.deal import get_deal
    from ..agent.loop import ask

    try:
        deal = get_deal(deal_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return StreamingResponse(_stream(ask, deal, body.question), media_type="text/event-stream")


@app.post("/deals/{deal_id}/memo")
async def memo_endpoint(deal_id: str, sequential: bool = Query(False)):
    from ..agent.deal import get_deal
    from ..agent.subagents import memo

    try:
        deal = get_deal(deal_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    def run_memo():
        return asyncio.run(memo(deal, sequential=sequential))

    return StreamingResponse(_stream(run_memo), media_type="text/event-stream")


# ---------------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------------


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    from ..agent.state import RunState

    try:
        rs = RunState.load(run_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    ledger = rs.load_ledger()
    return {
        "run_id": rs.run_id,
        "deal_id": rs.deal_id,
        "run_type": rs.run_type,
        "state": rs.state,
        "question": rs.question,
        "answer": rs.answer,
        "guard": rs.guard,
        "meta": rs.meta,
        "sections": rs.sections(),
        "transitions": rs.transitions(),
        "ledger": [i.as_dict() for i in ledger.items()],
    }


@app.post("/runs/{run_id}/resume")
async def resume_endpoint(run_id: str):
    from ..agent.loop import resume_run

    return StreamingResponse(_stream(resume_run, run_id), media_type="text/event-stream")


# ---------------------------------------------------------------------------------
# reviews
# ---------------------------------------------------------------------------------


@app.get("/reviews")
def reviews_endpoint(request: Request, status: str = Query("pending"), format: str = Query("json")):
    from ..hitl.queue import list_reviews

    rows = list_reviews(None if status == "all" else status)
    if format == "html":
        from .render import render_reviews

        return HTMLResponse(render_reviews(rows))
    return {"reviews": rows, "count": len(rows)}


@app.get("/reviews/{review_id}")
def review_detail(review_id: str) -> dict:
    from ..agent.state import RunState
    from ..hitl.queue import get_review

    try:
        review = get_review(review_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    evidence: list[dict] = []
    try:
        ledger = RunState.load(review["run_id"]).load_ledger()
        evidence = [i.as_dict() for i in ledger.items()]
    except KeyError:
        pass
    return {**review, "evidence": evidence}


@app.post("/reviews/{review_id}/decision")
def decision_endpoint(review_id: str, body: DecisionIn) -> dict:
    from ..hitl.queue import decide

    if body.decision not in ("approved", "edited", "rejected"):
        raise HTTPException(400, "decision must be approved | edited | rejected")
    try:
        return decide(review_id, body.decision, edited_text=body.edited_text, notes=body.notes)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------------


@app.get("/traces/{run_id}")
def trace_endpoint(run_id: str) -> dict:
    from ..obs.tracer import span_tree

    tree = span_tree(run_id)
    if not tree:
        raise HTTPException(404, f"no spans for run {run_id}")
    return {"run_id": run_id, "spans": tree}


@app.get("/health")
def health() -> dict:
    from .health import health_payload

    return health_payload()


@app.get("/")
def root() -> dict:
    cfg = get_config()
    return {
        "service": "dossier",
        "what_this_is": (
            "Agentic due diligence over public SEC filings from the FinanceBench benchmark. "
            "Not a client system, not investment advice."
        ),
        "graph_backend": cfg.graph_backend,
        "endpoints": ["/deals", "/runs/{id}", "/reviews", "/reviews?format=html", "/traces/{id}", "/health"],
    }
