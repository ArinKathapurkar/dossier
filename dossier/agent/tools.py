"""Tool schemas and handlers.

Design notes worth stating, because the tool surface is most of what determines whether an
agent behaves:

* **Every retrieval tool writes into the ledger and returns ids, not just text.** The model
  never sees a passage it cannot cite, which makes "cite your evidence" a property of the
  tool contract rather than an instruction it might ignore.
* **`compute` is a whitelisted AST evaluator, not `eval`.** Beyond the obvious safety point,
  it means a derived number carries the evidence ids of its inputs, so a reviewer checking
  "free cash flow was $4.8bn [C1]" can see it came from [F3] - [F7].
* **Tool results are wrapped in `<document>` tags.** Filing text is data. The system prompt
  says so and the wrapper makes the boundary machine-visible.
* **A tool never raises into the loop.** Exceptions become `{error, hint}` results (see
  guard/fallbacks.py) so a bad argument costs one turn instead of the run.
"""

from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from typing import Any

from ..config import get_config
from ..guard.fallbacks import tool_error
from ..obs.tracer import get_tracer

# ---------------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------------

SEARCH_FILINGS = {
    "name": "search_filings",
    "description": (
        "Hybrid search (dense vector + BM25 keyword + entity graph, fused with reciprocal "
        "rank fusion and reranked by a cross-encoder) over the indexed filing corpus. "
        "Returns passages with evidence ids you can cite. Use filters to narrow to one "
        "company, document type, or fiscal year when you know them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for, in natural language."},
            "company": {"type": "string", "description": "Restrict to one company as named in the corpus."},
            "doc_type": {
                "type": "string",
                "enum": ["10k", "10q", "8k", "Earnings"],
                "description": "Document type as stored in this corpus. Spelling variants such as '10-K' are accepted.",
            },
            "fiscal_period": {"type": "string", "description": "Fiscal year as a four-digit string, e.g. '2022'."},
            "k": {"type": "integer", "description": "How many passages to return (default 8, max 15)."},
        },
        "required": ["query"],
    },
}

GRAPH_QUERY = {
    "name": "graph_query",
    "description": (
        "Traverse the entity graph built from the filings' Business and Risk Factors "
        "sections. Use this for questions that cross companies -- suppliers, customers, "
        "competitors, shared risk exposures -- where repeated text search would be slow and "
        "would miss the connection. Returns neighbouring entities and the passages that "
        "established each edge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity to start from, e.g. a company or segment name."},
            "relation": {
                "type": "string",
                "description": "Optional relation filter: COMPETES_WITH, DEPENDS_ON, SELLS_TO, OPERATES_IN, EXPOSED_TO, REGULATED_BY, HAS_SEGMENT, OFFERS.",
            },
            "hops": {"type": "integer", "description": "1 or 2 (default 1)."},
        },
        "required": ["entity"],
    },
}

GET_FINANCIALS = {
    "name": "get_financials",
    "description": (
        "Look up exact XBRL facts from SEC companyfacts for one company. Prefer this over "
        "text search for any specific financial figure -- the values are exact and carry the "
        "form and filing date as provenance."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "concepts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "us-gaap concept names, e.g. Revenues, NetIncomeLoss, Assets, LongTermDebt, NetCashProvidedByUsedInOperatingActivities, PaymentsToAcquirePropertyPlantAndEquipment.",
            },
            "fiscal_years": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["company", "concepts"],
    },
}

COMPUTE = {
    "name": "compute",
    "description": (
        "Evaluate an arithmetic expression over values you have already retrieved. Bind each "
        "input name to an evidence id (F3, C1, ...) so the result stays traceable. Supports "
        "+ - * / ( ), and the helpers pct_change(old, new) and cagr(begin, end, years)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. 'ocf - capex' or 'pct_change(rev2021, rev2022)'."},
            "inputs": {
                "type": "object",
                "description": "Map from a name used in the expression to an evidence id, e.g. {\"ocf\": \"F3\", \"capex\": \"F7\"}.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["expression", "inputs"],
    },
}

COMPARE_PEERS = {
    "name": "compare_peers",
    "description": "Build a peer comparison table for one XBRL concept across several companies for one fiscal year.",
    "input_schema": {
        "type": "object",
        "properties": {
            "concept": {"type": "string"},
            "companies": {"type": "array", "items": {"type": "string"}},
            "fiscal_year": {"type": "integer"},
        },
        "required": ["concept", "companies", "fiscal_year"],
    },
}

REQUEST_HUMAN_REVIEW = {
    "name": "request_human_review",
    "description": (
        "Escalate a draft to a human reviewer and pause the run. Use this when you cannot "
        "ground a claim the section needs, when the evidence conflicts, or when the question "
        "asks for a judgement the filings do not support."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "section": {"type": "string"},
            "reason": {"type": "string", "description": "What specifically needs a human decision."},
            "draft": {"type": "string", "description": "The draft text as it stands."},
        },
        "required": ["section", "reason", "draft"],
    },
}

DRAFT_SECTION = {
    "name": "draft_section",
    "description": "Record a finished memo section with the evidence ids it cites.",
    "input_schema": {
        "type": "object",
        "properties": {
            "section": {"type": "string"},
            "markdown": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["section", "markdown", "evidence_ids"],
    },
}

FINISH = {
    "name": "finish",
    "description": (
        "Return the final answer. Every number in it must be supported by a cited evidence "
        "id. If the corpus does not contain the answer, say so here with an empty evidence "
        "list rather than answering from general knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer_markdown": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["answer_markdown", "evidence_ids"],
    },
}

ASK_TOOLS = [SEARCH_FILINGS, GRAPH_QUERY, GET_FINANCIALS, COMPUTE, COMPARE_PEERS, REQUEST_HUMAN_REVIEW, FINISH]
SECTION_TOOLS = [SEARCH_FILINGS, GRAPH_QUERY, GET_FINANCIALS, COMPUTE, COMPARE_PEERS, REQUEST_HUMAN_REVIEW, DRAFT_SECTION]

TERMINAL_TOOLS = {"finish", "draft_section", "request_human_review"}
# Tools with no side effects on the run's control flow can be dispatched concurrently.
PARALLEL_SAFE = {"search_filings", "graph_query", "get_financials", "compare_peers"}


# ---------------------------------------------------------------------------------
# compute: whitelisted AST evaluator
# ---------------------------------------------------------------------------------

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def pct_change(old: float, new: float) -> float:
    if old == 0:
        raise ValueError("pct_change: base value is zero")
    return (new - old) / old * 100.0


def cagr(begin: float, end: float, years: float) -> float:
    if begin <= 0 or years <= 0:
        raise ValueError("cagr: begin must be positive and years must be > 0")
    return ((end / begin) ** (1.0 / years) - 1.0) * 100.0


_FUNCS = {"pct_change": pct_change, "cagr": cagr, "abs": abs, "min": min, "max": max, "round": round}


def safe_eval(expression: str, variables: dict[str, float]) -> float:
    """Evaluate `expression` over `variables`. Rejects anything not in the whitelist."""
    tree = ast.parse(expression, mode="eval")

    def ev(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"unsupported constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"unknown name {node.id!r}; bind it in `inputs` to an evidence id")
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ValueError("only pct_change, cagr, abs, min, max and round may be called")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression element {type(node).__name__}")

    return ev(tree)


# ---------------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------------


@dataclass
class ToolContext:
    ledger: Any
    deal: Any = None
    run_state: Any = None
    retriever: Any = None
    section: str = ""
    # set by handlers to signal the loop
    finished: dict | None = None
    drafted: dict | None = None
    review_request: dict | None = None
    last_rerank_top: float | None = None
    searches: int = 0


def _wrap_documents(items: list[dict]) -> str:
    parts = []
    for it in items:
        parts.append(f"[{it['id']}] {it['citation']}\n<document>\n{it['text']}\n</document>")
    return "\n\n".join(parts)


def handle_search_filings(args: dict, ctx: ToolContext) -> str:
    cfg = get_config()
    k = min(int(args.get("k") or cfg.rerank_top_n), 15)
    filters = {kk: args[kk] for kk in ("company", "doc_type", "fiscal_period") if args.get(kk)}
    retriever = ctx.retriever
    if retriever is None:
        from ..retrieve.hybrid import get_retriever

        retriever = get_retriever()
    with get_tracer().span(
        "retrieval",
        "search_filings",
        attrs={"query": args.get("query", "")[:120], "k": k, "filters": filters,
               "channels": list(cfg.retrieval_channels), "rerank": cfg.retrieval_rerank},
    ) as span:
        chunks = retriever.search(
            args["query"],
            deal=ctx.deal,
            filters=filters or None,
            channels=cfg.retrieval_channels,
            rerank=cfg.retrieval_rerank,
            top_n=k,
        )
        span.attrs["hits"] = len(chunks)
        span.attrs["channel_hits"] = {c: sum(1 for x in chunks if c in x.channels_hit) for c in ("vector", "bm25", "graph")}
        if chunks:
            span.attrs["top_rerank_score"] = chunks[0].rerank_score
    ctx.searches += 1
    if chunks and chunks[0].rerank_score is not None:
        ctx.last_rerank_top = chunks[0].rerank_score
    if not chunks:
        return "No passages matched. The corpus may not contain this company or period. Consider saying so rather than answering."
    added = [ctx.ledger.add_chunk(c) for c in chunks]
    return _wrap_documents([{"id": a.id, "citation": a.citation, "text": a.text} for a in added])


def handle_graph_query(args: dict, ctx: ToolContext) -> str:
    from ..index.graph_store import normalize_entity

    retriever = ctx.retriever
    if retriever is None:
        from ..retrieve.hybrid import get_retriever

        retriever = get_retriever()
    graph = retriever.graph
    entity = args["entity"]
    hops = max(1, min(int(args.get("hops") or 1), 2))
    rel = args.get("relation")
    seeds = graph.find_entities(entity)
    if not seeds:
        return f"No entity matching {entity!r} is in the graph. The graph only covers Business and Risk Factors sections of the ingested filings."
    expanded = graph.expand(seeds, hops=hops, rel_filter=[rel] if rel else None)
    triples = []
    for eid in list(expanded)[:25]:
        triples.extend(graph.neighbors(eid))
    if rel:
        triples = [t for t in triples if t[1] == rel]
    triples = sorted(set(triples))[:40]
    chunk_ids = graph.chunks_for(list(expanded)[:10])[:6]
    rows = retriever.chunk_meta
    added = []
    for cid in chunk_ids:
        row = rows.get(cid)
        if not row:
            continue
        from ..retrieve.types import RetrievedChunk

        added.append(ctx.ledger.add_chunk(RetrievedChunk.from_row(row, 0.0, ["graph"])))
    lines = [f"{s} -[{r}]-> {d}" for s, r, d in triples] or ["(no edges)"]
    body = "Graph neighbourhood of " + normalize_entity(entity) + ":\n" + "\n".join(lines)
    if added:
        body += "\n\nProvenance passages:\n" + _wrap_documents([{"id": a.id, "citation": a.citation, "text": a.text} for a in added])
    return body


def _fact_rows(company: str, concepts: list[str], years: list[int] | None) -> list[dict]:
    from ..ingest.xbrl import query_facts

    rows = query_facts([company], concepts, years)
    # One row per (concept, fiscal_year): query_facts already orders latest-filing-first,
    # so the first row for a key is the most recently filed (restated) value.
    best: dict[tuple[str, Any], dict] = {}
    for r in rows:
        key = (r["concept"], r["fiscal_year"])
        if key not in best:
            best[key] = r
    return sorted(best.values(), key=lambda r: (r["concept"], -(r["fiscal_year"] or 0)))


def handle_get_financials(args: dict, ctx: ToolContext) -> str:
    company = args["company"]
    concepts = list(args.get("concepts") or [])
    years = [int(y) for y in (args.get("fiscal_years") or [])] or None
    rows = _fact_rows(company, concepts, years)
    if not rows:
        from ..ingest.xbrl import available_companies

        names = available_companies()
        close = [n for n in names if company.lower()[:5] in n.lower()][:5]
        return (
            f"No XBRL facts for company={company!r} concepts={concepts} years={years}. "
            f"{'Did you mean: ' + ', '.join(close) + '. ' if close else ''}"
            f"The facts store covers {len(names)} companies from the benchmark corpus."
        )
    added = [ctx.ledger.add_fact(r) for r in rows[:40]]
    lines = [f"[{a.id}] {a.text}" for a in added]
    return "XBRL facts (source: SEC companyfacts):\n" + "\n".join(lines)


def handle_compare_peers(args: dict, ctx: ToolContext) -> str:
    concept = args["concept"]
    companies = list(args.get("companies") or [])
    year = int(args["fiscal_year"])
    added = []
    missing = []
    for c in companies:
        rows = _fact_rows(c, [concept], [year])
        if not rows:
            missing.append(c)
            continue
        added.append(ctx.ledger.add_fact(rows[0]))
    if not added:
        return f"No facts for {concept} FY{year} across {companies}."
    header = f"| Company | {concept} FY{year} | Unit | Evidence |\n|---|---|---|---|"
    body = "\n".join(
        f"| {a.meta.get('company')} | {a.meta.get('value')} | {a.meta.get('unit')} | [{a.id}] |" for a in added
    )
    note = f"\n\nNo FY{year} value for: {', '.join(missing)}." if missing else ""
    return header + "\n" + body + note


def handle_compute(args: dict, ctx: ToolContext) -> str:
    expression = args["expression"]
    # Sorted, not insertion-ordered. The binding order arrives as JSON object key order from
    # the model, which is incidental -- and it leaked into the computed value's citation text
    # and its recorded input list, so two identical requests could produce different
    # provenance strings. Caught by a cassette replay; a run's evidence text must not depend
    # on how a JSON object happened to be serialised.
    inputs = dict(sorted((args.get("inputs") or {}).items()))
    variables: dict[str, float] = {}
    used_ids: list[str] = []
    for name, eid in inputs.items():
        item = ctx.ledger.get(eid)
        if item is None:
            return f"Evidence id {eid!r} is not in the ledger. Retrieve it first, then bind it."
        value = item.meta.get("value")
        if value is None and item.kind == "computed":
            value = item.meta.get("value")
        if value is None:
            return (
                f"Evidence {eid} is a text passage, not a numeric fact, so it cannot be a compute input. "
                f"Use get_financials for exact figures, or state the number with its [{eid}] citation instead of computing on it."
            )
        try:
            variables[name] = float(value)
        except (TypeError, ValueError):
            return f"Evidence {eid} has a non-numeric value {value!r}."
        used_ids.append(eid)
    try:
        result = safe_eval(expression, variables)
    except Exception as exc:
        return f"compute failed: {exc}"
    # The input list is reported sorted by evidence id, so "derives from [F1, F2]" reads the
    # same regardless of which order the model bound the names in.
    ordered_ids = sorted(set(used_ids), key=lambda i: (i[0], int(i[1:])))
    item = ctx.ledger.add_computed(
        expression,
        round(float(result), 6),
        ordered_ids,
        detail=", ".join(f"{k}={v}" for k, v in variables.items()),
    )
    return f"[{item.id}] {item.text}\nCite this value as [{item.id}]; it derives from {ordered_ids}."


def handle_request_human_review(args: dict, ctx: ToolContext) -> str:
    ctx.review_request = {
        "section": args.get("section") or ctx.section or "answer",
        "reason": args.get("reason", ""),
        "draft": args.get("draft", ""),
    }
    return "Escalated to a human reviewer. The run is now paused; do not continue."


def handle_draft_section(args: dict, ctx: ToolContext) -> str:
    ctx.drafted = {
        "section": args.get("section") or ctx.section,
        "markdown": args.get("markdown", ""),
        "evidence_ids": list(args.get("evidence_ids") or []),
    }
    return "Section recorded."


def handle_finish(args: dict, ctx: ToolContext) -> str:
    ctx.finished = {
        "answer_markdown": args.get("answer_markdown", ""),
        "evidence_ids": list(args.get("evidence_ids") or []),
    }
    return "Answer recorded."


HANDLERS = {
    "search_filings": handle_search_filings,
    "graph_query": handle_graph_query,
    "get_financials": handle_get_financials,
    "compare_peers": handle_compare_peers,
    "compute": handle_compute,
    "request_human_review": handle_request_human_review,
    "draft_section": handle_draft_section,
    "finish": handle_finish,
}


# Tools that read the corpus. These are recorded and replayed (see llm.Cassette) so a
# cassette run reproduces offline; the rest are pure and always execute live.
DATA_TOOLS = {"search_filings", "graph_query", "get_financials", "compare_peers"}


def _ledger_snapshot(ledger) -> set[str]:
    return set(ledger.ids())


def _ledger_items(ledger, ids: list[str]) -> list[dict]:
    return [ledger.get(i).as_dict() for i in ids if ledger.get(i) is not None]


def _restore_ledger_items(ledger, items: list[dict]) -> None:
    """Re-add recorded evidence with its original ids, so citations still resolve."""
    from .ledger import EvidenceItem

    for raw in items:
        if raw["id"] in ledger:
            continue
        item = EvidenceItem(
            id=raw["id"],
            kind=raw["kind"],
            citation=raw["citation"],
            text=raw["text"],
            source_key=raw.get("source_key", ""),
            meta=raw.get("meta", {}),
        )
        ledger._items[item.id] = item
        if item.source_key:
            ledger._by_source[item.source_key] = item.id
        prefix, number = item.id[0], int(item.id[1:])
        ledger._counters[prefix] = max(ledger._counters.get(prefix, 0), number)


def dispatch(name: str, args: dict, ctx: ToolContext) -> str:
    from .llm import active_cassette, llm_mode

    handler = HANDLERS.get(name)
    if handler is None:
        return json.dumps(tool_error(name, ValueError(f"unknown tool {name!r}"), hint=f"Available tools: {sorted(HANDLERS)}"))

    cassette = active_cassette()
    mode = llm_mode()
    recordable = cassette is not None and name in DATA_TOOLS
    tool_key = cassette.tool_key(name, args) if recordable else None

    with get_tracer().span("tool_call", name, attrs={"args": {k: str(v)[:120] for k, v in args.items()}}) as span:
        if recordable and mode == "replay":
            recorded = cassette.get_tool(tool_key)
            if recorded is None:
                from .llm import LLMReplayMiss

                raise LLMReplayMiss(
                    f"cassette {cassette.path.name} has no recorded result for tool call {tool_key}"
                )
            _restore_ledger_items(ctx.ledger, recorded["ledger_adds"])
            if name == "search_filings":
                ctx.searches += 1
                # Emit the retrieval span the live handler would have, so a replayed run's
                # trace has the same shape as the recorded one rather than a hole where the
                # retriever used to be.
                get_tracer().event(
                    "retrieval",
                    "search_filings",
                    attrs={"query": str(args.get("query", ""))[:120], "replayed": True,
                           "hits": len(recorded["ledger_adds"])},
                )
            span.attrs.update({"replayed": True, "result_chars": len(recorded["result"])})
            return recorded["result"]
        try:
            before = _ledger_snapshot(ctx.ledger) if recordable else set()
            out = handler(args, ctx)
            span.attrs["result_chars"] = len(out)
            if recordable and mode == "record":
                added = [i for i in ctx.ledger.ids() if i not in before]
                cassette.put_tool(tool_key, out, _ledger_items(ctx.ledger, added))
            return out
        except Exception as exc:  # noqa: BLE001 - converted to a model-visible result
            span.attrs["error"] = f"{type(exc).__name__}: {exc}"
            out = json.dumps(tool_error(name, exc))
            if recordable and mode == "record":
                cassette.put_tool(tool_key, out, [])
            return out
