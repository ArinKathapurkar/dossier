"""LLM entity/relation extraction over the narrative sections of each filing.

Cost control shapes this module more than anything else. Extracting over 32k chunks with a
frontier model would cost more than the rest of the project put together, so:

  * only the *narrative* pages are extracted -- Item 1 (Business) and Item 1A (Risk
    Factors) for 10-Ks, MD&A for 10-Q/8-K. Financial statement pages contribute numbers,
    which the XBRL store already holds exactly;
  * the cheap model tier does the extraction, with a **forced tool schema** so the output is
    structurally valid without a parsing retry loop;
  * every raw extraction is cached to `data/index/extractions/<doc>.json` keyed by page, so
    re-running `dossier index` costs nothing and a schema change can be replayed offline.

Cost and wall time are reported, because "we built a knowledge graph" means nothing without
what it cost to build.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from ..config import get_config
from ..obs.prompts import registry
from .graph_store import ENTITY_TYPES, RELATION_TYPES, Entity, GraphStore, Relation

EMIT_GRAPH_TOOL = {
    "name": "emit_graph",
    "description": "Return the entities and relations found in this passage.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False,
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "rel": {"type": "string", "enum": list(RELATION_TYPES)},
                        "dst": {"type": "string"},
                    },
                    "required": ["src", "rel", "dst"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities", "relations"],
        "additionalProperties": False,
    },
}

_ITEM1 = re.compile(r"item\s*1\s*[.\-:—]?\s*business", re.I)
_ITEM1A = re.compile(r"item\s*1a\s*[.\-:—]?\s*risk\s*factors", re.I)
_ITEM7 = re.compile(r"item\s*[27]\s*[.\-:—]?\s*management.s\s+discussion", re.I)
_MDNA = re.compile(r"management.s\s+discussion\s+and\s+analysis", re.I)


def narrative_pages(doc_type: str, pages: dict[int, str], max_pages: int = 25) -> list[int]:
    """Heuristically pick the pages worth extracting from.

    10-K: everything from the Item 1 header to the Item 1A section end, capped. If no header
    is found (scanned or unusually formatted filings), fall back to the first 25 pages, which
    is where Business and Risk Factors sit in essentially every 10-K.
    10-Q / 8-K / earnings: the MD&A pages.
    """
    dt = (doc_type or "").upper()
    ordered = sorted(pages)
    if not ordered:
        return []
    if "10K" in dt.replace("-", "") or "10-K" in dt:
        starts = [p for p in ordered if _ITEM1.search(pages[p] or "")]
        ends = [p for p in ordered if _ITEM1A.search(pages[p] or "")]
        if starts:
            begin = starts[-1] if len(starts) > 1 else starts[0]
            stop = next((e for e in ends if e > begin), begin + max_pages)
            window = [p for p in ordered if begin <= p <= min(stop + 12, begin + max_pages)]
            if window:
                return window[:max_pages]
        return ordered[:max_pages]
    hits = [p for p in ordered if _ITEM7.search(pages[p] or "") or _MDNA.search(pages[p] or "")]
    if hits:
        begin = hits[0]
        return [p for p in ordered if begin <= p < begin + 12][: max_pages // 2]
    return ordered[: max_pages // 2]


def _cache_path(doc_name: str) -> Path:
    return get_config().paths.extractions / f"{doc_name}.json"


def load_cache(doc_name: str) -> dict:
    p = _cache_path(doc_name)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(doc_name: str, cache: dict) -> None:
    p = _cache_path(doc_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, indent=1, sort_keys=True))


def extract_page(text: str, model: str | None = None) -> tuple[dict, float, int, int]:
    """Call the model once for one page. Returns (payload, cost_usd, tokens_in, tokens_out)."""
    from ..agent.llm import complete

    cfg = get_config()
    system, version = registry().get("entity_extraction")
    resp = complete(
        system=system,
        messages=[{"role": "user", "content": f"<document>\n{text[:12000]}\n</document>"}],
        tools=[EMIT_GRAPH_TOOL],
        tool_choice={"type": "tool", "name": "emit_graph"},
        model=model or cfg.cheap_model,
        max_tokens=2048,
        prompt_name="entity_extraction",
        prompt_version=version,
        effort=None,
    )
    payload = {"entities": [], "relations": []}
    for block in resp.tool_uses():
        if block.get("name") == "emit_graph":
            payload = block.get("input") or payload
            break
    return payload, resp.cost_usd, resp.tokens_in, resp.tokens_out


def build_graph(
    store: GraphStore,
    pages_by_doc: dict[str, dict[int, str]],
    doc_meta: dict[str, dict],
    chunks_by_doc_page: dict[tuple[str, int], list[str]],
    limit_docs: int | None = None,
    max_pages_per_doc: int = 25,
    use_cache: bool = True,
) -> dict:
    """Extract over every document's narrative pages and upsert into the graph store."""
    started = time.time()
    total_cost = 0.0
    calls = 0
    cached_hits = 0
    tokens_in = tokens_out = 0
    docs = sorted(pages_by_doc)[: limit_docs or None]
    for doc_name in docs:
        pages = pages_by_doc[doc_name]
        meta = doc_meta.get(doc_name, {})
        targets = narrative_pages(meta.get("doc_type", ""), pages, max_pages=max_pages_per_doc)
        cache = load_cache(doc_name) if use_cache else {}
        dirty = False
        entities: list[Entity] = []
        relations: list[Relation] = []
        for page in targets:
            text = (pages.get(page) or "").strip()
            if len(text) < 400:
                continue
            key = str(page)
            if key in cache:
                payload = cache[key]
                cached_hits += 1
            else:
                try:
                    payload, cost, tin, tout = extract_page(text)
                except Exception:
                    # A single page failing must not abort a 25-page document.
                    continue
                total_cost += cost
                tokens_in += tin
                tokens_out += tout
                calls += 1
                cache[key] = payload
                dirty = True
            provenance = chunks_by_doc_page.get((doc_name, page), [])
            for e in payload.get("entities", []) or []:
                if not e.get("name"):
                    continue
                entities.append(Entity(name=e["name"], type=e.get("type", ""), chunk_ids=list(provenance)))
            for r in payload.get("relations", []) or []:
                if not (r.get("src") and r.get("dst") and r.get("rel")):
                    continue
                relations.append(Relation(src=r["src"], rel=r["rel"], dst=r["dst"], chunk_ids=list(provenance)))
        if dirty:
            save_cache(doc_name, cache)
        if entities or relations:
            store.upsert(entities, relations)
    stats = store.stats()
    return {
        **stats,
        "documents": len(docs),
        "llm_calls": calls,
        "cached_pages": cached_hits,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(total_cost, 6),
        "elapsed_s": round(time.time() - started, 1),
    }
