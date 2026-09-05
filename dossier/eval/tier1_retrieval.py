"""Tier 1 -- retrieval quality. Free, deterministic, runs in CI.

FinanceBench gives, for each question, the page(s) that contain the answer. A retrieved
chunk is relevant if it comes from a gold page of the right document. That makes Recall@k,
nDCG@10 and MRR exact, with no LLM in the loop and no judgement calls.

Two things this reports that a bare recall number would hide:

* **Six configurations, not one.** vector / bm25 / graph alone, then rrf(vector+bm25),
  rrf(all three), rrf(all three)+rerank. An ablation is the only way to say what each
  component is worth, and two of the six turn out to matter much less than the
  architecture diagram implies.
* **Evidence match rate.** The fraction of gold evidence strings that appear in *some*
  chunk of the right page. This is the ceiling on achievable recall: if chunking or PDF
  text extraction lost the passage, no retriever can find it. Reporting recall without it
  attributes an ingestion loss to the retriever.

The gold page offset is measured at ingest (`evidence_page_num` is 0-indexed against our
1-indexed extraction) rather than assumed -- see ingest/financebench.verify_page_convention.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from ..config import get_config

CONFIGS: list[tuple[str, dict]] = [
    ("vector", {"channels": ("vector",), "rerank": False}),
    ("bm25", {"channels": ("bm25",), "rerank": False}),
    ("graph", {"channels": ("graph",), "rerank": False}),
    ("rrf(vector+bm25)", {"channels": ("vector", "bm25"), "rerank": False}),
    ("rrf(all three)", {"channels": ("vector", "bm25", "graph"), "rerank": False}),
    ("rrf(all three)+rerank", {"channels": ("vector", "bm25", "graph"), "rerank": True}),
]


def gold_pages(question: dict, offset: int) -> set[tuple[str, int]]:
    """`{(doc_name, our_page_num)}` for a question's evidence."""
    out = set()
    for ev in question.get("evidence") or []:
        doc = ev.get("doc_name")
        page = ev.get("evidence_page_num")
        if doc and page is not None:
            out.add((doc, int(page) + offset))
    return out


def recall_at_k(hits: list[bool], k: int) -> float:
    return 1.0 if any(hits[:k]) else 0.0


def mrr(hits: list[bool]) -> float:
    for i, h in enumerate(hits, start=1):
        if h:
            return 1.0 / i
    return 0.0


def ndcg_at_k(hits: list[bool], k: int, n_relevant: int) -> float:
    """Binary-gain nDCG. The ideal ranking puts every relevant chunk first; since we do not
    know how many chunks exist on a gold page, the ideal is capped at min(k, n_relevant)."""
    dcg = sum((1.0 / math.log2(i + 1)) for i, h in enumerate(hits[:k], start=1) if h)
    ideal_n = min(k, max(n_relevant, 1))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def evidence_match_rate(questions: list[dict], chunks_by_doc_page: dict[tuple[str, int], list[str]], offset: int) -> dict:
    """Ceiling check: does the gold evidence text actually survive into a chunk?"""
    from rapidfuzz import fuzz

    from ..ingest.pdf_text import normalize_ws

    total = matched = 0
    for q in questions:
        for ev in q.get("evidence") or []:
            doc, page = ev.get("doc_name"), ev.get("evidence_page_num")
            probe = normalize_ws(ev.get("evidence_text", ""))
            if not doc or page is None or len(probe) < 40:
                continue
            total += 1
            texts = chunks_by_doc_page.get((doc, int(page) + offset), [])
            if not texts:
                continue
            head = probe[:200]
            if any(head in t for t in texts):
                matched += 1
                continue
            if any(fuzz.partial_ratio(head, t) >= 90 for t in texts):
                matched += 1
    return {"evidence_rows": total, "matched": matched, "evidence_match_rate": round(matched / total, 4) if total else 0.0}


def _mini_corpus() -> dict:
    """The committed CI fixture: a small chunk set plus a 20-question slice."""
    from ..config import REPO_ROOT

    d = REPO_ROOT / "tests" / "fixtures" / "mini_corpus"
    return {
        "chunks": json.loads((d / "chunks.json").read_text()),
        "questions": json.loads((d / "questions.json").read_text()),
        "embeddings": d / "embeddings.npy",
        "rerank_scores": json.loads((d / "rerank_scores.json").read_text()) if (d / "rerank_scores.json").exists() else {},
    }


def _build_mini_retriever(mini: dict):
    """Retriever over the fixture, with pre-computed embeddings so CI needs no model."""
    import numpy as np

    from ..index.bm25_store import BM25Store
    from ..retrieve.hybrid import HybridRetriever

    chunks = mini["chunks"]
    vectors = np.load(mini["embeddings"])
    queries = json.loads((mini["embeddings"].parent / "query_embeddings.json").read_text())

    class _FixtureVectorStore:
        def search(self, query_vec, k=50, filters=None):
            sims = vectors @ np.asarray(query_vec, dtype=np.float32)
            order = np.argsort(-sims)[: k * 2]
            out = []
            for i in order:
                row = chunks[int(i)]
                if filters and not bm._matches(row, filters):
                    continue
                out.append({**row, "score": float(sims[int(i)])})
                if len(out) >= k:
                    break
            return out

    class _FixtureEmbedder:
        device = "fixture"

        def encode_query(self, query: str):
            key = query.strip()
            if key not in queries:
                raise KeyError(f"mini corpus has no pre-computed embedding for query {key[:60]!r}")
            return np.asarray(queries[key], dtype=np.float32)

    bm = BM25Store()
    bm.build_in_memory = True
    from rank_bm25 import BM25Okapi

    from ..index.bm25_store import tokenize

    bm.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    bm.meta = chunks

    class _EmptyGraph:
        def find_entities(self, *a, **k):
            return []

        def expand(self, *a, **k):
            return {}

        def chunks_for(self, *a, **k):
            return []

        def degree(self, *a, **k):
            return 1

    return HybridRetriever(vector_store=_FixtureVectorStore(), bm25_store=bm, graph_store=_EmptyGraph(), embedder=_FixtureEmbedder())


def _install_fixture_reranker(mini: dict):
    """Replace the cross-encoder with the committed scores, so CI runs the rerank row too.

    Returns a restore callable. Only the fixture's own questions have scores; anything else
    raises, which keeps this from silently masking a real reranker failure elsewhere.
    """
    from ..retrieve import rerank

    scores = mini.get("rerank_scores") or {}
    original = rerank._score

    def fixture_score(query: str, chunks):
        table = scores.get(query.strip())
        if table is None:
            raise KeyError(f"mini corpus has no pre-computed reranker scores for {query[:60]!r}")
        return [table.get(c.chunk_id, -12.0) for c in chunks]

    rerank._score = fixture_score
    return lambda: setattr(rerank, "_score", original)


def run_tier1(limit: int | None = None, mini: bool = False, configs: list[tuple[str, dict]] | None = None) -> dict:
    cfg = get_config()
    offset = cfg.gold_page_offset
    started = time.time()

    if mini:
        data = _mini_corpus()
        questions = data["questions"]
        chunk_rows = data["chunks"]
        retriever = _build_mini_retriever(data)
        restore_rerank = _install_fixture_reranker(data)
        rerank_available = bool(data.get("rerank_scores"))
    else:
        import pandas as pd

        from ..ingest.financebench import load_questions
        from ..retrieve.hybrid import HybridRetriever

        questions = load_questions()
        chunk_rows = pd.read_parquet(cfg.paths.chunks_parquet).to_dict("records")
        retriever = HybridRetriever()
        restore_rerank = None
        rerank_available = True

    ingested_docs = {c["doc_name"] for c in chunk_rows}
    scored_questions = [q for q in questions if q["doc_name"] in ingested_docs]
    skipped = len(questions) - len(scored_questions)
    if limit:
        scored_questions = scored_questions[:limit]

    chunks_by_doc_page: dict[tuple[str, int], list[str]] = {}
    page_of_chunk: dict[str, tuple[str, int]] = {}
    for c in chunk_rows:
        key = (c["doc_name"], int(c["page_num"]))
        chunks_by_doc_page.setdefault(key, []).append(c["text"])
        page_of_chunk[c["chunk_id"]] = key

    report: dict[str, Any] = {
        "mode": "mini" if mini else "full",
        "questions_in_benchmark": len(questions),
        "questions_scored": len(scored_questions),
        "questions_skipped_no_document": skipped,
        "documents_ingested": len(ingested_docs),
        "gold_page_offset": offset,
        "chunks": len(chunk_rows),
        **evidence_match_rate(scored_questions, chunks_by_doc_page, offset),
        "configs": {},
    }

    latencies: dict[str, list[float]] = {}
    rerank_latencies: list[float] = []

    for name, spec in configs or CONFIGS:
        if spec["rerank"] and not rerank_available:
            continue
        r5 = r10 = ndcg = rr = 0.0
        n = 0
        lat: list[float] = []
        for q in scored_questions:
            gold = gold_pages(q, offset)
            if not gold:
                continue
            n += 1
            t0 = time.time()
            try:
                results = retriever.search(
                    q["question"],
                    deal=None,
                    k=cfg.retrieve_k,
                    channels=spec["channels"],
                    rerank=spec["rerank"],
                    top_n=cfg.retrieve_k if not spec["rerank"] else 10,
                )
            except KeyError:
                # mini corpus has no embedding for this query; skip rather than fake one
                n -= 1
                continue
            elapsed = time.time() - t0
            lat.append(elapsed)
            if spec["rerank"]:
                rerank_latencies.append(elapsed)
            hits = [page_of_chunk.get(c.chunk_id) in gold for c in results]
            r5 += recall_at_k(hits, 5)
            r10 += recall_at_k(hits, 10)
            ndcg += ndcg_at_k(hits, 10, n_relevant=sum(len(chunks_by_doc_page.get(g, [])) for g in gold))
            rr += mrr(hits)
        latencies[name] = lat
        report["configs"][name] = {
            "n": n,
            "recall@5": round(r5 / n, 4) if n else 0.0,
            "recall@10": round(r10 / n, 4) if n else 0.0,
            "ndcg@10": round(ndcg / n, 4) if n else 0.0,
            "mrr": round(rr / n, 4) if n else 0.0,
            "latency_p50_ms": round(_pct(lat, 50) * 1000, 1),
            "latency_p95_ms": round(_pct(lat, 95) * 1000, 1),
        }

    if rerank_latencies:
        report["rerank_latency_p50_ms"] = round(_pct(rerank_latencies, 50) * 1000, 1)
        report["rerank_latency_p95_ms"] = round(_pct(rerank_latencies, 95) * 1000, 1)
    if restore_rerank is not None:
        restore_rerank()
    report["elapsed_s"] = round(time.time() - started, 1)
    return report


def _pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((p / 100) * (len(s) - 1)))))
    return s[idx]
