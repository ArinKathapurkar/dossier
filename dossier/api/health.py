"""Health payload: what is actually loaded, not just whether the process is up."""

from __future__ import annotations

from ..config import MODEL_PRICES, get_config


def health_payload() -> dict:
    cfg = get_config()
    out: dict = {
        "status": "ok",
        "models": {
            "primary": cfg.primary_model,
            "fallback": cfg.fallback_model,
            "cheap": cfg.cheap_model,
            "judge": cfg.judge_model,
        },
        "prices_usd_per_mtok": {
            m: {"input": p[0], "output": p[1]}
            for m, p in MODEL_PRICES.items()
            if m in {cfg.primary_model, cfg.fallback_model, cfg.cheap_model, cfg.judge_model}
        },
        "index": {},
        "graph": {},
    }
    try:
        from ..index.vector_store import VectorStore

        out["index"]["vectors"] = VectorStore().count()
    except Exception as exc:
        out["index"]["vectors"] = f"unavailable: {type(exc).__name__}"
    try:
        from ..index.bm25_store import BM25Store

        bm = BM25Store()
        out["index"]["bm25"] = len(bm.load().meta) if bm.exists() else 0
    except Exception as exc:
        out["index"]["bm25"] = f"unavailable: {type(exc).__name__}"
    try:
        from ..ingest.xbrl import connect

        conn = connect()
        out["index"]["xbrl_facts"] = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        out["index"]["companies_with_facts"] = conn.execute("SELECT COUNT(DISTINCT company) FROM facts").fetchone()[0]
        conn.close()
    except Exception as exc:
        out["index"]["xbrl_facts"] = f"unavailable: {type(exc).__name__}"
    try:
        from ..index.graph_store import open_graph_store

        out["graph"] = open_graph_store(cfg.graph_backend).stats()
    except Exception as exc:
        out["graph"] = {"backend": "unavailable", "error": type(exc).__name__, "detail": str(exc)[:120]}
    try:
        from ..ingest.financebench import load_manifest

        m = load_manifest()
        out["corpus"] = m["counts"]
    except Exception:
        out["corpus"] = {"documents": 0}
    try:
        from ..hitl.queue import list_reviews

        out["pending_reviews"] = len(list_reviews("pending"))
    except Exception:
        out["pending_reviews"] = 0
    return out
