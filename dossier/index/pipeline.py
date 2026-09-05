"""`dossier index` orchestration: embeddings -> LanceDB, BM25, entity graph."""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd

from ..config import get_config
from .bm25_store import BM25Store
from .vector_store import VectorStore


def load_chunks() -> list[dict]:
    cfg = get_config()
    if not cfg.paths.chunks_parquet.exists():
        raise FileNotFoundError("no chunks found -- run `dossier ingest` first")
    return pd.read_parquet(cfg.paths.chunks_parquet).to_dict("records")


def run_index(
    graph_backend: str = "networkx",
    skip_graph: bool = False,
    skip_vectors: bool = False,
    limit_docs: int | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    cfg.paths.ensure()
    started = time.time()
    chunks = load_chunks()
    report: dict[str, Any] = {"chunks": len(chunks)}

    if skip_vectors:
        # Rebuilding the graph does not require re-encoding 32k chunks.
        vs = VectorStore()
        report.update(
            {"device": "skipped", "mps_parity": {"ran": False}, "embed_s": 0.0,
             "vectors": vs.count(), "vector_index_type": "unchanged", "bm25_docs": len(chunks), "bm25_s": 0.0}
        )
    else:
        # ---- vectors -------------------------------------------------------------
        from .embed import Embedder

        embedder = Embedder()
        report["device"] = embedder.device
        report["mps_parity"] = embedder.parity
        t0 = time.time()
        vectors = embedder.encode_passages([c["text"] for c in chunks])
        report["embed_s"] = round(time.time() - t0, 1)
        vs = VectorStore()
        build = vs.build(chunks, vectors)
        report["vectors"] = build["rows"]
        report["vector_index_type"] = build["index_type"]

        # ---- bm25 ----------------------------------------------------------------
        t0 = time.time()
        bm = BM25Store().build(chunks)
        report["bm25_docs"] = bm["docs"]
        report["bm25_s"] = round(time.time() - t0, 1)

    # ---- graph -------------------------------------------------------------------
    if skip_graph:
        report["graph"] = {"skipped": True}
    else:
        from ..ingest.financebench import load_manifest
        from .extract_entities import build_graph
        from .graph_store import open_graph_store

        pages = pd.read_parquet(cfg.paths.pages_parquet)
        pages_by_doc: dict[str, dict[int, str]] = {}
        for r in pages.itertuples():
            pages_by_doc.setdefault(r.doc_name, {})[int(r.page_num)] = r.text
        doc_meta = {d["doc_name"]: d for d in load_manifest()["documents"]}
        chunks_by_doc_page: dict[tuple[str, int], list[str]] = {}
        for c in chunks:
            chunks_by_doc_page.setdefault((c["doc_name"], int(c["page_num"])), []).append(c["chunk_id"])
        store = open_graph_store(graph_backend)
        report["graph"] = build_graph(store, pages_by_doc, doc_meta, chunks_by_doc_page, limit_docs=limit_docs)
        if hasattr(store, "save"):
            store.save()

    report["elapsed_s"] = round(time.time() - started, 1)
    (cfg.paths.reports / "index.json").write_text(json.dumps(report, indent=2, default=str))
    return report
