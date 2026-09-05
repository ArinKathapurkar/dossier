"""Hybrid retrieval: vector + BM25 + graph, fused with reciprocal rank fusion.

Why RRF rather than a weighted score blend: the three channels produce scores on
incomparable scales (cosine similarity in [-1,1], unbounded BM25, a hop-distance
heuristic). Normalising them requires per-corpus calibration that drifts. RRF uses only
the *rank* within each channel, so adding or removing a channel never requires retuning
weights -- which is what made the Tier 1 ablation table (§7.1) cheap to produce.

    rrf(d) = sum over channels of 1 / (k + rank_c(d)),  k = 60

Ties are broken on chunk_id so the fused order is byte-stable run to run.
"""

from __future__ import annotations

from typing import Any

from ..config import get_config
from ..index.bm25_store import BM25Store
from ..index.vector_store import VectorStore
from .types import RetrievedChunk

CHANNELS = ("vector", "bm25", "graph")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]], k: int = 60, weights: dict[str, float] | None = None
) -> list[tuple[str, float, list[str]]]:
    """Fuse `{channel: [doc_id ordered best-first]}` into `[(doc_id, score, channels)]`."""
    scores: dict[str, float] = {}
    hits: dict[str, list[str]] = {}
    for channel, ids in ranked_lists.items():
        w = (weights or {}).get(channel, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            hits.setdefault(doc_id, []).append(channel)
    return sorted(
        ((d, s, hits[d]) for d, s in scores.items()),
        key=lambda t: (-t[1], t[0]),
    )


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        graph_store: Any | None = None,
        embedder: Any | None = None,
    ):
        self.cfg = get_config()
        self._vs = vector_store
        self._bm = bm25_store
        self._graph = graph_store
        self._embedder = embedder
        self._chunk_meta: dict[str, dict] | None = None

    # -- lazy components ----------------------------------------------------------
    @property
    def vector_store(self) -> VectorStore:
        if self._vs is None:
            self._vs = VectorStore()
        return self._vs

    @property
    def bm25(self) -> BM25Store:
        if self._bm is None:
            self._bm = BM25Store()
            if self._bm.exists():
                self._bm.load()
        return self._bm

    @property
    def graph(self):
        if self._graph is None:
            from ..index.graph_store import open_graph_store

            self._graph = open_graph_store()
        return self._graph

    @property
    def embedder(self):
        if self._embedder is None:
            from ..index.embed import Embedder

            self._embedder = Embedder()
        return self._embedder

    @property
    def chunk_meta(self) -> dict[str, dict]:
        """chunk_id -> row. Sourced from the BM25 store, which already holds every chunk
        with its metadata, so graph-only hits can be resolved without a vector lookup."""
        if self._chunk_meta is None:
            self._chunk_meta = {m["chunk_id"]: m for m in self.bm25.meta}
        return self._chunk_meta

    # -- channels ------------------------------------------------------------------
    def _vector(self, query: str, k: int, filters: dict | None) -> list[dict]:
        from ..guard.fallbacks import with_fallback

        def primary() -> list[dict]:
            return self.vector_store.search(self.embedder.encode_query(query), k=k, filters=filters)

        def fallback() -> list[dict]:
            # LanceDB unavailable is survivable: BM25 alone still answers keyword-shaped
            # questions, which is most of FinanceBench.
            return self.bm25.search(query, k=k, filters=filters)

        return with_fallback(primary, fallback, span_name="vector_store", frm="lancedb", to="bm25")

    def _bm25(self, query: str, k: int, filters: dict | None) -> list[dict]:
        return self.bm25.search(query, k=k, filters=filters)

    def _graph_channel(self, query: str, k: int, deal: Any | None, filters: dict | None) -> list[dict]:
        """Entities named in the query (plus the deal's target and peers) -> 1-2 hop
        expansion -> the provenance chunks recorded on those nodes and edges.

        Scoring is (hop distance, degree-normalised): a neighbour reached in one hop from a
        low-degree node is a more specific signal than one reached from a hub.
        """
        seeds: list[str] = []
        try:
            seeds.extend(self.graph.find_entities(query))
            if deal is not None:
                for name in [getattr(deal, "target", None), *getattr(deal, "peers", [])]:
                    if name:
                        seeds.extend(self.graph.find_entities(name, limit=3))
        except Exception:
            return []
        seeds = list(dict.fromkeys(s for s in seeds if s))
        if not seeds:
            return []
        expanded = self.graph.expand(seeds, hops=self.cfg.graph_hops)
        scored: dict[str, float] = {}
        for eid, hop in expanded.items():
            deg = max(self.graph.degree(eid), 1)
            weight = (1.0 / (1 + hop)) * (1.0 / (1 + deg) ** 0.5)
            for cid in self.graph.chunks_for([eid]):
                scored[cid] = max(scored.get(cid, 0.0), weight)
        rows: list[dict] = []
        for cid, score in sorted(scored.items(), key=lambda t: (-t[1], t[0])):
            row = self.chunk_meta.get(cid)
            if row is None:
                continue
            if filters and not self.bm25._matches(row, filters):
                continue
            rows.append({**row, "score": score})
            if len(rows) >= k:
                break
        return rows

    # -- public --------------------------------------------------------------------
    def search(
        self,
        query: str,
        deal: Any | None = None,
        k: int | None = None,
        channels: tuple[str, ...] | set[str] = CHANNELS,
        filters: dict | None = None,
        rerank: bool = False,
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        cfg = self.cfg
        k = k or cfg.retrieve_k
        channels = tuple(c for c in CHANNELS if c in set(channels))
        per_channel: dict[str, list[dict]] = {}
        if "vector" in channels:
            per_channel["vector"] = self._vector(query, k, filters)
        if "bm25" in channels:
            per_channel["bm25"] = self._bm25(query, k, filters)
        if "graph" in channels:
            per_channel["graph"] = self._graph_channel(query, k, deal, filters)

        rows_by_id: dict[str, dict] = {}
        for rows in per_channel.values():
            for r in rows:
                rows_by_id.setdefault(r["chunk_id"], r)

        if len(per_channel) == 1:
            only = next(iter(per_channel))
            fused = [(r["chunk_id"], float(r["score"]), [only]) for r in per_channel[only]]
        else:
            fused = reciprocal_rank_fusion(
                {c: [r["chunk_id"] for r in rows] for c, rows in per_channel.items()}, k=cfg.rrf_k
            )

        out = [
            RetrievedChunk.from_row(rows_by_id[cid], score, chans)
            for cid, score, chans in fused[:k]
            if cid in rows_by_id
        ]
        if rerank and out:
            from .rerank import rerank_chunks

            out = rerank_chunks(query, out, top_n=top_n or cfg.rerank_top_n)
        elif top_n:
            out = out[:top_n]
        return out


_RETRIEVER: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def reset_retriever(retriever: HybridRetriever | None = None) -> None:
    global _RETRIEVER
    _RETRIEVER = retriever


def search(query: str, deal: Any | None = None, **kwargs) -> list[RetrievedChunk]:
    return get_retriever().search(query, deal=deal, **kwargs)
