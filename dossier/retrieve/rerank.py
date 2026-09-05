"""Cross-encoder reranking.

The first stage optimises recall over 50 candidates; the reranker optimises precision over
the 8 that actually enter the model's context. A cross-encoder sees the query and the
passage together, so it can tell "FY2018 capital expenditure" from "FY2017 capital
expenditure" -- a distinction bi-encoders routinely miss and one that matters on every
metrics-generated FinanceBench question.

It is also the component most likely to be slow or absent (model download, MPS quirks), so
it is wrapped in the same fallback helper as everything else: on failure or timeout the
RRF order is returned and a `fallback` span is emitted.
"""

from __future__ import annotations

import concurrent.futures
import threading

from ..config import get_config
from .types import RetrievedChunk

_MODEL = None
_LOCK = threading.Lock()


def load_reranker():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import CrossEncoder

                from ..index.embed import pick_device

                cfg = get_config()
                _MODEL = CrossEncoder(cfg.rerank_model, device=pick_device(), max_length=512)
    return _MODEL


def reset_reranker() -> None:
    global _MODEL
    _MODEL = None


def _score(query: str, chunks: list[RetrievedChunk]) -> list[float]:
    model = load_reranker()
    pairs = [(query, c.text) for c in chunks]
    return [float(s) for s in model.predict(pairs, show_progress_bar=False)]


def rerank_chunks(query: str, chunks: list[RetrievedChunk], top_n: int = 8) -> list[RetrievedChunk]:
    from ..guard.fallbacks import with_fallback

    cfg = get_config()

    def primary() -> list[RetrievedChunk]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_score, query, chunks)
            scores = future.result(timeout=cfg.rerank_timeout_s)
        for c, s in zip(chunks, scores, strict=True):
            c.rerank_score = s
        return sorted(chunks, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))[:top_n]

    def fallback() -> list[RetrievedChunk]:
        return chunks[:top_n]

    return with_fallback(primary, fallback, span_name="reranker", frm="cross_encoder", to="rrf_order")
