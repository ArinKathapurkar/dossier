"""BM25 keyword store.

Dense retrieval is weak exactly where 10-K questions are strongest: exact line-item names
("Purchases of property, plant and equipment"), ticker symbols, and fiscal-year strings.
BM25 covers that, and fusing the two (retrieve/hybrid.py) beats either alone -- Tier 1
measures by how much.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from ..config import get_config

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Store:
    def __init__(self, path: Path | None = None):
        self.path = path or get_config().paths.bm25
        self.bm25 = None
        self.meta: list[dict] = []

    def build(self, chunks: list[dict]) -> dict:
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.meta = [
            {
                "chunk_id": c["chunk_id"],
                "doc_name": c["doc_name"],
                "company": c["company"],
                "doc_type": c["doc_type"],
                "fiscal_period": str(c["fiscal_period"]),
                "page_num": int(c["page_num"]),
                "text": c["text"],
            }
            for c in chunks
        ]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            pickle.dump({"bm25": self.bm25, "meta": self.meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return {"docs": len(self.meta), "path": str(self.path)}

    def load(self) -> BM25Store:
        with self.path.open("rb") as fh:
            blob = pickle.load(fh)
        self.bm25 = blob["bm25"]
        self.meta = blob["meta"]
        return self

    def exists(self) -> bool:
        return self.path.exists() and self.path.stat().st_size > 0

    def _matches(self, row: dict, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        for key in ("company", "doc_type", "fiscal_period", "doc_name"):
            want = filters.get(key)
            if want is None:
                continue
            got = str(row.get(key, ""))
            if isinstance(want, (list, tuple, set)):
                if got not in {str(w) for w in want}:
                    return False
            elif got != str(want):
                return False
        return True

    def search(self, query: str, k: int = 50, filters: dict | None = None) -> list[dict]:
        if self.bm25 is None:
            self.load()
        scores = self.bm25.get_scores(tokenize(query))
        # Filters are applied post-hoc: BM25Okapi has no index-level predicate, and the
        # corpus is small enough that scoring everything then filtering is cheap.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.meta[i]["chunk_id"]))
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            row = self.meta[i]
            if not self._matches(row, filters):
                continue
            out.append({**row, "score": float(scores[i])})
            if len(out) >= k:
                break
        return out
