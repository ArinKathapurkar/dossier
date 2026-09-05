"""LanceDB vector store.

LanceDB is embedded (no server, no hosted dependency), stores the chunk metadata columns
alongside the vector so filters are a WHERE clause rather than a post-hoc pass, and
persists to a directory that can be committed as a CI fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_config

TABLE = "chunks"
# Below this row count an IVF-PQ index costs more in recall than it saves in latency; a
# flat scan over a few hundred thousand 384-d vectors is a few milliseconds.
IVF_MIN_ROWS = 100_000


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def build_filter(filters: dict[str, Any] | None) -> str | None:
    if not filters:
        return None
    clauses = []
    for key in ("company", "doc_type", "fiscal_period", "doc_name"):
        val = filters.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple, set)):
            if not val:
                continue
            joined = ", ".join(f"'{_sql_escape(str(v))}'" for v in val)
            clauses.append(f"{key} IN ({joined})")
        else:
            clauses.append(f"{key} = '{_sql_escape(str(val))}'")
    return " AND ".join(clauses) if clauses else None


class VectorStore:
    def __init__(self, path: Path | None = None):
        self.path = path or get_config().paths.lancedb
        self._db = None
        self._table = None

    def _connect(self):
        if self._db is None:
            import lancedb

            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.path))
        return self._db

    @property
    def table(self):
        if self._table is None:
            self._table = self._connect().open_table(TABLE)
        return self._table

    def exists(self) -> bool:
        try:
            return TABLE in self._connect().table_names()
        except Exception:
            return False

    def build(self, chunks: list[dict], vectors: np.ndarray) -> dict:
        db = self._connect()
        rows = []
        for c, v in zip(chunks, vectors, strict=True):
            rows.append(
                {
                    "chunk_id": c["chunk_id"],
                    "doc_name": c["doc_name"],
                    "company": c["company"],
                    "doc_type": c["doc_type"],
                    "fiscal_period": str(c["fiscal_period"]),
                    "page_num": int(c["page_num"]),
                    "text": c["text"],
                    "vector": np.asarray(v, dtype=np.float32),
                }
            )
        if TABLE in db.table_names():
            db.drop_table(TABLE)
        tbl = db.create_table(TABLE, data=rows)
        index_type = "flat"
        if len(rows) >= IVF_MIN_ROWS:
            try:
                tbl.create_index(metric="cosine", num_partitions=256, num_sub_vectors=48)
                index_type = "ivf_pq"
            except Exception:
                index_type = "flat (index build failed)"
        self._table = tbl
        return {"rows": len(rows), "index_type": index_type, "path": str(self.path)}

    def count(self) -> int:
        try:
            return self.table.count_rows()
        except Exception:
            return 0

    def search(self, query_vec: np.ndarray, k: int = 50, filters: dict | None = None) -> list[dict]:
        q = self.table.search(np.asarray(query_vec, dtype=np.float32)).metric("cosine").limit(k)
        where = build_filter(filters)
        if where:
            q = q.where(where, prefilter=True)
        out = []
        for row in q.to_list():
            row = dict(row)
            row.pop("vector", None)
            # LanceDB reports cosine *distance*; convert so bigger is better everywhere.
            row["score"] = 1.0 - float(row.pop("_distance", 0.0))
            out.append(row)
        return out

    def get(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        joined = ", ".join(f"'{_sql_escape(c)}'" for c in chunk_ids)
        rows = self.table.search().where(f"chunk_id IN ({joined})").limit(len(chunk_ids)).to_list()
        out = []
        for r in rows:
            r = dict(r)
            r.pop("vector", None)
            out.append(r)
        return out
