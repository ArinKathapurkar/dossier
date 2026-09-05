"""Build the committed CI fixture: a small corpus with pre-computed embeddings.

CI has no GPU, no model cache, and a wall-clock budget. Downloading bge-small and encoding
even a slice of the corpus on every push would dominate the job. So the fixture ships the
vectors themselves: three documents' chunks, a 20-question slice of FinanceBench, the
chunk embeddings as a `.npy`, and the query embeddings keyed by question text.

That makes Tier 1 in CI a pure numpy dot product against committed arrays -- deterministic,
seconds long, and a real regression gate rather than a smoke test.

`python -m dossier.eval.mini_corpus` regenerates it from the full index.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import REPO_ROOT, get_config

OUT = REPO_ROOT / "tests" / "fixtures" / "mini_corpus"
# The four documents carrying the most FinanceBench questions, across four sectors
# (semiconductors, financial services, aerospace, consumer staples). Chosen for question
# density: a fixture with two questions per document could not detect a regression.
DEFAULT_DOCS = ("AMD_2022_10K", "AMERICANEXPRESS_2022_10K", "BOEING_2022_10K", "PEPSICO_2022_10K")


def build(doc_names: tuple[str, ...] = DEFAULT_DOCS, max_questions: int = 20) -> dict:
    from ..index.embed import Embedder
    from ..ingest.financebench import load_questions

    cfg = get_config()
    OUT.mkdir(parents=True, exist_ok=True)

    chunks = pd.read_parquet(cfg.paths.chunks_parquet)
    chunks = chunks[chunks.doc_name.isin(doc_names)].sort_values(["doc_name", "page_num", "chunk_id"])
    rows = [
        {
            "chunk_id": r.chunk_id,
            "doc_name": r.doc_name,
            "company": r.company,
            "doc_type": r.doc_type,
            "fiscal_period": str(r.fiscal_period),
            "page_num": int(r.page_num),
            "text": r.text,
        }
        for r in chunks.itertuples()
    ]

    questions = [q for q in load_questions() if q["doc_name"] in doc_names][:max_questions]

    embedder = Embedder()
    vectors = embedder.encode_passages([r["text"] for r in rows])
    query_vecs = {q["question"]: embedder.encode_query(q["question"]).tolist() for q in questions}

    # Reranker scores are pre-computed too, so CI can run the *full* six-config ablation
    # rather than skipping the row that matters most. Only the fused candidate set for each
    # question is scored -- the reranker never sees more than that at query time either.
    rerank_scores = _precompute_rerank(rows, vectors, questions, query_vecs)

    (OUT / "chunks.json").write_text(json.dumps(rows))
    (OUT / "questions.json").write_text(json.dumps(questions, indent=1))
    np.save(OUT / "embeddings.npy", vectors.astype(np.float32))
    (OUT / "query_embeddings.json").write_text(json.dumps(query_vecs))
    (OUT / "rerank_scores.json").write_text(json.dumps(rerank_scores))
    (OUT / "README.md").write_text(
        "# mini corpus\n\n"
        "Committed CI fixture. Regenerate with `python -m dossier.eval.mini_corpus`.\n\n"
        f"- documents: {', '.join(doc_names)}\n"
        f"- chunks: {len(rows)}\n"
        f"- questions: {len(questions)}\n"
        f"- embedding model: {cfg.embed_model} ({cfg.embed_dim}-d, normalized)\n\n"
        "`embeddings.npy` holds the chunk vectors in the row order of `chunks.json`; "
        "`query_embeddings.json` maps each question's text to its query vector; "
        "`rerank_scores.json` maps each question to the cross-encoder score of every chunk "
        "in its fused candidate set. Together these let CI run the full Tier 1 ablation, "
        "reranker row included, with no model download.\n"
    )
    return {
        "documents": list(doc_names),
        "chunks": len(rows),
        "questions": len(questions),
        "vector_shape": list(vectors.shape),
        "rerank_pairs": sum(len(v) for v in rerank_scores.values()),
        "dir": str(OUT),
    }


def _precompute_rerank(rows, vectors, questions, query_vecs) -> dict:
    """Score the fused candidate set for each question with the real cross-encoder."""
    from rank_bm25 import BM25Okapi

    from ..config import get_config as _cfg
    from ..index.bm25_store import tokenize
    from ..retrieve.hybrid import reciprocal_rank_fusion
    from ..retrieve.rerank import load_reranker

    cfg = _cfg()
    bm = BM25Okapi([tokenize(r["text"]) for r in rows])
    by_index = {i: r["chunk_id"] for i, r in enumerate(rows)}
    text_by_id = {r["chunk_id"]: r["text"] for r in rows}
    model = load_reranker()

    out: dict[str, dict[str, float]] = {}
    for q in questions:
        question = q["question"]
        qv = np.asarray(query_vecs[question], dtype=np.float32)
        sims = vectors @ qv
        vec_ids = [by_index[int(i)] for i in np.argsort(-sims)[: cfg.retrieve_k]]
        bm_scores = bm.get_scores(tokenize(question))
        bm_ids = [by_index[int(i)] for i in np.argsort(-bm_scores)[: cfg.retrieve_k] if bm_scores[int(i)] > 0]
        fused = reciprocal_rank_fusion({"vector": vec_ids, "bm25": bm_ids}, k=cfg.rrf_k)
        candidates = [cid for cid, _, _ in fused[: cfg.retrieve_k]]
        pairs = [(question, text_by_id[c]) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False) if pairs else []
        out[question] = {c: float(s) for c, s in zip(candidates, scores, strict=True)}
    return out


def main() -> None:
    rep = build()
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
