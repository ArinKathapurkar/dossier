# mini corpus

Committed CI fixture. Regenerate with `python -m dossier.eval.mini_corpus`.

- documents: AMD_2022_10K, AMERICANEXPRESS_2022_10K, BOEING_2022_10K, PEPSICO_2022_10K
- chunks: 2397
- questions: 20
- embedding model: BAAI/bge-small-en-v1.5 (384-d, normalized)

`embeddings.npy` holds the chunk vectors in the row order of `chunks.json`; `query_embeddings.json` maps each question's text to its query vector; `rerank_scores.json` maps each question to the cross-encoder score of every chunk in its fused candidate set. Together these let CI run the full Tier 1 ablation, reranker row included, with no model download.
