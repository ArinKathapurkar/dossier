"""`dossier ingest` orchestration: manifest -> PDFs -> page text -> chunks -> XBRL facts."""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd

from ..config import get_config
from . import financebench as fb
from . import tickers, xbrl
from .chunk import WhitespaceTokenizer, chunk_pages
from .pdf_text import iter_document_pages


def _embedding_tokenizer():
    """The embedding model's tokenizer, so chunk windows match what the encoder sees.

    Falls back to a whitespace tokenizer when transformers/the model is unavailable
    (offline CI), which keeps ingest runnable at the cost of approximate window sizes.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(get_config().embed_model), "hf"
    except Exception:
        return WhitespaceTokenizer(), "whitespace"


def run_ingest(limit_docs: int | None = None, skip_xbrl: bool = False) -> dict[str, Any]:
    cfg = get_config()
    cfg.paths.ensure()
    report: dict[str, Any] = {"started": time.time()}

    rows = fb.load_dataset_rows()
    docs, questions = fb.build_manifest(rows)
    if limit_docs:
        keep = {d.doc_name for d in docs[:limit_docs]}
        docs = [d for d in docs if d.doc_name in keep]
        questions = [q for q in questions if q.doc_name in keep]
    report["questions"] = len(questions)
    report["documents_in_benchmark"] = len(docs)

    docs = fb.download_pdfs(docs, cfg.paths.raw_pdfs)
    fetched = [d for d in docs if d.local_path]
    report["documents_fetched"] = len(fetched)
    report["fetch_sources"] = {
        s: sum(1 for d in docs if d.source == s) for s in {d.source for d in docs if d.source}
    }
    report["documents_missing"] = [d.doc_name for d in docs if not d.local_path]
    fb.write_manifest(docs, questions)

    # ---- page text -------------------------------------------------------------
    names = [d.doc_name for d in fetched]
    page_rows: list[dict] = []
    extractors: dict[str, int] = {}
    for doc_name, page_num, text, extractor in iter_document_pages(cfg.paths.raw_pdfs, names):
        page_rows.append({"doc_name": doc_name, "page_num": page_num, "text": text})
        extractors[extractor] = extractors.get(extractor, 0) + 1
    report["pages"] = len(page_rows)
    report["extractors"] = extractors
    pd.DataFrame(page_rows).to_parquet(cfg.paths.pages_parquet, index=False)

    # ---- page-index convention check (must run before any indexing) -------------
    pages_by_doc: dict[str, dict[int, str]] = {}
    for r in page_rows:
        pages_by_doc.setdefault(r["doc_name"], {})[r["page_num"]] = r["text"]
    q_dicts = [q.__dict__ for q in questions]
    report["page_convention"] = fb.verify_page_convention(q_dicts, pages_by_doc)

    # ---- chunks -----------------------------------------------------------------
    tok, tok_kind = _embedding_tokenizer()
    report["tokenizer"] = tok_kind
    doc_meta = {d.doc_name: {"company": d.company, "doc_type": d.doc_type, "doc_period": d.doc_period} for d in docs}
    chunks = chunk_pages(
        ((r["doc_name"], r["page_num"], r["text"]) for r in page_rows),
        doc_meta,
        tok,
        window=cfg.chunk_tokens,
        overlap=cfg.chunk_overlap,
    )
    pd.DataFrame([c.as_dict() for c in chunks]).to_parquet(cfg.paths.chunks_parquet, index=False)
    report["chunks"] = len(chunks)

    # ---- XBRL -------------------------------------------------------------------
    if not skip_xbrl:
        companies = sorted({d.company for d in docs})
        cik_map, missing = tickers.build_cik_map(companies)
        report["companies"] = len(companies)
        report["companies_with_cik"] = len(cik_map)
        report["companies_without_cik"] = missing
        report["xbrl"] = xbrl.load_facts(cik_map)
    report["elapsed_s"] = round(time.time() - report.pop("started"), 1)

    (cfg.paths.reports / "ingest.json").write_text(json.dumps(report, indent=2, default=str))
    return report
