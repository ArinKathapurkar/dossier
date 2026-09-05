"""Page-level text extraction.

Page granularity is deliberate: FinanceBench's ground truth is a page number, so keeping
pages as the atomic unit means retrieval can be scored exactly, with no alignment step
that could itself introduce error.

pypdf is the primary extractor. Some filings (image-heavy or unusual producers) yield
empty pages under pypdf, so any document with a high empty-page ratio is re-extracted
with PyMuPDF, which uses a different text engine.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def _extract_pypdf(path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return out


def _extract_pymupdf(path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(str(path)) as doc:
        return [page.get_text() or "" for page in doc]


def extract_pages(path: Path, empty_ratio_threshold: float = 0.25) -> tuple[list[str], str]:
    """Return (page_texts, extractor_used). Page i of the list is page i+1 of the PDF."""
    try:
        pages = _extract_pypdf(path)
        extractor = "pypdf"
    except Exception:
        pages, extractor = [], "pypdf-failed"

    empty = sum(1 for p in pages if len(normalize_ws(p)) < 20)
    if not pages or (empty / max(len(pages), 1)) > empty_ratio_threshold:
        try:
            alt = _extract_pymupdf(path)
            alt_empty = sum(1 for p in alt if len(normalize_ws(p)) < 20)
            if alt and (not pages or alt_empty < empty):
                return alt, "pymupdf"
        except Exception:
            pass
    return pages, extractor


def iter_document_pages(pdf_dir: Path, doc_names: list[str]) -> Iterator[tuple[str, int, str, str]]:
    """Yield `(doc_name, page_num_1_indexed, text, extractor)` for every page."""
    for name in doc_names:
        path = pdf_dir / f"{name}.pdf"
        if not path.exists():
            continue
        pages, extractor = extract_pages(path)
        for i, text in enumerate(pages, start=1):
            yield name, i, text, extractor
