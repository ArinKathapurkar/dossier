"""Chunking.

Chunks never cross a page boundary. That costs a little context at page edges but keeps
the mapping chunk -> page exact, which is what lets Tier 1 score retrieval against
FinanceBench's page-level ground truth without any fuzzy alignment.

Windows are measured in *embedding-model tokens*, not characters or words, so a chunk is
never silently truncated by the encoder.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from .pdf_text import normalize_ws


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> Sequence[int]: ...

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_name: str
    company: str
    doc_type: str
    fiscal_period: str
    page_num: int
    text: str

    def as_dict(self) -> dict:
        return asdict(self)


class WhitespaceTokenizer:
    """Deterministic stand-in used by unit tests and by CI, which has no model download."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self._vocab = getattr(self, "_vocab", {})
        ids = []
        for tok in text.split():
            ids.append(self._vocab.setdefault(tok, len(self._vocab)))
        self._rev = {v: k for k, v in self._vocab.items()}
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        rev = getattr(self, "_rev", {})
        return " ".join(rev.get(i, "") for i in ids).strip()


def chunk_page(
    text: str,
    tokenizer: Tokenizer,
    window: int = 350,
    overlap: int = 50,
) -> list[str]:
    """Split one page's text into overlapping token windows.

    A page shorter than `window` yields exactly one chunk. Empty pages yield none.
    """
    text = normalize_ws(text)
    if not text:
        return []
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(ids) <= window:
        return [text]
    step = window - overlap
    out: list[str] = []
    for start in range(0, len(ids), step):
        piece = ids[start : start + window]
        if not piece:
            break
        decoded = normalize_ws(tokenizer.decode(piece, skip_special_tokens=True))
        if decoded:
            out.append(decoded)
        if start + window >= len(ids):
            break
    return out


def make_chunk_id(doc_name: str, page_num: int, ordinal: int) -> str:
    """Stable, content-independent id: re-running ingest reproduces the same ids."""
    raw = f"{doc_name}:{page_num}:{ordinal}"
    return f"{doc_name}#p{page_num}#{ordinal}#{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


def chunk_pages(
    pages: Iterable[tuple[str, int, str]],
    doc_meta: dict[str, dict],
    tokenizer: Tokenizer,
    window: int = 350,
    overlap: int = 50,
) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_name, page_num, text in pages:
        meta = doc_meta.get(doc_name, {})
        for ordinal, piece in enumerate(chunk_page(text, tokenizer, window, overlap)):
            out.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_name, page_num, ordinal),
                    doc_name=doc_name,
                    company=meta.get("company", ""),
                    doc_type=meta.get("doc_type", ""),
                    fiscal_period=str(meta.get("doc_period", "")),
                    page_num=page_num,
                    text=piece,
                )
            )
    return out
