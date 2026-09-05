"""Load the FinanceBench benchmark and fetch the underlying filing PDFs.

FinanceBench (Patronus AI) is 150 analyst questions over public 10-K / 10-Q / 8-K /
earnings-release PDFs. Each question carries a gold answer, a justification, and one or
more evidence entries with a **1-indexed page number**, which is what makes deterministic
retrieval evaluation possible (see eval/tier1_retrieval.py).

Schema verified against the HF datasets-server at build time:

    financebench_id, company, doc_name, question_type, question_reasoning,
    domain_question_num, question, answer, justification, dataset_subset_label,
    evidence[{evidence_text, doc_name, evidence_page_num, evidence_text_full_page}],
    gics_sector, doc_type, doc_period (int), doc_link

PDFs live in the FinanceBench GitHub repo under `pdfs/<doc_name>.pdf`; `doc_link` (an SEC
URL) is the fallback and requires a descriptive User-Agent per SEC's access policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from ..config import get_config

HF_DATASET = "PatronusAI/financebench"
GITHUB_PDF_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


@dataclass
class DocRecord:
    doc_name: str
    company: str
    doc_type: str
    doc_period: int
    doc_link: str
    gics_sector: str
    source: str | None = None  # "github" | "doc_link" | None if not fetched
    local_path: str | None = None
    bytes: int | None = None


@dataclass
class QuestionRecord:
    financebench_id: str
    company: str
    doc_name: str
    doc_type: str
    doc_period: int
    question: str
    answer: str
    justification: str
    question_type: str
    question_reasoning: str
    gics_sector: str
    evidence: list[dict]


def load_dataset_rows() -> list[dict]:
    """Load the 150 FinanceBench rows via `datasets`."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    return [dict(row) for row in ds]


def build_manifest(rows: list[dict]) -> tuple[list[DocRecord], list[QuestionRecord]]:
    docs: dict[str, DocRecord] = {}
    questions: list[QuestionRecord] = []
    for r in rows:
        name = r["doc_name"]
        if name not in docs:
            docs[name] = DocRecord(
                doc_name=name,
                company=r["company"],
                doc_type=r["doc_type"],
                doc_period=int(r["doc_period"]),
                doc_link=r.get("doc_link") or "",
                gics_sector=r.get("gics_sector") or "",
            )
        questions.append(
            QuestionRecord(
                financebench_id=r["financebench_id"],
                company=r["company"],
                doc_name=name,
                doc_type=r["doc_type"],
                doc_period=int(r["doc_period"]),
                question=r["question"],
                answer=r["answer"],
                justification=r.get("justification") or "",
                question_type=r.get("question_type") or "",
                question_reasoning=r.get("question_reasoning") or "",
                gics_sector=r.get("gics_sector") or "",
                evidence=[dict(e) for e in (r.get("evidence") or [])],
            )
        )
    return list(docs.values()), questions


def _fetch(client: httpx.Client, url: str, attempts: int = 3) -> bytes | None:
    delay = 1.0
    for i in range(attempts):
        try:
            resp = client.get(url, follow_redirects=True, timeout=90.0)
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                return resp.content
            # SEC throttles with 403/429; back off and retry.
            if resp.status_code in (403, 429, 500, 502, 503):
                time.sleep(delay)
                delay *= 2
                continue
            return None
        except httpx.HTTPError:
            if i == attempts - 1:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def download_pdfs(docs: list[DocRecord], dest: Path, force: bool = False) -> list[DocRecord]:
    """Fetch each document, GitHub first then `doc_link`. Records which source served it."""
    cfg = get_config()
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": cfg.sec_user_agent}
    with httpx.Client(headers=headers) as client:
        for d in docs:
            path = dest / f"{d.doc_name}.pdf"
            if path.exists() and path.stat().st_size > 0 and not force:
                d.local_path = str(path)
                d.bytes = path.stat().st_size
                d.source = d.source or "cached"
                continue
            blob = _fetch(client, f"{GITHUB_PDF_BASE}/{d.doc_name}.pdf")
            source = "github"
            if blob is None and d.doc_link:
                # SEC rate limit is 10 req/s; one document at a time is well under it.
                time.sleep(0.15)
                blob = _fetch(client, d.doc_link)
                source = "doc_link"
            if blob is None:
                d.source = None
                continue
            path.write_bytes(blob)
            d.local_path = str(path)
            d.bytes = len(blob)
            d.source = source
    return docs


def write_manifest(docs: list[DocRecord], questions: list[QuestionRecord]) -> dict:
    cfg = get_config()
    cfg.paths.ensure()
    manifest = {
        "dataset": HF_DATASET,
        "documents": [asdict(d) for d in docs],
        "counts": {
            "documents": len(docs),
            "documents_fetched": sum(1 for d in docs if d.local_path),
            "questions": len(questions),
            "companies": len({d.company for d in docs}),
        },
    }
    cfg.paths.manifest.write_text(json.dumps(manifest, indent=2))
    cfg.paths.questions.write_text(json.dumps([asdict(q) for q in questions], indent=2))
    return manifest


def load_manifest() -> dict:
    return json.loads(get_config().paths.manifest.read_text())


def load_questions() -> list[dict]:
    return json.loads(get_config().paths.questions.read_text())


def verify_page_convention(questions: list[dict], pages_by_doc: dict[str, dict[int, str]], n: int = 3) -> dict:
    """Confirm `evidence_page_num` is 1-indexed against extracted page text.

    For the first `n` evidence rows we can resolve, compare a normalized fragment of the
    gold evidence against the extracted text at page P (1-indexed) and at P+1 (what we'd
    see if the benchmark were 0-indexed). Whichever wins tells us the convention. This
    runs before any indexing, because getting it wrong silently halves Tier 1 recall.
    """
    from .pdf_text import normalize_ws

    checks: list[dict] = []
    for q in questions:
        for ev in q.get("evidence", []):
            doc = ev.get("doc_name")
            page = ev.get("evidence_page_num")
            if doc not in pages_by_doc or page is None:
                continue
            probe = normalize_ws(ev.get("evidence_text", ""))[:120]
            if len(probe) < 40:
                continue
            pages = pages_by_doc[doc]
            as_one = normalize_ws(pages.get(int(page), ""))
            as_zero = normalize_ws(pages.get(int(page) + 1, ""))
            checks.append(
                {
                    "doc_name": doc,
                    "evidence_page_num": int(page),
                    "hit_if_1_indexed": probe[:60] in as_one,
                    "hit_if_0_indexed": probe[:60] in as_zero,
                }
            )
            break
        if len(checks) >= n:
            break
    one = sum(c["hit_if_1_indexed"] for c in checks)
    zero = sum(c["hit_if_0_indexed"] for c in checks)
    return {
        "checks": checks,
        "one_indexed_hits": one,
        "zero_indexed_hits": zero,
        "convention": "1-indexed" if one >= zero else "0-indexed",
    }
