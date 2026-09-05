"""Shared retrieval types."""

from __future__ import annotations

from dataclasses import dataclass, field


def format_citation(company: str, doc_type: str, fiscal_period: str, page: int) -> str:
    company = (company or "").strip()
    doc_type = (doc_type or "").strip()
    period = str(fiscal_period or "").strip()
    head = " ".join(p for p in (company, doc_type, period) if p)
    return f"{head}, p.{page}"


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    citation: str
    score: float
    channels_hit: list[str] = field(default_factory=list)
    doc_name: str = ""
    company: str = ""
    doc_type: str = ""
    fiscal_period: str = ""
    page_num: int = 0
    rerank_score: float | None = None

    @classmethod
    def from_row(cls, row: dict, score: float, channels: list[str]) -> RetrievedChunk:
        return cls(
            chunk_id=row["chunk_id"],
            text=row.get("text", ""),
            citation=format_citation(
                row.get("company", ""), row.get("doc_type", ""), row.get("fiscal_period", ""), int(row.get("page_num", 0))
            ),
            score=score,
            channels_hit=channels,
            doc_name=row.get("doc_name", ""),
            company=row.get("company", ""),
            doc_type=row.get("doc_type", ""),
            fiscal_period=str(row.get("fiscal_period", "")),
            page_num=int(row.get("page_num", 0)),
        )

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "citation": self.citation,
            "score": round(self.score, 6),
            "channels_hit": self.channels_hit,
            "doc_name": self.doc_name,
            "company": self.company,
            "doc_type": self.doc_type,
            "fiscal_period": self.fiscal_period,
            "page_num": self.page_num,
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }
