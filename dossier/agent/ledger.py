"""The evidence ledger -- the agent's working memory.

This is the core context-engineering idea in the project. Retrieved passages are large; a
run that retrieves forty of them cannot keep them all in the conversation. But the model
still needs to be able to *cite* any of them, and the output guard needs the full text to
verify a number against.

So the ledger separates the two:

  * the **conversation** carries a compact index -- one line per item: id, citation, a
    one-line excerpt. That is what the model reads when deciding what it already knows.
  * the **ledger** holds the full text, out of band, keyed by id.

Ids are typed by provenance so a reader can tell at a glance where a claim came from:

    E1, E2 ...   retrieved filing passages
    F1, F2 ...   XBRL facts
    C1, C2 ...   values computed by the `compute` tool, carrying their input ids

Deduplication is by chunk_id, so re-retrieving a passage in a later turn reuses its
existing id rather than creating a second one -- which matters, because a model that sees
the same passage under two ids will cite both and inflate its apparent support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MAX_EXCERPT = 220


@dataclass
class EvidenceItem:
    id: str
    kind: str  # "chunk" | "fact" | "computed"
    citation: str
    text: str
    source_key: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def excerpt(self, limit: int = MAX_EXCERPT) -> str:
        t = " ".join((self.text or "").split())
        return t if len(t) <= limit else t[: limit - 1] + "…"

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "citation": self.citation,
            "text": self.text,
            "source_key": self.source_key,
            "meta": self.meta,
        }


class Ledger:
    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}
        self._by_source: dict[str, str] = {}
        self._counters = {"E": 0, "F": 0, "C": 0}

    # -- construction ---------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, eid: str) -> bool:
        return eid in self._items

    def ids(self) -> list[str]:
        return list(self._items)

    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def get(self, eid: str) -> EvidenceItem | None:
        return self._items.get(eid)

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}{self._counters[prefix]}"

    def _add(self, prefix: str, kind: str, citation: str, text: str, source_key: str, meta: dict) -> EvidenceItem:
        if source_key and source_key in self._by_source:
            return self._items[self._by_source[source_key]]
        item = EvidenceItem(
            id=self._next_id(prefix), kind=kind, citation=citation, text=text, source_key=source_key, meta=meta
        )
        self._items[item.id] = item
        if source_key:
            self._by_source[source_key] = item.id
        return item

    def add_chunk(self, chunk) -> EvidenceItem:
        return self._add(
            "E",
            "chunk",
            chunk.citation,
            chunk.text,
            source_key=f"chunk:{chunk.chunk_id}",
            meta={
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "company": chunk.company,
                "page_num": chunk.page_num,
                "channels_hit": list(chunk.channels_hit),
                "rerank_score": chunk.rerank_score,
            },
        )

    def add_fact(self, row: dict) -> EvidenceItem:
        # The fiscal year shown is the one derived from the period end date, not the
        # filing's `fy` tag -- see ingest/xbrl.py for why those differ.
        period = row.get("fiscal_year") or row.get("fy") or (row.get("period_end") or "")
        citation = f"{row.get('company', '')} XBRL companyfacts {row.get('concept', '')} FY{period} ({row.get('form', '')} filed {row.get('filed', '')})"
        span = f"{row.get('period_start')} to {row.get('period_end')}" if row.get("period_start") else f"as of {row.get('period_end')}"
        text = (
            f"{row.get('company', '')} {row.get('concept', '')} = {row.get('value')} {row.get('unit', '')} "
            f"for fiscal year {period} ({span}), per {row.get('form', '')} filed {row.get('filed', '')}."
        )
        key = f"fact:{row.get('company')}:{row.get('concept')}:{row.get('unit')}:{row.get('period_start')}:{row.get('period_end')}"
        return self._add("F", "fact", citation, text, source_key=key, meta=dict(row))

    def add_computed(self, expression: str, value: Any, input_ids: list[str], detail: str = "") -> EvidenceItem:
        cites = ", ".join(input_ids) if input_ids else "no evidence inputs"
        citation = f"computed: {expression} from [{cites}]"
        text = f"{expression} = {value}" + (f" ({detail})" if detail else "")
        item = self._add("C", "computed", citation, text, source_key="", meta={"expression": expression, "value": value, "inputs": list(input_ids)})
        return item

    # -- rendering -------------------------------------------------------------------
    def render_index(self, limit: int | None = None, excerpt_chars: int = MAX_EXCERPT) -> str:
        """The compact block injected into the system prompt each turn."""
        items = self.items()
        if not items:
            return "(no evidence gathered yet)"
        if limit and len(items) > limit:
            items = items[-limit:]
        lines = [f"[{i.id}] {i.citation} :: {i.excerpt(excerpt_chars)}" for i in items]
        return "\n".join(lines)

    def render_full(self, ids: list[str]) -> str:
        out = []
        for eid in ids:
            item = self._items.get(eid)
            if item:
                out.append(f"[{item.id}] {item.citation}\n<document>\n{item.text}\n</document>")
        return "\n\n".join(out)

    def numeric_sources(self, ids: list[str]) -> list[EvidenceItem]:
        return [self._items[i] for i in ids if i in self._items]

    # -- persistence -------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {"counters": self._counters, "items": [i.as_dict() for i in self._items.values()]},
            default=str,
        )

    @classmethod
    def from_json(cls, blob: str) -> Ledger:
        led = cls()
        data = json.loads(blob) if blob else {"counters": {"E": 0, "F": 0, "C": 0}, "items": []}
        led._counters = {**{"E": 0, "F": 0, "C": 0}, **data.get("counters", {})}
        for raw in data.get("items", []):
            item = EvidenceItem(
                id=raw["id"],
                kind=raw["kind"],
                citation=raw["citation"],
                text=raw["text"],
                source_key=raw.get("source_key", ""),
                meta=raw.get("meta", {}),
            )
            led._items[item.id] = item
            if item.source_key:
                led._by_source[item.source_key] = item.id
        return led

    def slice_for(self, ids: list[str]) -> Ledger:
        """A sub-agent's own view: same ids, only the items it retrieved."""
        led = Ledger()
        led._counters = dict(self._counters)
        for eid in ids:
            item = self._items.get(eid)
            if item:
                led._items[eid] = item
                if item.source_key:
                    led._by_source[item.source_key] = eid
        return led

    def merge(self, other: Ledger) -> dict[str, str]:
        """Merge another ledger in, remapping ids that collide. Returns `{old: new}`."""
        remap: dict[str, str] = {}
        for item in other.items():
            if item.source_key and item.source_key in self._by_source:
                remap[item.id] = self._by_source[item.source_key]
                continue
            prefix = item.id[0]
            new = self._add(prefix, item.kind, item.citation, item.text, item.source_key, item.meta)
            remap[item.id] = new.id
        return remap
