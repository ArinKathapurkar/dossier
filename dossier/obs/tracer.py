"""Span tracer.

Written before the agent, deliberately: if tracing is added afterwards it always ends up
partial. Every LLM call, tool call, guard evaluation, fallback, context compaction and
review wait is a span with a parent, a duration and typed attributes, persisted to
`runs/runs.sqlite`. `dossier trace <run_id>` renders the tree; `dossier cost` aggregates
the same rows.

OpenTelemetry export is optional and lazy: when OTEL_EXPORTER_OTLP_ENDPOINT is set the
same spans are mirrored to an OTLP collector, but the package imports fine without the
`otel` extra installed.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_config

SPAN_KINDS = (
    "run",
    "turn",
    "llm_call",
    "tool_call",
    "guard",
    "fallback",
    "compaction",
    "review_wait",
    "retrieval",
    "extraction",
    "judge",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id    TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    parent_id  TEXT,
    kind       TEXT NOT NULL,
    name       TEXT NOT NULL,
    start_ts   REAL NOT NULL,
    end_ts     REAL,
    attrs      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
CREATE INDEX IF NOT EXISTS idx_spans_kind ON spans(kind);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or get_config().paths.runs_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn


@dataclass
class Span:
    span_id: str
    run_id: str
    parent_id: str | None
    kind: str
    name: str
    start_ts: float
    end_ts: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Thread-safe, run-scoped span recorder.

    `current_run` and the parent stack are per-thread so the parallel section sub-agents
    do not interleave their trees. Async sub-agents call `bind(run_id, parent)` on their
    own task.
    """

    def __init__(self, db: Path | None = None):
        self._conn = connect(db)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._otel = None
        self._otel_tried = False

    # -- context ------------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return getattr(self._local, "run_id", "unbound")

    def bind(self, run_id: str, parent_id: str | None = None) -> None:
        self._local.run_id = run_id
        self._local.stack = [parent_id] if parent_id else []

    @property
    def _stack(self) -> list[str]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    # -- otel ---------------------------------------------------------------------
    def _otel_tracer(self):
        if self._otel_tried:
            return self._otel
        self._otel_tried = True
        if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            return None
        try:
            from opentelemetry import trace as otel_trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider(resource=Resource.create({"service.name": "dossier"}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            otel_trace.set_tracer_provider(provider)
            self._otel = otel_trace.get_tracer("dossier")
        except Exception:
            self._otel = None
        return self._otel

    # -- writing ------------------------------------------------------------------
    def _write(self, span: Span) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO spans (span_id, run_id, parent_id, kind, name, start_ts, end_ts, attrs) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    span.span_id,
                    span.run_id,
                    span.parent_id,
                    span.kind,
                    span.name,
                    span.start_ts,
                    span.end_ts,
                    json.dumps(span.attrs, default=str),
                ),
            )
            self._conn.commit()

    @contextlib.contextmanager
    def span(self, kind: str, name: str, attrs: dict | None = None, run_id: str | None = None) -> Iterator[Span]:
        s = Span(
            span_id=uuid.uuid4().hex[:16],
            run_id=run_id or self.run_id,
            parent_id=self._stack[-1] if self._stack else None,
            kind=kind,
            name=name,
            start_ts=time.time(),
            attrs=dict(attrs or {}),
        )
        self._stack.append(s.span_id)
        otel = self._otel_tracer()
        otel_cm = otel.start_as_current_span(f"{kind}:{name}") if otel else contextlib.nullcontext()
        try:
            with otel_cm:
                yield s
        except Exception as exc:
            s.attrs["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Unwind to this span's own id rather than popping blindly. A `bind()` issued
            # while this span was open resets the thread's stack underneath us -- a caller
            # bug, but one that used to take the traced work down with `IndexError: pop
            # from empty list` rather than merely distorting the trace. Popping by identity
            # keeps the span recorded and the stack coherent, so a mis-scoped bind degrades
            # the trace shape instead of failing the run it was supposed to observe.
            stack = self._stack
            if s.span_id in stack:
                del stack[stack.index(s.span_id) :]
            s.end_ts = time.time()
            s.attrs.setdefault("duration_ms", round((s.end_ts - s.start_ts) * 1000, 2))
            self._write(s)

    def event(self, kind: str, name: str, attrs: dict | None = None, run_id: str | None = None) -> None:
        """A zero-duration span. Used for fallbacks and other point-in-time facts."""
        now = time.time()
        self._write(
            Span(
                span_id=uuid.uuid4().hex[:16],
                run_id=run_id or self.run_id,
                parent_id=self._stack[-1] if self._stack else None,
                kind=kind,
                name=name,
                start_ts=now,
                end_ts=now,
                attrs=dict(attrs or {}),
            )
        )

    # -- reading ------------------------------------------------------------------
    def spans_for(self, run_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM spans WHERE run_id = ? ORDER BY start_ts ASC", (run_id,)
        ).fetchall()
        return [{**dict(r), "attrs": json.loads(r["attrs"])} for r in rows]

    def count_kind(self, run_id: str, kind: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM spans WHERE run_id = ? AND kind = ?", (run_id, kind)
        ).fetchone()[0]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


_TRACER: Tracer | None = None
_TRACER_LOCK = threading.Lock()


def get_tracer() -> Tracer:
    global _TRACER
    if _TRACER is None:
        with _TRACER_LOCK:
            if _TRACER is None:
                _TRACER = Tracer()
    return _TRACER


def reset_tracer(db: Path | None = None) -> Tracer:
    """Tests point the tracer at a temp database."""
    global _TRACER
    with _TRACER_LOCK:
        if _TRACER is not None:
            _TRACER.close()
        _TRACER = Tracer(db)
    return _TRACER


def span_tree(run_id: str) -> list[dict]:
    spans = get_tracer().spans_for(run_id)
    by_parent: dict[str | None, list[dict]] = {}
    for s in spans:
        by_parent.setdefault(s["parent_id"], []).append(s)

    def build(parent: str | None) -> list[dict]:
        out = []
        for s in by_parent.get(parent, []):
            out.append({**s, "children": build(s["span_id"])})
        return out

    roots = build(None)
    # Spans whose parent is outside this run (sub-agent roots) would otherwise be orphaned.
    known = {s["span_id"] for s in spans}
    for s in spans:
        if s["parent_id"] and s["parent_id"] not in known:
            roots.append({**s, "children": build(s["span_id"])})
    return roots


def print_trace(run_id: str, console) -> None:
    from rich.tree import Tree

    tree_data = span_tree(run_id)
    if not tree_data:
        console.print(f"[yellow]no spans recorded for run {run_id}[/]")
        return

    total_cost = 0.0

    def label(s: dict) -> str:
        a = s["attrs"]
        dur = a.get("duration_ms", 0)
        bits = [f"[bold]{s['kind']}[/]:{s['name']}", f"[dim]{dur:.0f}ms[/]"]
        if a.get("model"):
            bits.append(f"[cyan]{a['model']}[/]")
        if a.get("tokens_in") is not None:
            bits.append(f"[dim]in {a['tokens_in']} out {a.get('tokens_out', 0)}[/]")
        if a.get("cost_usd"):
            bits.append(f"[green]${a['cost_usd']:.4f}[/]")
        if a.get("prompt_version"):
            bits.append(f"[magenta]{a.get('prompt_name', '?')}@{a['prompt_version']}[/]")
        if s["kind"] == "fallback":
            bits.append(f"[red]{a.get('from', '?')}->{a.get('to', '?')}[/]")
        if a.get("error"):
            bits.append(f"[red]{a['error']}[/]")
        return "  ".join(bits)

    def walk(node: dict, parent_tree: Tree) -> None:
        nonlocal total_cost
        total_cost += float(node["attrs"].get("cost_usd") or 0)
        child = parent_tree.add(label(node))
        for c in node["children"]:
            walk(c, child)

    root = Tree(f"[bold]run {run_id}[/]")
    for n in tree_data:
        walk(n, root)
    console.print(root)
    console.print(f"[dim]total cost ${total_cost:.4f}[/]")
