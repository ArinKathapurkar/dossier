"""The fallback chain.

One helper, used everywhere a component can degrade rather than fail:

    reranker      -> RRF order
    Neo4j         -> NetworkX
    primary model -> fallback model
    LanceDB       -> BM25-only
    tool exception-> a structured {error, hint} result handed back to the model

Every degradation emits a span with `kind="fallback"`, which means the system's health is
a query rather than a guess: `dossier trace` shows it in the tree and Tier 2 counts
fallback spans per run. Silent degradation is the failure mode this is built to prevent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class FallbackTriggered(Exception):
    """Raised by a primary that wants to degrade without being an error."""


def emit_fallback(span_name: str, frm: str, to: str, error: str | None = None, **attrs: Any) -> None:
    from ..obs.tracer import get_tracer

    get_tracer().event(
        "fallback",
        name=span_name,
        attrs={"from": frm, "to": to, "error": error, **attrs},
    )


def with_fallback[T](  # noqa: UP047 - explicit TypeVar kept for readability
    primary: Callable[[], T],
    fallback: Callable[[], T],
    span_name: str,
    frm: str = "primary",
    to: str = "fallback",
    reraise: tuple[type[BaseException], ...] = (),
) -> T:
    """Run `primary`; on any exception (except `reraise`) run `fallback` and record it."""
    started = time.time()
    try:
        return primary()
    except reraise:
        raise
    except BaseException as exc:  # includes TimeoutError from futures
        emit_fallback(
            span_name,
            frm,
            to,
            error=f"{type(exc).__name__}: {exc}"[:300],
            elapsed_ms=round((time.time() - started) * 1000, 2),
        )
        return fallback()


def tool_error(tool: str, exc: BaseException, hint: str = "") -> dict:
    """Turn a tool exception into a result the model can act on.

    A crashed tool ends the run; a structured error lets the model retry with different
    arguments, which in practice recovers most tool failures within one turn.
    """
    emit_fallback("tool_error", frm=tool, to="structured_error", error=f"{type(exc).__name__}: {exc}"[:300])
    return {
        "error": f"{type(exc).__name__}: {exc}"[:300],
        "hint": hint or "Check the argument names and types, then retry once with narrower arguments.",
    }
