"""Thin wrapper over the Anthropic Messages API.

Deliberately thin. The point of this project is to show the orchestration, so the loop,
the context management and the state machine are all written out in `loop.py` rather than
delegated to a framework. What lives here is only the stuff every call needs:

  * a **lazy** client -- importing `dossier` must never require an API key, because CI
    runs the whole offline test suite with no key set;
  * **retry** with exponential backoff on 429 / 529 / 5xx;
  * **model fallback** to the cheaper tier when the primary keeps failing, recorded as a
    `fallback` span so the degradation is countable;
  * **tracing** of tokens, latency, model, cost and prompt version on every call;
  * **record / replay** so the agent's behaviour can be regression-tested in CI with no
    API key and no spend (`DOSSIER_LLM_MODE=record|replay|live`).

Replay keys on a hash of (model, system, messages, tools), so a replayed run reproduces
the exact tool-call sequence -- that is what makes `tests/regression` meaningful.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import get_config
from ..obs.cost import call_cost
from ..obs.tracer import get_tracer

_CLIENT = None


class LLMReplayMiss(RuntimeError):
    """Raised in replay mode when no cassette entry matches the request."""


class NoAPIKey(RuntimeError):
    pass


def get_client():
    """Construct the Anthropic client on first use, never at import time."""
    global _CLIENT
    if _CLIENT is None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise NoAPIKey(
                "ANTHROPIC_API_KEY is not set. Offline commands (ingest, index --skip-graph, "
                "eval tier1/tier2, unit and regression tests) do not need it."
            )
        _CLIENT = anthropic.Anthropic(max_retries=0)  # retries are handled here, with spans
    return _CLIENT


def reset_client() -> None:
    global _CLIENT
    _CLIENT = None


@dataclass
class LLMResponse:
    content: list[dict]
    stop_reason: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_s: float
    fell_back: bool = False
    raw: dict = field(default_factory=dict)

    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()

    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]


def _request_key(model: str, system: Any, messages: list[dict], tools: list[dict] | None) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "messages": messages, "tools": tools or []},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _block_to_dict(block: Any) -> dict:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return dict(block)


class Cassette:
    """A recorded set of exchanges for one regression case.

    Two kinds of exchange are recorded, because replaying only one of them would not make
    a run reproducible offline:

    * **API exchanges**, keyed by a hash of (model, system, messages, tools);
    * **data-tool results**, keyed by (tool name, arguments, occurrence). CI has no LanceDB
      index, no BM25 pickle and no XBRL database, so a replay that re-executed
      `search_filings` against the real corpus would not run there at all -- and even
      locally it would make the "identical twice" property depend on index state rather
      than on the agent. Recording the tool result *and the ledger items it added* makes
      replay a property of the cassette alone.

    Pure tools (`compute`, `finish`, `draft_section`, `request_human_review`) are always
    executed live: they touch no index, and executing them keeps the control flow real.
    """

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict] = {}
        self.order: list[str] = []
        self.tools: dict[str, dict] = {}
        self._tool_counts: dict[str, int] = {}
        if path.exists():
            blob = json.loads(path.read_text())
            self.entries = blob.get("entries", {})
            self.order = blob.get("order", [])
            self.tools = blob.get("tools", {})
            self.meta = blob.get("meta", {})
        else:
            self.meta = {}

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, request: dict, response: dict) -> None:
        if key not in self.entries:
            self.order.append(key)
        self.entries[key] = {"request": request, "response": response}

    # -- data-tool recording -------------------------------------------------------
    def tool_key(self, name: str, args: dict) -> str:
        """A key that distinguishes repeat calls with identical arguments."""
        base = hashlib.sha256(json.dumps({"tool": name, "args": args}, sort_keys=True, default=str).encode()).hexdigest()[:16]
        n = self._tool_counts.get(base, 0)
        self._tool_counts[base] = n + 1
        return f"{name}:{base}:{n}"

    def get_tool(self, key: str) -> dict | None:
        return self.tools.get(key)

    def put_tool(self, key: str, result: str, ledger_adds: list[dict]) -> None:
        self.tools[key] = {"result": result, "ledger_adds": ledger_adds}

    def reset_tool_counts(self) -> None:
        self._tool_counts = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"meta": self.meta, "order": self.order, "entries": self.entries, "tools": self.tools},
                indent=2,
                sort_keys=True,
            )
        )


_ACTIVE_CASSETTE: Cassette | None = None


def set_cassette(cassette: Cassette | None) -> None:
    global _ACTIVE_CASSETTE
    _ACTIVE_CASSETTE = cassette


def active_cassette() -> Cassette | None:
    return _ACTIVE_CASSETTE


def llm_mode() -> str:
    return os.environ.get("DOSSIER_LLM_MODE", get_config().llm_mode).lower()


# Injectable fault used by the model-fallback cassette: a callable that raises on the
# first N primary-model calls so the fallback path can be recorded deterministically.
_FAULT_INJECTOR = None


def set_fault_injector(fn) -> None:
    global _FAULT_INJECTOR
    _FAULT_INJECTOR = fn


# `output_config.effort` is not accepted by every model: Haiku 4.5 returns a 400 for it.
# This bit the model-fallback path specifically -- the primary tier accepted effort, then a
# fallback to the cheaper tier reused the same kwargs and failed with an unhelpful 400, so
# the degradation path was broken exactly when it was needed. Effort is therefore attached
# per attempt, per model, rather than once for the request.
_NO_EFFORT_PREFIXES = ("claude-haiku",)


def supports_effort(model: str) -> bool:
    return not model.startswith(_NO_EFFORT_PREFIXES)


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status in (408, 409, 429, 500, 502, 503, 529):
        return True
    name = type(exc).__name__
    return name in {"APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError", "OverloadedError"}


def complete(
    system: str | list[dict],
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    tool_choice: dict | None = None,
    prompt_name: str = "",
    prompt_version: str = "",
    effort: str | None = "low",
    span_name: str | None = None,
) -> LLMResponse:
    """One Messages API call with retry, model fallback, tracing and record/replay."""
    cfg = get_config()
    model = model or cfg.primary_model
    max_tokens = max_tokens or cfg.max_tokens
    mode = llm_mode()
    tracer = get_tracer()

    def _finish(payload: dict, model_used: str, latency: float, fell_back: bool) -> LLMResponse:
        usage = payload.get("usage") or {}
        tin = int(usage.get("input_tokens") or 0)
        tout = int(usage.get("output_tokens") or 0)
        return LLMResponse(
            content=[_block_to_dict(b) for b in payload.get("content", [])],
            stop_reason=payload.get("stop_reason") or "end_turn",
            model=model_used,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=call_cost(model_used, tin, tout),
            latency_s=latency,
            fell_back=fell_back,
            raw=payload,
        )

    with tracer.span(
        "llm_call",
        span_name or prompt_name or "messages.create",
        attrs={"model": model, "prompt_name": prompt_name, "prompt_version": prompt_version, "mode": mode},
    ) as span:
        # ---- replay ---------------------------------------------------------------
        if mode == "replay":
            cassette = active_cassette()
            if cassette is None:
                raise LLMReplayMiss("DOSSIER_LLM_MODE=replay but no cassette is active")
            key = _request_key(model, system, messages, tools)
            entry = cassette.get(key)
            if entry is None:
                raise LLMReplayMiss(
                    f"cassette {cassette.path.name} has no entry for request {key} "
                    f"(model={model}, {len(messages)} message(s))"
                )
            payload = entry["response"]
            resp = _finish(payload, payload.get("model", model), 0.0, bool(payload.get("_fell_back")))
            span.attrs.update(
                {
                    "tokens_in": resp.tokens_in,
                    "tokens_out": resp.tokens_out,
                    "cost_usd": 0.0,  # replay is free; keep cost honest
                    "replayed": True,
                    "cassette_key": key,
                    "stop_reason": resp.stop_reason,
                }
            )
            return resp

        # ---- live / record --------------------------------------------------------
        client = get_client()
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        attempt_models = [model]
        if cfg.fallback_model and cfg.fallback_model != model:
            attempt_models.append(cfg.fallback_model)

        last_exc: BaseException | None = None
        started = time.time()
        for tier_idx, use_model in enumerate(attempt_models):
            for attempt in range(cfg.llm_retries):
                try:
                    if _FAULT_INJECTOR is not None:
                        _FAULT_INJECTOR(use_model, attempt)
                    call_started = time.time()
                    attempt_kwargs = {**kwargs, "model": use_model}
                    if effort and supports_effort(use_model):
                        attempt_kwargs["output_config"] = {"effort": effort}
                    raw = client.messages.create(**attempt_kwargs)
                    latency = time.time() - call_started
                    payload = raw.model_dump(exclude_none=True)
                    fell_back = tier_idx > 0
                    resp = _finish(payload, use_model, latency, fell_back)
                    span.attrs.update(
                        {
                            "model": use_model,
                            "tokens_in": resp.tokens_in,
                            "tokens_out": resp.tokens_out,
                            "cost_usd": round(resp.cost_usd, 6),
                            "latency_s": round(latency, 3),
                            "stop_reason": resp.stop_reason,
                            "attempts": attempt + 1,
                            "fell_back": fell_back,
                        }
                    )
                    if mode == "record" and active_cassette() is not None:
                        payload_to_store = dict(payload)
                        payload_to_store["_fell_back"] = fell_back
                        active_cassette().put(
                            _request_key(model, system, messages, tools),
                            {"model": model, "system": system, "messages": messages, "tools": tools or []},
                            payload_to_store,
                        )
                    return resp
                except BaseException as exc:  # noqa: BLE001 - classified below
                    last_exc = exc
                    if not _is_retryable(exc):
                        raise
                    if attempt < cfg.llm_retries - 1:
                        time.sleep(min(2**attempt, 8) * (1 + random.random() * 0.1))
            if tier_idx + 1 < len(attempt_models):
                from ..guard.fallbacks import emit_fallback

                emit_fallback(
                    "model_tier",
                    frm=use_model,
                    to=attempt_models[tier_idx + 1],
                    error=f"{type(last_exc).__name__}: {last_exc}"[:200] if last_exc else None,
                )
        span.attrs["error"] = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
        span.attrs["elapsed_s"] = round(time.time() - started, 3)
        raise last_exc if last_exc else RuntimeError("llm call failed with no exception")
