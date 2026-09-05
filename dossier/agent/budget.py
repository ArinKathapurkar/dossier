"""Token-budget manager.

A diligence run that searches ten times accumulates ten large tool results. Left alone the
conversation grows until it either exceeds the context window or costs more per turn than
the answer is worth.

The strategy here is deliberately simple and deliberately lossy in only one direction:
when the conversation would exceed `max_context_tokens`, the **oldest tool results** are
replaced by a one-paragraph summary that *keeps the evidence ids*. Nothing is actually
lost -- the ledger still holds the full text of every id, and the model can re-read any of
them. What is dropped is the verbatim passage text sitting in the message history, which is
the largest and least reusable part of the context.

Summarising the oldest first (rather than the least relevant) is on purpose: relevance
scoring at compaction time would need another model call, and the oldest results are the
ones the model has already acted on.

Every compaction emits a `compaction` span with the tokens before and after, so the README
number for "tokens saved" is measured rather than asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import get_config


def approx_tokens(text: str) -> int:
    """Cheap deterministic estimate: ~4 characters per token for English prose.

    Deliberately not a model call and deliberately not the real tokenizer -- the budget
    manager runs on every turn, and being 5% off is fine for a threshold decision while a
    network round trip per turn is not. Tests inject an exact fake tokenizer instead.
    """
    return max(1, len(text) // 4)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(_content_text(block.get("content")))
                elif block.get("type") == "tool_use":
                    parts.append(str(block.get("input", "")))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def message_tokens(message: dict, counter: Callable[[str], int] = approx_tokens) -> int:
    return counter(_content_text(message.get("content")))


def conversation_tokens(messages: list[dict], counter: Callable[[str], int] = approx_tokens) -> int:
    return sum(message_tokens(m, counter) for m in messages)


def _ids_in(text: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"\b([EFC]\d+)\b", text)), key=lambda s: (s[0], int(s[1:])))


class BudgetManager:
    def __init__(
        self,
        max_context_tokens: int | None = None,
        counter: Callable[[str], int] = approx_tokens,
        keep_recent: int = 4,
    ):
        self.max_context_tokens = max_context_tokens or get_config().max_context_tokens
        self.counter = counter
        # Never compact the last `keep_recent` messages: the model is mid-thought on them.
        self.keep_recent = keep_recent
        self.compactions = 0
        self.tokens_saved = 0

    def total(self, messages: list[dict]) -> int:
        return conversation_tokens(messages, self.counter)

    def needs_compaction(self, messages: list[dict], incoming_tokens: int = 0) -> bool:
        return self.total(messages) + incoming_tokens > self.max_context_tokens

    def _summarize_tool_result(self, block: dict) -> dict:
        body = _content_text(block.get("content"))
        ids = _ids_in(body)
        head = " ".join(body.split())[:160]
        summary = (
            f"[compacted tool result] {head}… "
            f"Evidence recorded in the ledger: {', '.join(ids) if ids else 'none'}. "
            f"Full text remains available by evidence id."
        )
        return {**block, "content": summary}

    def compact(self, messages: list[dict], run_id: str | None = None) -> list[dict]:
        """Replace the oldest tool results with summaries until under budget."""
        from ..obs.tracer import get_tracer

        before = self.total(messages)
        out = [dict(m) for m in messages]
        head_limit = max(0, len(out) - self.keep_recent)
        changed = 0
        for idx in range(head_limit):
            if self.total(out) <= self.max_context_tokens:
                break
            content = out[idx].get("content")
            if not isinstance(content, list):
                continue
            new_blocks = []
            touched = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result" and not block.get("_compacted"):
                    new_blocks.append({**self._summarize_tool_result(block), "_compacted": True})
                    touched = True
                else:
                    new_blocks.append(block)
            if touched:
                out[idx] = {**out[idx], "content": new_blocks}
                changed += 1
        after = self.total(out)
        if changed:
            self.compactions += 1
            self.tokens_saved += before - after
            get_tracer().event(
                "compaction",
                name="oldest_tool_results",
                attrs={
                    "tokens_before": before,
                    "tokens_after": after,
                    "tokens_saved": before - after,
                    "messages_compacted": changed,
                    "max_context_tokens": self.max_context_tokens,
                },
                run_id=run_id,
            )
        return out

    def maybe_compact(self, messages: list[dict], run_id: str | None = None) -> list[dict]:
        if not self.needs_compaction(messages):
            return messages
        return self.compact(messages, run_id=run_id)

    def stats(self) -> dict:
        return {
            "compactions": self.compactions,
            "tokens_saved": self.tokens_saved,
            "max_context_tokens": self.max_context_tokens,
        }


def strip_internal_keys(messages: list[dict]) -> list[dict]:
    """Remove our bookkeeping keys before the messages go to the API."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = [{k: v for k, v in b.items() if not k.startswith("_")} if isinstance(b, dict) else b for b in content]
        out.append({"role": m["role"], "content": content})
    return out
