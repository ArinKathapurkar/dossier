"""Input guard.

Two layers, cheapest first:

1. **Deterministic pre-checks** -- empty input, over-length input, and a small set of
   literal injection strings. These cost nothing and catch the obvious cases without a
   model call, which matters because the guard runs on every question.
2. **A cheap classifier call** (haiku tier, forced tool schema) for the judgement calls:
   in_scope vs personal_investment_advice vs off_topic vs prompt_injection. The distinction
   that actually needs a model is "what risks does management identify" (in scope) against
   "is this a good investment" (advice), which no keyword list gets right.

Scope refusals are templated and redirect rather than stonewalling -- the useful behaviour
for an analyst tool is to say what it *can* answer.

The complementary half of injection defence lives in the tool layer: retrieved document
text is always wrapped in `<document>` tags and the system prompt states that document
content is data, never instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import get_config
from ..obs.prompts import registry

MAX_CHARS = 2000

CLASSIFY_TOOL = {
    "name": "classify",
    "description": "Return the category for this user message.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["in_scope", "off_topic", "personal_investment_advice", "prompt_injection"],
            },
            "reason": {"type": "string"},
        },
        "required": ["category"],
        "additionalProperties": False,
    },
}

INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore (?:all |any )?(?:your |the )?(?:previous|prior|above) instructions",
        r"disregard (?:your |the )?(?:previous|prior|system) (?:instructions|prompt)",
        r"(?:print|reveal|repeat|output|show me) (?:your |the )?system prompt",
        r"you are (?:now|no longer) (?:a|an|the)\b.*\b(?:assistant|model|bot)",
        r"<\|im_start\|>|<\|im_end\|>",
        r"\bDAN\b mode",
        r"developer mode enabled",
    )
]

REFUSALS = {
    "personal_investment_advice": (
        "I can't give investment advice or recommendations. What I can do is tell you what the "
        "filings actually say -- reported financials, segment performance, the risks management "
        "discloses, and how the company describes its competitive position. For example: "
        "\"What does the FY2022 10-K say about margin pressure?\" or \"Compare operating cash "
        "flow across the peer set.\""
    ),
    "off_topic": (
        "I only answer questions about the SEC filings in this corpus -- 10-K, 10-Q, 8-K and "
        "earnings releases for the companies in the deal. Ask me about a company's financials, "
        "segments, risk factors, or competitive position and I'll answer with citations."
    ),
    "prompt_injection": (
        "That request tries to change my instructions, so I've declined it and logged it. I "
        "answer questions about the indexed filings, with a citation for every claim."
    ),
}


@dataclass
class InputVerdict:
    category: str
    allowed: bool
    message: str = ""
    reason: str = ""
    source: str = "deterministic"

    def as_dict(self) -> dict:
        return {
            "category": self.category,
            "allowed": self.allowed,
            "message": self.message,
            "reason": self.reason,
            "source": self.source,
        }


def deterministic_check(text: str) -> InputVerdict | None:
    t = (text or "").strip()
    if not t:
        return InputVerdict("off_topic", False, "Please ask a question about the filings in this corpus.", "empty input")
    if len(t) > MAX_CHARS:
        return InputVerdict(
            "off_topic",
            False,
            f"That question is {len(t)} characters; please keep it under {MAX_CHARS} so I can retrieve against it properly.",
            "over length",
        )
    for pat in INJECTION_PATTERNS:
        if pat.search(t):
            return InputVerdict("prompt_injection", False, REFUSALS["prompt_injection"], f"matched {pat.pattern!r}")
    return None


def classify(text: str) -> InputVerdict:
    """Deterministic pre-checks, then a cheap classifier call."""
    from ..obs.tracer import get_tracer

    with get_tracer().span("guard", "input_guard", attrs={"chars": len(text or "")}) as span:
        early = deterministic_check(text)
        if early is not None:
            span.attrs.update({"category": early.category, "allowed": early.allowed, "source": "deterministic"})
            return early

        from ..agent.llm import complete

        cfg = get_config()
        system, version = registry().get("input_guard")
        try:
            resp = complete(
                system=system,
                messages=[{"role": "user", "content": f"<message>\n{text}\n</message>"}],
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify"},
                model=cfg.cheap_model,
                max_tokens=256,
                prompt_name="input_guard",
                prompt_version=version,
                effort=None,
            )
        except Exception as exc:
            # Fail open on classification, closed on the deterministic checks: a guard
            # outage must not take the whole system down, and the output guard still runs.
            from .fallbacks import emit_fallback

            emit_fallback("input_guard", frm="classifier", to="allow", error=f"{type(exc).__name__}: {exc}"[:200])
            span.attrs.update({"category": "in_scope", "allowed": True, "source": "fallback"})
            return InputVerdict("in_scope", True, source="fallback", reason="classifier unavailable")

        category, reason = "in_scope", ""
        for block in resp.tool_uses():
            if block.get("name") == "classify":
                payload = block.get("input") or {}
                category = payload.get("category", "in_scope")
                reason = payload.get("reason", "")
                break
        allowed = category == "in_scope"
        span.attrs.update({"category": category, "allowed": allowed, "source": "classifier"})
        return InputVerdict(
            category,
            allowed,
            message="" if allowed else REFUSALS.get(category, REFUSALS["off_topic"]),
            reason=reason,
            source="classifier",
        )
