"""Output grounding guard -- deterministic, no model calls.

An LLM-based grader would be the easy build here, and it is the wrong one. The failure this
guard exists to catch is a fluent, confident, wrong number. A grader model is fluent-and-
confident by construction and shares the generator's blind spots; worse, it makes the check
non-deterministic, so CI cannot gate on it and the same answer can pass on Tuesday and fail
on Wednesday. Everything below is string and arithmetic, runs in milliseconds, costs
nothing, and produces the same verdict every time.

Four checks:

1. **Citation validity** -- every `[E#]/[F#]/[C#]` in the answer resolves in the ledger.
2. **Numeric grounding** -- every number in the answer either appears in the text of a
   *cited* passage, equals a cited fact's value within 0.5% after unit normalization, or is
   a computed `C#` value.
3. **Forward-looking attribution** -- sentences containing expects / anticipates / will /
   guidance / projected must attribute to management or carry an evidence id.
4. **Abstention shape** -- an answer citing nothing must actually say it found nothing,
   rather than answering from the model's own knowledge with no citations.

Known false-positive modes are documented in docs/DESIGN.md; the important ones are years
and section references (handled by the ignore rules below) and figures a filing states in
words rather than digits.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

CITATION_RE = re.compile(r"\[((?:[EFC]\d+)(?:\s*,\s*[EFC]\d+)*)\]")
ID_RE = re.compile(r"\b([EFC]\d+)\b")

# A number with optional $ / % / thousands separators / decimal / scale word.
NUMBER_RE = re.compile(
    r"(?P<neg>\(|-)?\s*(?P<dollar>\$)?\s*(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<scale>billion|bn|million|mm|mn|thousand|k\b)?\s*(?P<pct>%)?",
    re.I,
)

SCALES = {
    "billion": 1e9,
    "bn": 1e9,
    "million": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}

FORWARD_LOOKING = re.compile(
    r"\b(expects?|expected|anticipates?|anticipated|will|guidance|projected|projects?|forecasts?|outlook|plans to)\b",
    re.I,
)
ATTRIBUTION = re.compile(
    r"\b(management|the company states|the company expects|the filing states|according to the (?:filing|company|10-K|10-Q)|"
    r"the company (?:said|reported|disclosed|anticipates|projects))\b",
    re.I,
)
ABSTENTION = re.compile(
    r"(not found in the indexed filings|not in the indexed (?:filings|corpus)|no (?:passages|documents|filings) (?:matched|were found)|"
    r"the (?:indexed )?corpus does not (?:contain|include)|i (?:could|can)not find|is not available in the (?:indexed )?filings|"
    r"not disclosed in the (?:indexed )?filings)",
    re.I,
)

NUMERIC_TOLERANCE = 0.005  # 0.5%
# Years, small ordinals, page/item references and enumerations are not financial claims.
IGNORE_BARE_INTEGERS_BELOW = 32
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


@dataclass
class Violation:
    kind: str
    detail: str
    sentence: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GuardReport:
    passed: bool
    violations: list[Violation] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    # `pass` is a keyword; the spec's field name is exposed on the dict form.
    def as_dict(self) -> dict:
        return {
            "pass": self.passed,
            "violations": [v.as_dict() for v in self.violations],
            "checks": self.checks,
        }

    def to_prompt(self) -> str:
        if self.passed:
            return "The output guard passed."
        lines = ["The output guard rejected your answer. Fix each item and call the tool again.", ""]
        for v in self.violations:
            lines.append(f"- ({v.kind}) {v.detail}" + (f'  In: "{v.sentence.strip()[:200]}"' if v.sentence else ""))
        lines.append("")
        lines.append(
            "Rules: every number must come from a cited passage, a cited XBRL fact, or a compute() "
            "result. If you cannot ground a figure, remove it or retrieve evidence for it. If the "
            "corpus does not contain the answer, say so explicitly and cite nothing."
        )
        return "\n".join(lines)


def cited_ids(text: str) -> list[str]:
    out: list[str] = []
    for group in CITATION_RE.findall(text or ""):
        for eid in ID_RE.findall(group):
            if eid not in out:
                out.append(eid)
    return out


def normalize_number(match: re.Match) -> tuple[float, bool] | None:
    raw = match.group("num")
    if not raw:
        return None
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        return None
    scale = (match.group("scale") or "").lower().strip()
    if scale:
        value *= SCALES.get(scale, 1.0)
    if match.group("neg"):
        value = -value
    return value, bool(match.group("pct"))


def extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    """Return `(surface, normalized_value, is_percent)` for each number in `text`."""
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        if not m.group("num"):
            continue
        parsed = normalize_number(m)
        if parsed is None:
            continue
        value, is_pct = parsed
        surface = m.group(0).strip()
        out.append((surface, value, is_pct))
    return out


def _is_ignorable(surface: str, value: float, is_pct: bool, sentence: str) -> bool:
    if is_pct:
        return False
    bare = surface.replace("$", "").replace(",", "").strip()
    if "." not in bare and float(value).is_integer():
        iv = int(value)
        # Fiscal years and small counts / list numbers.
        if 1900 <= iv <= 2100:
            return True
        if abs(iv) < IGNORE_BARE_INTEGERS_BELOW and "$" not in surface:
            return True
    # "Item 1A", "p.42", "Note 12" are document references, not claims.
    if re.search(r"\b(item|note|page|p\.|section|exhibit|part)\s*$", sentence[: sentence.find(surface)] or "", re.I):
        return True
    return False


def _matches_value(candidate: float, target: float, tolerance: float = NUMERIC_TOLERANCE) -> bool:
    if target == 0:
        return abs(candidate) < 1e-9
    return abs(candidate - target) / abs(target) <= tolerance


def _appears_in_text(surface: str, value: float, is_pct: bool, text: str) -> bool:
    """Does this number appear in a source passage, allowing for scale wording?"""
    bare = surface.replace("$", "").replace("%", "").strip()
    digits = re.sub(r"[^\d.,]", "", bare)
    if digits and digits in text:
        return True
    # A passage may render 1,577 where the answer says $1.577 billion, or vice versa.
    for src_surface, src_value, src_pct in extract_numbers(text):
        if src_pct != is_pct:
            # A percentage in the answer may be stated as a plain number in the passage.
            if not (is_pct and not src_pct):
                continue
        if _matches_value(src_value, value):
            return True
        # scale-insensitive comparison: 1577 (millions) vs 1,577,000,000
        for factor in (1e3, 1e6, 1e9):
            if _matches_value(src_value * factor, value) or _matches_value(src_value, value * factor):
                return True
        del src_surface
    return False


def check(answer: str, ledger, evidence_ids: list[str] | None = None) -> GuardReport:
    """Run all four checks. `ledger` is an agent.ledger.Ledger (or anything with `.get`)."""
    answer = answer or ""
    violations: list[Violation] = []
    inline_ids = cited_ids(answer)
    declared_ids = list(evidence_ids or [])
    all_ids = list(dict.fromkeys(inline_ids + declared_ids))

    # ---- 1. citation validity ----------------------------------------------------
    unresolved = [i for i in all_ids if ledger.get(i) is None]
    for eid in unresolved:
        violations.append(Violation("citation_invalid", f"citation [{eid}] does not resolve to any evidence in the ledger"))
    resolved = [i for i in all_ids if ledger.get(i) is not None]

    # ---- 4. abstention shape (checked early: it changes what else applies) --------
    if not resolved:
        if ABSTENTION.search(answer):
            return GuardReport(
                passed=not unresolved,
                violations=violations,
                checks={
                    "citation_validity": not unresolved,
                    "numeric_grounding": True,
                    "forward_looking": True,
                    "abstention": True,
                    "numbers_checked": 0,
                    "abstained": True,
                },
            )
        violations.append(
            Violation(
                "abstention_shape",
                "the answer cites no evidence but does not state that the information was not found in the indexed filings",
            )
        )

    # ---- 2. numeric grounding -----------------------------------------------------
    cited_text = " ".join((ledger.get(i).text or "") for i in resolved)
    fact_values: list[float] = []
    computed_values: list[float] = []
    for eid in resolved:
        item = ledger.get(eid)
        if item.kind == "fact":
            try:
                fact_values.append(float(item.meta.get("value")))
            except (TypeError, ValueError):
                pass
        elif item.kind == "computed":
            try:
                computed_values.append(float(item.meta.get("value")))
            except (TypeError, ValueError):
                pass

    numbers_checked = 0
    ungrounded: list[tuple[str, str]] = []
    for sentence in SENTENCE_SPLIT.split(answer):
        for surface, value, is_pct in extract_numbers(sentence):
            if _is_ignorable(surface, value, is_pct, sentence):
                continue
            numbers_checked += 1
            grounded = (
                any(_matches_value(v, value) for v in computed_values)
                or any(
                    _matches_value(v, value) or any(_matches_value(v * f, value) or _matches_value(v, value * f) for f in (1e3, 1e6, 1e9))
                    for v in fact_values
                )
                or _appears_in_text(surface, value, is_pct, cited_text)
            )
            if not grounded:
                ungrounded.append((surface, sentence))
    for surface, sentence in ungrounded:
        violations.append(Violation("numeric_ungrounded", f"the figure {surface!r} does not appear in any cited passage, fact or computed value", sentence))

    # ---- 3. forward-looking attribution -------------------------------------------
    fl_unattributed = 0
    for sentence in SENTENCE_SPLIT.split(answer):
        if not FORWARD_LOOKING.search(sentence):
            continue
        if ATTRIBUTION.search(sentence) or CITATION_RE.search(sentence):
            continue
        fl_unattributed += 1
        violations.append(
            Violation(
                "forward_looking_unattributed",
                "a forward-looking statement is asserted without attributing it to management or citing evidence",
                sentence,
            )
        )

    checks = {
        "citation_validity": not unresolved,
        "numeric_grounding": not ungrounded,
        "forward_looking": fl_unattributed == 0,
        "abstention": True,
        "numbers_checked": numbers_checked,
        "citations_checked": len(all_ids),
        "abstained": False,
    }
    if any(v.kind == "abstention_shape" for v in violations):
        checks["abstention"] = False
    return GuardReport(passed=not violations, violations=violations, checks=checks)
