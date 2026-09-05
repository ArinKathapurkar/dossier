"""Tool handlers: the compute evaluator, and provenance that does not depend on chance."""

import pytest

from dossier.agent.ledger import Ledger
from dossier.agent.tools import ToolContext, cagr, handle_compute, pct_change, safe_eval


@pytest.fixture
def ctx():
    led = Ledger()
    led._add("F", "fact", "3M ocf FY2018", "3M ocf", "fact:ocf", {"value": 6439000000.0})
    led._add("F", "fact", "3M capex FY2018", "3M capex", "fact:capex", {"value": 1577000000.0})
    led._add("E", "chunk", "3M 10K 2018, p.60", "some prose", "chunk:c1", {})
    return ToolContext(ledger=led)


# -- the whitelisted evaluator ------------------------------------------------------

def test_arithmetic():
    assert safe_eval("a - b", {"a": 10, "b": 3}) == 7
    assert safe_eval("(a + b) / 2", {"a": 10, "b": 4}) == 7


def test_helpers():
    assert pct_change(100, 125) == 25.0
    assert round(cagr(100, 121, 2), 6) == 10.0
    assert round(safe_eval("pct_change(old, new)", {"old": 200, "new": 150}), 6) == -25.0


def test_division_and_growth_edge_cases():
    with pytest.raises(ValueError):
        pct_change(0, 5)
    with pytest.raises(ValueError):
        cagr(0, 100, 3)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('rm -rf /')",
        "open('/etc/passwd').read()",
        "[x for x in range(10)]",
        "a.__class__",
        "lambda: 1",
        "print('hi')",
        "'string'",
    ],
)
def test_only_whitelisted_constructs_are_allowed(expr):
    with pytest.raises((ValueError, SyntaxError, TypeError)):
        safe_eval(expr, {"a": 1})


def test_unbound_name_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="bind it in"):
        safe_eval("revenue * 2", {})


# -- provenance ---------------------------------------------------------------------

def test_computed_value_carries_its_inputs(ctx):
    out = handle_compute({"expression": "ocf - capex", "inputs": {"ocf": "F1", "capex": "F2"}}, ctx)
    assert "[C1]" in out
    item = ctx.ledger.get("C1")
    assert item.meta["value"] == 4862000000.0
    assert item.meta["inputs"] == ["F1", "F2"]


def test_provenance_does_not_depend_on_json_key_order(ctx):
    """Regression: the binding order arrived as JSON object key order from the model, which
    is incidental, and it leaked into the computed value's citation text and input list."""
    a = handle_compute({"expression": "ocf - capex", "inputs": {"ocf": "F1", "capex": "F2"}}, ctx)
    b = handle_compute({"expression": "ocf - capex", "inputs": {"capex": "F2", "ocf": "F1"}}, ctx)
    assert a.replace("C1", "C") == b.replace("C2", "C")
    assert ctx.ledger.get("C1").meta["inputs"] == ctx.ledger.get("C2").meta["inputs"] == ["F1", "F2"]
    assert ctx.ledger.get("C1").citation.replace("C1", "C") == ctx.ledger.get("C2").citation.replace("C2", "C")


def test_a_text_passage_cannot_be_a_compute_input(ctx):
    out = handle_compute({"expression": "x * 2", "inputs": {"x": "E1"}}, ctx)
    assert "not a numeric fact" in out
    assert ctx.ledger.get("C1") is None


def test_unknown_evidence_id_is_reported_not_raised(ctx):
    out = handle_compute({"expression": "x * 2", "inputs": {"x": "F9"}}, ctx)
    assert "not in the ledger" in out


def test_a_bad_expression_returns_a_message_rather_than_crashing(ctx):
    out = handle_compute({"expression": "ocf ** capex ** ocf", "inputs": {"ocf": "F1", "capex": "F2"}}, ctx)
    assert "compute failed" in out or "[C1]" in out
