"""Output guard: the numeric-grounding check is the load-bearing one."""

import json

import pytest

from dossier.agent.ledger import Ledger
from dossier.eval.tier2_grounding import _DictLedger, load_fixtures
from dossier.guard import output_guard


@pytest.fixture
def led():
    led = Ledger()
    led._add("E", "chunk", "3M 10K 2018, p.60",
             "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420).",
             "chunk:c1", {})
    led._add("F", "fact", "3M XBRL NetCashProvidedByUsedInOperatingActivities FY2018",
             "3M operating cash flow = 6439000000.0 USD for fiscal year 2018.",
             "fact:k1", {"value": 6439000000.0, "unit": "USD"})
    return led


def test_number_present_in_a_cited_passage_passes(led):
    report = output_guard.check("Capex was $1,577 million [E1].", led, ["E1"])
    assert report.passed, report.as_dict()


def test_invented_number_is_caught(led):
    report = output_guard.check("Capex was $1,890 million [E1].", led, ["E1"])
    assert not report.passed
    assert {v.kind for v in report.violations} == {"numeric_ungrounded"}
    assert "1,890" in report.violations[0].detail


def test_number_matching_a_cited_fact_passes_across_unit_scaling(led):
    assert output_guard.check("Operating cash flow was $6.439 billion [F1].", led, ["F1"]).passed


def test_wrong_scale_on_a_cited_fact_is_caught(led):
    report = output_guard.check("Operating cash flow was $6.4 million [F1].", led, ["F1"])
    assert not report.passed


def test_unresolvable_citation_is_caught(led):
    report = output_guard.check("Capex was $1,577 million [E7].", led, ["E7"])
    assert "citation_invalid" in {v.kind for v in report.violations}


def test_unattributed_forward_looking_is_caught(led):
    report = output_guard.check("Capex was $1,577 million [E1]. Spending will rise next year.", led, ["E1"])
    assert "forward_looking_unattributed" in {v.kind for v in report.violations}


def test_attributed_forward_looking_passes(led):
    assert output_guard.check(
        "Capex was $1,577 million [E1]. Management expects spending to rise next year [E1].", led, ["E1"]
    ).passed


def test_explicit_abstention_passes_with_no_evidence(led):
    assert output_guard.check("This is not found in the indexed filings.", Ledger(), []).passed


def test_silent_no_answer_is_caught(led):
    report = output_guard.check("3M had roughly $32 billion of revenue.", Ledger(), [])
    kinds = {v.kind for v in report.violations}
    assert "abstention_shape" in kinds


def test_years_and_item_references_are_not_treated_as_claims(led):
    assert output_guard.check(
        "Per Item 8 of the 2018 Form 10-K, capex was $1,577 million [E1].", led, ["E1"]
    ).passed


def test_number_extraction_normalizes_scale_and_sign():
    nums = dict((s, v) for s, v, _ in output_guard.extract_numbers("$1.5 billion, (2,000), 12%"))
    assert any(abs(v - 1.5e9) < 1 for v in nums.values())
    assert any(v == -2000 for v in nums.values())


def test_every_tier2_fixture_produces_its_documented_verdict():
    failures = []
    for fx in load_fixtures():
        report = output_guard.check(fx["answer"], _DictLedger(fx.get("ledger", [])), fx.get("evidence_ids"))
        if report.passed != fx["expect_pass"]:
            failures.append((fx["name"], fx["expect_pass"], report.passed, [v.kind for v in report.violations]))
        expected = set(fx.get("expect_violations", []))
        if expected and not expected.issubset({v.kind for v in report.violations}):
            failures.append((fx["name"], "violations", sorted(expected), [v.kind for v in report.violations]))
    assert not failures, json.dumps(failures, indent=1)


def test_guard_report_renders_actionable_revise_text(led):
    report = output_guard.check("Capex was $1,890 million [E1].", led, ["E1"])
    text = report.to_prompt()
    assert "1,890" in text and "compute()" in text
