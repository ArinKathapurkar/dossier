"""Reviewer decisions exported as regression cases must stay well-formed and enforceable."""

from __future__ import annotations

import pytest

from dossier.eval.regression import load_cases
from dossier.guard import output_guard

CASES = load_cases()
pytestmark = pytest.mark.skipif(not CASES, reason="no reviewer decisions exported yet")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_case_is_well_formed(case):
    assert case["question"]
    assert case["decision"] in ("approved", "edited", "rejected")
    assert isinstance(case["expect"], dict)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_expected_properties_are_ones_the_guard_can_enforce(case):
    known = {"guard_must_pass", "previously_violated", "must_not_contain", "must_cite",
             "must_address_notes", "must_abstain"}
    assert set(case["expect"]).issubset(known), f"unenforceable property in {case['case_id']}"


def test_must_not_contain_properties_are_actually_detectable():
    """A 'must not contain' figure has to be something extract_numbers would find."""
    for case in CASES:
        for surface in case["expect"].get("must_not_contain", []):
            assert output_guard.extract_numbers(surface), f"{surface!r} is not a detectable figure"
