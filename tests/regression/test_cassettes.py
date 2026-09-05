"""Replay every recorded cassette. Runs in CI with no ANTHROPIC_API_KEY.

Each cassette carries an `expect` block recorded alongside the exchanges: the tool-call
sequence, the final workflow state, the guard verdict, the span kinds and whether the run
escalated. Replay must reproduce all of them, and must do so *identically twice* -- the
determinism property is what makes a behaviour change visible instead of dismissible as
sampling noise.

A cassette miss raises rather than falling through to the live API, so a test that drifts
out of sync with its recording fails loudly instead of quietly spending money.
"""

from __future__ import annotations

import pytest

from dossier.eval.record_cassettes import SCENARIOS
from dossier.eval.regression import cassette, list_cassettes
from dossier.obs.tracer import get_tracer

CASSETTES = list_cassettes()
pytestmark = pytest.mark.skipif(not CASSETTES, reason="no cassettes recorded yet")


def _run(name: str):
    fn = SCENARIOS[name]
    with cassette(name, mode="replay"):
        return fn()


def _observed(result) -> dict:
    spans = get_tracer().spans_for(result.run_id)
    return {
        "tool_sequence": [e["tool"] for e in result.events if e.get("type") == "tool_call"],
        "final_state": result.state,
        "guard_pass": result.guard_pass,
        "span_kinds": sorted({s["kind"] for s in spans}),
        "revisions": result.revisions,
        "escalated": bool(result.review_id),
    }


@pytest.mark.parametrize("path", CASSETTES, ids=lambda p: p.stem)
def test_replay_matches_the_recorded_expectations(path, runs_db):
    import json

    meta = json.loads(path.read_text())["meta"]
    name = meta["scenario"]
    if name not in SCENARIOS:
        pytest.skip(f"cassette {name} has no matching scenario")
    expect = meta["expect"]
    observed = _observed(_run(name))

    assert observed["tool_sequence"] == expect["tool_sequence"], "tool-call sequence drifted"
    assert observed["final_state"] == expect["final_state"], "final workflow state drifted"
    assert observed["guard_pass"] == expect["guard_pass"], "guard verdict drifted"
    assert observed["escalated"] == expect["escalated"], "HITL escalation drifted"
    assert observed["revisions"] == expect["revisions"], "revise-loop count drifted"
    # Span kinds are asserted as a superset check on the recorded kinds: replay skips the
    # live retry path, so a recorded run may legitimately have kinds a replay cannot emit.
    missing = set(expect["span_kinds"]) - set(observed["span_kinds"]) - {"fallback"}
    assert not missing, f"replay lost span kinds {sorted(missing)}"


@pytest.mark.parametrize("path", CASSETTES, ids=lambda p: p.stem)
def test_replaying_twice_is_identical(path, runs_db):
    import json

    name = json.loads(path.read_text())["meta"]["scenario"]
    if name not in SCENARIOS:
        pytest.skip(f"cassette {name} has no matching scenario")
    first = _observed(_run(name))
    second = _observed(_run(name))
    assert first["tool_sequence"] == second["tool_sequence"]
    assert first["final_state"] == second["final_state"]
    assert first["guard_pass"] == second["guard_pass"]


def test_replay_miss_raises_rather_than_calling_the_api(runs_db):
    from dossier.agent.llm import LLMReplayMiss, complete

    with cassette("plain_answer", mode="replay"), pytest.raises(LLMReplayMiss):
        complete(system="a system prompt no cassette ever saw", messages=[{"role": "user", "content": "x"}])


def test_no_cassette_recorded_an_api_key(runs_db):
    """A cassette must never contain a credential -- they are committed to the repo."""
    for path in CASSETTES:
        body = path.read_text()
        assert "sk-ant-" not in body, f"{path.name} contains what looks like an API key"
        assert "x-api-key" not in body.lower(), f"{path.name} contains a credential header"
