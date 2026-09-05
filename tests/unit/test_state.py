"""State machine: transitions are validated, persisted, and resumable."""

import pytest

from dossier.agent import state as S
from dossier.agent.ledger import Ledger


def test_run_starts_in_plan(runs_db):
    rs = S.RunState.create("ask", question="q")
    assert rs.state == S.PLAN


def test_valid_path_to_final(runs_db):
    rs = S.RunState.create("ask")
    for target in (S.GATHER, S.ANALYZE, S.DRAFT, S.FINAL):
        rs.transition(target)
    assert rs.state == S.FINAL


def test_plan_cannot_jump_straight_to_final(runs_db):
    rs = S.RunState.create("ask")
    with pytest.raises(S.InvalidTransition):
        rs.transition(S.FINAL)


def test_final_is_terminal(runs_db):
    rs = S.RunState.create("ask")
    rs.transition(S.GATHER)
    rs.transition(S.ANALYZE)
    rs.transition(S.FINAL)
    with pytest.raises(S.InvalidTransition):
        rs.transition(S.GATHER)


def test_unknown_state_is_rejected(runs_db):
    rs = S.RunState.create("ask")
    with pytest.raises(S.InvalidTransition):
        rs.transition("SHIPPING")


def test_fail_is_reachable_from_anywhere_and_records_the_reason(runs_db):
    rs = S.RunState.create("ask")
    rs.transition(S.GATHER)
    rs.fail("tool exploded")
    assert rs.state == S.FAILED
    assert rs.meta["failure"] == "tool exploded"


def test_review_resumes_into_draft(runs_db):
    rs = S.RunState.create("memo_section")
    rs.transition(S.GATHER)
    rs.transition(S.ANALYZE)
    rs.transition(S.REVIEW)
    rs.transition(S.DRAFT, "review approved")
    assert rs.state == S.DRAFT


def test_transitions_are_persisted_and_replayable(runs_db):
    rs = S.RunState.create("ask", question="what was capex")
    rs.transition(S.GATHER, "first tool call")
    rs.transition(S.ANALYZE, "terminal tool")
    reloaded = S.RunState.load(rs.run_id)
    assert reloaded.state == S.ANALYZE
    assert [t["to_state"] for t in reloaded.transitions()] == [S.PLAN, S.GATHER, S.ANALYZE]
    assert "first tool call" in [t["reason"] for t in reloaded.transitions()]


def test_messages_and_ledger_survive_a_reload(runs_db, chunk_factory):
    rs = S.RunState.create("ask", question="q")
    rs.save_messages([{"role": "user", "content": "hello"}])
    led = Ledger()
    led.add_chunk(chunk_factory(chunk_id="c1"))
    rs.save_ledger(led)
    reloaded = S.RunState.load(rs.run_id)
    assert reloaded.load_messages() == [{"role": "user", "content": "hello"}]
    assert reloaded.load_ledger().get("E1").text == led.get("E1").text


def test_summary_mentions_the_path_and_any_review_notes(runs_db):
    rs = S.RunState.create("ask")
    rs.transition(S.GATHER)
    rs.save_meta(review_notes="tighten the peer comparison")
    summary = rs.summary()
    assert "GATHER" in summary and "tighten the peer comparison" in summary


def test_unknown_run_raises(runs_db):
    with pytest.raises(KeyError):
        S.RunState.load("run_nonexistent")
