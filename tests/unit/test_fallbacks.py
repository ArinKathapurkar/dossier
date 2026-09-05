"""Fallback chain: degrade, record a span, never crash the run."""

from dossier.guard.fallbacks import tool_error, with_fallback
from dossier.obs.tracer import get_tracer


def test_primary_result_is_returned_when_it_works(runs_db):
    assert with_fallback(lambda: "primary", lambda: "fallback", "x") == "primary"
    assert get_tracer().count_kind("test_run", "fallback") == 0


def test_failure_falls_back_and_records_a_span(runs_db):
    def boom():
        raise RuntimeError("model failed to load")

    assert with_fallback(boom, lambda: "rrf order", "reranker", frm="cross_encoder", to="rrf_order") == "rrf order"
    spans = [s for s in get_tracer().spans_for("test_run") if s["kind"] == "fallback"]
    assert len(spans) == 1
    assert spans[0]["attrs"]["from"] == "cross_encoder"
    assert spans[0]["attrs"]["to"] == "rrf_order"
    assert "RuntimeError" in spans[0]["attrs"]["error"]


def test_timeout_counts_as_a_fallback(runs_db):
    def slow():
        raise TimeoutError("exceeded 10s")

    assert with_fallback(slow, lambda: [], "reranker") == []
    assert get_tracer().count_kind("test_run", "fallback") == 1


def test_reraise_list_bypasses_the_fallback(runs_db):
    import pytest

    class Fatal(Exception):
        pass

    def boom():
        raise Fatal()

    with pytest.raises(Fatal):
        with_fallback(boom, lambda: "x", "y", reraise=(Fatal,))


def test_tool_error_returns_a_structured_result_the_model_can_act_on(runs_db):
    out = tool_error("get_financials", ValueError("unknown concept"), hint="Use a us-gaap concept name.")
    assert "ValueError" in out["error"]
    assert out["hint"] == "Use a us-gaap concept name."
    assert get_tracer().count_kind("test_run", "fallback") == 1
