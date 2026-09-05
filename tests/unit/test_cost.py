"""Cost ledger: one price table, one arithmetic path, aggregations over real spans."""

from dossier.config import MODEL_PRICES
from dossier.obs.cost import call_cost, price_for, run_cost, summarize


def test_price_lookup_is_exact_for_known_models():
    assert price_for("claude-sonnet-5") == MODEL_PRICES["claude-sonnet-5"]


def test_dated_snapshot_ids_resolve_to_their_base_model():
    assert price_for("claude-haiku-4-5-20251001") == MODEL_PRICES["claude-haiku-4-5"]


def test_unknown_model_falls_back_to_the_most_expensive_tier():
    # Conservative on purpose: a run cost cap must never be under-counted.
    assert price_for("claude-something-unreleased") == (5.00, 25.00)


def test_call_cost_arithmetic():
    # 1M in + 1M out on sonnet-5 = $2 + $10
    assert call_cost("claude-sonnet-5", 1_000_000, 1_000_000) == 12.0
    assert call_cost("claude-sonnet-5", 0, 0) == 0.0


def test_run_cost_sums_span_attributes(runs_db):
    from dossier.obs.tracer import get_tracer

    t = get_tracer()
    t.bind("cost_run")
    t.event("llm_call", "a", {"model": "claude-sonnet-5", "tokens_in": 1000, "tokens_out": 500, "cost_usd": 0.007})
    t.event("llm_call", "b", {"model": "claude-haiku-4-5", "tokens_in": 2000, "tokens_out": 100, "cost_usd": 0.0025})
    assert abs(run_cost("cost_run") - 0.0095) < 1e-9


def test_summarize_groups_by_prompt_version(runs_db):
    from dossier.obs.tracer import get_tracer

    t = get_tracer()
    t.bind("sum_run")
    t.event("llm_call", "a", {"model": "claude-sonnet-5", "prompt_name": "system_analyst",
                              "prompt_version": "aaaa1111", "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.01})
    t.event("llm_call", "b", {"model": "claude-sonnet-5", "prompt_name": "system_analyst",
                              "prompt_version": "bbbb2222", "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.02})
    rows = {r["key"]: r for r in summarize("prompt_version")}
    assert "system_analyst@aaaa1111" in rows
    assert rows["system_analyst@bbbb2222"]["cost_usd"] == 0.02
