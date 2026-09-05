"""Cost ledger.

One price table in config.py, one function that turns (model, tokens_in, tokens_out) into
dollars, and one aggregation over the recorded spans. Correcting a price re-costs history
because nothing stores a dollar figure that was computed anywhere else.
"""

from __future__ import annotations

import json

from ..config import MODEL_PRICES, get_config

# Unknown models fall back to the most expensive tier so cost caps stay conservative.
_DEFAULT_PRICE = (5.00, 25.00)


def price_for(model: str) -> tuple[float, float]:
    if model in MODEL_PRICES:
        return MODEL_PRICES[model]
    for known, price in MODEL_PRICES.items():
        if model.startswith(known):
            return price
    return _DEFAULT_PRICE


def call_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = price_for(model)
    return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout


def run_cost(run_id: str) -> float:
    from .tracer import get_tracer

    total = 0.0
    for s in get_tracer().spans_for(run_id):
        total += float(s["attrs"].get("cost_usd") or 0)
    return round(total, 6)


def summarize(by: str = "run_type") -> list[dict]:
    """Aggregate llm_call spans. `by` is one of run_type | prompt_version | model."""
    from .tracer import get_tracer

    tracer = get_tracer()
    rows = tracer._conn.execute(
        "SELECT run_id, kind, name, attrs FROM spans WHERE kind IN ('llm_call','judge')"
    ).fetchall()
    run_types = {
        r["run_id"]: r["name"]
        for r in tracer._conn.execute("SELECT run_id, name FROM spans WHERE kind = 'run'").fetchall()
    }
    agg: dict[str, dict] = {}
    for r in rows:
        a = json.loads(r["attrs"])
        if by == "prompt_version":
            key = f"{a.get('prompt_name', '?')}@{a.get('prompt_version', '?')}"
        elif by == "model":
            key = a.get("model", "?")
        else:
            key = run_types.get(r["run_id"], "unknown")
        slot = agg.setdefault(key, {"key": key, "calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
        slot["calls"] += 1
        slot["tokens_in"] += int(a.get("tokens_in") or 0)
        slot["tokens_out"] += int(a.get("tokens_out") or 0)
        slot["cost_usd"] += float(a.get("cost_usd") or 0)
    out = sorted(agg.values(), key=lambda r: -r["cost_usd"])
    for r in out:
        r["cost_usd"] = round(r["cost_usd"], 6)
    return out


def price_table() -> dict[str, dict[str, float]]:
    cfg = get_config()
    return {
        m: {"input_per_mtok": p[0], "output_per_mtok": p[1]}
        for m, p in MODEL_PRICES.items()
        if m in {cfg.primary_model, cfg.fallback_model, cfg.cheap_model, cfg.judge_model}
    }
