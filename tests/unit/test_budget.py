"""Budget manager: compacts the oldest tool results, keeps ids, records what it saved."""

from dossier.agent.budget import BudgetManager, approx_tokens, conversation_tokens, strip_internal_keys


def exact_tokens(text: str) -> int:
    """A deterministic fake tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def _tool_result(tool_use_id: str, body: str) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": body}]}


def _conversation(n: int, words: int = 100) -> list[dict]:
    msgs = [{"role": "user", "content": "what was capital expenditure"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "search_filings", "input": {}}]})
        msgs.append(_tool_result(f"t{i}", f"[E{i + 1}] 3M 10K 2018, p.60 " + " ".join(["passage"] * words)))
    return msgs


def test_under_budget_is_untouched(runs_db):
    bm = BudgetManager(max_context_tokens=10_000, counter=exact_tokens)
    msgs = _conversation(2)
    assert bm.maybe_compact(msgs) is msgs
    assert bm.compactions == 0


def test_compaction_reduces_tokens_below_the_cap(runs_db):
    bm = BudgetManager(max_context_tokens=300, counter=exact_tokens, keep_recent=2)
    msgs = _conversation(6, words=100)
    before = conversation_tokens(msgs, exact_tokens)
    out = bm.compact(msgs, run_id="test_run")
    after = conversation_tokens(out, exact_tokens)
    assert after < before
    assert bm.compactions == 1
    assert bm.tokens_saved == before - after


def test_compaction_preserves_evidence_ids(runs_db):
    bm = BudgetManager(max_context_tokens=200, counter=exact_tokens, keep_recent=2)
    out = bm.compact(_conversation(6, words=100), run_id="test_run")
    compacted = [
        b for m in out if isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("_compacted")
    ]
    assert compacted, "expected at least one compacted tool result"
    for block in compacted:
        assert "E" in block["content"]
        assert "ledger" in block["content"].lower()


def test_recent_messages_are_never_compacted(runs_db):
    bm = BudgetManager(max_context_tokens=50, counter=exact_tokens, keep_recent=4)
    msgs = _conversation(6, words=100)
    out = bm.compact(msgs, run_id="test_run")
    for m in out[-4:]:
        if isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    assert not b.get("_compacted")


def test_compaction_emits_a_span(runs_db):
    from dossier.obs.tracer import get_tracer

    bm = BudgetManager(max_context_tokens=200, counter=exact_tokens, keep_recent=2)
    bm.compact(_conversation(6, words=100), run_id="span_run")
    spans = get_tracer().spans_for("span_run")
    kinds = [s["kind"] for s in spans]
    assert "compaction" in kinds
    attrs = next(s["attrs"] for s in spans if s["kind"] == "compaction")
    assert attrs["tokens_saved"] > 0


def test_internal_keys_are_stripped_before_the_api_sees_them():
    msgs = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x", "_compacted": True}]}]
    clean = strip_internal_keys(msgs)
    assert "_compacted" not in clean[0]["content"][0]


def test_approx_tokens_is_monotonic():
    assert approx_tokens("a" * 4) <= approx_tokens("a" * 400)
