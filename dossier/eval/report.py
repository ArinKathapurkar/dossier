"""Rendering for eval reports. Same data the JSON under `runs/reports/` carries."""

from __future__ import annotations

from typing import Any


def _table(console, title: str, columns: list[str], rows: list[list[Any]]) -> None:
    from rich.table import Table

    t = Table(title=title, header_style="bold")
    for i, c in enumerate(columns):
        t.add_column(c, justify="left" if i == 0 else "right")
    for r in rows:
        t.add_row(*[str(x) for x in r])
    console.print(t)


def print_tier1(rep: dict, console) -> None:
    rows = []
    for name, m in rep["configs"].items():
        rows.append([name, m["recall@5"], m["recall@10"], m["ndcg@10"], m["mrr"], m["latency_p50_ms"], m["latency_p95_ms"]])
    _table(
        console,
        f"Tier 1 retrieval ({rep['mode']}, n={rep['questions_scored']} questions over {rep['documents_ingested']} documents)",
        ["config", "R@5", "R@10", "nDCG@10", "MRR", "p50 ms", "p95 ms"],
        rows,
    )
    console.print(
        f"[dim]evidence match rate {rep['evidence_match_rate']:.3f} "
        f"({rep['matched']}/{rep['evidence_rows']} gold evidence strings found in a chunk of the right page) "
        f"— this is the ceiling on achievable recall. "
        f"gold page offset {rep['gold_page_offset']:+d}. {rep['elapsed_s']}s[/]"
    )


def _rate(value) -> str:
    return "n/a (not scored)" if value is None else f"{value:.3f}"


def print_tier2(rep: dict, console) -> None:
    m = rep["metrics"]
    _table(
        console,
        f"Tier 2 grounding and guardrails (n={rep['answers_checked']} answers, {rep['fixtures']} fixtures)",
        ["metric", "value"],
        [
            ["citation validity", _rate(m["citation_validity"])],
            ["numeric grounding", _rate(m["numeric_grounding"])],
            ["forward-looking attribution", _rate(m["forward_looking_attribution"])],
            [f"abstention accuracy ({rep['ooc_scored']}/{rep['ooc_questions']} out-of-corpus)", _rate(m["abstention_accuracy"])],
            ["fixture verdict accuracy", _rate(m["fixture_accuracy"])],
            ["fallback spans per run", f"{m['fallback_spans_per_run']:.2f}"],
        ],
    )


def print_tier3(rep: dict, console) -> None:
    rows = [
        [qt, v["n"], f"{v['correct']:.3f}", f"{v['partially_correct']:.3f}", f"{v['incorrect']:.3f}", f"{v['abstained']:.3f}"]
        for qt, v in rep["by_question_type"].items()
    ]
    rows.append(
        [
            "ALL",
            rep["n"],
            f"{rep['overall']['correct']:.3f}",
            f"{rep['overall']['partially_correct']:.3f}",
            f"{rep['overall']['incorrect']:.3f}",
            f"{rep['overall']['abstained']:.3f}",
        ]
    )
    _table(console, f"Tier 3 answer correctness (n={rep['n']})", ["question type", "n", "correct", "partial", "incorrect", "abstained"], rows)
    console.print(
        f"[dim]mean cost ${rep['mean_cost_usd']:.4f}/question · mean latency {rep['mean_latency_s']:.1f}s · "
        f"revise-loop rate {rep['revise_rate']:.3f} · HITL escalation {rep['hitl_rate']:.3f} · "
        f"fallback spans per 100 runs {rep['fallback_per_100_runs']:.1f} · total ${rep['total_cost_usd']:.2f}[/]"
    )


def print_compare(rep: dict, console) -> None:
    _table(
        console,
        f"Prompt comparison — tier {rep['tier']}: A={rep['a']} vs B={rep['b']}",
        ["metric", "A", "B", "Δ"],
        [[r["metric"], r["a"], r["b"], f"{r['delta']:+.4f}"] for r in rep["rows"]],
    )
