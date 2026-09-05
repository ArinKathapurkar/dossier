"""`dossier` command line.

Every command prints a one-line summary and writes machine-readable JSON under
`runs/reports/`, so the numbers quoted in the README can be regenerated rather than
retyped.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="Agentic due-diligence copilot over SEC filings.")
console = Console()

eval_app = typer.Typer(help="Evaluation harness (tiers 1-3, comparisons, regression export).")
app.add_typer(eval_app, name="eval")
review_app = typer.Typer(help="Human-in-the-loop review queue.")
app.add_typer(review_app, name="review")
deal_app = typer.Typer(help="Deals: the unit of cross-run memory.")
app.add_typer(deal_app, name="deal")


def _emit(name: str, payload: dict) -> None:
    from .config import get_config

    reports = get_config().paths.reports
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------------
# ingest / index
# ---------------------------------------------------------------------------------


@app.command()
def ingest(
    limit_docs: int = typer.Option(0, help="Only ingest the first N documents (smoke runs)."),
    skip_xbrl: bool = typer.Option(False, help="Skip the SEC companyfacts pull."),
) -> None:
    """Build the manifest, fetch PDFs, extract pages, chunk, and load XBRL facts."""
    from .ingest.pipeline import run_ingest

    rep = run_ingest(limit_docs=limit_docs or None, skip_xbrl=skip_xbrl)
    conv = rep.get("page_convention", {}).get("convention", "?")
    xb = rep.get("xbrl", {})
    console.print(
        f"[bold green]ingest[/] docs {rep['documents_fetched']}/{rep['documents_in_benchmark']} · "
        f"pages {rep['pages']} · chunks {rep['chunks']} · "
        f"facts {xb.get('rows_total', 0)} · "
        f"CIK match {rep.get('companies_with_cik', 0)}/{rep.get('companies', 0)} · "
        f"page-index {conv} · {rep['elapsed_s']}s"
    )


@app.command()
def index(
    graph: str = typer.Option("networkx", help="Graph backend: networkx | neo4j."),
    skip_graph: bool = typer.Option(False, help="Skip entity extraction (which costs API tokens)."),
    skip_vectors: bool = typer.Option(False, help="Reuse the existing vector and BM25 indexes."),
    limit_docs: int = typer.Option(0, help="Only extract entities from the first N documents."),
) -> None:
    """Build the vector index, the BM25 index, and the entity graph."""
    from .index.pipeline import run_index

    rep = run_index(graph_backend=graph, skip_graph=skip_graph, skip_vectors=skip_vectors, limit_docs=limit_docs or None)
    g = rep.get("graph", {})
    console.print(
        f"[bold green]index[/] vectors {rep['vectors']} ({rep['vector_index_type']}) · "
        f"bm25 {rep['bm25_docs']} · device {rep['device']} · "
        f"graph {g.get('entities', 0)}e/{g.get('relations', 0)}r "
        f"(${g.get('cost_usd', 0):.4f}, {g.get('elapsed_s', 0)}s) · {rep['elapsed_s']}s"
    )


# ---------------------------------------------------------------------------------
# deals, ask, memo, resume
# ---------------------------------------------------------------------------------


@deal_app.command("create")
def deal_create(
    target: str = typer.Option(..., "--target"),
    peers: list[str] = typer.Option([], "--peers"),
    thesis: str = typer.Option("", "--thesis"),
) -> None:
    from .agent.deal import create_deal

    d = create_deal(target=target, peers=list(peers), thesis=thesis)
    console.print(f"[bold green]deal[/] {d.id} target={d.target} peers={', '.join(d.peers) or '-'}")


@deal_app.command("list")
def deal_list() -> None:
    from .agent.deal import list_deals

    for d in list_deals():
        console.print(f"{d.id}  {d.target}  peers={','.join(d.peers)}  findings={len(d.findings)}")


@app.command()
def ask(deal_id: str, question: str, sequential: bool = typer.Option(False, hidden=True)) -> None:
    """Answer one analyst question against the indexed filings."""
    from .agent.deal import get_deal
    from .agent.loop import ask as run_ask

    res = run_ask(get_deal(deal_id), question)
    console.print(res.answer_markdown or "(no answer)")
    console.print(
        f"\n[dim]run {res.run_id} · state {res.state} · turns {res.turns} · "
        f"${res.cost_usd:.4f} · guard {'pass' if res.guard_pass else 'FAIL'} · "
        f"evidence {len(res.evidence_ids)}[/]"
    )


@app.command()
def memo(
    deal_id: str,
    sequential: bool = typer.Option(False, help="Run section sub-agents one at a time (for timing comparison)."),
) -> None:
    """Draft a five-section diligence memo with parallel section sub-agents."""
    import asyncio

    from .agent.deal import get_deal
    from .agent.subagents import memo as run_memo

    res = asyncio.run(run_memo(get_deal(deal_id), sequential=sequential))
    console.print(res["markdown"])
    console.print(
        f"\n[dim]run {res['run_id']} · state {res['state']} · sections {len(res['sections'])} · "
        f"${res['cost_usd']:.4f} · {res['elapsed_s']}s ({'sequential' if sequential else 'parallel'})[/]"
    )


@app.command()
def resume(run_id: str) -> None:
    """Continue a run that stopped for human review."""
    from .agent.loop import resume_run

    res = resume_run(run_id)
    console.print(res.answer_markdown or "(no answer)")
    console.print(f"\n[dim]run {res.run_id} · state {res.state} · ${res.cost_usd:.4f}[/]")


# ---------------------------------------------------------------------------------
# review queue
# ---------------------------------------------------------------------------------


@review_app.command("list")
def review_list(status: str = typer.Option("pending")) -> None:
    from .hitl.queue import list_reviews

    rows = list_reviews(status=None if status == "all" else status)
    for r in rows:
        console.print(f"{r['id']}  run={r['run_id']}  section={r['section']}  status={r['status']}  {r['reason'][:70]}")
    console.print(f"[dim]{len(rows)} review(s)[/]")


@review_app.command("show")
def review_show(review_id: str) -> None:
    from .hitl.queue import get_review

    r = get_review(review_id)
    console.print(f"[bold]{r['section']}[/] · run {r['run_id']} · {r['status']}")
    console.print(f"[yellow]reason:[/] {r['reason']}\n")
    console.print(r["draft"])
    if r.get("guard_report"):
        console.print("\n[red]guard violations:[/]")
        for v in json.loads(r["guard_report"]).get("violations", []):
            console.print(f"  · {v['kind']}: {v['detail']}")


@review_app.command("decide")
def review_decide(
    review_id: str,
    approve: bool = typer.Option(False, "--approve"),
    edit: Path = typer.Option(None, "--edit", help="Markdown file with the reviewer's edited text."),
    reject: bool = typer.Option(False, "--reject"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    from .hitl.queue import decide

    if edit:
        decision, text = "edited", edit.read_text()
    elif approve:
        decision, text = "approved", None
    elif reject:
        decision, text = "rejected", None
    else:
        raise typer.BadParameter("one of --approve / --edit / --reject is required")
    res = decide(review_id, decision, edited_text=text, notes=notes)
    console.print(f"[bold green]review {review_id}[/] -> {decision}; run {res['run_id']} state {res['state']}")


# ---------------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------------


@app.command()
def trace(run_id: str) -> None:
    """Print the span tree for a run with durations, tokens and cost."""
    from .obs.tracer import print_trace

    print_trace(run_id, console)


@app.command()
def cost(by: str = typer.Option("run_type", help="run_type | prompt_version | model")) -> None:
    """Summarize spend from the recorded spans."""
    from .obs.cost import summarize

    rows = summarize(by)
    for r in rows:
        console.print(f"{r['key']:<40} calls {r['calls']:>5}  in {r['tokens_in']:>9}  out {r['tokens_out']:>8}  ${r['cost_usd']:.4f}")
    _emit("cost", {"by": by, "rows": rows})


@app.command()
def prompts() -> None:
    """List the prompt registry with content-hash versions."""
    from .obs.prompts import registry

    for name, version in sorted(registry().versions().items()):
        console.print(f"{name:<28} {version}")


# ---------------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------------


@eval_app.command("tier1")
def eval_tier1(
    limit: int = typer.Option(0, help="Only score the first N questions."),
    mini: bool = typer.Option(False, help="Use the committed mini-corpus fixture (offline, CI)."),
    filtered: bool = typer.Option(False, help="Apply each question's company as a filter (what the agent issues)."),
) -> None:
    from .eval.tier1_retrieval import run_tier1

    rep = run_tier1(limit=limit or None, mini=mini, filtered=filtered)
    from .eval.report import print_tier1

    print_tier1(rep, console)
    _emit(("tier1_mini" if mini else "tier1") + ("_filtered" if filtered else ""), rep)


@eval_app.command("tier2")
def eval_tier2(
    mini: bool = typer.Option(False, help="Fixtures only; no run history required."),
    run_ooc: bool = typer.Option(False, help="First run the 15 out-of-corpus questions (costs API tokens)."),
) -> None:
    from .eval.report import print_tier2
    from .eval.tier2_grounding import run_ooc_questions, run_tier2

    if run_ooc:
        ooc = run_ooc_questions()
        console.print(
            f"[dim]out-of-corpus probe: {ooc['abstained']}/{ooc['questions']} abstained, ${ooc['cost_usd']:.4f}[/]"
        )
        _emit("tier2_ooc", ooc)
    rep = run_tier2(fixtures_only=mini)
    print_tier2(rep, console)
    _emit("tier2_mini" if mini else "tier2", rep)


@eval_app.command("tier3")
def eval_tier3(limit: int = typer.Option(50, help="Questions to run. Costs API tokens.")) -> None:
    from .eval.report import print_tier3
    from .eval.tier3_judge import run_tier3

    rep = run_tier3(limit=limit)
    print_tier3(rep, console)
    _emit("tier3", rep)


@eval_app.command("compare")
def eval_compare(
    tier: int = typer.Option(1, "--tier"),
    prompts_a: str = typer.Argument(..., metavar="A"),
    prompts_b: str = typer.Argument(..., metavar="B"),
    limit: int = typer.Option(0),
) -> None:
    """Run a tier under two prompt sets (git ref or directory) and diff the metrics."""
    from .eval.compare import run_compare

    rep = run_compare(tier=tier, a=prompts_a, b=prompts_b, limit=limit or None)
    from .eval.report import print_compare

    print_compare(rep, console)
    _emit(f"compare_tier{tier}", rep)


@eval_app.command("export-reviews")
def eval_export_reviews() -> None:
    """Turn reviewer decisions into regression cases."""
    from .eval.regression import export_reviews

    rep = export_reviews()
    console.print(f"[bold green]exported[/] {rep['written']} regression case(s) -> {rep['dir']}")


# ---------------------------------------------------------------------------------
# service / demo
# ---------------------------------------------------------------------------------


@app.command()
def serve(port: int = typer.Option(8000), host: str = typer.Option("0.0.0.0")) -> None:
    import uvicorn

    uvicorn.run("dossier.api.app:app", host=host, port=port, log_level="info")


@app.command()
def demo(sequential: bool = typer.Option(False)) -> None:
    """Run the canonical demo deal: three asks plus a memo."""
    from .demo import run_demo

    rep = run_demo(sequential=sequential)
    console.print(f"[bold green]demo[/] deal {rep['deal_id']} · asks {len(rep['asks'])} · memo {rep['memo']['state']}")
    _emit("demo", rep)


@app.command()
def health() -> None:
    """Index counts, graph backend, model tiers -- same payload as GET /health."""
    from .api.health import health_payload

    console.print_json(json.dumps(health_payload()))


if __name__ == "__main__":  # pragma: no cover
    app()
