"""The five-section memo assembly, run end to end against a scripted model.

The live memo run is the one thing this repository could not measure -- the API credit
balance was exhausted before a five-section run completed. What that left unproven was not
the model's prose, which no test can assert anyway, but the *assembly*: five sub-agents
running concurrently, five independent ledgers merging into one, and every section's
citations being rewritten through its own remap before the synthesizer sees them.

That assembly is the highest-risk code in the project. Each sub-agent numbers its evidence
from E1, so five sections all have an E1 meaning five different passages. If `Ledger.merge`
and `remap_ids` disagree by even one index, the memo's citations point at the wrong
passages -- silently, in a document whose entire selling point is that its claims are
traceable. It is exactly the failure this system exists to prevent, and it would not show
up as a crash.

So the model is scripted here and the retriever is a fixture, but everything between them
is the real thing: `subagents.memo`, the real `run_loop`, the real tool dispatch, the real
ledger merge, the real output guard, the real state machine and the real persistence.

The retrieval fixture is built to make a remap bug fatal rather than lucky. Every section
retrieves one passage unique to it and one passage shared by all five, and the order is
flipped between sections -- so `E1` means the shared passage in Business Overview and the
unique passage in Financial Profile. An implementation that merged ledgers without
remapping, or that remapped with an off-by-one, produces a memo whose citations resolve to
the wrong text, and `test_every_citation_resolves_to_the_passage_its_section_cited` fails.

What this does **not** claim: nothing here measures model quality, cost, or the
parallel-versus-sequential wall time. A scripted model returns instantly, so timing it
would measure nothing. Those numbers still need an account with credit, and the README
says so rather than quoting a number from this test.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from dossier.agent import state as S
from dossier.agent import subagents
from dossier.agent.deal import Deal
from dossier.agent.llm import LLMResponse
from dossier.retrieve.types import RetrievedChunk

# The passage every section retrieves. Its numbers are the ones the drafts are allowed to
# quote: the output guard requires a number in the memo to appear in a passage the memo
# actually cites, so a typo here fails the run rather than passing quietly.
SHARED = RetrievedChunk(
    chunk_id="shared::costco::rev",
    text="Total revenue was $242,290 million in fiscal 2023, compared with $226,954 million in fiscal 2022.",
    citation="Costco 10K 2023, p.31",
    score=0.91,
    channels_hit=["vector", "bm25"],
    doc_name="COSTCO_2023_10K",
    company="Costco",
    doc_type="10K",
    fiscal_period="2023",
    page_num=31,
    rerank_score=0.88,
)

# One passage per section, unique to it. The text is what the assertion checks a citation
# resolved to, so each must be distinguishable from every other.
UNIQUE = {
    "Business Overview": (
        "unique::business",
        "The company operates 861 warehouses worldwide and sells memberships to individual and business members.",
        14,
    ),
    "Financial Profile": (
        "unique::financial",
        "Membership fee revenue totaled $4,580 million in fiscal 2023, recognized ratably over the membership term.",
        33,
    ),
    "Competitive Position": (
        "unique::competitive",
        "The company competes with wholesale clubs, supermarkets, internet retailers and category killers.",
        9,
    ),
    "Key Risks": (
        "unique::risks",
        "A substantial portion of merchandise is sourced from suppliers outside the United States, exposing operations to trade policy.",
        22,
    ),
    "Open Diligence Questions": (
        "unique::questions",
        "Renewal rates for paid memberships were disclosed on a rolling basis and vary by region.",
        35,
    ),
}

# Sections that see the shared passage first. Flipping the order between sections is what
# makes E1 mean different passages in different sections.
SHARED_FIRST = {"Business Overview", "Competitive Position", "Open Diligence Questions"}


def _chunk_for(section: str) -> RetrievedChunk:
    cid, text, page = UNIQUE[section]
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        citation=f"Costco 10K 2023, p.{page}",
        score=0.87,
        channels_hit=["vector"],
        doc_name="COSTCO_2023_10K",
        company="Costco",
        doc_type="10K",
        fiscal_period="2023",
        page_num=page,
        rerank_score=0.81,
    )


def _section_of(text: str) -> str:
    for name in UNIQUE:
        if name in text:
            return name
    raise AssertionError(f"no section name found in: {text[:200]!r}")


class ScriptedRetriever:
    """Returns two passages per section, ordered so that E1 differs across sections."""

    graph = None

    def search(self, query, deal=None, filters=None, channels=None, rerank=True, top_n=8):
        section = _section_of(query)
        unique = _chunk_for(section)
        return [SHARED, unique] if section in SHARED_FIRST else [unique, SHARED]


# Matches the `[E1] citation\n<document>\ntext\n</document>` blocks the search tool returns,
# so the scripted model reads its evidence ids out of the tool result the way a real model
# would rather than assuming an ordering.
_DOC_RE = re.compile(r"\[([EFC]\d+)\][^\n]*\n<document>\n(.*?)\n</document>", re.S)


def _ids_from_tool_result(content) -> dict[str, str]:
    """Map passage text -> evidence id, from the last tool result in the conversation."""
    blob = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                c = block.get("content")
                blob += c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [])
    else:
        blob = str(content)
    return {text.strip(): eid for eid, text in _DOC_RE.findall(blob)}


def _resp(content, stop_reason="tool_use") -> LLMResponse:
    return LLMResponse(
        content=content,
        stop_reason=stop_reason,
        model="scripted",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.0,
        latency_s=0.0,
    )


def scripted_section_complete(system=None, messages=None, tools=None, **kw) -> LLMResponse:
    """Two turns per section: search, then draft citing both retrieved passages."""
    section = _section_of(str(messages[0]["content"]))
    if len(messages) == 1:
        return _resp(
            [
                {"type": "text", "text": f"Retrieving evidence for {section}."},
                {
                    "type": "tool_use",
                    "id": "tu_search",
                    "name": "search_filings",
                    "input": {"query": f"{section} Costco fiscal 2023", "company": "Costco"},
                },
            ]
        )

    by_text = _ids_from_tool_result(messages[-1]["content"])
    shared_id = by_text[SHARED.text]
    unique_text = UNIQUE[section][1]
    unique_id = by_text[unique_text]

    # The unique sentence is quoted verbatim so the assertion can check which passage a
    # citation resolved to. Numbers appear only in the shared sentence, which is cited.
    markdown = (
        f"{unique_text} [{unique_id}] "
        f"Total revenue was $242,290 million in fiscal 2023, compared with $226,954 million in fiscal 2022. [{shared_id}]"
    )
    return _resp(
        [
            {
                "type": "tool_use",
                "id": "tu_draft",
                "name": "draft_section",
                "input": {"section": section, "markdown": markdown, "evidence_ids": [unique_id, shared_id]},
            }
        ]
    )


def scripted_synthesizer_complete(system=None, messages=None, **kw) -> LLMResponse:
    """Echo the already-remapped section bodies.

    A real synthesizer rewrites prose. This one must not: the property under test is that
    the ids reaching it are already correct, so anything it changed would only obscure a
    remap bug.
    """
    user = str(messages[0]["content"])
    body = user.split("## Section drafts", 1)[1].strip()
    return _resp([{"type": "text", "text": f"# Diligence memo: Costco\n\n{body}"}], stop_reason="end_turn")


@pytest.fixture
def scripted(monkeypatch):
    monkeypatch.setattr("dossier.agent.loop.complete", scripted_section_complete)
    monkeypatch.setattr("dossier.agent.subagents.complete", scripted_synthesizer_complete)
    monkeypatch.setattr("dossier.retrieve.hybrid.get_retriever", lambda *a, **k: ScriptedRetriever())
    return Deal(id="deal_test", target="Costco", peers=["Kroger", "Walmart"], thesis="Membership durability")


def _memo(deal, sequential=False) -> dict:
    return asyncio.run(subagents.memo(deal, sequential=sequential))


def _section_bodies(markdown: str) -> dict[str, str]:
    out, current = {}, None
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            out[current] = ""
        elif current:
            out[current] += line + "\n"
    return out


def test_all_five_sections_are_drafted_and_assembled(scripted, runs_db):
    result = _memo(scripted)

    assert [s["section"] for s in result["sections"]] == [name for name, _ in subagents.SECTIONS]
    bodies = _section_bodies(result["markdown"])
    for name in UNIQUE:
        assert name in bodies, f"{name} missing from the assembled memo"
        assert bodies[name].strip(), f"{name} assembled empty"


def test_every_citation_resolves_to_the_passage_its_section_cited(scripted, runs_db):
    """The remap property: a citation must resolve to the text that section retrieved.

    This is the assertion the whole file exists for. Five sections each numbered their
    evidence from E1 over different passages; after the merge, every id in the memo must
    point at the passage its own section meant.
    """
    result = _memo(scripted)
    run = S.RunState.load(result["run_id"])
    merged = run.load_ledger()

    bodies = _section_bodies(result["markdown"])
    for section, (_, unique_text, _) in UNIQUE.items():
        ids = re.findall(r"\[([EFC]\d+)\]", bodies[section])
        assert len(ids) == 2, f"{section} lost a citation in assembly: {ids}"

        unique_item = merged.get(ids[0])
        shared_item = merged.get(ids[1])
        assert unique_item is not None and shared_item is not None, f"{section} cites an id not in the merged ledger"
        assert unique_item.text == unique_text, (
            f"{section}'s first citation {ids[0]} resolves to {unique_item.text[:60]!r}, "
            f"but that section cited {unique_text[:60]!r} -- the ledger merge remapped it wrong"
        )
        assert shared_item.text == SHARED.text, (
            f"{section}'s second citation {ids[1]} resolves to the wrong passage after merge"
        )


def test_each_sections_recorded_evidence_ids_match_the_ids_in_its_prose(scripted, runs_db):
    """The id list travels with the prose, or the memo's provenance is a lie.

    `markdown` and `evidence_ids` are remapped separately in `memo`, so they can drift
    apart: the prose can cite the right passages while the recorded list -- what gets
    persisted, and what the memo-level guard is handed as the cited set -- still points at
    the sub-agent's pre-merge numbering. Nothing else in the suite compares the two.
    """
    result = _memo(scripted)
    merged = S.RunState.load(result["run_id"]).load_ledger()
    bodies = _section_bodies(result["markdown"])

    for draft in result["sections"]:
        in_prose = re.findall(r"\[([EFC]\d+)\]", bodies[draft["section"]])
        assert sorted(draft["evidence_ids"]) == sorted(in_prose), (
            f"{draft['section']} records {draft['evidence_ids']} but cites {in_prose} -- "
            "the id list was not remapped alongside the prose"
        )
        for eid in draft["evidence_ids"]:
            assert merged.get(eid) is not None, f"{draft['section']} records id {eid}, absent from the merged ledger"


def test_the_shared_passage_is_stored_once_and_the_unique_ones_survive(scripted, runs_db):
    """Merge dedupes on source key, so five retrievals of one passage make one entry."""
    result = _memo(scripted)
    merged = S.RunState.load(result["run_id"]).load_ledger()

    texts = [item.text for item in merged.items()]
    assert texts.count(SHARED.text) == 1, "the shared passage was duplicated across sections"
    for _, unique_text, _ in UNIQUE.values():
        assert texts.count(unique_text) == 1, f"lost or duplicated {unique_text[:40]!r}"
    # five unique + one shared, and nothing invented along the way
    assert len(merged) == 6 == result["evidence_count"]


def test_the_assembled_memo_passes_the_output_guard(scripted, runs_db):
    """Every number in the memo comes from a passage the memo cites, so the guard clears."""
    result = _memo(scripted)
    assert result["guard_pass"], result["guard_report"]
    assert not result["guard_report"]["violations"]


def test_the_parent_run_reaches_final_and_persists_its_sections(scripted, runs_db):
    result = _memo(scripted)
    assert result["state"] == S.FINAL
    assert not result["pending_reviews"]

    run = S.RunState.load(result["run_id"])
    stored = {s["section"] for s in run.sections()}
    assert stored == set(UNIQUE)
    assert run.meta["section_runs"] and len(run.meta["section_runs"]) == 5


def test_sequential_and_parallel_assemble_the_same_memo(scripted, runs_db):
    """`--sequential` must differ only in scheduling, never in what it produces."""
    parallel = _memo(scripted, sequential=False)
    sequential = _memo(scripted, sequential=True)

    assert sequential["sequential"] is True and parallel["sequential"] is False
    assert sequential["markdown"] == parallel["markdown"]
    assert sequential["evidence_count"] == parallel["evidence_count"]
