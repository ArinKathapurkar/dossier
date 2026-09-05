# dossier

An agentic due-diligence copilot over SEC filings. Given a target company and a peer set, it
answers analyst questions and drafts a diligence memo where **every claim is tied to a cited
filing passage or an XBRL fact**, and any section it cannot ground is routed to a human
before it ships.

---

## What this is, and what it isn't

**What it is.** A working system that reads **public SEC filings** — 10-K, 10-Q, 8-K and
earnings releases for 32 companies — and answers questions about them with citations, run
against a public benchmark so the claims below can be checked.

**What it isn't:**

- **Not a client system.** [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench)
  (Patronus AI) is a benchmark, not a customer. Nobody's deal ran through this.
- **Not investment advice.** It describes what filings say. Asked for a recommendation it
  refuses and redirects, and that refusal is tested.
- **Not validated by practising analysts.** The reviewer interface works and is tested; it
  has not been used in a real diligence workflow, and no claim is made about how it would
  perform in one.
- **Not fine-tuned.** Parameter-efficient fine-tuning is an explicit **non-goal** for this
  phase. The entity-extraction cache is a ready-made training set for it and the plan is
  written up in [`docs/DESIGN.md` §10](docs/DESIGN.md), but nothing here is trained.

**Every number below was produced by a command in this repository**, and every command
writes its output to `runs/reports/`. Where a measurement contradicts the architecture — and
one of them does, loudly — the measurement is what shipped.

---

## Architecture

```
                 ┌──────────────────────────────────────────────────────────────┐
                 │  FastAPI service (REST + SSE)          dossier CLI           │
                 └──────────────┬───────────────────────────────┬───────────────┘
                                ▼                               ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │  Agent runtime                                                             │
   │   · workflow state machine  PLAN→GATHER→ANALYZE→DRAFT→REVIEW→FINAL (SQLite)│
   │   · evidence ledger (working memory)   · token-budget manager              │
   │   · tool router  · parallel section sub-agents  · synthesizer              │
   │   · input guard  · output guard (numeric grounding)  · fallback chain      │
   └───────┬──────────────────┬──────────────────┬──────────────────┬───────────┘
           ▼                  ▼                  ▼                  ▼
   ┌──────────────┐  ┌───────────────┐  ┌────────────────┐  ┌──────────────────┐
   │ Hybrid       │  │ Structured    │  │ HITL review    │  │ Observability    │
   │ retrieval    │  │ facts (XBRL)  │  │ queue + UI     │  │ tracer · cost    │
   │ vector+BM25  │  │ SQLite        │  │ SQLite+Jinja2  │  │ prompt registry  │
   │ +graph → RRF │  └───────────────┘  └────────────────┘  └──────────────────┘
   │ → reranker   │
   └──────────────┘
           ▲
   ┌──────────────────────────────────────────────────────────────────────────┐
   │  Eval harness: Tier 1 retrieval · Tier 2 grounding/guardrails ·          │
   │  Tier 3 LLM judge · offline regression cassettes (run in CI, no API key) │
   └──────────────────────────────────────────────────────────────────────────┘
```

The agent loop is written directly against the Anthropic Messages API with tool use. No
LangChain, LlamaIndex, CrewAI or AutoGen — the loop, the context management and the state
machine are the substance here, and a framework would have made them configuration.

---

## How the guardrails and human review work

*Two paragraphs, no engineering background needed.*

**Every number has to come from somewhere.** When the system answers, it writes prose with
tags like `[E3]` or `[F2]` attached to each claim. Those tags point at a specific passage
from a specific page of a specific filing, or at a specific figure from the SEC's structured
financial data. Before an answer is returned, an automatic check pulls out every number in
it and verifies that the number actually appears in one of the cited sources, or was
calculated from cited sources by the system's own calculator. A number that cannot be traced
that way is rejected. The check is ordinary arithmetic and text matching, not another AI
model — so it gives the same verdict every time, cannot be talked out of its answer, and
runs on every response rather than on a sample.

**When the system cannot ground something, a person decides.** If a check fails, the system
is told what is wrong and given two chances to fix it. If it still cannot produce a grounded
answer, the draft does not ship: it goes into a review queue with the specific problem
attached, and the run stops. A reviewer sees the draft, the exact violations, and the
underlying filing passages side by side, and can approve it, edit it, or reject it with
notes — after which the run picks up where it stopped and finishes. The system can also
escalate on its own initiative, before any check fails, when a question asks for something
the filings do not support. Declining is treated as a correct outcome, not a failure.

Full risk-to-mitigation table, and an explicit list of what is *not* covered, in
[`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md).

---

## Measured results

### Corpus and index

`dossier ingest && dossier index --graph networkx` → `runs/reports/{ingest,index}.json`

| | |
|---|---|
| documents ingested / in benchmark | **84 / 84** |
| pages extracted | **12,013** (11,914 pypdf, 99 re-extracted with PyMuPDF) |
| chunks | **32,468** (350-token windows, never crossing a page) |
| questions | **150** · companies **32** |
| companies matched to a CIK | **30 / 32** (2 unmatched, logged not dropped) |
| XBRL facts | **55,670** |
| graph entities / relations | **4,604 / 5,781** |
| graph extraction cost / wall time | **$5.03** · **772 s** (1,003 model calls) |
| embedding wall time | **203 s** on MPS |
| MPS/CPU parity | mean cosine **1.000000** (threshold 0.999) — MPS trusted |

### The benchmark's page numbers are 0-indexed

The specification this was built from said `evidence_page_num` is 1-indexed. Rather than
assume, ingest locates a fragment of every gold evidence string in the extracted text and
counts which offset wins:

| offset | matched |
|---|---|
| **+1** | **159** |
| none | 30 |
| +2 | 12 |
| 0 | 7 |
| −1 | 5 |

189 rows checked. **Offset +1** — the benchmark is 0-indexed. Assuming otherwise would have
halved Tier 1 recall and looked exactly like a bad retriever.

### Tier 1 — retrieval (free, deterministic, gated in CI)

`dossier eval tier1` · 150 questions, 84 documents, corpus-wide:

| config | R@5 | R@10 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| **vector** | **0.3533** | **0.4400** | **0.1624** | **0.2407** | **18.0** | **43.8** |
| bm25 | 0.1067 | 0.1333 | 0.0631 | 0.0855 | 181.0 | 415.7 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3 | 2.7 |
| rrf(vector+bm25) | 0.2000 | 0.3333 | 0.1146 | 0.1582 | 212.5 | 465.4 |
| rrf(all three) | 0.1733 | 0.2800 | 0.1018 | 0.1403 | 213.7 | 454.7 |
| rrf(all three)+rerank | 0.2800 | 0.4200 | 0.1545 | 0.2078 | 1551.0 | 2030.7 |

`dossier eval tier1 --filtered` · with each question's company as a filter, which is what the
agent actually issues:

| config | R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| **vector** | **0.4067** | **0.5000** | **0.1911** | **0.2765** |
| rrf(all three)+rerank | 0.3333 | 0.4600 | 0.1741 | 0.2316 |

**Evidence match rate: 0.577** — only 109 of 189 gold evidence strings survive PDF extraction
and chunking into a chunk of the right page. That is the ceiling on achievable recall, so
vector's R@10 of 0.44 is **76% of what is reachable**, not 44% of what is possible. Reporting
recall without this number blames the retriever for an ingestion loss.

Reranker latency: **p50 1,551 ms / p95 2,031 ms** end to end, against **18 ms / 44 ms** for
dense retrieval alone — roughly 86×.

**The ablation changed what shipped.** Dense retrieval alone beats every fused and reranked
configuration, in both regimes. So `search_filings` now defaults to the vector channel with
reranking off, set by `DOSSIER_RETRIEVAL_CHANNELS` / `DOSSIER_RETRIEVAL_RERANK` rather than
hardcoded. A weight sweep confirmed there is no RRF weighting at which BM25 adds anything, so
it was not tuned into the default — that would have been fitting a knob to make a component
look useful. The graph scores exactly zero for a structural reason: it is built from Item 1
and Item 1A, and FinanceBench questions are single-figure lookups answered from
financial-statement pages. Neither component is dead code (BM25 is the LanceDB fallback; the
graph backs `graph_query`, which the memo path uses), but neither earns a place in the
default retrieval path on this benchmark. Reasoning in full in
[`docs/EVAL.md` §1.4](docs/EVAL.md).

### Tier 2 — grounding and guardrails (free, deterministic, in CI)

`dossier eval tier2 --run-ooc` → `runs/reports/tier2.json`

| metric | value |
|---|---|
| citation validity rate | **0.984** |
| numeric grounding rate | **0.887** |
| forward-looking attribution rate | **0.984** |
| **abstention accuracy on out-of-corpus questions** | **1.000** (15 / 15) |
| guard fixture verdict accuracy | **1.000** (12 / 12) |
| fallback spans per run | 0.057 |

62 answers checked: 12 hand-written fixtures plus 50 stored answers from real runs. Five of
the fixtures fail on purpose — an invented number, a citation to a nonexistent id, an
unattributed projection, a silent no-answer, a right-digits-wrong-scale figure — because
measuring the guard only on real answers would let a guard that passes everything score 100%.

The 15 out-of-corpus questions name companies not in the corpus (Tesla, NVIDIA, Alphabet,
Apple, …) or fiscal years outside it (3M FY2035, Microsoft FY1994). All 15 abstained.

### A caught hallucination

The guard's core case, as a unit test
(`tests/unit/test_output_guard.py::test_invented_number_is_caught`) and as a fixture:

```
ledger [E1]  3M 10K 2018, p.60
             "Purchases of property, plant and equipment (PP&E) (1,577) (1,373) (1,420)."

answer       "3M's FY2018 capital expenditure was $1,890 million [E1]."

guard        FAIL  numeric_ungrounded
             the figure '$1,890' does not appear in any cited passage, fact or computed value
```

The citation is real, the passage is real, the sentence is fluent — and the number is not in
it. The run then gets two chances to revise, and escalates to a human if it still cannot
ground the claim.

### Tier 3 — answer correctness

`dossier eval tier3 --limit 50` → `runs/reports/tier3.json`

| question type | n | correct | partial | incorrect | abstained |
|---|---|---|---|---|---|
| domain-relevant | 22 | 0.773 | 0.136 | 0.000 | 0.091 |
| metrics-generated | 17 | 0.824 | 0.000 | 0.118 | 0.059 |
| novel-generated | 11 | 0.727 | 0.000 | 0.182 | 0.091 |
| **ALL** | **50** | **0.780** | **0.060** | **0.080** | **0.080** |

| | |
|---|---|
| mean cost per question | **$0.1292** |
| mean latency per question | **28.3 s** |
| revise-loop rate | **0.580** |
| HITL escalation rate | **0.080** |
| fallback spans per 100 runs | 0.0 |
| total cost for the 50-question run | $6.68 |

Judge: `prompts/judge.md` on the sonnet tier, cached by (answer hash, prompt version).

**78% correct, 84% correct-or-partial**, against a retrieval ceiling of 0.577 evidence match
rate — the agent recovers a lot of what single-shot retrieval misses by searching again,
falling back to exact XBRL facts, and computing rather than recalling.

**The revise-loop rate is the number to look at: 58% of first drafts were rejected by
the output guard** and rewritten. That is the guard doing real work on real answers, not a
formality — and it is why abstention (8.0%) and incorrect (8.0%) are both low
while retrieval recall is only 0.44. It also explains the mean cost: a run that revises pays
for two or three extra turns.

### Memo run

The five-section memo path is implemented (`agent/subagents.py`), unit-tested for the
ledger-merge and id-remapping logic that makes it correct, and exercised by the
`hitl_enqueue` / `hitl_resume_*` cassettes, which run the `memo_section` path with a real
section prompt through escalation and resume.

**A full five-section memo has not been run end to end.** The Anthropic credit balance for
this account was exhausted during the build — after the entity graph ($5.03), the 50-question
Tier 3 run ($6.68), the 15 out-of-corpus probes ($0.18) and 22 cassette recordings ($0.97).
The parallel-versus-`--sequential` wall-time comparison and total memo cost are therefore
**not measured, and are not stated**. `dossier memo <deal_id>` and `dossier demo` will produce
them on an account with credit; both write their numbers to `runs/reports/`.

Measured from the recorded runs that did complete:

| | |
|---|---|
| budget-manager compaction on a forced 600-token ceiling | 1 compaction, **10,919 tokens saved** |
| cost cap firing | run ends with a partial answer and `budget_capped: true` |

### Repository

`pygount --format=summary --suffix=py dossier tests`

| | files | code | comments |
|---|---|---|---|
| package | 45 | **5,375** | 1,018 |
| tests | 14 | **604** | 49 |

| | |
|---|---|
| unit tests | **88**, all offline |
| regression cassettes | **22**, replayed offline in CI ($0.97 to record) |
| Tier 1 CI floor | nDCG@10 ≥ 0.08 on `rrf(all three)+rerank`, ≥ 0.12 on `vector` |

---

## The demo run

`dossier demo` creates the demo deal (`data/demo_deal.json`: Costco against Coca-Cola,
General Mills and Kraft Heinz) and runs three questions plus a memo. The memo half could not
complete — see above — so what follows is a **real recorded run** from the regression
cassettes, not a reconstruction. Reproduce it with `dossier trace run_e112a35e5111`.

**Question.** *What was 3M free cash flow in FY2018? Compute it from operating cash flow
minus capital expenditure.*

```
run run_e112a35e5111
└── run:ask  7097ms
    ├── turn:turn_1  1974ms
    │   └── llm_call:system_analyst  1973ms  claude-sonnet-5  in 3280 out 148  $0.0080
    │       system_analyst@dae74f73
    ├── tool_call:get_financials  41ms
    ├── turn:turn_2  1206ms
    │   └── llm_call:system_analyst  1205ms  claude-sonnet-5  in 3884 out 93  $0.0087
    │       system_analyst@dae74f73
    ├── tool_call:compute  0ms
    ├── turn:turn_3  3865ms
    │   └── llm_call:system_analyst  3864ms  claude-sonnet-5  in 4111 out 434  $0.0126
    │       system_analyst@dae74f73
    ├── tool_call:finish  0ms
    └── guard:output_guard  1ms
total cost $0.0293
```

**Answer.**

> 3M's FY2018 free cash flow was **$4,862 million**.
>
> - Operating cash flow (FY2018): $6,439 million **[F1]**
> - Capital expenditures / PaymentsToAcquirePropertyPlantAndEquipment (FY2018): $1,577 million **[F2]**
> - Free cash flow = OCF − Capex = $6,439M − $1,577M = **$4,862M** **[C1]**
>
> Source: 3M XBRL companyfacts, as reported in the 10-K filed 2021-02-04, fiscal year 2018
> (2018-01-01 to 2018-12-31).

**Evidence ledger.**

```
[F1] 3M XBRL companyfacts NetCashProvidedByUsedInOperatingActivities FY2018 (10-K filed 2021-02-04)
[F2] 3M XBRL companyfacts PaymentsToAcquirePropertyPlantAndEquipment FY2018 (10-K filed 2021-02-04)
[C1] computed: ocf - capex from [F1, F2]
```

Three things this shows. The derived figure is not asserted — it is a `compute` call whose
result carries the ids it was derived from, so a reviewer can check the arithmetic. The
figures come from XBRL rather than from prose, so they are exact and carry a filing date.
And the guard verified all three numbers against those sources before the answer was
returned; `$4,862` matches `[C1]`, and `[C1]` matches `[F1] − [F2]`.

*(The FY2018 capex figure here is $1,577M — the gold FinanceBench answer. Getting it right
required fixing how SEC `companyfacts` fiscal years are read; see
[`docs/DESIGN.md` §5.2](docs/DESIGN.md).)*

---

## Quickstart

```bash
git clone https://github.com/ArinKathapurkar/dossier && cd dossier
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

# offline — no API key needed
ruff check . && pytest tests/unit tests/regression -q
dossier eval tier1 --mini && dossier eval tier2 --mini

# build the corpus (~7 min) and the indexes (~4 min)
dossier ingest
dossier index --skip-graph            # add --graph networkx for the entity graph (~$5)

# needs an API key in .env
dossier deal create --target Costco --peers Walmart --peers PepsiCo
dossier ask <deal_id> "What was Costco's FY2021 free cash flow?"
dossier memo <deal_id>
dossier serve &  curl -s localhost:8000/health
```

---

## Notes on this build

- **Neo4j was exercised, but every number here came from NetworkX.** The Docker daemon was
  stopped at the start of the build and came up partway through, so the retrieval and eval
  numbers above were all produced against the **NetworkX** backend. Neo4j was then loaded
  with the same graph (`dossier graph sync --to neo4j`, 4,604 entities and 5,781 relations in
  2.1 s) and five parity tests assert that both backends answer `find_entities`, `expand`,
  `neighbors` and `chunks_for` identically. `docker compose up --build` was run end to end:
  the container serves `/health` with the full index, and reports the graph from Neo4j when
  `DOSSIER_GRAPH_BACKEND=neo4j`. The Neo4j-to-NetworkX fallback is covered by a unit test and
  a recorded cassette.
- **The API credit balance ran out** before `dossier demo` and a full five-section memo run
  could complete. Those numbers are stated as unmeasured rather than estimated — see the
  Memo run section above and [`docs/EVAL.md` §6](docs/EVAL.md).
- **Model tiers.** Primary `claude-sonnet-5`, fallback and extraction tier
  `claude-haiku-4-5`, judge `claude-sonnet-5`. All ids are config values, verified to resolve
  against `/v1/models` at preflight.
- **CI runs with no API key**, deliberately. If a test starts calling the live API it fails
  the build rather than spending money.

---

## Documentation

| | |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | why it is built this way — context engineering, state and resume, retrieval tradeoffs, guard design and its false-positive modes, what was rejected |
| [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) | install, ingest, ask, review, read a trace, add a tool, add a prompt version, Neo4j, Docker, the evals |
| [`docs/RESPONSIBLE_AI.md`](docs/RESPONSIBLE_AI.md) | risk → mitigation, what is *not* mitigated, how a reviewer decision becomes a regression test |
| [`docs/EVAL.md`](docs/EVAL.md) | every table above with the command that produced it |

MIT licensed.
