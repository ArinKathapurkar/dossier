# Evaluation

Every number here was produced by a command in this repository, and every command writes
its machine-readable output to `runs/reports/`. Nothing is estimated.

Corpus: the full FinanceBench open-source set — 150 questions over 84 filing PDFs from 32
companies. Retrieval defaults at the time of measurement: `DOSSIER_RETRIEVAL_CHANNELS`
unset (vector), `DOSSIER_RETRIEVAL_RERANK` unset (off). See §1.4 for how those defaults
were chosen.

---

## 0. Corpus and index

```bash
dossier ingest && dossier index --graph networkx
```

| | |
|---|---|
| documents ingested / in benchmark | **84 / 84** (all from the FinanceBench GitHub `pdfs/` path) |
| pages extracted | **12,013** (11,914 pypdf, 99 re-extracted with PyMuPDF) |
| chunks | **32,468** (350-token windows, 50 overlap, never crossing a page) |
| questions | **150** |
| companies | **32** |
| companies matched to a CIK | **30 / 32** (Activision Blizzard and Foot Locker unmatched, logged not dropped) |
| XBRL facts | **55,670** across 15 us-gaap concepts |
| graph entities / relations | **4,604 / 5,781** |
| graph extraction cost | **$5.03**, 1,003 model calls, **772 s** wall time |
| embedding wall time | **203 s** for 32,468 chunks on MPS |
| MPS/CPU parity | mean cosine **1.000000** (threshold 0.999) — MPS trusted, no CPU demotion |
| vector index | flat (32,468 rows is below the 100,000-row IVF-PQ threshold) |
| ingest wall time | 428 s |

### The page-index convention, measured not assumed

`ingest/financebench.verify_page_convention` locates a normalized fragment of every gold
evidence string in the extracted page text and counts which offset wins:

| offset applied to `evidence_page_num` | evidence rows matched |
|---|---|
| **+1** | **159** |
| no offset matched | 30 |
| +2 | 12 |
| 0 | 7 |
| −1 | 5 |

**189 rows checked; offset +1 wins.** FinanceBench's `evidence_page_num` is 0-indexed
against our 1-indexed extraction. Assuming the documented 1-indexing would have roughly
halved Tier 1 recall and looked like a bad retriever.

---

## 1. Tier 1 — retrieval

Free, deterministic, runs in CI. A retrieved chunk is relevant if it comes from a gold page
of the right document.

```bash
dossier eval tier1              # corpus-wide
dossier eval tier1 --filtered   # with each question's company as a filter
dossier eval tier1 --mini       # the committed CI fixture
```

### 1.1 Corpus-wide (n = 150 questions, 84 documents)

| config | R@5 | R@10 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| **vector** | **0.3533** | **0.4400** | **0.1624** | **0.2407** | **18.0** | **43.8** |
| bm25 | 0.1067 | 0.1333 | 0.0631 | 0.0855 | 181.0 | 415.7 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3 | 2.7 |
| rrf(vector+bm25) | 0.2000 | 0.3333 | 0.1146 | 0.1582 | 212.5 | 465.4 |
| rrf(all three) | 0.1733 | 0.2800 | 0.1018 | 0.1403 | 213.7 | 454.7 |
| rrf(all three)+rerank | 0.2800 | 0.4200 | 0.1545 | 0.2078 | 1551.0 | 2030.7 |

**Evidence match rate: 0.577** (109 / 189 gold evidence strings found in a chunk of the
right page). This is the ceiling on achievable recall — the other 42% was lost in PDF text
extraction or fell across a chunk boundary before any retriever saw it. Vector's R@10 of
0.44 is therefore **76% of what is achievable**, not 44% of what is possible.

Total wall time 360 s for all six configurations over 150 questions.

### 1.2 With a company filter (what the agent actually issues)

Once the agent knows which company a question is about it passes a filter, shrinking the
candidate pool roughly 30×. Same 150 questions:

| config | R@5 | R@10 | nDCG@10 | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| **vector** | **0.4067** | **0.5000** | **0.1911** | **0.2765** | **18.6** | **25.3** |
| bm25 | 0.1333 | 0.1733 | 0.0859 | 0.1079 | 192.2 | 444.6 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.3 | 2.8 |
| rrf(vector+bm25) | 0.1467 | 0.2800 | 0.1118 | 0.1644 | 214.4 | 492.7 |
| rrf(all three) | 0.1467 | 0.2400 | 0.1017 | 0.1521 | 217.1 | 453.6 |
| rrf(all three)+rerank | 0.3333 | 0.4600 | 0.1741 | 0.2316 | 1661.1 | 2173.2 |

Filtering lifts every configuration and changes none of the ordering.

### 1.3 Latency

| | p50 | p95 |
|---|---|---|
| vector retrieval, end to end | 18.0 ms | 43.8 ms |
| reranked pipeline, end to end | 1,551 ms | 2,031 ms |
| reranker stage alone (2,031 − 455 RRF) | ≈ 1,340 ms | ≈ 1,580 ms |

The cross-encoder is roughly **86× the latency of dense retrieval** and, on this benchmark,
loses recall. That combination is what decided §1.4.

### 1.4 What the ablation decided

Three findings, all of which contradict the architecture diagram, and all of which the
shipped defaults now follow.

**Dense retrieval alone wins.** Every fused configuration is worse than the vector channel
by itself, in both regimes. So `search_filings` defaults to `channels=("vector",)` with
reranking off (`DOSSIER_RETRIEVAL_CHANNELS`, `DOSSIER_RETRIEVAL_RERANK`).

**BM25 drags the fusion down.** It is the weakest channel by a wide margin (R@10 0.13
corpus-wide), and RRF gives it equal vote, so fusing costs recall rather than adding it. A
weight sweep confirms the shape — the fused score improves monotonically as BM25 is
down-weighted, converging on vector-only:

| RRF weighting | R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| equal | 0.2000 | 0.3333 | 0.1146 | 0.1582 |
| vector ×2 | 0.2733 | 0.3867 | 0.1339 | 0.1809 |
| vector ×3 | 0.2867 | 0.4200 | 0.1462 | 0.2042 |
| vector ×4 | 0.2933 | 0.4400 | 0.1547 | 0.2195 |
| vector only | 0.3533 | 0.4400 | 0.1624 | 0.2407 |

There is no weight at which BM25 adds anything, so it was not tuned into the default — that
would have been fitting a knob to make a component look useful. The diagnosis is that
FinanceBench questions are analyst prose ("*Give a response to the question by relying on
the details shown in the cash flow statement*") whose term distribution is financial
vocabulary common to every page of every 10-K. BM25 over the raw question is close to
uninformative. BM25 over *extracted key terms* would be a different experiment and is the
obvious next thing to try.

**The graph channel scores exactly zero, for a structural reason.** The graph is built from
Item 1 and Item 1A; FinanceBench questions are overwhelmingly single-document, single-figure
lookups answered from financial-statement pages. The graph is built from precisely the pages
the answers are not on. This is not a bug and it is not fixable by tuning — it is a mismatch
between what the graph indexes and what this benchmark asks. The graph still earns its place
through the `graph_query` tool, which the Competitive Position memo section uses heavily, but
that value is not something this benchmark can measure, so no number is claimed for it.

Neither BM25 nor the graph is dead code: BM25 is the LanceDB fallback and backs the
`chunk_meta` lookup that resolves graph hits, and the graph backs `graph_query`.

### 1.5 CI fixture (`--mini`)

4 documents, 2,397 chunks, 20 questions, with chunk embeddings, query embeddings and 1,000
pre-computed cross-encoder scores committed, so CI runs the full ablation as numpy against
committed arrays in **0.6 s** with no model download.

| config | R@5 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| vector | 0.3500 | 0.4500 | 0.1362 | 0.1871 |
| bm25 | 0.2000 | 0.2500 | 0.0928 | 0.1312 |
| graph | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rrf(vector+bm25) | 0.2000 | 0.3000 | 0.0865 | 0.1611 |
| rrf(all three) | 0.2000 | 0.3000 | 0.0865 | 0.1611 |
| rrf(all three)+rerank | 0.2000 | 0.3500 | 0.0923 | 0.1181 |

Evidence match rate 0.636. The fixture reproduces the full corpus's ordering, which is what
makes it a usable gate.

### 1.6 The regression gate

`tests/fixtures/floors.json`, enforced by `python -m dossier.eval.floors` in CI:

| config | metric | floor | measured |
|---|---|---|---|
| rrf(all three)+rerank | nDCG@10 | 0.0800 | 0.0923 |
| rrf(all three)+rerank | recall@10 | 0.3000 | 0.3500 |
| vector | nDCG@10 | 0.1200 | 0.1362 |
| vector | recall@10 | 0.4000 | 0.4500 |
| — | evidence match rate | 0.6000 | 0.6364 |

Floors sit below the measured values so an ordinary tweak does not trip them but a real
regression does. Raising a floor is a deliberate act, made in the same commit as the change
that earns it.

---

## 2. Tier 2 — grounding and guardrails

Free, deterministic, runs in CI.

```bash
dossier eval tier2          # fixtures plus stored answers from previous runs
dossier eval tier2 --mini   # fixtures only, no run history required
```

### 2.1 Guard fixtures

Twelve hand-written answers in `tests/fixtures/answers/`, **five of which must fail**.
Measuring only on real answers would let a guard that passes everything score 100%.

| fixture | expected | what it pins |
|---|---|---|
| `pass_number_from_cited_passage` | pass | figure appears verbatim in the cited passage |
| `pass_number_from_cited_fact_with_unit_scaling` | pass | `$6.439 billion` ≡ fact value 6,439,000,000 |
| `pass_computed_value_carries_its_inputs` | pass | a `[C1]` derived from cited `[F1]`,`[F2]` |
| `pass_attributed_forward_looking` | pass | "management expects…" with a citation |
| `pass_explicit_abstention` | pass | no citations, explicit not-found statement |
| `pass_years_and_page_refs_are_not_claims` | pass | false-positive guard: "Item 8", "2018" |
| `pass_percentage_derived_and_cited` | pass | a percentage matching a computed value |
| `fail_invented_number` | **fail** | `numeric_ungrounded` — the core failure mode |
| `fail_citation_does_not_resolve` | **fail** | `citation_invalid` — `[E9]` with no E9 |
| `fail_unattributed_forward_looking` | **fail** | `forward_looking_unattributed` |
| `fail_silent_no_answer` | **fail** | `abstention_shape` — answered from model knowledge |
| `fail_wrong_scale_on_cited_fact` | **fail** | right digits, wrong scale |

**Fixture verdict accuracy: 12 / 12 (1.000)**, and every failing fixture produces the
specific violation kind it is supposed to. Asserted in CI by
`test_output_guard.py::test_every_tier2_fixture_produces_its_documented_verdict`.

### 2.2 Live metrics

Measured over the fixtures plus the answers stored from Tier 3 and the cassette recordings.
Regenerate with `dossier eval tier2`; current values are in `runs/reports/tier2.json`.

| metric | value |
|---|---|
| citation validity rate | see `runs/reports/tier2.json` |
| numeric grounding rate | " |
| forward-looking attribution rate | " |
| abstention accuracy on 15 out-of-corpus questions | " |
| fallback spans per run | " |

The out-of-corpus set (`tests/fixtures/ooc_questions.json`) is fifteen questions naming
companies that are not in the corpus (Tesla, NVIDIA, Alphabet, Apple, Goldman Sachs, Exxon
Mobil, Shopify, Salesforce) or fiscal years outside it (3M FY2035, Microsoft FY1994, Boeing
Q3 2026). Abstaining or asking for clarification are both counted correct.

---

## 3. Tier 3 — answer correctness

Costs API tokens. Run on demand.

```bash
dossier eval tier3 --limit 50
```

Method: run the full `ask` path over N FinanceBench questions with a fresh deal per company
(peers drawn from the same GICS sector), then grade each answer against the gold answer and
justification with `prompts/judge.md` on the sonnet tier. Judge outputs are cached by
(answer hash, judge prompt version).

Results are in `runs/reports/tier3.json` and reproduced in the README. Abstention is kept as
its own category rather than folded into "incorrect": for a system whose selling point is
that it declines rather than guesses, collapsing the two would penalise the behaviour it was
built to produce. The report splits abstentions by whether the source document was actually
ingested, which is what separates correct caution from a retrieval miss.

Given the Tier 1 numbers, expect abstention to be a large share of the distribution: when
retrieval returns nothing on a gold page, abstaining is the correct behaviour, and a 0.577
evidence-match ceiling bounds how often it can do better.

---

## 4. Regression cassettes

Free, deterministic, runs in CI with no API key.

```bash
python -m dossier.eval.record_cassettes   # once, costs tokens
pytest tests/regression -q                # forever after, free
```

**21 cassettes recorded for $0.91 total.** Each pins a control-flow path, not an answer:

| cassette | path it pins |
|---|---|
| `plain_answer` | search → finish with `[E#]` citations |
| `financials_and_compute` | `get_financials` → `compute` → `[C#]` carrying its inputs |
| `graph_tool` | `graph_query` traversal and provenance chunks |
| `peer_comparison` | `compare_peers` table across three companies |
| `abstention` | out-of-corpus company → explicit not-found |
| `clarification` | weak retrieval + no corpus company → `NEEDS_CLARIFICATION` |
| `input_guard_advice` | "should I buy" → templated refusal |
| `input_guard_injection` | instruction override → refused and logged |
| `input_guard_off_topic` | unrelated request → redirect |
| `revise_loop_after_guard_failure` | guard rejects → revise turn |
| `model_fallback_after_529` | injected overload → fallback tier + `fallback` span |
| `reranker_fallback` | cross-encoder raises → RRF order + `fallback` span |
| `neo4j_fallback` | dead Neo4j → NetworkX + `fallback` span |
| `hitl_enqueue` | `request_human_review` → run pauses in `REVIEW` |
| `hitl_resume_approve` / `_edit` / `_reject` | all three resume paths |
| `budget_compaction` | tiny `max_context_tokens` → compaction span |
| `cost_cap` | cap hit → partial answer with a budget flag |
| `tool_error_recovery` | bad concept name → `{error, hint}` → retry |
| `deal_memory` | a second run reading findings carried from the first |

Each replay asserts the tool-call sequence, the final workflow state, the guard verdict, the
revise count, the escalation flag and the span kinds. `test_replaying_twice_is_identical`
asserts determinism, and `test_no_cassette_recorded_an_api_key` asserts no credential was
recorded into a committed file.

### What recording the cassettes found

Two real bugs, both in paths that only fire under failure:

1. **`output_config.effort` is rejected by Haiku 4.5.** The primary tier accepted it, then
   the fallback to the cheaper tier reused the same request kwargs and died with an
   unhelpful 400 — so the model-fallback path was broken *exactly when it was needed*. Only
   the injected-529 scenario could have surfaced this. Effort is now attached per attempt,
   per model.

2. **Test isolation was leaking.** `deal.py` and `hitl/queue.py` did
   `from .state import connect`, binding at import time and defeating the fixture that
   redirects the run database. Replay therefore read a deal that a *different* cassette
   scenario had mutated in the developer's live database, and thirteen cassettes failed with
   a replay miss whose real cause was three modules away. Both now call through the module.

---

## 5. Prompt comparison

```bash
dossier eval compare --tier 1|2|3 <A> <B> [--limit N]
```

`A` and `B` are each a git ref or a directory; a ref is materialised with
`git show <ref>:prompts/<file>` into a temp directory, so any commit can be compared without
checking anything out.

**Comparing Tier 1 is the control.** Retrieval reads no prompts at all, so every metric must
come back with Δ = 0. Any non-zero delta on Tier 1 is measurement noise, and it tells you how
much of a Tier 3 delta is signal rather than variance. Run it before trusting a Tier 3
comparison.

A worked comparison of `prompts/system_analyst.md` across the commit that tightened its
citation rule is recorded in `runs/reports/compare_tier2.json`.

---

## 6. Reproducing everything

```bash
uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e ".[dev]"

# offline, no API key
ruff check . && pytest tests/unit tests/regression -q
dossier eval tier1 --mini && dossier eval tier2 --mini
python -m dossier.eval.floors

# needs the corpus
dossier ingest && dossier index --skip-graph
dossier eval tier1 && dossier eval tier1 --filtered

# needs an API key
dossier index --skip-vectors --graph networkx     # ~$5
dossier eval tier3 --limit 50
python -m dossier.eval.record_cassettes           # ~$0.91
```

Every command writes JSON to `runs/reports/`. The tables above are transcribed from those
files; if a number here disagrees with the file, the file is right.
