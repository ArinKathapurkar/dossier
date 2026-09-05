# Design

This document explains why `dossier` is built the way it is: the decisions that were
actually decisions, the ones that turned out to be wrong, and the things that were
deliberately not built.

It assumes you have read the README's first section — this analyzes **public SEC filings
on a public benchmark**. It is not a client system and not investment advice.

---

## 1. The problem shape

A diligence question is not a search query. "What was FY2018 capital expenditure" has one
right answer sitting in one table on one page; "how capital-intensive is this business"
needs three numbers from two sources and a division; "what should we ask management" needs
the system to know what it *failed* to find. A single retrieval-then-generate pass handles
the first and quietly fabricates the other two.

So the system is an agent with tools, and the design pressure is almost entirely about two
things: **what goes into the model's context**, and **what happens when the answer cannot
be grounded**. Everything below follows from those.

---

## 2. Context engineering

### 2.1 The evidence ledger

The central idea. Retrieved passages are large — eight chunks at ~350 tokens is 3k tokens
per search, and a thorough run searches five or six times. Keeping all of it in the
conversation is expensive and, past a point, actively harmful: irrelevant context degrades
what the model attends to.

But the model must still be able to *cite* any passage it has seen, and the output guard
needs the full text of a passage to verify a number against it. So the two are separated:

| | conversation | ledger |
|---|---|---|
| holds | one line per item: id, citation, ~220-char excerpt | full text, keyed by id |
| grows | slowly (a line per item) | freely (out of band) |
| read by | the model, every turn | the guard, and the model on demand |

Ids are typed by provenance — `E#` retrieved passage, `F#` XBRL fact, `C#` computed value —
so a reader can tell where a claim came from without following the citation. Deduplication
is by chunk id: re-retrieving a passage reuses its existing id. That matters more than it
looks. A model that sees the same passage under two ids will cite both, which inflates the
apparent support for a claim in a way a reader cannot detect.

### 2.2 What goes in the system prompt, and in what order

Every turn rebuilds the system prompt as:

```
prompts/system_analyst.md          (stable)
diligence checklist                (stable)
section prompt, for sub-agents     (stable per sub-agent)
deal: target, peers, thesis        (stable per run)
findings carried from earlier runs (stable per run)
workflow state summary             (volatile)
evidence ledger index              (volatile, grows each turn)
```

Stable content first, volatile content last. That ordering exists for prompt caching: a
cache is a prefix match, so anything that changes between turns must sit after everything
that does not. Putting the ledger index at the top would invalidate the cached prefix on
every single turn.

### 2.3 The budget manager

When the conversation would exceed `max_context_tokens` (default 60k), the **oldest tool
results** are replaced with a one-paragraph summary that keeps their evidence ids.

Two choices worth defending:

*Oldest, not least-relevant.* Relevance-ranked compaction needs another model call per
compaction, and the oldest results are the ones the model has already acted on — it has
extracted what it needed and moved on. Ranking would spend money to make a marginally
better version of a decision that is usually obvious.

*Summaries keep the ids.* Nothing is actually lost: the ledger still has the full text, so
the model can re-read any compacted passage by citing its id. What is dropped is the
verbatim text sitting in the message history — the largest and least reusable part of the
context.

Every compaction emits a span with tokens before and after, so "the budget manager saved N
tokens" is a query, not a claim.

### 2.4 Sub-agents and why the ledgers must be remapped

Five memo sections written in one conversation would carry the Business Overview's forty
passages while drafting Key Risks. So each section is its own conversation, its own ledger,
and its own prompt file, run concurrently. They share only the deal.

The subtle part is the merge. Each sub-agent numbers its own evidence from `E1`, so five
sub-agents produce five different passages all called `E1`. `Ledger.merge` returns an id
remap per section, and **each section's markdown is rewritten through its remap before the
synthesizer ever sees it**. Skipping that step produces a memo whose citations point at the
wrong passages — which is precisely the class of error the whole system exists to prevent,
introduced by the system itself. It is tested (`test_ledger.py::test_merge_remaps_...`).

The synthesizer has no retrieval tools. It cannot introduce a fact no section retrieved.

---

## 3. State, resume, and memory

### 3.1 The state machine

```
PLAN → GATHER → ANALYZE → DRAFT → REVIEW → FINAL
                   ↘ NEEDS_CLARIFICATION
    (any state) → FAILED
```

Transitions are validated, so `PLAN → FINAL` raises rather than producing an answer that
skipped evidence gathering. That turns "the model answered from memory" from a quality
problem you notice later into a state-machine error you notice immediately.

Every transition is persisted to `runs/runs.sqlite` along with the messages, the ledger and
any drafted sections. A run that stops for review is a row on disk, not a suspended
process — which is the only way a human step can take a day.

### 3.2 Resume

`dossier resume <run_id>` reloads the messages and the ledger, appends the reviewer's
decision as a normal user turn, and continues the same loop. A resumed run is structurally
identical to one that never paused; there is no separate resume code path to drift.

The three decisions resume differently:

- **approved** — the draft is accepted, any note applied
- **edited** — the reviewer's text is authoritative and used verbatim
- **rejected** — the notes go back to the model as a redo instruction

### 3.3 Deal memory

A `Deal` carries `findings`: short cited statements accepted from earlier runs, injected
into later runs' system prompts. The constraint that makes this safe is that **a finding
must carry evidence ids**; uncited statements are rejected at the API rather than stored.
Without that rule, one hallucination in run 1 becomes premise in runs 2 through 40, and the
output guard would happily pass it because it arrived as context rather than as a claim.

---

## 4. Retrieval

### 4.1 Why three channels

| channel | what it is good at | what it misses |
|---|---|---|
| dense vector | paraphrase, conceptual questions | exact line-item names, fiscal-year discrimination |
| BM25 | "Purchases of property, plant and equipment", ticker symbols, FY strings | anything phrased differently from the filing |
| entity graph | cross-document joins — this company's supplier's risk disclosure | anything not in Item 1 / Item 1A / MD&A |

The graph is the only one that answers a question requiring a *join*. Vector and keyword
search both answer "which passage looks like this query"; neither answers "which companies
does the target depend on, and what did those filings say".

### 4.2 Why RRF rather than a weighted score blend

The three channels produce scores on incomparable scales: cosine similarity in [-1, 1],
unbounded BM25, and a hop-distance heuristic. Normalising them requires per-corpus
calibration that drifts as the corpus changes.

Reciprocal rank fusion uses only the *rank* within each channel:

```
rrf(d) = Σ_channels 1 / (k + rank_c(d)),   k = 60
```

Adding or removing a channel therefore needs no retuning — which is exactly what made the
six-configuration ablation in `docs/EVAL.md` cheap to produce. Ties break on chunk id, so
the fused order is byte-stable run to run.

### 4.3 Why a reranker, and what happens when it is not there

The first stage optimises recall over 50 candidates; the cross-encoder optimises precision
over the 8 that actually enter the model's context. A cross-encoder sees query and passage
together, so it can distinguish "FY2018 capital expenditure" from "FY2017 capital
expenditure" — a distinction bi-encoders routinely miss and one that matters on every
metrics-generated FinanceBench question.

It is also the component most likely to be slow or absent, so it is wrapped in the standard
fallback: on failure or a >10s timeout, the RRF order is returned and a `fallback` span is
emitted. Retrieval degrades; it does not fail.

### 4.4 Where graph retrieval does *not* help

Measured, not assumed — see `docs/EVAL.md`. The graph channel contributes almost nothing to
Tier 1 on FinanceBench, and the reason is structural rather than a bug: FinanceBench
questions are overwhelmingly **single-document, single-figure** questions ("what was X's
FY2018 capex"), answered from a financial statement page. The graph is built from Item 1
and Item 1A, which are exactly the pages those answers are *not* on.

This is worth stating plainly because the architecture diagram implies the graph is a third
of the retrieval story and the benchmark says it is not. The honest position is that the
graph earns its place on the memo path — Competitive Position sections use `graph_query`
heavily, and cross-company diligence questions are the ones a deal team actually asks —
while contributing near zero on a benchmark composed of single-document lookups. A
benchmark that measured cross-document questions would show the opposite. We do not have
one, so the number stands as measured.

### 4.5 Chunking

Chunks never cross a page boundary. That costs a little context at page edges, but it keeps
the mapping chunk → page exact, which is what makes Tier 1 scoring exact against
FinanceBench's page-level ground truth. Windows are measured in the embedding model's own
tokens (350, 50 overlap), so a chunk is never silently truncated by the encoder.

---

## 5. Two things the data got wrong, and how they were caught

Both of these were found by *measuring* something the specification told us to assume. They
are the most useful things in this document.

### 5.1 `evidence_page_num` is 0-indexed

FinanceBench's evidence entries carry a page number, and every reasonable reading says it
is 1-indexed. `ingest/financebench.verify_page_convention` checks instead of assuming: it
locates a normalized fragment of every gold evidence string in the extracted page text and
counts which offset wins.

| offset | matches |
|---|---|
| **+1** | **159** |
| none | 30 |
| +2 | 12 |
| 0 | 7 |
| −1 | 5 |

Offset +1 — the benchmark's page numbers are 0-indexed against our 1-indexed extraction.
Had we assumed 1-indexing, Tier 1 recall would have been roughly halved, and the obvious
diagnosis would have been "the retriever is bad". The offset is now a measured config value
(`Config.gold_page_offset`) with the measurement recorded in `runs/reports/ingest.json`.

### 5.2 SEC `companyfacts` tags every value with the *filing's* fiscal year

Found by running the agent, not by reading the docs. Asked for 3M's FY2018 capital
expenditure, the agent returned $1,420M — the 2016 figure — and then, to its credit, noted
in its own answer that the record's period-end metadata said `2016-12-31`, which was
inconsistent with the FY2018 tag.

The cause: a FY2018 10-K reports 2016, 2017 and 2018 columns, and `companyfacts` tags all
three `fy=2018, fp=FY`. Selecting on `fy` returns whichever row sorts first. The fix is to
select on a fiscal year derived from the period **end date**, and to restrict duration
concepts to roughly-annual spans (300–400 days) so a quarterly figure never answers an
annual question. FY2018 capex is now $1,577M, which matches the benchmark's gold answer.

The general lesson: the agent's own caveat was the bug report. An agent that is required to
state its provenance surfaces data errors that a system returning bare numbers would hide.

---

## 6. Guardrails

### 6.1 Why the output guard is deterministic

An LLM-based grounding grader was considered and rejected. Three reasons:

1. **It shares the generator's blind spots.** The failure being defended against is a
   fluent, confident, wrong number. A grader model is fluent-and-confident by construction.
2. **It is non-deterministic**, so CI cannot gate on it and the same answer can pass on
   Tuesday and fail on Wednesday.
3. **It costs money per answer**, which means it gets sampled rather than run on everything.

Everything in `guard/output_guard.py` is string and arithmetic work. It runs in
milliseconds, costs nothing, and returns the same verdict every time — which is why Tier 2
can be a CI gate rather than a report someone reads occasionally.

### 6.2 The four checks

1. **Citation validity** — every `[E#]/[F#]/[C#]` resolves in the ledger.
2. **Numeric grounding** — every number in the answer appears in a cited passage's text,
   equals a cited fact's value within 0.5% after unit normalization, or is a `C#` computed
   value. `$1.577 billion`, `$1,577 million` and `1577` all normalize to the same value.
3. **Forward-looking attribution** — sentences containing *expects / anticipates / will /
   guidance / projected* must attribute to management or carry an evidence id.
4. **Abstention shape** — an answer citing nothing must actually say it found nothing.

### 6.3 Known false-positive modes

The guard is tuned to be strict, and strictness has a cost. Documented, with mitigations:

| mode | example | handling |
|---|---|---|
| fiscal years read as claims | "in 2018, capex was …" | bare integers 1900–2100 are ignored |
| document references | "Item 8", "Note 12", "p.42" | ignored when preceded by item/note/page/section |
| small counts | "four reportable segments" | bare integers < 32 without `$` are ignored |
| figures a filing writes in words | "approximately three billion dollars" | **not handled** — will flag as ungrounded |
| a number correct at a scale the passage does not state | passage says "1,577" in a millions-denominated table; answer says "$1,577,000,000" | handled by scale-insensitive comparison at 10³/10⁶/10⁹ |
| a computed figure the model states without calling `compute` | | flagged, correctly — this is the behaviour we want |

The last row is worth noting: the guard deliberately fails an answer whose arithmetic is
*right* but untraced. That is not a false positive. A derived number without its inputs is
not checkable by a reviewer, and the `compute` tool exists to make that cheap.

Twelve fixtures in `tests/fixtures/answers/` pin these behaviours, five of which fail on
purpose. Measuring the guard only on real answers would let a guard that passes everything
score 100%.

### 6.4 Input guard

Two layers, cheapest first: deterministic pre-checks (empty, over-length, literal injection
strings) and then a haiku-tier classifier for the judgement calls. The distinction that
actually needs a model is "what risks does management identify" (in scope) against "is this
a good investment" (advice) — no keyword list gets that right.

The classifier **fails open** if it errors: a guard outage must not take the system down,
and the deterministic layer plus the output guard still run. The deterministic layer fails
closed. That asymmetry is deliberate.

The other half of injection defence is in the tool layer: retrieved document text is always
wrapped in `<document>` tags, and the system prompt states that document content is data,
never instructions.

### 6.5 The revise loop and automatic escalation

Guard failure → the report goes back to the model as a normal turn (max 2 revisions). Still
failing → `request_human_review` fires automatically with the guard report attached. The
answer is never shipped ungrounded; it is either fixed or escalated.

---

## 7. Fallbacks

One helper, `with_fallback(primary, fallback, span_name)`, used everywhere:

| primary | fallback | trigger |
|---|---|---|
| cross-encoder reranker | RRF order | load failure or >10s |
| Neo4j | NetworkX | connection failure |
| primary model tier | fallback model tier | 3 failed retries on 429/529/5xx |
| LanceDB | BM25-only | any vector store error |
| a tool | `{error, hint}` result to the model | any exception |

Every degradation emits a span with `kind="fallback"`. That is the whole point: the system's
health becomes a query rather than a guess. `dossier trace` shows it in the tree, Tier 2
counts fallback spans per run, and Tier 3 reports fallbacks per 100 runs. **Silent
degradation is the failure mode this is built to prevent** — a reranker that quietly stopped
loading would look like a gradual quality decline with no proximate cause.

---

## 8. Evaluation design

Three tiers, split by what they cost:

| tier | measures | cost | runs in CI |
|---|---|---|---|
| 1 | retrieval against page-level ground truth | free | yes |
| 2 | grounding and guardrails | free | yes |
| 3 | answer correctness via LLM judge | API tokens | no |
| cassettes | agent control flow | free | yes |

**Tier 1 reports the evidence match rate next to recall.** That is the fraction of gold
evidence strings that survive PDF extraction and chunking into *some* chunk of the right
page — the ceiling on achievable recall. Reporting recall without it attributes an
ingestion loss to the retriever, which sends you optimising the wrong component.

**Tier 3 keeps abstention as its own category.** For a system whose selling point is that it
declines rather than guesses, folding abstention into "incorrect" penalises the behaviour it
was built to produce, and folding it into "correct" hides a recall failure. The report
splits abstentions by whether the source document was ingested, which is what separates
correct caution from a retrieval miss.

**Cassettes test control flow, not answers.** The paths that matter — a guard rejection, a
model fallback after a 529, a reranker timeout, a compaction, a cost cap, a review round
trip — are forced deterministically with injected faults, because waiting for them to
happen naturally is not a test strategy. Replay must be identical twice; that property is
what makes a behaviour change visible rather than dismissible as sampling noise.

---

## 9. Tradeoffs rejected

**An agent framework (LangChain / LlamaIndex / CrewAI).** The loop, the context management
and the state machine are the substance of this project. A framework would have made them
configuration, and configuration is not a demonstration of understanding. The concrete cost
of not using one is maybe 300 lines in `agent/loop.py`; the concrete benefit is that the
compaction policy, the tool dispatch order and the revise loop are all readable and all
testable.

**A hosted vector database.** LanceDB is embedded: no server, no credentials, no network in
the hot path, and the index is a directory that can be committed as a CI fixture. At 32k
chunks the entire index is smaller than one of the PDFs. A hosted store would add an
operational dependency to buy latency that a flat scan over 32k 384-d vectors already
provides in single-digit milliseconds.

**An LLM-based output guard.** Covered in §6.1. The short version: it would have made the
CI gate impossible.

**IVF-PQ indexing.** Measured against the row count and rejected: below ~100k rows the
quantization costs more recall than the index saves latency. `vector_store.py` builds one
automatically above that threshold and records which it used.

**Re-ranking with the primary model.** Would work and would cost roughly 40× the
cross-encoder per query, for a task a 110M-parameter model does well.

**Fine-tuning.** An explicit non-goal for this phase — see §10 and the README.

---

## 10. Future

Neither of these is built. They are listed because they are the natural next steps, not as
claimed work.

**Parameter-efficient fine-tuning of the entity extractor.** Graph construction cost $5.03
and 12.9 minutes of frontier-model calls for 1,003 pages, and every raw extraction is
cached in `data/index/extractions/`. That cache is a ready-made supervised dataset: ~1,000
(page text → entities, relations) pairs in a fixed schema. A LoRA adapter on a ~1B open
model, trained on that cache and evaluated against the frontier model's outputs on held-out
documents, is the obvious way to make re-extraction over a larger corpus economical. The
evaluation is well-posed because the frontier outputs are the labels and the schema is
closed — entity-type and relation-type accuracy, plus edge-level precision and recall
against the held-out frontier extractions.

**A gRPC `Ask` service** mirroring the REST endpoint, for callers that want a typed
contract and streaming without SSE framing. The agent layer is already transport-agnostic;
this is a second adapter over the same `ask` / `memo` entry points.

---

## 11. Engineering decisions taken during the build

Recorded here rather than interrupting to ask.

**Model tiers.** The specification named `claude-sonnet-4-5` as primary. Both that id and
the current-generation `claude-sonnet-5` resolve against `/v1/models`, and the current
generation was chosen as the primary tier with `claude-haiku-4-5` as the fallback and
extraction tier. The ids are config values (`DOSSIER_PRIMARY_MODEL`, etc.), verified to
resolve at preflight, and the price table in `config.py` is the single place cost is
defined.

**Effort setting.** Agent calls run at `output_config.effort = "low"`. A diligence loop is
tool-dispatch-heavy rather than reasoning-heavy, and the per-run cost cap is $1.50; low
effort keeps a typical `ask` under three cents.

**Entity extraction scope.** Capped at 14 narrative pages per document. Extracting every
page of 84 filings would have cost roughly $40 for pages whose content the XBRL store
already holds exactly. The cap is a parameter (`max_pages_per_doc`), and the cache means
raising it only pays for the new pages.

**Graph backend in this build.** The Docker daemon was stopped when the build started, so
the NetworkX backend produced every retrieval and eval number in this repository. Docker came
up partway through; Neo4j was then loaded with the same graph via `dossier graph sync --to
neo4j` (4,604 entities, 5,781 relations, 2.1 s) and five parity tests now assert that both
backends answer the interface identically. `docker compose up --build` was verified end to
end. The measured numbers were not re-run against Neo4j because the backends are asserted
equivalent and re-running would not produce different retrieval results -- the graph channel
scores zero on this benchmark either way (§4.4).

**A sync command rather than a second extraction.** Exercising Neo4j could have meant
re-running entity extraction against it for another $5. `dossier graph sync --to neo4j`
copies the built graph instead: extraction is the expensive step, the store is not.

**Mini-corpus reranker scores.** Rather than skipping the reranked row in CI, the fixture
ships pre-computed cross-encoder scores for each question's fused candidate set (1,000
pairs). CI therefore runs the full six-configuration ablation as numpy against committed
arrays, and the regression gate covers the configuration that actually ships.
