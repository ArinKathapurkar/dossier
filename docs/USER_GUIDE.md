# User guide

How to install `dossier`, build its indexes, ask it things, review what it drafts, read a
trace, extend it, and run the evaluations.

---

## Install

Python 3.12 and [`uv`](https://docs.astral.sh/uv/). No other system dependencies.

```bash
git clone https://github.com/ArinKathapurkar/dossier
cd dossier
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Verify the install with the offline suite — this needs no API key:

```bash
ruff check .
pytest tests/unit tests/regression -q
```

### The API key

Only the agent, the entity extractor, the input-guard classifier and Tier 3 need one.
Copy `.env.example` to `.env` and fill in `ANTHROPIC_API_KEY`. `.env` is gitignored; the key
is loaded with `python-dotenv` and never printed.

```bash
cp .env.example .env
$EDITOR .env
```

Everything else — ingest, indexing without the graph, Tier 1, Tier 2, the unit tests and the
cassette replays — runs offline.

---

## Build the corpus

```bash
dossier ingest
```

Fetches the FinanceBench dataset from HuggingFace, downloads the 84 filing PDFs, extracts
page text, chunks within page boundaries, and pulls SEC XBRL `companyfacts` for every
company it can map to a CIK. Takes about seven minutes on a laptop and writes
`runs/reports/ingest.json`.

Useful flags:

| flag | effect |
|---|---|
| `--limit-docs N` | only ingest the first N documents (smoke runs) |
| `--skip-xbrl` | skip the SEC pull |

The report includes a `page_convention` block — the measured offset between FinanceBench's
`evidence_page_num` and our page numbering. Check it after any change to PDF extraction; if
the winning offset moves, Tier 1 is scoring against the wrong pages.

## Build the indexes

```bash
dossier index --graph networkx
```

Encodes every chunk with `BAAI/bge-small-en-v1.5` into LanceDB, builds the BM25 index, and
extracts the entity graph. The model downloads on first use (~130 MB).

| flag | effect |
|---|---|
| `--skip-graph` | skip entity extraction — this is the part that costs API tokens |
| `--skip-vectors` | rebuild only the graph, reusing the existing vector and BM25 indexes |
| `--limit-docs N` | extract entities from the first N documents only |
| `--graph neo4j` | use the Neo4j backend (see below) |

Entity extraction cost $5.03 for all 84 documents and caches every raw extraction to
`data/index/extractions/`, so re-running is free. Deleting that directory is what makes it
cost money again.

**MPS parity guard.** Before trusting Apple's Metal backend, the embedder encodes 32 chunks
on both MPS and CPU and requires mean cosine ≥ 0.999. If it fails, encoding silently drops
to CPU and the report says so. PyTorch's MPS backend has historically produced *wrong*
transformer output rather than an error, which would look like a bad retriever.

---

## Ask questions

```bash
dossier deal create --target Costco --peers Walmart --peers "General Mills" --peers PepsiCo \
                    --thesis "Membership-model retail screen"
# → deal_a1b2c3d4e5

dossier ask deal_a1b2c3d4e5 "What was Costco's FY2021 free cash flow?"
```

The answer carries citations: `[E3]` a filing passage, `[F2]` an XBRL fact, `[C1]` a value
the system computed from cited inputs. The footer reports the run id, the final state, the
turn count, the cost and whether the output guard passed.

Three outcomes are all normal:

- **an answer with citations** — the usual case
- **an abstention** — "not found in the indexed filings"; correct when the corpus lacks the
  document
- **a clarifying question** — the run ends in `NEEDS_CLARIFICATION` when retrieval came back
  weak and the question named no company in the corpus

### Draft a memo

```bash
dossier memo deal_a1b2c3d4e5
dossier memo deal_a1b2c3d4e5 --sequential   # same work, one section at a time
```

Five section sub-agents run concurrently — Business Overview, Financial Profile, Competitive
Position, Key Risks, Open Diligence Questions — each with its own conversation and its own
evidence ledger, then a synthesizer merges them. `--sequential` exists so the parallel
speed-up can be measured rather than asserted.

If any section escalates, the memo ends in `REVIEW` rather than `FINAL` and the pending
reviews are listed.

### The canned demo

```bash
dossier demo
```

Creates the demo deal, asks three questions chosen to exercise three different paths (text
lookup, financials + compute, out-of-corpus abstention), then runs the memo. Writes
`runs/reports/demo.json`.

---

## Review a flagged section

A run reaches the queue two ways: the agent calls `request_human_review` itself, or the
output guard rejects a draft twice.

```bash
dossier review list
dossier review show rev_9f3a2b1c4d
```

`show` prints the draft, why it stopped, and the specific guard violations.

Decide, and the run resumes from persisted state:

```bash
dossier review decide rev_9f3a2b1c4d --approve --notes "Fine as written."
dossier review decide rev_9f3a2b1c4d --edit corrected.md
dossier review decide rev_9f3a2b1c4d --reject --notes "Do not characterise reserve adequacy."
```

- **approve** — the draft is accepted, any note applied
- **edit** — your text is authoritative and used verbatim
- **reject** — your notes go back to the model as a redo instruction

Or use the web page:

```bash
dossier serve &
open http://localhost:8000/reviews?format=html
```

Draft on the left, guard violations and the cited filing passages on the right; approve,
edit or reject inline. Plain HTML and a little JavaScript — no build step.

### Turn decisions into tests

```bash
dossier eval export-reviews
```

Writes each decided review to `tests/regression/cases/*.json` with the properties the
answer must satisfy afterwards — must cite, must not contain, must abstain. These run in CI.
A correction made once becomes a check that runs forever.

---

## Read a trace

```bash
dossier trace run_484d37147b11
```

A tree of spans — `run → turn → llm_call | tool_call | guard | fallback | compaction |
review_wait` — with durations, models, token counts, cost and the prompt version each call
used, ending with the run total.

What to look for:

| you see | it means |
|---|---|
| a `fallback` span | something degraded: reranker, Neo4j, model tier, or a tool error |
| a `compaction` span | the conversation hit the context budget; `tokens_saved` says how much |
| `guard` with `pass: false` | a revise turn followed, or the run escalated |
| many `tool_call: search_filings` | retrieval is not finding it; check the query and filters |

Aggregate spend across runs:

```bash
dossier cost                        # by run type
dossier cost --by prompt_version    # which prompt version cost what
dossier cost --by model             # tier split, including fallbacks
```

And the prompt registry:

```bash
dossier prompts                     # name → content-hash version
```

---

## Extend it

### Add a tool

1. Define the schema in `dossier/agent/tools.py`, next to the others. Write the
   `description` for the model, not for a human reading the code — it is the only thing
   deciding whether the tool gets called at the right moment.
2. Write `handle_<name>(args, ctx) -> str`. Rules:
   - anything the model may cite must go through `ctx.ledger` and come back as an id;
   - wrap document text in `<document>` tags (`_wrap_documents` does this);
   - never raise — return a helpful string, or let `dispatch` convert the exception into a
     `{error, hint}` result the model can act on.
3. Register it in `HANDLERS`, and add it to `ASK_TOOLS` and/or `SECTION_TOOLS`.
4. If it is side-effect free, add its name to `PARALLEL_SAFE` so it can be dispatched
   concurrently with other reads.
5. Record a cassette that exercises it, so its behaviour is pinned in CI.

### Add a prompt version and compare it

Prompts are plain Markdown in `prompts/`, versioned by the first eight hex characters of
their content hash. Git history is the version history.

```bash
$EDITOR prompts/system_analyst.md
git commit -am "Tighten the citation rule in the analyst prompt"

dossier eval compare --tier 3 HEAD~1 HEAD --limit 30
```

`A` and `B` are each a git ref or a directory, so an uncommitted draft can be measured
before it is committed:

```bash
cp -r prompts /tmp/prompts_v2 && $EDITOR /tmp/prompts_v2/system_analyst.md
dossier eval compare --tier 2 prompts /tmp/prompts_v2
```

Comparing Tier 1 is a useful control: retrieval reads no prompts, so any difference it
reports is noise — which tells you how much of a Tier 3 difference is signal.

### Run with Neo4j

```bash
open -a Docker                       # macOS; wait for `docker info` to succeed
docker compose up -d neo4j
dossier graph sync --to neo4j        # copy the built graph -- no re-extraction, no cost
dossier graph stats --backend neo4j
DOSSIER_GRAPH_BACKEND=neo4j dossier ask <deal_id> "Which suppliers does the target name?"
```

`dossier graph sync` exists so the Neo4j path can be exercised without paying for entity
extraction twice. `pytest tests/unit/test_neo4j_integration.py` then asserts that both
backends answer `find_entities`, `expand`, `neighbors` and `chunks_for` identically; it skips
itself when no container is reachable, so it never breaks CI.

Stopping the container mid-run is a supported failure: the store falls back to NetworkX and
emits a `fallback` span, which `dossier trace` will show. That is the intended behaviour, and
it is what `test_neo4j_request_falls_back_to_networkx_with_a_span` asserts.

### Run in Docker

```bash
docker compose up --build -d
curl -s localhost:8000/health | python -m json.tool
docker compose down
```

`data/` and `runs/` are bind-mounted, so an index built on the host is reused rather than
rebuilt. The embedding and reranker models download on first search into a named volume; the
first query after a fresh build is slow for that reason.

---

## The service

```bash
dossier serve --port 8000
```

| Method | Path | What it does |
|---|---|---|
| POST | `/deals` | create a deal |
| GET | `/deals/{id}` | deal and its carried findings |
| POST | `/deals/{id}/ask` | SSE stream of agent events, ending `final` or `review_required` |
| POST | `/deals/{id}/memo` | SSE stream for the memo run |
| GET | `/runs/{run_id}` | state, answer, ledger, guard report, transition history |
| POST | `/runs/{run_id}/resume` | continue a paused run |
| GET | `/reviews` | pending reviews; `?format=html` for the reviewer page |
| POST | `/reviews/{id}/decision` | decide, and resume the run |
| GET | `/traces/{run_id}` | span tree as JSON |
| GET | `/health` | index counts, graph backend, model tiers, pending reviews |

```bash
curl -sN -X POST localhost:8000/deals/deal_a1b2c3d4e5/ask \
     -H 'content-type: application/json' \
     -d '{"question":"What was FY2021 operating cash flow?"}'
```

---

## Run the evaluations

```bash
dossier eval tier1                # retrieval ablation, free, ~10 min on the full corpus
dossier eval tier1 --mini         # the committed CI fixture, seconds
dossier eval tier2                # grounding and guardrails, free
dossier eval tier3 --limit 50     # answer correctness — costs API tokens
python -m dossier.eval.floors     # the CI regression gate
```

Every command writes machine-readable JSON to `runs/reports/`, which is where the README's
numbers come from — they are regenerated, not retyped.

### Record cassettes

```bash
python -m dossier.eval.record_cassettes                       # all scenarios
python -m dossier.eval.record_cassettes --only abstention     # one
```

Costs API tokens once; after that the recordings replay for free in CI. Re-record when a
change deliberately alters the agent's control flow, and read the diff before committing —
a cassette diff is the clearest statement of what a prompt change did.

### Regenerate the CI fixture

```bash
python -m dossier.eval.mini_corpus
```

Rebuilds `tests/fixtures/mini_corpus/` from the full index: chunks, questions, chunk
embeddings, query embeddings and pre-computed cross-encoder scores. Needs the full corpus
ingested and indexed first.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `no chunks found -- run dossier ingest first` | indexes built before ingest | `dossier ingest` |
| `ANTHROPIC_API_KEY is not set` | agent command without a key | fill in `.env`, or use an offline command |
| `LLMReplayMiss` | replay mode with a stale cassette | re-record that scenario, or check `DOSSIER_LLM_MODE` |
| Tier 1 recall near zero after an ingest change | page offset moved | check `page_convention` in `runs/reports/ingest.json` |
| every run has a `fallback` span for the reranker | model failed to download | check network; the run still works on the RRF order |
| `dossier eval tier2` reports abstention accuracy 0 | no out-of-corpus questions have been run yet | run them through `dossier ask` first, or use `--mini` for fixtures only |
| MPS parity check demoted to CPU | Metal kernel mismatch on this PyTorch build | expected and safe; encoding is ~3× slower |

---

## Configuration

Everything is in `dossier/config.py`, overridable by environment variable:

| variable | default | what it does |
|---|---|---|
| `DOSSIER_PRIMARY_MODEL` | `claude-sonnet-5` | the agent's model |
| `DOSSIER_FALLBACK_MODEL` | `claude-haiku-4-5` | used after 3 failed retries |
| `DOSSIER_CHEAP_MODEL` | `claude-haiku-4-5` | entity extraction, input guard |
| `DOSSIER_JUDGE_MODEL` | `claude-sonnet-5` | Tier 3 grading |
| `DOSSIER_MAX_TURNS` | 24 | agent loop bound |
| `DOSSIER_MAX_CONTEXT_TOKENS` | 60000 | compaction trigger |
| `DOSSIER_RUN_COST_CAP` | 1.50 | per-run dollar cap |
| `DOSSIER_GRAPH_BACKEND` | `networkx` | `networkx` or `neo4j` |
| `DOSSIER_LLM_MODE` | `live` | `live`, `record` or `replay` |
| `DOSSIER_SEC_USER_AGENT` | — | required by SEC on API requests |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | when set, spans also export via OTLP |

---

## Regenerate the README's size figures

```bash
pygount --format=summary --suffix=py dossier
pygount --format=summary --suffix=py tests
```

`pygount` is in the `dev` extra. Every other number in the README comes from a JSON file
under `runs/reports/`, written by the command named next to it.
