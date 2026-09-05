"""Central configuration.

Everything tunable lives here so that a run's behaviour is describable by a single
object, and so the eval harness can vary one knob at a time. Values come from
defaults, overridden by environment variables (loaded from a repo-root `.env` when
present). Nothing here reads the API key eagerly -- see `agent/llm.py`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv_once() -> None:
    """Load `.env` from the repo root if python-dotenv is installed.

    Import-safe: a missing file or a missing dependency is not an error, because CI
    deliberately runs with no `.env` and no API key.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a hard dependency in practice
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


_load_dotenv_once()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


# --------------------------------------------------------------------------------------
# Model pricing. USD per 1M tokens. Editable: obs/cost.py reads this table and nothing
# else, so correcting a price re-costs every historical span via `dossier cost`.
# --------------------------------------------------------------------------------------
MODEL_PRICES: dict[str, tuple[float, float]] = {
    # model_id: (input $/MTok, output $/MTok)
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    raw_pdfs: Path = REPO_ROOT / "data" / "raw" / "pdfs"
    raw_xbrl: Path = REPO_ROOT / "data" / "raw" / "xbrl"
    processed: Path = REPO_ROOT / "data" / "processed"
    chunks_parquet: Path = REPO_ROOT / "data" / "processed" / "chunks.parquet"
    pages_parquet: Path = REPO_ROOT / "data" / "processed" / "pages.parquet"
    facts_sqlite: Path = REPO_ROOT / "data" / "processed" / "facts.sqlite"
    manifest: Path = REPO_ROOT / "data" / "manifest.json"
    questions: Path = REPO_ROOT / "data" / "questions.json"
    demo_deal: Path = REPO_ROOT / "data" / "demo_deal.json"
    index_dir: Path = REPO_ROOT / "data" / "index"
    lancedb: Path = REPO_ROOT / "data" / "index" / "lancedb"
    bm25: Path = REPO_ROOT / "data" / "index" / "bm25.pkl"
    graph_json: Path = REPO_ROOT / "data" / "index" / "graph.json"
    extractions: Path = REPO_ROOT / "data" / "index" / "extractions"
    runs_db: Path = REPO_ROOT / "runs" / "runs.sqlite"
    reports: Path = REPO_ROOT / "runs" / "reports"
    prompts: Path = REPO_ROOT / "prompts"

    def ensure(self) -> None:
        for p in (
            self.raw_pdfs,
            self.raw_xbrl,
            self.processed,
            self.index_dir,
            self.extractions,
            self.reports,
            self.runs_db.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    # ---- models -----------------------------------------------------------------
    # Verified against GET /v1/models at preflight; see docs/DESIGN.md.
    primary_model: str = field(default_factory=lambda: _env("DOSSIER_PRIMARY_MODEL", "claude-sonnet-5"))
    fallback_model: str = field(default_factory=lambda: _env("DOSSIER_FALLBACK_MODEL", "claude-haiku-4-5"))
    cheap_model: str = field(default_factory=lambda: _env("DOSSIER_CHEAP_MODEL", "claude-haiku-4-5"))
    judge_model: str = field(default_factory=lambda: _env("DOSSIER_JUDGE_MODEL", "claude-sonnet-5"))
    max_tokens: int = field(default_factory=lambda: _env_int("DOSSIER_MAX_TOKENS", 4096))
    llm_retries: int = 3

    # ---- agent loop --------------------------------------------------------------
    max_turns: int = field(default_factory=lambda: _env_int("DOSSIER_MAX_TURNS", 24))
    max_context_tokens: int = field(default_factory=lambda: _env_int("DOSSIER_MAX_CONTEXT_TOKENS", 60_000))
    run_cost_cap_usd: float = field(default_factory=lambda: _env_float("DOSSIER_RUN_COST_CAP", 1.50))
    max_revisions: int = 2
    # Below this reranker score the loop is not confident it retrieved anything useful.
    retrieval_confidence_threshold: float = field(
        default_factory=lambda: _env_float("DOSSIER_RETRIEVAL_CONF", -2.0)
    )

    # ---- retrieval ---------------------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_batch: int = 64
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_timeout_s: float = 10.0
    chunk_tokens: int = 350
    chunk_overlap: int = 50
    rrf_k: int = 60
    retrieve_k: int = 50
    rerank_top_n: int = 8
    graph_hops: int = 1
    fuzzy_threshold: int = 90

    # ---- infra -------------------------------------------------------------------
    graph_backend: str = field(default_factory=lambda: _env("DOSSIER_GRAPH_BACKEND", "networkx"))
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "dossierdev"))
    sec_user_agent: str = field(
        default_factory=lambda: _env("DOSSIER_SEC_USER_AGENT", "dossier research contact@example.com")
    )
    llm_mode: str = field(default_factory=lambda: _env("DOSSIER_LLM_MODE", "live"))
    cassette_dir: Path = REPO_ROOT / "tests" / "regression" / "cassettes"

    paths: Paths = field(default_factory=Paths)


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()


def reset_config_cache() -> None:
    """Tests mutate the environment then call this to pick up new values."""
    get_config.cache_clear()
