"""Shared fixtures.

Two invariants every test in this suite relies on:

1. **No test may call the live API.** CI runs with no ANTHROPIC_API_KEY; the autouse
   fixture below also clears it locally so a test that reaches for the network fails on
   a developer machine the same way it would in CI, rather than quietly spending money.
2. **No test may write to the developer's runs database.** The tracer and the state
   machine are both redirected to a per-test temporary SQLite file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _no_live_api(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DOSSIER_LLM_MODE", "replay")
    from dossier.agent import llm

    llm.reset_client()
    yield
    llm.reset_client()


@pytest.fixture
def runs_db(tmp_path, monkeypatch):
    """Point the tracer and the run store at a temp database."""
    db = tmp_path / "runs.sqlite"

    import dossier.agent.state as state
    import dossier.obs.tracer as tracer

    real_state_connect = state.connect
    real_tracer_connect = tracer.connect
    monkeypatch.setattr(state, "connect", lambda path=None: real_state_connect(db))
    monkeypatch.setattr(tracer, "connect", lambda path=None: real_tracer_connect(db))
    t = tracer.reset_tracer(db)
    t.bind("test_run")
    yield db
    tracer.reset_tracer(db)


@pytest.fixture
def ledger():
    from dossier.agent.ledger import Ledger

    return Ledger()


@pytest.fixture
def chunk_factory():
    from dossier.retrieve.types import RetrievedChunk

    def make(chunk_id="c1", text="Revenue was $1,234 million in fiscal 2022.", page=7, company="3M", channels=("vector",)):
        return RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            citation=f"{company} 10K 2022, p.{page}",
            score=0.9,
            channels_hit=list(channels),
            doc_name=f"{company}_2022_10K",
            company=company,
            doc_type="10K",
            fiscal_period="2022",
            page_num=page,
        )

    return make
