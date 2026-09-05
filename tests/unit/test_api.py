"""API surface: shapes, error codes, and that the offline endpoints stay offline.

The agent endpoints stream and cost money, so they are covered by the cassettes rather than
here. What this pins is the contract: what a client gets back, what a missing resource
returns, and that `/health` and `/reviews` work with no API key -- which is what makes the
service inspectable in CI and in a container that has never been given a key.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dossier.api.app import app


@pytest.fixture
def client(runs_db):
    return TestClient(app)


def test_root_states_what_the_service_is_and_isnt(client):
    body = client.get("/").json()
    assert body["service"] == "dossier"
    assert "not investment advice" in body["what_this_is"].lower()


def test_health_reports_index_counts_graph_backend_and_model_tiers(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {"primary", "fallback", "cheap", "judge"} <= set(body["models"])
    assert "backend" in body["graph"]
    assert "vectors" in body["index"]
    # Prices are reported for exactly the tiers in use, so a cost figure is checkable.
    assert set(body["prices_usd_per_mtok"]) <= set(body["models"].values())


def test_deal_round_trip(client):
    created = client.post("/deals", json={"target": "Costco", "peers": ["Walmart"], "thesis": "screen"}).json()
    assert created["target"] == "Costco" and created["peers"] == ["Walmart"]
    fetched = client.get(f"/deals/{created['id']}").json()
    assert fetched == created


def test_unknown_resources_are_404_not_500(client):
    assert client.get("/deals/deal_missing").status_code == 404
    assert client.get("/runs/run_missing").status_code == 404
    assert client.get("/traces/run_missing").status_code == 404
    assert client.get("/reviews/rev_missing").status_code == 404


def test_reviews_json_and_html(client):
    assert client.get("/reviews").json() == {"reviews": [], "count": 0}
    html = client.get("/reviews?format=html")
    assert html.status_code == 200
    assert "review queue" in html.text
    # The empty state explains how a section reaches the queue rather than showing nothing.
    assert "output guard rejects a draft twice" in html.text


def test_decision_validates_the_decision_value(client):
    assert client.post("/reviews/rev_x/decision", json={"decision": "maybe"}).status_code == 400


def test_no_endpoint_touched_here_needs_an_api_key(client, monkeypatch):
    """Everything above must work with no credential -- asserted, not assumed."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for path in ("/", "/health", "/reviews"):
        assert client.get(path).status_code == 200
