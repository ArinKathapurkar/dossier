"""Neo4j backend parity, skipped when no container is reachable.

The two graph backends sit behind one interface, so the meaningful test is not "does Cypher
run" but "do both backends answer the same questions the same way". This loads the committed
NetworkX graph into Neo4j and compares the interface methods node for node.

Run it with:  docker compose up -d neo4j && dossier graph sync --to neo4j
"""

from __future__ import annotations

import pytest

from dossier.index.graph_store import Neo4jGraphStore, NetworkXGraphStore

pytestmark = pytest.mark.skipif(
    not Neo4jGraphStore.available(), reason="no Neo4j container reachable"
)


@pytest.fixture(scope="module")
def stores():
    nx_store = NetworkXGraphStore()
    if nx_store.stats()["entities"] == 0:
        pytest.skip("no graph built -- run `dossier index --graph networkx` first")
    neo = Neo4jGraphStore()
    if neo.stats()["entities"] == 0:
        pytest.skip("Neo4j is empty -- run `dossier graph sync --to neo4j` first")
    yield nx_store, neo
    neo.close()


def test_both_backends_hold_the_same_graph(stores):
    nx_store, neo = stores
    a, b = nx_store.stats(), neo.stats()
    assert (a["entities"], a["relations"]) == (b["entities"], b["relations"])


def test_entity_lookup_agrees(stores):
    nx_store, neo = stores
    for probe in ("3M", "Costco", "Boeing"):
        nx_hits = set(nx_store.find_entities(probe, limit=5))
        neo_hits = set(neo.find_entities(probe, limit=5))
        if not nx_hits:
            continue
        assert nx_hits & neo_hits, f"{probe!r}: {nx_hits} vs {neo_hits}"


def test_one_hop_expansion_agrees(stores):
    nx_store, neo = stores
    seeds = nx_store.find_entities("3M", limit=1)
    if not seeds:
        pytest.skip("target entity not in the graph")
    assert set(nx_store.expand(seeds, hops=1)) == set(neo.expand(seeds, hops=1))


def test_neighbors_agree(stores):
    nx_store, neo = stores
    seeds = nx_store.find_entities("3M", limit=1)
    if not seeds:
        pytest.skip("target entity not in the graph")
    assert set(nx_store.neighbors(seeds[0])) == set(neo.neighbors(seeds[0]))


def test_provenance_chunks_agree(stores):
    nx_store, neo = stores
    seeds = nx_store.find_entities("3M", limit=1)
    if not seeds:
        pytest.skip("target entity not in the graph")
    assert set(nx_store.chunks_for(seeds)) == set(neo.chunks_for(seeds))
