"""NetworkX graph store: normalization, expansion, provenance, persistence."""

import pytest

from dossier.index.graph_store import (
    Entity,
    NetworkXGraphStore,
    Relation,
    normalize_entity,
    open_graph_store,
)


@pytest.fixture
def store(tmp_path):
    s = NetworkXGraphStore(path=tmp_path / "graph.json", autoload=False)
    s.upsert(
        [
            Entity("Apple Inc.", "Company", ["c1"]),
            Entity("Foxconn", "Supplier", ["c2"]),
            Entity("Samsung Electronics Co., Ltd.", "Competitor", ["c3"]),
            Entity("semiconductor supply shortage", "RiskFactor", ["c4"]),
        ],
        [
            Relation("Apple Inc.", "DEPENDS_ON", "Foxconn", ["c2"]),
            Relation("Apple", "COMPETES_WITH", "Samsung Electronics", ["c3"]),
            Relation("Foxconn", "EXPOSED_TO", "semiconductor supply shortage", ["c4"]),
        ],
    )
    return s


def test_normalization_collapses_corporate_suffixes():
    assert normalize_entity("Apple Inc.") == normalize_entity("APPLE, INC") == "apple"
    assert normalize_entity("Samsung Electronics Co., Ltd.") == "samsung electronics"
    assert normalize_entity("") == ""


def test_relations_using_a_variant_name_land_on_the_same_node(store):
    # "Apple Inc." and "Apple" both normalize to 'apple', so there is one node, not two.
    assert store.stats()["entities"] == 4


def test_exact_and_substring_lookup(store):
    assert "apple" in store.find_entities("Apple Inc.")
    assert "foxconn" in store.find_entities("What does Apple say about Foxconn?")


def test_one_hop_expansion(store):
    hood = store.expand(["apple"], hops=1)
    assert hood["apple"] == 0
    assert hood["foxconn"] == 1
    assert "semiconductor supply shortage" not in hood, "two hops away"


def test_two_hop_expansion_reaches_the_suppliers_risk(store):
    hood = store.expand(["apple"], hops=2)
    assert hood["semiconductor supply shortage"] == 2


def test_relation_filter_restricts_traversal(store):
    hood = store.expand(["apple"], hops=1, rel_filter=["COMPETES_WITH"])
    assert "samsung electronics" in hood
    assert "foxconn" not in hood


def test_hops_are_clamped_to_two(store):
    assert store.expand(["apple"], hops=9) == store.expand(["apple"], hops=2)


def test_chunks_for_returns_provenance(store):
    chunks = store.chunks_for(["apple"])
    assert "c1" in chunks  # the entity's own provenance
    assert "c2" in chunks  # and its outgoing edge's
    assert "c3" in chunks


def test_neighbors_returns_triples(store):
    triples = store.neighbors("Apple Inc.")
    assert ("apple", "DEPENDS_ON", "foxconn") in triples


def test_self_edges_and_blanks_are_dropped(store):
    before = store.stats()["relations"]
    store.upsert([], [Relation("Apple", "COMPETES_WITH", "Apple Inc.", []), Relation("", "DEPENDS_ON", "x", [])])
    assert store.stats()["relations"] == before


def test_round_trips_through_json(store, tmp_path):
    store.save()
    reloaded = NetworkXGraphStore(path=store.path)
    assert reloaded.stats() == store.stats()
    assert reloaded.expand(["apple"], hops=2) == store.expand(["apple"], hops=2)


def test_neo4j_request_falls_back_to_networkx_with_a_span(runs_db, monkeypatch):
    from dossier.obs.tracer import get_tracer

    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:9")  # nothing listening
    from dossier.config import reset_config_cache

    reset_config_cache()
    store = open_graph_store("neo4j")
    reset_config_cache()
    assert isinstance(store, NetworkXGraphStore)
    spans = [s for s in get_tracer().spans_for("test_run") if s["kind"] == "fallback"]
    assert spans and spans[0]["attrs"]["to"] == "networkx"
