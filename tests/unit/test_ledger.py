"""The evidence ledger: typed ids, dedup by source, compact index, round-tripping."""

from dossier.agent.ledger import Ledger


def test_chunk_ids_are_sequential_and_typed(ledger, chunk_factory):
    a = ledger.add_chunk(chunk_factory(chunk_id="c1"))
    b = ledger.add_chunk(chunk_factory(chunk_id="c2"))
    assert (a.id, b.id) == ("E1", "E2")


def test_same_chunk_retrieved_twice_reuses_its_id(ledger, chunk_factory):
    a = ledger.add_chunk(chunk_factory(chunk_id="c1"))
    b = ledger.add_chunk(chunk_factory(chunk_id="c1", text="different excerpt of the same chunk"))
    assert a.id == b.id
    assert len(ledger) == 1


def test_facts_and_computed_values_get_their_own_id_space(ledger, chunk_factory):
    ledger.add_chunk(chunk_factory())
    f = ledger.add_fact({"company": "3M", "concept": "Revenues", "unit": "USD", "fiscal_year": 2018,
                         "period_start": "2018-01-01", "period_end": "2018-12-31", "value": 32765000000.0,
                         "form": "10-K", "filed": "2019-02-07"})
    c = ledger.add_computed("a - b", 5.0, ["F1"])
    assert (f.id, c.id) == ("F1", "C1")


def test_computed_value_carries_its_inputs(ledger):
    item = ledger.add_computed("ocf - capex", 4862.0, ["F1", "F2"])
    assert item.meta["inputs"] == ["F1", "F2"]
    assert "F1" in item.citation and "F2" in item.citation


def test_index_is_compact_but_full_text_is_retained(ledger, chunk_factory):
    long_text = "x " * 500
    item = ledger.add_chunk(chunk_factory(text=long_text))
    index = ledger.render_index()
    assert len(index) < 400, "the index block must stay small -- it goes in every turn"
    assert len(ledger.get(item.id).text) > 900, "the ledger must keep the full text"


def test_round_trips_through_json(ledger, chunk_factory):
    ledger.add_chunk(chunk_factory(chunk_id="c1"))
    ledger.add_computed("2 + 2", 4, [])
    restored = Ledger.from_json(ledger.to_json())
    assert restored.ids() == ledger.ids()
    assert restored.get("E1").text == ledger.get("E1").text
    # Counters survive, so a resumed run does not reissue E1.
    assert restored.add_chunk(chunk_factory(chunk_id="c9")).id == "E2"


def test_merge_remaps_colliding_ids_and_dedups_shared_sources(chunk_factory):
    a, b = Ledger(), Ledger()
    a.add_chunk(chunk_factory(chunk_id="shared"))
    a.add_chunk(chunk_factory(chunk_id="only_a"))
    b.add_chunk(chunk_factory(chunk_id="shared"))   # B calls this E1 too
    b.add_chunk(chunk_factory(chunk_id="only_b"))
    remap = a.merge(b)
    assert remap["E1"] == "E1", "a chunk both sub-agents retrieved keeps one id"
    assert remap["E2"] == "E3", "B's second chunk is renumbered after A's"
    assert len(a) == 3
