"""Reciprocal rank fusion: rank-only, deterministic, channel-count agnostic."""

from dossier.retrieve.hybrid import reciprocal_rank_fusion
from dossier.retrieve.types import format_citation


def test_single_channel_preserves_order():
    fused = reciprocal_rank_fusion({"vector": ["a", "b", "c"]}, k=60)
    assert [d for d, _, _ in fused] == ["a", "b", "c"]


def test_document_in_both_channels_outranks_a_higher_single_channel_hit():
    # 'b' is 2nd in both channels; 'a' is 1st in one and absent from the other.
    fused = reciprocal_rank_fusion({"vector": ["a", "b"], "bm25": ["c", "b"]}, k=60)
    assert fused[0][0] == "b"
    assert set(fused[0][2]) == {"vector", "bm25"}


def test_scores_use_only_rank_not_channel_score():
    # Same ranks, wildly different underlying scores would be identical here by design.
    fused = reciprocal_rank_fusion({"vector": ["x"], "bm25": ["x"]}, k=60)
    assert fused[0][1] == 2 * (1 / 61)


def test_ties_break_on_id_so_the_order_is_stable():
    a = reciprocal_rank_fusion({"vector": ["z", "y"], "bm25": ["y", "z"]}, k=60)
    b = reciprocal_rank_fusion({"bm25": ["y", "z"], "vector": ["z", "y"]}, k=60)
    assert [d for d, _, _ in a] == [d for d, _, _ in b] == ["y", "z"]


def test_weights_are_applied_per_channel():
    fused = dict((d, s) for d, s, _ in reciprocal_rank_fusion({"vector": ["a"], "bm25": ["b"]}, k=60, weights={"vector": 2.0}))
    assert fused["a"] > fused["b"]


def test_citation_format():
    assert format_citation("3M", "10K", "2018", 60) == "3M 10K 2018, p.60"
    assert format_citation("3M", "", "", 4) == "3M, p.4"


# -- filters ------------------------------------------------------------------------

def test_doc_type_filters_match_across_spellings():
    """Regression: the corpus stores FinanceBench's own '10k' label while every model
    writes '10-K'. An exact match returned nothing and the agent reported "not found in the
    indexed filings" -- a false abstention with no error anywhere."""
    from dossier.index.vector_store import build_filter, doc_type_variants

    for spelling in ("10-K", "10K", "10k"):
        variants = doc_type_variants(spelling)
        assert "10k" in variants and "10K" in variants
    where = build_filter({"company": "3M", "doc_type": "10-K"})
    assert "'10k'" in where and "company = '3M'" in where


def test_bm25_and_vector_filters_agree_on_doc_type():
    from dossier.index.bm25_store import BM25Store

    store = BM25Store()
    row = {"company": "3M", "doc_type": "10k", "fiscal_period": "2018", "doc_name": "3M_2018_10K"}
    for spelling in ("10-K", "10K", "10k"):
        assert store._matches(row, {"doc_type": spelling}), spelling
    assert not store._matches(row, {"doc_type": "8k"})


def test_other_filters_stay_exact():
    from dossier.index.vector_store import build_filter

    where = build_filter({"company": "3M", "fiscal_period": "2018"})
    assert where == "company = '3M' AND fiscal_period = '2018'"


def test_sql_injection_in_a_filter_value_is_escaped():
    from dossier.index.vector_store import build_filter

    where = build_filter({"company": "3M' OR '1'='1"})
    assert where == "company = '3M'' OR ''1''=''1'"
