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
