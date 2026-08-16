"""Hermetic tests for deterministic Reciprocal Rank Fusion."""

from __future__ import annotations

from rag.rrf import reciprocal_rank_fusion


def test_rrf_favors_items_ranked_high_in_both_lists():
    dense = ["c1", "c2", "c3"]
    sparse = ["c1", "c3", "c2"]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0].chunk_id == "c1"  # rank 1 in both


def test_rrf_includes_items_present_in_only_one_list():
    dense = ["c1", "c2"]
    sparse = ["c3"]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    ids = {f.chunk_id for f in fused}
    assert ids == {"c1", "c2", "c3"}


def test_rrf_score_formula_is_exact():
    dense = ["c1"]
    sparse = ["c1"]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0].rrf_score == 1 / 61 + 1 / 61


def test_rrf_tie_break_is_ascending_chunk_id():
    # Two chunks with identical fused scores (both absent from the
    # opposite list, both rank 1 in their own list) -- must break the tie
    # deterministically on chunk_id.
    dense = ["z_chunk"]
    sparse = ["a_chunk"]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert fused[0].chunk_id == "a_chunk"
    assert fused[1].chunk_id == "z_chunk"


def test_rrf_is_deterministic_across_repeated_calls():
    dense = ["c3", "c1", "c2"]
    sparse = ["c2", "c3", "c1"]
    f1 = reciprocal_rank_fusion(dense, sparse)
    f2 = reciprocal_rank_fusion(dense, sparse)
    assert [f.chunk_id for f in f1] == [f.chunk_id for f in f2]


def test_rrf_empty_lists_produce_empty_result():
    assert reciprocal_rank_fusion([], []) == []
