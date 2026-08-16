"""Deterministic Reciprocal Rank Fusion (Checkpoint Phase 3,
architecture.md §9.2/§9.5).

RRF score for a chunk = sum over each ranked list it appears in of
1 / (k + rank), rank 1-based. A chunk absent from a list contributes 0
for that list. Stable tie-breaking: ties in fused score break on
ascending chunk_id, giving a total deterministic order every time.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedResult:
    chunk_id: str
    rrf_score: float
    dense_rank: int | None
    sparse_rank: int | None


def reciprocal_rank_fusion(
    dense_ranked: list[str],
    sparse_ranked: list[str],
    k: int = DEFAULT_RRF_K,
) -> list[FusedResult]:
    """`dense_ranked`/`sparse_ranked` are chunk_id lists already in rank
    order (best first). Returns chunks fused and sorted by descending RRF
    score, ties broken by ascending chunk_id."""
    dense_rank_of = {cid: i + 1 for i, cid in enumerate(dense_ranked)}
    sparse_rank_of = {cid: i + 1 for i, cid in enumerate(sparse_ranked)}
    all_ids = set(dense_ranked) | set(sparse_ranked)

    fused: list[FusedResult] = []
    for cid in all_ids:
        score = 0.0
        d_rank = dense_rank_of.get(cid)
        s_rank = sparse_rank_of.get(cid)
        if d_rank is not None:
            score += 1.0 / (k + d_rank)
        if s_rank is not None:
            score += 1.0 / (k + s_rank)
        fused.append(FusedResult(chunk_id=cid, rrf_score=score, dense_rank=d_rank, sparse_rank=s_rank))

    fused.sort(key=lambda r: (-r.rrf_score, r.chunk_id))
    return fused
