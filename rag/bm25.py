"""Deterministic BM25-compatible sparse scoring (Checkpoint Phase 3,
architecture.md §9.2). Pure Python (via `rank_bm25`), no external service,
no randomness -- tokenization is deterministic whitespace/punctuation
splitting applied identically to indexed chunk text and queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Deterministic, locale-independent tokenization: lowercase word
    tokens via a Unicode-aware regex. Applied identically to indexed
    chunk text and queries -- the one shared tokenization rule."""
    return _TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class BM25Index:
    chunk_ids: tuple[str, ...]
    _bm25: BM25Okapi

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns up to top_k (chunk_id, score) pairs, descending score,
        stable tie-break on chunk_id ascending (deterministic total
        order)."""
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.chunk_ids, scores),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]


def build_index(chunk_ids: list[str], texts: list[str]) -> BM25Index:
    if len(chunk_ids) != len(texts):
        raise ValueError("chunk_ids and texts must be the same length")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return BM25Index(chunk_ids=tuple(chunk_ids), _bm25=bm25)
