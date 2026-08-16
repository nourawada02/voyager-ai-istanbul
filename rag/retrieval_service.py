"""Provider-independent retrieval service (Checkpoint Phase 3).

Wraps dense (Qdrant) + sparse (BM25) retrieval and RRF fusion behind one
`RetrievalService.search()` call, producing a `contracts/RetrievalResult.schema.json`
-shaped dict. Degrades cleanly when Qdrant is unavailable: returns a
result with `zero_result=True` and an empty item list rather than
raising, so the grounded-answer service can fall back to the structured
catalog (answer_service.py) instead of crashing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag import qdrant_store, rrf
from rag.bm25 import BM25Index


@dataclass(frozen=True)
class RetrievalService:
    client: QdrantClient | None
    collection_name: str
    bm25_index: BM25Index | None
    embed_query_fn: Callable[[str], list[float]]
    collection_fingerprint: str

    def search(
        self,
        query: str,
        query_language: str,
        chunk_config: str,
        top_k: int,
        mode: str = "dense",
        district_id: str | None = None,
        poi_id: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()

        if self.client is None:
            return self._empty_result(query, query_language, chunk_config, top_k, mode, t0)

        query_filter = _build_filter(district_id, poi_id)

        try:
            query_vector = self.embed_query_fn(query)
            dense_hits = qdrant_store.dense_search(
                self.client, self.collection_name, query_vector, max(top_k, 20), query_filter
            )
        except Exception:
            return self._empty_result(query, query_language, chunk_config, top_k, mode, t0)

        dense_ranked = [h.chunk_id for h in dense_hits]
        dense_score_by_id = {h.chunk_id: h.score for h in dense_hits}
        source_by_id = {h.chunk_id: h.payload.get("source_id", "") for h in dense_hits}

        if mode == "dense":
            top = dense_ranked[:top_k]
            items = [
                {
                    "rank": i + 1,
                    "chunk_id": cid,
                    "source_id": source_by_id.get(cid, ""),
                    "score": dense_score_by_id.get(cid, 0.0),
                    "dense_score": dense_score_by_id.get(cid, 0.0),
                }
                for i, cid in enumerate(top)
            ]
        elif mode == "hybrid_rrf":
            sparse_ranked = [cid for cid, _ in (self.bm25_index.search(query, max(top_k, 20)) if self.bm25_index else [])]
            sparse_score_by_id = dict(self.bm25_index.search(query, max(top_k, 20))) if self.bm25_index else {}
            fused = rrf.reciprocal_rank_fusion(dense_ranked, sparse_ranked)[:top_k]
            items = [
                {
                    "rank": i + 1,
                    "chunk_id": f.chunk_id,
                    "source_id": source_by_id.get(f.chunk_id, ""),
                    "score": f.rrf_score,
                    "dense_score": dense_score_by_id.get(f.chunk_id, 0.0),
                    "sparse_score": sparse_score_by_id.get(f.chunk_id, 0.0),
                }
                for i, f in enumerate(fused)
            ]
        else:
            raise ValueError(f"unknown retrieval mode {mode!r}")

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "schema_version": "1.0.0",
            "query": query,
            "query_language": query_language,
            "retrieval_mode": mode,
            "chunk_config": chunk_config,
            "top_k": top_k,
            "collection_fingerprint": self.collection_fingerprint,
            "items": items,
            "latency_ms": latency_ms,
            "zero_result": len(items) == 0,
        }

    def _empty_result(self, query, query_language, chunk_config, top_k, mode, t0) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "query": query,
            "query_language": query_language,
            "retrieval_mode": mode,
            "chunk_config": chunk_config,
            "top_k": top_k,
            "collection_fingerprint": self.collection_fingerprint,
            "items": [],
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "zero_result": True,
        }


def _build_filter(district_id: str | None, poi_id: str | None) -> qmodels.Filter | None:
    conditions = []
    if district_id is not None:
        conditions.append(qmodels.FieldCondition(key="district_id", match=qmodels.MatchValue(value=district_id)))
    if poi_id is not None:
        conditions.append(qmodels.FieldCondition(key="poi_id", match=qmodels.MatchValue(value=poi_id)))
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)
