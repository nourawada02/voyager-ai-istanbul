"""Qdrant collection management (Checkpoint Phase 3, architecture.md §9.2).

Supports two modes via the same API:
- local-mode (embedded, on-disk or in-memory) -- used by hermetic tests
  and can be used for the real experiment too; no server, no Docker.
- real server mode (`url=...`) -- used by the one real-server integration
  test, when a reachable Qdrant server exists.

Every collection carries a fixed-id fingerprint point (`FINGERPRINT_POINT_ID`)
recording the exact chunking/embedding configuration it was built with.
`ensure_collection` fails closed (raises `CollectionFingerprintMismatch`)
if an existing collection's stored fingerprint disagrees with what the
caller is about to ingest, rather than silently mixing incompatible
vectors from two different configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

FINGERPRINT_POINT_ID = 0
DENSE_VECTOR_NAME = "dense"


class CollectionFingerprintMismatch(RuntimeError):
    """Raised when an existing Qdrant collection's recorded config
    fingerprint does not match the fingerprint of the ingestion run about
    to write to it -- fail closed, never silently mix configurations."""


class QdrantUnavailableError(RuntimeError):
    """Raised when a Qdrant server (real-server mode) cannot be reached.
    Callers (retrieval_service.py) catch this to degrade cleanly."""


def local_client(path: str | None = None) -> QdrantClient:
    """Embedded local-mode client -- `path=None` means fully in-memory
    (hermetic tests); a real path persists to disk between runs."""
    return QdrantClient(location=path or ":memory:")


def server_client(url: str, timeout: float = 5.0) -> QdrantClient:
    try:
        client = QdrantClient(url=url, timeout=timeout)
        client.get_collections()  # forces a real round trip now, not lazily later
        return client
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any connectivity failure degrades the same way
        raise QdrantUnavailableError(f"Qdrant server at {url!r} is not reachable") from exc


def ensure_collection(client: QdrantClient, name: str, dim: int, fingerprint: str) -> None:
    """Creates the collection if absent. If present, validates its stored
    fingerprint matches; raises CollectionFingerprintMismatch on any
    disagreement -- never silently proceeds."""
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={DENSE_VECTOR_NAME: qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
        )
        client.upsert(
            collection_name=name,
            points=[
                qmodels.PointStruct(
                    id=FINGERPRINT_POINT_ID,
                    vector={DENSE_VECTOR_NAME: [0.0] * dim},
                    payload={"__fingerprint__": fingerprint, "__is_fingerprint_marker__": True},
                )
            ],
        )
        return

    marker = client.retrieve(collection_name=name, ids=[FINGERPRINT_POINT_ID])
    if not marker or marker[0].payload.get("__fingerprint__") != fingerprint:
        stored = marker[0].payload.get("__fingerprint__") if marker else None
        raise CollectionFingerprintMismatch(
            f"collection {name!r} was built with fingerprint {stored!r}, expected {fingerprint!r}"
        )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunk_ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict[str, Any]],
) -> None:
    """Deterministic, idempotent: point ids are the (stable) chunk_ids
    hashed to Qdrant-compatible integer ids -- upserting the same
    chunk_id/vector/payload twice produces the same final state, never a
    duplicate point."""
    points = [
        qmodels.PointStruct(id=_point_id(cid), vector={DENSE_VECTOR_NAME: vec}, payload={**payload, "chunk_id": cid})
        for cid, vec, payload in zip(chunk_ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points)


def _point_id(chunk_id: str) -> int:
    """Qdrant point ids must be int or UUID; derive a stable positive int
    from the deterministic chunk_id string."""
    import hashlib

    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def dense_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    query_filter: qmodels.Filter | None = None,
) -> list[DenseHit]:
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    hits = []
    for point in result.points:
        if point.payload.get("__is_fingerprint_marker__"):
            continue
        hits.append(DenseHit(chunk_id=point.payload["chunk_id"], score=point.score, payload=point.payload))
    return hits
