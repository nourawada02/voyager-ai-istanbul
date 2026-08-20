"""Ingestion orchestration for one chunk config (Checkpoint Phase 3).

Loads normalized document records (rag/documents/*.json, written by
build_documents.py), applies heading-aware recursive chunking, embeds
every chunk with the pinned E5 model (passage: prefix), and upserts into
a Qdrant collection named for the config. Fails closed on a fingerprint
mismatch (qdrant_store.ensure_collection) rather than silently mixing
configurations. Also builds the parallel in-memory BM25 sparse index for
the same chunk set (bm25.py), returned alongside so a caller can run
dense, sparse, or RRF-fused retrieval without re-chunking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from rag import bm25, qdrant_store
from rag.chunking import ChunkRecord, Tokenizer, chunk_document
from rag.ids import chunk_id as make_chunk_id
from rag.ids import config_fingerprint, corpus_fingerprint

RAG_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = RAG_ROOT / "documents"

CHUNK_CONFIGS: dict[str, dict[str, int]] = {
    "A": {"chunk_tokens": 350, "overlap_tokens": 50, "top_k": 3},
    "B": {"chunk_tokens": 350, "overlap_tokens": 50, "top_k": 5},
    "C": {"chunk_tokens": 700, "overlap_tokens": 100, "top_k": 3},
    "D": {"chunk_tokens": 700, "overlap_tokens": 100, "top_k": 5},
}


@dataclass(frozen=True)
class LoadedDocument:
    source_id: str
    title: str
    language: str
    content_type: str
    poi_id: str | None
    district_id: str | None
    checksum: str
    text: str
    # RAG-FIRST SYSTEM B CHECKPOINT R.1 (additive; default matches every
    # document record written before this checkpoint, which has none of
    # these keys -- .get() below, never raw[...]):
    interest_tags: tuple[str, ...] = ()
    lat: float | None = None
    lon: float | None = None
    side: str | None = None
    display_name: str | None = None
    preferred_period: str = "any"
    entity_kind: str = "general_knowledge"
    publisher_type: str = "wikipedia"
    transformation_method: str = "llm_paraphrase_of_cited_source"
    derived_from_source: str | None = None


def load_documents() -> list[LoadedDocument]:
    docs = []
    for path in sorted(DOCUMENTS_DIR.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        docs.append(
            LoadedDocument(
                source_id=raw["source_id"],
                title=raw["title"],
                language=raw["language"],
                content_type=raw["content_type"],
                poi_id=raw["poi_id"],
                district_id=raw["district_id"],
                checksum=raw["checksum"],
                text=raw["text"],
                interest_tags=tuple(raw.get("interest_tags") or ()),
                lat=raw.get("lat"),
                lon=raw.get("lon"),
                side=raw.get("side"),
                display_name=raw.get("display_name"),
                preferred_period=raw.get("preferred_period", "any"),
                entity_kind=raw.get("entity_kind", "general_knowledge"),
                publisher_type=raw.get("publisher_type", "wikipedia"),
                transformation_method=raw.get("transformation_method", "llm_paraphrase_of_cited_source"),
                derived_from_source=raw.get("derived_from_source"),
            )
        )
    if not docs:
        raise RuntimeError("no documents found -- run `python -m rag.build_documents` first")
    return docs


def collection_name(chunk_config: str) -> str:
    return f"istanbul_rag_{chunk_config}"


def _ingestion_config_dict(chunk_config: str, embedding_fp: dict[str, Any]) -> dict[str, Any]:
    cfg = CHUNK_CONFIGS[chunk_config]
    return {
        "chunk_config": chunk_config,
        "chunk_tokens": cfg["chunk_tokens"],
        "overlap_tokens": cfg["overlap_tokens"],
        "embedding": embedding_fp,
    }


@dataclass(frozen=True)
class IngestionResult:
    chunk_config: str
    collection_name: str
    chunk_ids: list[str]
    chunk_texts: list[str]
    chunk_payloads: list[dict[str, Any]]
    fingerprint: str
    corpus_fp: str
    bm25_index: Any


def ingest_config(
    client: QdrantClient,
    chunk_config: str,
    tokenizer: Tokenizer,
    embed_fn,
    embedding_fp: dict[str, Any],
    documents: list[LoadedDocument] | None = None,
) -> IngestionResult:
    """Runs ingestion for one config (A-D) against `client`. `embed_fn`
    takes a list[str] of already-`passage:`-prefixed-or-not text (the
    caller decides prefixing -- see embeddings.embed_passages) and
    returns list[list[float]]. Idempotent: re-running with an unchanged
    corpus+config reproduces the identical chunk_id set and upserts the
    identical vectors/payloads."""
    docs = documents if documents is not None else load_documents()
    cfg = CHUNK_CONFIGS[chunk_config]

    all_chunk_ids: list[str] = []
    all_texts: list[str] = []
    all_payloads: list[dict[str, Any]] = []

    checksums = {d.source_id: d.checksum for d in docs}
    corpus_fp = corpus_fingerprint(checksums)
    ingestion_cfg = _ingestion_config_dict(chunk_config, embedding_fp)
    fp = config_fingerprint(ingestion_cfg)

    for doc in docs:
        chunks: list[ChunkRecord] = chunk_document(
            doc.text, doc.title, tokenizer, cfg["chunk_tokens"], cfg["overlap_tokens"]
        )
        for chunk in chunks:
            cid = make_chunk_id(doc.source_id, chunk_config, chunk.chunk_index)
            all_chunk_ids.append(cid)
            all_texts.append(chunk.text)
            all_payloads.append(
                {
                    "source_id": doc.source_id,
                    "language": doc.language,
                    "content_type": doc.content_type,
                    "heading_path": list(chunk.heading_path),
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "chunk_index": chunk.chunk_index,
                    "poi_id": doc.poi_id,
                    "district_id": doc.district_id,
                    "source_checksum": doc.checksum,
                    "chunk_config": chunk_config,
                    # RAG-FIRST SYSTEM B CHECKPOINT R.1: entity-linkage
                    # fields consumed by phase4.knowledge.qdrant_client's
                    # RetrievedChunk and phase4.rag_candidates -- always
                    # present in the payload (None/() for general-
                    # knowledge chunks), never a second lookup required.
                    "interest_tags": list(doc.interest_tags),
                    "lat": doc.lat,
                    "lon": doc.lon,
                    "side": doc.side,
                    "display_name": doc.display_name,
                    "preferred_period": doc.preferred_period,
                    "entity_kind": doc.entity_kind,
                    # Manual QA remediation Q.1 (§A): the exact ingestion
                    # config fingerprint this point was written under --
                    # same value as the collection's own fingerprint marker
                    # point, but recorded per-chunk too so a caller can
                    # answer "which ingestion run produced this point"
                    # without a separate lookup.
                    "ingestion_version": fp,
                }
            )

    coll_name = collection_name(chunk_config)
    dim = len(embed_fn([all_texts[0]])[0]) if all_texts else 0
    qdrant_store.ensure_collection(client, coll_name, dim, fp)

    vectors = embed_fn(all_texts)
    qdrant_store.upsert_chunks(client, coll_name, all_chunk_ids, vectors, all_payloads)

    sparse_index = bm25.build_index(all_chunk_ids, all_texts)

    return IngestionResult(
        chunk_config=chunk_config,
        collection_name=coll_name,
        chunk_ids=all_chunk_ids,
        chunk_texts=all_texts,
        chunk_payloads=all_payloads,
        fingerprint=fp,
        corpus_fp=corpus_fp,
        bm25_index=sparse_index,
    )
