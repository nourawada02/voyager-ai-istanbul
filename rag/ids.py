"""Deterministic ID and fingerprint helpers (Checkpoint Phase 3).

Every ID here is a pure function of stable inputs -- never a random UUID,
never wall-clock time, never insertion order -- so ingestion is exactly
reproducible: the same corpus + same chunking config always produces the
same chunk_id set, byte for byte.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(source_id: str) -> str:
    """The document id is simply the source id -- one document per
    source, no separate namespace needed."""
    return source_id


def chunk_id(source_id: str, chunk_config: str, chunk_index: int) -> str:
    """Deterministic, stable across re-ingestion: a pure function of
    (source_id, chunk_config, chunk_index). Re-ingesting the identical
    corpus under the identical config reproduces identical chunk_ids --
    required for idempotent re-ingestion (upsert, not duplicate)."""
    raw = f"{source_id}:{chunk_config}:{chunk_index}"
    return f"chunk_{sha256_hex(raw)[:24]}"


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def config_fingerprint(config: dict[str, Any]) -> str:
    """A pure function of the chunking/embedding configuration (chunk
    tokens, overlap, embedding model + revision, prefixes) -- used to
    detect a mismatch between what a Qdrant collection was built with and
    what the current ingestion run would use, and to fail closed rather
    than silently mixing incompatible vectors."""
    return sha256_hex(_canonical_json(config))


def corpus_fingerprint(source_checksums: dict[str, str]) -> str:
    """A pure function of every source document's own checksum -- changes
    if and only if the corpus content changes, independent of ingestion
    order."""
    return sha256_hex(_canonical_json(dict(sorted(source_checksums.items()))))


def ingestion_fingerprint(corpus_fp: str, config_fp: str) -> str:
    return sha256_hex(f"{corpus_fp}:{config_fp}")
