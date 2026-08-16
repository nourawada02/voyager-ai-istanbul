"""Builds normalized document records + the corpus manifest from
`corpus_sources.py` (Checkpoint Phase 3).

Writes:
- rag/documents/<source_id>.json  -- one normalized document record per
  source, with a real sha256 checksum of the exact stored text.
- rag/manifests/corpus_manifest.json -- one manifest listing every
  document's full provenance (source_id, title, publisher, url, language,
  content_type, retrieved_at, license/attribution, checksum, poi_id,
  district_id) plus the corpus fingerprint (ids.corpus_fingerprint).

Deterministic and idempotent: re-running with unchanged corpus_sources.py
reproduces byte-identical output (retrieved_at is fixed at the actual
real fetch date recorded below, not wall-clock "now", so re-runs don't
drift).
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.corpus_sources import DOCUMENTS, LICENSE, PUBLISHER
from rag.ids import corpus_fingerprint, document_id, sha256_hex

RAG_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = RAG_ROOT / "documents"
MANIFESTS_DIR = RAG_ROOT / "manifests"

# The real date this fetch-and-extract pass was performed (see the
# conversation's tool-call record) -- fixed, not wall-clock, so rebuilds
# are reproducible.
RETRIEVED_AT = "2026-08-15T00:00:00Z"


def build() -> dict:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    checksums: dict[str, str] = {}

    for doc in DOCUMENTS:
        checksum = sha256_hex(doc.text)
        checksums[doc.source_id] = checksum

        record = {
            "schema_version": "1.0.0",
            "source_id": doc.source_id,
            "document_id": document_id(doc.source_id),
            "title": doc.title,
            "publisher": PUBLISHER,
            "url": doc.url,
            "language": doc.language,
            "content_type": doc.content_type,
            "retrieved_at": RETRIEVED_AT,
            "license": LICENSE,
            "attribution": f"{PUBLISHER}, \"{doc.title}\", {doc.language.upper()} Wikipedia, {LICENSE}",
            "checksum": checksum,
            "poi_id": doc.poi_id,
            "district_id": doc.district_id,
            "topic_group": doc.topic_group,
            "text": doc.text,
        }
        out_path = DOCUMENTS_DIR / f"{doc.source_id}.json"
        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

        manifest_entries.append(
            {
                "source_id": doc.source_id,
                "title": doc.title,
                "publisher": PUBLISHER,
                "url": doc.url,
                "language": doc.language,
                "content_type": doc.content_type,
                "retrieved_at": RETRIEVED_AT,
                "license": LICENSE,
                "checksum": checksum,
                "poi_id": doc.poi_id,
                "district_id": doc.district_id,
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "corpus_fingerprint": corpus_fingerprint(checksums),
        "document_count": len(DOCUMENTS),
        "language_counts": {
            lang: sum(1 for d in DOCUMENTS if d.language == lang) for lang in ("en", "tr", "ar")
        },
        "documents": manifest_entries,
    }
    manifest_path = MANIFESTS_DIR / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    return manifest


if __name__ == "__main__":
    m = build()
    print(f"wrote {m['document_count']} documents, corpus_fingerprint={m['corpus_fingerprint'][:16]}...")
    print(f"language_counts={m['language_counts']}")
