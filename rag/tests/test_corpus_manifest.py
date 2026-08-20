"""Tests over the real, already-built corpus manifest and documents
(rag/manifests/corpus_manifest.json, rag/documents/*.json) -- not
synthetic fixtures, since the whole point is verifying the real corpus's
own checksums/provenance/language distribution."""

from __future__ import annotations

import json
from pathlib import Path

from rag.ids import sha256_hex

RAG_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = RAG_ROOT / "manifests" / "corpus_manifest.json"
DOCUMENTS_DIR = RAG_ROOT / "documents"

_REQUIRED_FIELDS = {
    "source_id", "title", "publisher", "url", "language", "content_type",
    "retrieved_at", "license", "checksum",
}

_FORBIDDEN_SUBSTRINGS = (
    "current price", "current hours", "opening hours today", "today's weather",
    "current availability",
)


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_exists_and_has_expected_document_count():
    # RAG-FIRST SYSTEM B R.1 CORRECTION: corpus expanded from the frozen
    # Phase 3 20 documents to 49 (R.1) to 69 (R.1 correction: expanded
    # Arabic/Turkish coverage, official/primary sources, and one more
    # real food-schedulable entity, Cicek Pasaji).
    m = _manifest()
    assert m["document_count"] == 69
    assert len(m["documents"]) == 69


def test_manifest_language_counts_match_actual_documents():
    m = _manifest()
    counts = {"en": 0, "tr": 0, "ar": 0}
    for doc in m["documents"]:
        counts[doc["language"]] += 1
    assert m["language_counts"] == counts
    assert counts["en"] >= 1 and counts["tr"] >= 1 and counts["ar"] >= 1


def test_every_manifest_entry_has_required_provenance_fields():
    m = _manifest()
    for doc in m["documents"]:
        missing = _REQUIRED_FIELDS - set(doc.keys())
        assert not missing, f"{doc['source_id']} missing {missing}"
        assert doc["license"] == "CC BY-SA 4.0"
        assert doc["url"].startswith("https://")


def test_every_document_file_checksum_matches_its_manifest_entry():
    m = _manifest()
    for entry in m["documents"]:
        doc_path = DOCUMENTS_DIR / f"{entry['source_id']}.json"
        assert doc_path.exists(), entry["source_id"]
        record = json.loads(doc_path.read_text(encoding="utf-8"))
        assert record["checksum"] == entry["checksum"]
        assert sha256_hex(record["text"]) == entry["checksum"], "checksum must be a real hash of the stored text"


def test_corpus_fingerprint_is_reproducible_from_manifest_checksums():
    from rag.ids import corpus_fingerprint

    m = _manifest()
    checksums = {d["source_id"]: d["checksum"] for d in m["documents"]}
    assert corpus_fingerprint(checksums) == m["corpus_fingerprint"]


def test_no_document_text_contains_excluded_dynamic_content_markers():
    """Spot-check the corpus boundary: none of the real document texts
    should contain obviously time-sensitive phrasing patterns."""
    for path in DOCUMENTS_DIR.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        lowered = record["text"].lower()
        for forbidden in _FORBIDDEN_SUBSTRINGS:
            assert forbidden not in lowered, f"{record['source_id']} contains {forbidden!r}"


def test_poi_and_district_ids_on_documents_are_syntactically_canonical():
    import re

    poi_pattern = re.compile(r"^poi_[a-z0-9_]+$")
    district_pattern = re.compile(r"^district_[a-z0-9_]+$")
    for path in DOCUMENTS_DIR.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["poi_id"] is not None:
            assert poi_pattern.match(record["poi_id"]), record["source_id"]
        if record["district_id"] is not None:
            assert district_pattern.match(record["district_id"]), record["source_id"]
