"""Tests for rag/ingest_cli.py (Manual QA remediation Q.1, §A).

Two groups:
- Hermetic document-validation tests (no network) -- prove a malformed or
  unsafe demo document is rejected with a clear, specific reason.
- Real-Qdrant-server tests, mirroring rag/tests/test_qdrant_real_server.py's
  own pattern (a genuine networked Qdrant instance, a unique throwaway
  collection per test, never the shared production 'istanbul_rag_B') --
  proves the required demo-ingestion workflow end to end: ingest a small
  synthetic document, retrieve a unique fact from it, re-ingest
  unchanged, and prove the point count does not increase.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid

import pytest

from rag import ingest_cli
from rag.tests.test_qdrant_real_server import QDRANT_TEST_URL, _get_real_server_client

# --- hermetic: document validation (no network) --------------------------------------


def _write(tmp_path, name: str, content: dict | str):
    path = tmp_path / name
    if isinstance(content, dict):
        path.write_text(json.dumps(content), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _valid_record(**overrides) -> dict:
    record = {
        "source_id": "demo_landmark",
        "title": "Demo Landmark",
        "language": "en",
        "content_type": "attraction",
        "text": "A short, safe demo fact about a fictional landmark.",
    }
    record.update(overrides)
    return record


def test_valid_document_record_is_accepted(tmp_path):
    path = _write(tmp_path, "doc.json", _valid_record())
    docs = ingest_cli._load_demo_documents(path)
    assert len(docs) == 1
    assert docs[0].source_id == "demo_landmark"
    assert docs[0].checksum  # always computed, never trusted from the file


def test_checksum_is_always_recomputed_never_trusted_from_the_file(tmp_path):
    path = _write(tmp_path, "doc.json", _valid_record(checksum="not-a-real-checksum"))
    docs = ingest_cli._load_demo_documents(path)
    from rag.ids import sha256_hex

    assert docs[0].checksum == sha256_hex(_valid_record()["text"])


def test_non_json_extension_is_rejected(tmp_path):
    path = _write(tmp_path, "doc.txt", "plain text, not json")
    with pytest.raises(ingest_cli.DocumentRejected, match="unsupported file type"):
        ingest_cli._load_demo_documents(path)


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "doc.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ingest_cli.DocumentRejected, match="not valid UTF-8 JSON"):
        ingest_cli._load_demo_documents(path)


def test_missing_required_field_is_rejected(tmp_path):
    record = _valid_record()
    del record["text"]
    path = _write(tmp_path, "doc.json", record)
    with pytest.raises(ingest_cli.DocumentRejected, match="missing required field"):
        ingest_cli._load_demo_documents(path)


def test_unsupported_language_is_rejected(tmp_path):
    path = _write(tmp_path, "doc.json", _valid_record(language="fr"))
    with pytest.raises(ingest_cli.DocumentRejected, match="not in supported set"):
        ingest_cli._load_demo_documents(path)


def test_empty_text_is_rejected(tmp_path):
    path = _write(tmp_path, "doc.json", _valid_record(text=""))
    with pytest.raises(ingest_cli.DocumentRejected, match="text is empty"):
        ingest_cli._load_demo_documents(path)


def test_oversized_text_is_rejected(tmp_path):
    path = _write(tmp_path, "doc.json", _valid_record(text="x" * (ingest_cli._MAX_TEXT_CHARS + 1)))
    with pytest.raises(ingest_cli.DocumentRejected, match="exceeds"):
        ingest_cli._load_demo_documents(path)


def test_unrecognized_field_is_rejected(tmp_path):
    path = _write(tmp_path, "doc.json", _valid_record(unexpected_field="not part of the approved shape"))
    with pytest.raises(ingest_cli.DocumentRejected, match="unrecognized field"):
        ingest_cli._load_demo_documents(path)


def test_directory_with_no_json_files_is_rejected(tmp_path):
    (tmp_path / "readme.md").write_text("not a document", encoding="utf-8")
    with pytest.raises(ingest_cli.DocumentRejected, match="no \\*.json files found"):
        ingest_cli._load_demo_documents(tmp_path)


def test_directory_of_valid_documents_loads_all(tmp_path):
    _write(tmp_path, "a.json", _valid_record(source_id="demo_a"))
    _write(tmp_path, "b.json", _valid_record(source_id="demo_b"))
    docs = ingest_cli._load_demo_documents(tmp_path)
    assert {d.source_id for d in docs} == {"demo_a", "demo_b"}


# --- real Qdrant server: end-to-end demo ingestion workflow --------------------------


@pytest.fixture
def throwaway_chunk_config(monkeypatch):
    """A unique, disposable chunk-config name (Manual QA remediation
    Q.1) -- collection_name() derives 'istanbul_rag_<name>' from it, so
    every test gets its own real, throwaway Qdrant collection, and the
    shared production 'istanbul_rag_B' collection is never touched."""
    from rag import ingest as ingest_module

    name = f"TESTQ1_{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(ingest_module.CHUNK_CONFIGS, name, {"chunk_tokens": 200, "overlap_tokens": 20, "top_k": 3})
    return name


@pytest.fixture
def real_client_for_cleanup():
    client = _get_real_server_client()
    try:
        yield client
    finally:
        client.close()


def _args(chunk_config: str, **overrides) -> argparse.Namespace:
    base = {"qdrant_url": QDRANT_TEST_URL, "chunk_config": chunk_config}
    base.update(overrides)
    return argparse.Namespace(**base)


def test_status_reports_collection_missing_before_any_ingestion(throwaway_chunk_config, real_client_for_cleanup):
    exit_code = ingest_cli.cmd_status(_args(throwaway_chunk_config))
    assert exit_code == 1  # not ready


def test_bootstrap_populates_a_genuinely_missing_collection_and_is_idempotent(throwaway_chunk_config, real_client_for_cleanup):
    """Required verification (Manual QA remediation Q.1, §E.6): "Qdrant
    bootstraps from an empty test collection" -- proves the real
    `bootstrap` subcommand (not just `ingest`) against a collection that
    genuinely does not exist yet, using the real approved corpus, and
    that re-running it a second time is a safe no-op."""
    collection_name = f"istanbul_rag_{throwaway_chunk_config}"
    try:
        status_before = ingest_cli._collection_status(
            real_client_for_cleanup, collection_name, ingest_cli._expected_fingerprint(throwaway_chunk_config)
        )
        assert status_before["state"] == "collection_missing"

        exit_code = ingest_cli.cmd_bootstrap(_args(throwaway_chunk_config))
        assert exit_code == 0
        first_count = real_client_for_cleanup.count(collection_name).count
        assert first_count > 1  # the fingerprint marker plus real approved-corpus chunks

        # Re-running bootstrap against an already-populated, matching
        # collection must be a safe no-op -- never a duplicate re-ingest.
        exit_code = ingest_cli.cmd_bootstrap(_args(throwaway_chunk_config))
        assert exit_code == 0
        second_count = real_client_for_cleanup.count(collection_name).count
        assert second_count == first_count
    finally:
        try:
            real_client_for_cleanup.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass


def test_demo_document_ingest_retrieve_and_idempotent_reingest(tmp_path, throwaway_chunk_config, real_client_for_cleanup):
    """The exact required test (Manual QA remediation Q.1, §A demo
    ingestion): a small synthetic demo document is ingested, a unique
    fact from it is retrieved, ingestion is re-run unchanged, and the
    point count does not increase the second time."""
    unique_fact_text = (
        "The Zephyrion Test Kiosk (a fictional Q.1 test fixture, never a real place) "
        "is painted entirely in the invented color glimmercrust violet."
    )
    doc_path = tmp_path / "demo.json"
    doc_path.write_text(json.dumps(_valid_record(
        source_id="q1_zephyrion_test_kiosk", title="Zephyrion Test Kiosk", text=unique_fact_text,
    )), encoding="utf-8")

    collection_name = f"istanbul_rag_{throwaway_chunk_config}"
    try:
        exit_code = ingest_cli.cmd_ingest(_args(throwaway_chunk_config, source=str(doc_path)))
        assert exit_code == 0
        first_count = real_client_for_cleanup.count(collection_name).count
        assert first_count >= 1  # the fingerprint marker plus at least the one real chunk

        # Retrieve a unique fact from the demo document -- proves it is
        # genuinely searchable, not just stored.
        from rag import embeddings, qdrant_store

        query_vector = embeddings.embed_query("What color is the Zephyrion Test Kiosk painted?")
        hits = qdrant_store.dense_search(real_client_for_cleanup, collection_name, query_vector, top_k=3)
        assert hits, "expected at least one real hit from the demo document"
        assert hits[0].payload["source_id"] == "q1_zephyrion_test_kiosk"
        assert "glimmercrust violet" in hits[0].payload["text"]

        # Re-run ingestion, completely unchanged -- must be a no-op:
        # the point count must NOT increase.
        exit_code = ingest_cli.cmd_ingest(_args(throwaway_chunk_config, source=str(doc_path)))
        assert exit_code == 0
        second_count = real_client_for_cleanup.count(collection_name).count
        assert second_count == first_count
    finally:
        try:
            real_client_for_cleanup.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 -- best-effort test cleanup, never masks a real assertion failure
            pass


def test_changed_document_content_updates_the_same_point_not_a_duplicate(tmp_path, throwaway_chunk_config, real_client_for_cleanup):
    collection_name = f"istanbul_rag_{throwaway_chunk_config}"
    doc_path = tmp_path / "demo.json"
    try:
        doc_path.write_text(json.dumps(_valid_record(source_id="q1_changing_doc", text="Original fact about a fictional place.")), encoding="utf-8")
        ingest_cli.cmd_ingest(_args(throwaway_chunk_config, source=str(doc_path)))
        first_count = real_client_for_cleanup.count(collection_name).count

        doc_path.write_text(json.dumps(_valid_record(source_id="q1_changing_doc", text="A DIFFERENT fact about the same fictional place, now updated.")), encoding="utf-8")
        exit_code = ingest_cli.cmd_ingest(_args(throwaway_chunk_config, source=str(doc_path)))
        assert exit_code == 0
        second_count = real_client_for_cleanup.count(collection_name).count
        # Same number of points (one chunk each way) -- the changed
        # content overwrote the same deterministic point id, never added
        # a duplicate alongside the old version.
        assert second_count == first_count
    finally:
        try:
            real_client_for_cleanup.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass
