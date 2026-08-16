"""Tests that re-ingestion is idempotent and that a chunk-config change
is detected and fails closed, using the real corpus documents and a
hermetic embedded Qdrant client with a fast fake embedding function (no
need for the slow real E5 model to prove these mechanics)."""

from __future__ import annotations

import pytest

from rag import ingest, qdrant_store
from rag.chunking import SimpleWhitespaceTokenizer
from rag.tests.conftest import fake_embed


@pytest.fixture
def documents():
    return ingest.load_documents()[:3]  # small subset, fast


def test_ingest_config_is_idempotent(documents):
    client = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    fp = {"model_name": "fake", "revision": "0", "dim": 16}

    r1 = ingest.ingest_config(client, "A", tok, fake_embed, fp, documents)
    r2 = ingest.ingest_config(client, "A", tok, fake_embed, fp, documents)

    assert r1.chunk_ids == r2.chunk_ids
    count = client.count(r1.collection_name).count
    assert count == len(r1.chunk_ids) + 1  # + the one fingerprint marker point, never duplicated


def test_ingest_config_produces_deterministic_chunk_ids_across_fresh_runs(documents):
    client1 = qdrant_store.local_client(path=None)
    client2 = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    fp = {"model_name": "fake", "revision": "0", "dim": 16}

    r1 = ingest.ingest_config(client1, "B", tok, fake_embed, fp, documents)
    r2 = ingest.ingest_config(client2, "B", tok, fake_embed, fp, documents)
    assert r1.chunk_ids == r2.chunk_ids
    assert r1.fingerprint == r2.fingerprint


def test_changing_chunk_config_on_the_same_collection_fails_closed(documents):
    client = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    fp_v1 = {"model_name": "fake", "revision": "0", "dim": 16}
    fp_v2 = {"model_name": "fake", "revision": "1", "dim": 16}  # different embedding fingerprint -> different config fp

    ingest.ingest_config(client, "C", tok, fake_embed, fp_v1, documents)
    with pytest.raises(qdrant_store.CollectionFingerprintMismatch):
        ingest.ingest_config(client, "C", tok, fake_embed, fp_v2, documents)


def test_all_four_configs_produce_non_empty_chunk_sets_for_the_real_corpus():
    client = qdrant_store.local_client(path=None)
    tok = SimpleWhitespaceTokenizer()
    fp = {"model_name": "fake", "revision": "0", "dim": 16}
    documents = ingest.load_documents()
    for config_name in ingest.CHUNK_CONFIGS:
        result = ingest.ingest_config(client, config_name, tok, fake_embed, fp, documents)
        assert len(result.chunk_ids) > 0, config_name
        assert len(result.chunk_ids) == len(set(result.chunk_ids)), f"{config_name}: duplicate chunk_ids"
