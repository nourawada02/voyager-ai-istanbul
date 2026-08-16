"""Hermetic Qdrant tests -- real qdrant-client, embedded in-memory mode,
no network server. Covers collection setup, fingerprint validation,
idempotent upsert, dense search, and metadata filtering."""

from __future__ import annotations

import pytest
from qdrant_client.http import models as qmodels

from rag import qdrant_store
from rag.tests.conftest import FAKE_DIM, fake_embed, fake_embed_query


def test_ensure_collection_creates_and_records_fingerprint(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_a", FAKE_DIM, "fp_123")
    names = {c.name for c in qdrant_client_hermetic.get_collections().collections}
    assert "coll_a" in names


def test_ensure_collection_is_idempotent_for_matching_fingerprint(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_b", FAKE_DIM, "fp_x")
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_b", FAKE_DIM, "fp_x")  # must not raise


def test_ensure_collection_raises_on_fingerprint_mismatch(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_c", FAKE_DIM, "fp_old")
    with pytest.raises(qdrant_store.CollectionFingerprintMismatch):
        qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_c", FAKE_DIM, "fp_new")


def test_upsert_and_dense_search_roundtrip(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_d", FAKE_DIM, "fp1")
    chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
    texts = ["Hagia Sophia was built in 532.", "The Grand Bazaar opened in 1461.", "Topkapi housed sultans."]
    vectors = fake_embed(texts)
    payloads = [{"source_id": f"src{i}", "district_id": None, "poi_id": None} for i in range(3)]
    qdrant_store.upsert_chunks(qdrant_client_hermetic, "coll_d", chunk_ids, vectors, payloads)

    query_vec = fake_embed_query("Hagia Sophia was built in 532.")
    hits = qdrant_store.dense_search(qdrant_client_hermetic, "coll_d", query_vec, top_k=3)
    assert hits[0].chunk_id == "chunk_1"  # identical text -> nearest neighbor


def test_upsert_is_idempotent_no_duplicate_points(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_e", FAKE_DIM, "fp1")
    ids = ["chunk_x"]
    vectors = fake_embed(["some text"])
    payloads = [{"source_id": "src1", "district_id": None, "poi_id": None}]
    qdrant_store.upsert_chunks(qdrant_client_hermetic, "coll_e", ids, vectors, payloads)
    qdrant_store.upsert_chunks(qdrant_client_hermetic, "coll_e", ids, vectors, payloads)
    count = qdrant_client_hermetic.count("coll_e").count
    assert count == 2  # 1 fingerprint marker point + 1 real chunk point, never duplicated


def test_metadata_filter_restricts_results(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_f", FAKE_DIM, "fp1")
    ids = ["c1", "c2"]
    vectors = fake_embed(["text about fatih", "text about beyoglu"])
    payloads = [
        {"source_id": "s1", "district_id": "district_fatih", "poi_id": None},
        {"source_id": "s2", "district_id": "district_beyoglu", "poi_id": None},
    ]
    qdrant_store.upsert_chunks(qdrant_client_hermetic, "coll_f", ids, vectors, payloads)

    query_vec = fake_embed_query("anything")
    flt = qmodels.Filter(must=[qmodels.FieldCondition(key="district_id", match=qmodels.MatchValue(value="district_fatih"))])
    hits = qdrant_store.dense_search(qdrant_client_hermetic, "coll_f", query_vec, top_k=5, query_filter=flt)
    assert all(h.payload["district_id"] == "district_fatih" for h in hits)
    assert len(hits) == 1


def test_fingerprint_marker_point_never_returned_as_a_search_hit(qdrant_client_hermetic):
    qdrant_store.ensure_collection(qdrant_client_hermetic, "coll_g", FAKE_DIM, "fp1")
    ids = ["c1"]
    vectors = fake_embed(["real chunk"])
    payloads = [{"source_id": "s1", "district_id": None, "poi_id": None}]
    qdrant_store.upsert_chunks(qdrant_client_hermetic, "coll_g", ids, vectors, payloads)

    query_vec = fake_embed_query("real chunk")
    hits = qdrant_store.dense_search(qdrant_client_hermetic, "coll_g", query_vec, top_k=10)
    assert all("chunk_id" in h.payload for h in hits)
    assert all(h.chunk_id != "" for h in hits)
