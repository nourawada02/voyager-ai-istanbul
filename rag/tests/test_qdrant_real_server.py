"""Real Qdrant-SERVER integration test (network, not embedded mode).

Genuine client code exercising a real networked Qdrant instance via
`qdrant_store.server_client()` -- never a mock. Requires an actual Qdrant
server reachable at QDRANT_TEST_URL (default http://localhost:6333),
started from a pinned image (see docs/adr/0005-phase3-rag-ownership.md /
the Phase 3 remediation report for the exact tag+digest and launch
command) -- never `latest`, never persistent, always cleaned up by the
caller after the run.

If QDRANT_REQUIRE_SERVER=1 is set, an unreachable server is a hard test
FAILURE, never a skip -- this module never silently hides the absence of
a real server behind pytest.skip.
"""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client.http import models as qmodels

from rag import qdrant_store
from rag.tests.conftest import FAKE_DIM, fake_embed, fake_embed_query

QDRANT_TEST_URL = os.environ.get("QDRANT_TEST_URL", "http://localhost:6333")
_REQUIRE_SERVER = os.environ.get("QDRANT_REQUIRE_SERVER") == "1"


def _get_real_server_client():
    try:
        return qdrant_store.server_client(QDRANT_TEST_URL, timeout=5.0)
    except qdrant_store.QdrantUnavailableError:
        if _REQUIRE_SERVER:
            pytest.fail(f"QDRANT_REQUIRE_SERVER=1 but no Qdrant server is reachable at {QDRANT_TEST_URL!r}")
        pytest.fail(
            f"real Qdrant server not reachable at {QDRANT_TEST_URL!r} in this environment -- "
            "this is a genuine, disclosed environmental limitation, not a code defect; "
            "see the Phase 3 remediation report"
        )
        raise  # unreachable, keeps type-checkers happy; pytest.fail always raises


@pytest.fixture
def real_client():
    client = _get_real_server_client()
    try:
        yield client
    finally:
        client.close()  # clean client/server shutdown of this connection


@pytest.fixture
def collection_name():
    # unique per test run, never reused across parallel/rerun invocations
    return f"rag_real_server_test_{uuid.uuid4().hex[:12]}"


def test_server_is_ready_and_reachable(real_client):
    """Proves readiness through the real network API, not just a raw TCP
    connect: get_collections() is a real authenticated round trip."""
    result = real_client.get_collections()
    assert isinstance(result.collections, list)


def test_collection_creation_and_dense_vector_configuration(real_client, collection_name):
    qdrant_store.ensure_collection(real_client, collection_name, FAKE_DIM, "fp_v1")
    try:
        info = real_client.get_collection(collection_name)
        vectors_config = info.config.params.vectors
        dense = vectors_config[qdrant_store.DENSE_VECTOR_NAME]
        assert dense.size == FAKE_DIM
        assert dense.distance == qmodels.Distance.COSINE
    finally:
        real_client.delete_collection(collection_name)


def test_fingerprint_marker_point_is_persisted_on_the_real_server(real_client, collection_name):
    qdrant_store.ensure_collection(real_client, collection_name, FAKE_DIM, "fp_marker_test")
    try:
        marker = real_client.retrieve(collection_name=collection_name, ids=[qdrant_store.FINGERPRINT_POINT_ID])
        assert marker
        assert marker[0].payload["__fingerprint__"] == "fp_marker_test"
    finally:
        real_client.delete_collection(collection_name)


def test_idempotent_ingestion_against_the_real_server(real_client, collection_name):
    qdrant_store.ensure_collection(real_client, collection_name, FAKE_DIM, "fp_idempotent")
    try:
        chunk_ids = ["real_c1", "real_c2"]
        vectors = fake_embed(["Hagia Sophia was built in 532.", "The Grand Bazaar opened in 1461."])
        payloads = [
            {"source_id": "wiki_en_hagia_sophia", "district_id": "district_fatih", "poi_id": "poi_hagia_sophia"},
            {"source_id": "wiki_en_grand_bazaar", "district_id": "district_fatih", "poi_id": "poi_grand_bazaar"},
        ]
        qdrant_store.upsert_chunks(real_client, collection_name, chunk_ids, vectors, payloads)
        qdrant_store.upsert_chunks(real_client, collection_name, chunk_ids, vectors, payloads)  # re-ingest
        count = real_client.count(collection_name).count
        assert count == 3  # 2 real points + 1 fingerprint marker, never duplicated
    finally:
        real_client.delete_collection(collection_name)


def test_real_vector_upsert_dense_search_and_metadata_filter(real_client, collection_name):
    qdrant_store.ensure_collection(real_client, collection_name, FAKE_DIM, "fp_search")
    try:
        chunk_ids = ["real_c1", "real_c2", "real_c3"]
        vectors = fake_embed(
            ["Hagia Sophia was built in 532.", "The Grand Bazaar opened in 1461.", "Topkapi housed sultans."]
        )
        payloads = [
            {"source_id": "s1", "district_id": "district_fatih", "poi_id": "poi_hagia_sophia"},
            {"source_id": "s2", "district_id": "district_fatih", "poi_id": "poi_grand_bazaar"},
            {"source_id": "s3", "district_id": "district_beyoglu", "poi_id": None},
        ]
        qdrant_store.upsert_chunks(real_client, collection_name, chunk_ids, vectors, payloads)

        query_vec = fake_embed_query("Hagia Sophia was built in 532.")
        hits = qdrant_store.dense_search(real_client, collection_name, query_vec, top_k=3)
        assert hits[0].chunk_id == "real_c1"

        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(key="district_id", match=qmodels.MatchValue(value="district_fatih"))]
        )
        filtered = qdrant_store.dense_search(real_client, collection_name, query_vec, top_k=3, query_filter=flt)
        assert all(h.payload["district_id"] == "district_fatih" for h in filtered)
        assert len(filtered) == 2
    finally:
        real_client.delete_collection(collection_name)


def test_collection_config_mismatch_is_rejected_by_the_real_server(real_client, collection_name):
    qdrant_store.ensure_collection(real_client, collection_name, FAKE_DIM, "fp_original")
    try:
        with pytest.raises(qdrant_store.CollectionFingerprintMismatch):
            qdrant_store.ensure_collection(real_client, collection_name, FAKE_DIM, "fp_different")
    finally:
        real_client.delete_collection(collection_name)


def test_client_close_is_a_clean_shutdown_and_a_fresh_client_can_reconnect(collection_name):
    client1 = _get_real_server_client()
    qdrant_store.ensure_collection(client1, collection_name, FAKE_DIM, "fp_reconnect")
    client1.close()

    client2 = _get_real_server_client()
    try:
        info = client2.get_collection(collection_name)
        assert info is not None
    finally:
        client2.delete_collection(collection_name)
        client2.close()
