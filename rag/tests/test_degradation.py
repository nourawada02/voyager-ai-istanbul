"""Tests for clean degradation when Qdrant is unavailable -- structured
catalog only, no descriptive claim beyond it, clearly marked."""

from __future__ import annotations

from rag.answer_service import GroundedAnswerService
from rag.qdrant_store import QdrantUnavailableError, server_client
from rag.retrieval_service import RetrievalService


def test_server_client_raises_typed_error_when_unreachable():
    import pytest

    with pytest.raises(QdrantUnavailableError):
        server_client("http://127.0.0.1:1", timeout=1.0)  # port 1: nothing listens, fails fast


def test_retrieval_service_degrades_cleanly_when_client_is_none():
    svc = RetrievalService(client=None, collection_name="x", bm25_index=None, embed_query_fn=lambda q: [], collection_fingerprint="fp")
    result = svc.search("any query", "en", "A", top_k=3)
    assert result["zero_result"] is True
    assert result["items"] == []


def test_answer_service_degraded_response_uses_only_structured_catalog_fact():
    retrieval = RetrievalService(client=None, collection_name="x", bm25_index=None, embed_query_fn=lambda q: [], collection_fingerprint="fp")
    svc = GroundedAnswerService(retrieval=retrieval, provider=None, chunk_config="A", top_k=3, retrieval_mode="dense")
    result = svc.answer("Tell me about Hagia Sophia", "en", {}, {}, poi_id="poi_hagia_sophia")
    assert result.degraded is True
    assert "Degraded mode" in result.answer_text
    assert "district_fatih" in result.answer_text  # only the catalog fact, nothing invented
    assert "historic_religious_site" in result.answer_text


def test_answer_service_degraded_response_without_poi_id_is_plain_refusal():
    retrieval = RetrievalService(client=None, collection_name="x", bm25_index=None, embed_query_fn=lambda q: [], collection_fingerprint="fp")
    svc = GroundedAnswerService(retrieval=retrieval, provider=None, chunk_config="A", top_k=3, retrieval_mode="dense")
    result = svc.answer("What is the weather?", "en", {}, {})
    from rag.answer_service import REFUSAL_TEXT

    assert result.answer_text == REFUSAL_TEXT
