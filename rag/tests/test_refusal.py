"""Tests for the canonical refusal path -- exact string, and triggered
whenever retrieval is empty, the model reports answer_status
'insufficient', or every claim fails validation."""

from __future__ import annotations

from dataclasses import dataclass

from rag.answer_service import REFUSAL_TEXT, GroundedAnswerService
from rag.retrieval_service import RetrievalService
from rag.tests.conftest import StubStructuredProvider


def _empty_retrieval_service() -> RetrievalService:
    return RetrievalService(
        client=None,  # None client -> always empty/degraded result, no network needed
        collection_name="unused",
        bm25_index=None,
        embed_query_fn=lambda q: [],
        collection_fingerprint="fp",
    )


@dataclass
class _StubRetrieval:
    items: list

    def search(self, **kwargs):
        return {
            "schema_version": "1.0.0", "query": kwargs["query"], "query_language": kwargs["query_language"],
            "retrieval_mode": kwargs["mode"], "chunk_config": kwargs["chunk_config"], "top_k": kwargs["top_k"],
            "collection_fingerprint": "fp", "items": self.items, "latency_ms": 1.0, "zero_result": False,
        }


def test_exact_refusal_text_constant_matches_architecture_template():
    assert REFUSAL_TEXT == "I don't have grounded information on this in my current sources."


def test_zero_retrieval_result_triggers_refusal():
    provider = StubStructuredProvider(responses=[{"answer_status": "grounded", "claims": [{"text": "x", "chunk_ids": ["c1"]}]}])
    svc = GroundedAnswerService(retrieval=_empty_retrieval_service(), provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")
    result = svc.answer("Any question", "en", {}, {})
    assert result.refused is True
    assert REFUSAL_TEXT in result.answer_text
    assert provider.calls == []  # provider must never be called when retrieval is empty


def test_no_provider_available_also_degrades_to_refusal_or_catalog():
    svc = GroundedAnswerService(
        retrieval=_empty_retrieval_service(), provider=None, chunk_config="A", top_k=3, retrieval_mode="dense",
    )
    result = svc.answer("Any question", "en", {}, {})
    assert result.refused is True


def test_model_reported_insufficient_triggers_exact_canonical_refusal():
    provider = StubStructuredProvider(responses=[{"answer_status": "insufficient", "claims": []}])
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")
    result = svc.answer("Any question", "en", {"c1": "some chunk text"}, {"s1": {"title": "T", "url": "https://x"}})
    assert result.answer_text == REFUSAL_TEXT
    assert result.refused is True
    assert result.citations == []


def test_all_claims_failing_validation_falls_back_to_refusal():
    # A claim citing a chunk_id that was never retrieved -> dropped -> zero valid claims -> refusal.
    provider = StubStructuredProvider(
        responses=[{"answer_status": "grounded", "claims": [{"text": "Fact", "chunk_ids": ["never_retrieved"]}]}]
    )
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")
    result = svc.answer("Any question", "en", {"c1": "some chunk text"}, {"s1": {"title": "T", "url": "https://x"}})
    assert result.answer_text == REFUSAL_TEXT
    assert result.refused is True


def test_live_data_question_refuses_before_retrieval_or_provider_call():
    provider = StubStructuredProvider(
        responses=[{"answer_status": "grounded", "claims": [{"text": "Sunny", "chunk_ids": ["c1"]}]}]
    )
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")

    for question, language in (
        ("What is the weather in Istanbul today?", "en"),
        ("İstanbul'da bugün hava durumu nasıl?", "tr"),
        ("كيف هو الطقس في إسطنبول اليوم؟", "ar"),
    ):
        result = svc.answer(question, language, {"c1": "old weather"}, {"s1": {}})
        assert result.answer_text == REFUSAL_TEXT
        assert result.refused is True
        assert result.degradation_reason == "live_data_required"

    assert provider.calls == []
