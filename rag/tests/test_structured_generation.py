"""Tests for structured-JSON generation mechanics: bounded retries,
StructuredGenerationFailure, and the deterministic extractive fallback
path in GroundedAnswerService."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag.answer_service import GroundedAnswerService, _claim_is_supported_by_chunks
from rag.llm_providers import ProviderTransportError, StructuredGenerationFailure, _decode_last_valid_json_object
from rag.tests.conftest import StubStructuredProvider


@dataclass
class _StubRetrieval:
    items: list

    def search(self, **kwargs):
        return {
            "schema_version": "1.0.0", "query": kwargs["query"], "query_language": kwargs["query_language"],
            "retrieval_mode": kwargs["mode"], "chunk_config": kwargs["chunk_config"], "top_k": kwargs["top_k"],
            "collection_fingerprint": "fp", "items": self.items, "latency_ms": 1.0, "zero_result": False,
        }


def test_claim_evidence_check_accepts_high_overlap():
    assert _claim_is_supported_by_chunks(
        "Hagia Sophia was built in 532 AD.", ["Hagia Sophia was built in 532 AD by Justinian."]
    )


def test_claim_evidence_check_rejects_unrelated_text():
    assert not _claim_is_supported_by_chunks(
        "The Grand Bazaar has four thousand shops.", ["Hagia Sophia was built in 532 AD by Justinian."]
    )


def test_claim_evidence_check_rejects_empty_claim():
    assert not _claim_is_supported_by_chunks("", ["some chunk text"])


def test_generate_json_retries_on_invalid_json_then_succeeds():
    from rag.llm_providers import OllamaProvider

    call_count = {"n": 0}
    responses = ["not json at all", '{"answer_status": "grounded", "claims": []}']

    class _FlakyProvider(OllamaProvider):
        def generate(self, system: str, user: str) -> str:
            call_count["n"] += 1
            return responses[call_count["n"] - 1]

    provider = _FlakyProvider(model="fake")

    def validate(data):
        assert data["answer_status"] in ("grounded", "insufficient")

    result = provider.generate_json("sys", "usr", validate, max_retries=2)
    assert result["answer_status"] == "grounded"
    assert call_count["n"] == 2  # first attempt failed, second succeeded


def test_generate_json_raises_structured_generation_failure_after_exhausting_retries():
    from rag.llm_providers import OllamaProvider

    class _AlwaysBrokenProvider(OllamaProvider):
        def generate(self, system: str, user: str) -> str:
            return "still not json"

    provider = _AlwaysBrokenProvider(model="fake")

    def validate(data):
        pass

    with pytest.raises(StructuredGenerationFailure):
        provider.generate_json("sys", "usr", validate, max_retries=2)


def test_embedded_multiple_json_objects_selects_last_schema_valid_object():
    raw = (
        '{"analysis": "intermediate"}\n'
        '{"faithfulness": 2, "correctness": 2, "relevance": 2, "reason": "grounded"}'
    )

    def validate(data):
        assert data.get("faithfulness") in (0, 1, 2)
        assert data.get("correctness") in (0, 1, 2)
        assert data.get("relevance") in (0, 1, 2)
        assert isinstance(data.get("reason"), str)

    assert _decode_last_valid_json_object(raw, validate) == {
        "faithfulness": 2,
        "correctness": 2,
        "relevance": 2,
        "reason": "grounded",
    }


def test_multiple_schema_valid_objects_selects_final_answer():
    raw = (
        '{"faithfulness": 0, "correctness": 0, "relevance": 0, "reason": "draft"}\n'
        '{"faithfulness": 2, "correctness": 2, "relevance": 2, "reason": "final"}'
    )

    def validate(data):
        for key in ("faithfulness", "correctness", "relevance"):
            if data.get(key) not in (0, 1, 2):
                raise ValueError(key)
        if not isinstance(data.get("reason"), str):
            raise ValueError("reason")

    assert _decode_last_valid_json_object(raw, validate)["reason"] == "final"


def test_structured_generation_failure_triggers_deterministic_extractive_fallback():
    provider = StubStructuredProvider(raise_failure=True)
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")

    chunk_texts = {"c1": "Hagia Sophia was built in 532 AD. It remained a cathedral for centuries."}
    source_meta = {"s1": {"title": "Hagia Sophia", "url": "https://en.wikipedia.org/wiki/Hagia_Sophia", "language": "en"}}

    result = svc.answer("When was Hagia Sophia built?", "en", chunk_texts, source_meta)

    assert result.used_fallback is True
    assert result.degraded is True
    assert result.answer_text.startswith("Hagia Sophia was built in 532 AD")
    assert result.citations == [
        {
            "schema_version": "1.0.0", "source_id": "s1", "title": "Hagia Sophia",
            "url": "https://en.wikipedia.org/wiki/Hagia_Sophia", "retrieved_at": None,
            "chunk_ids": ["c1"], "language": "en",
        }
    ]


@dataclass
class _TransportFailingProvider:
    """Simulates a generator whose provider fails at the transport layer
    (connection reset, timeout, TLS handshake failure) after its own
    bounded retry budget -- distinct from StubStructuredProvider's
    raise_failure (which simulates a JSON-quality failure)."""

    name: str = "qwen"
    model: str = "test-model"

    def generate_json(self, system, user, validate, max_retries=2):
        raise ProviderTransportError(self.name, 3)


def test_generator_transport_failure_produces_a_safe_degraded_result_not_a_crash():
    """Requirement: a generator transport failure after bounded retries
    must produce a safe degraded/provider-failed result via the existing
    GroundedAnswerService degradation model -- never an uncaught
    exception, never a stack trace leaking into the public result."""
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(
        retrieval=retrieval, provider=_TransportFailingProvider(), chunk_config="A", top_k=3, retrieval_mode="dense"
    )

    chunk_texts = {"c1": "Hagia Sophia was built in 532 AD. It remained a cathedral for centuries."}
    source_meta = {"s1": {"title": "Hagia Sophia", "url": "https://en.wikipedia.org/wiki/Hagia_Sophia", "language": "en"}}

    result = svc.answer("When was Hagia Sophia built?", "en", chunk_texts, source_meta)

    # Real evidence existed, so the deterministic extractive fallback
    # succeeds -- but it is honestly labeled as a provider failure, not
    # an ordinary structured-generation failure.
    assert result.used_fallback is True
    assert result.degraded is True
    assert result.degradation_reason is not None
    assert result.degradation_reason.startswith("provider_transport_failed")
    assert result.answer_text.startswith("Hagia Sophia was built in 532 AD")
    # Never a stack trace or the raw exception's internals in the public result.
    assert "Traceback" not in result.answer_text
    assert "ProviderTransportError" not in result.answer_text


def test_generator_transport_failure_with_no_extractable_evidence_refuses_safely():
    """When even the fallback finds no safe verbatim evidence, the
    result must still be the canonical refusal -- degraded and clearly
    tagged as a provider failure, never a fabricated answer."""
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(
        retrieval=retrieval, provider=_TransportFailingProvider(), chunk_config="A", top_k=3, retrieval_mode="dense"
    )

    result = svc.answer(
        "Completely unrelated question about something else entirely",
        "en",
        {"c1": "Zebras have black and white stripes."},
        {"s1": {"title": "T", "url": "https://x", "language": "en"}},
    )

    assert result.refused is True
    assert result.degraded is True
    assert result.degradation_reason is not None
    assert result.degradation_reason.startswith("provider_transport_failed")


def test_extractive_fallback_with_no_retrieved_items_refuses():
    from rag.answer_service import REFUSAL_TEXT

    provider = StubStructuredProvider(raise_failure=True)
    retrieval = _StubRetrieval(items=[])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")
    # zero_result path triggers before the provider is ever consulted when items=[] AND search() reports zero_result;
    # our stub retrieval always reports zero_result=False, so this exercises the extractive-fallback "no items" branch.
    result = svc._extractive_fallback("Any question", [], {}, {})
    assert result.answer_text == REFUSAL_TEXT
    assert result.refused is True


def test_model_insufficient_with_direct_turkish_evidence_uses_grounded_extractive_fallback():
    provider = StubStructuredProvider(responses=[{"answer_status": "insufficient", "claims": []}])
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")

    result = svc.answer(
        "Ayasofya ne zaman inşa edildi?",
        "tr",
        {"c1": "Ayasofya, 532-537 yılları arasında inşa edilmiştir."},
        {"s1": {"title": "Ayasofya", "url": "https://example.test", "language": "tr"}},
    )

    assert result.refused is False
    assert result.used_fallback is True
    assert result.answer_text == "Ayasofya, 532-537 yılları arasında inşa edilmiştir."
    assert result.citations[0]["chunk_ids"] == ["c1"]


def test_live_data_gate_cannot_be_bypassed_even_with_relevant_looking_retrieved_evidence():
    """Regression for the live gt_09 false-answer bug: a corpus chunk that
    superficially overlaps a live-data question (mentions the same POI,
    "ticket price", "admission fee", etc) must never let the extractive
    fallback answer a price/availability/weather/hours/accessibility
    question. The live-data gate (rag.answer_service._requires_live_data)
    runs before retrieval and is unconditional -- fallback logic (and the
    provider) is never even reached, regardless of what the corpus
    contains."""
    provider = StubStructuredProvider(raise_failure=True)
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")

    chunk_texts = {
        "c1": "Hagia Sophia's current ticket price and admission fee are managed by the museum authority.",
    }
    source_meta = {"s1": {"title": "Hagia Sophia", "url": "https://example.test", "language": "en"}}

    result = svc.answer("What is the current ticket price for Hagia Sophia?", "en", chunk_texts, source_meta)

    assert result.refused is True
    assert result.degradation_reason == "live_data_required"
    assert result.used_fallback is False
    assert result.citations == []
    assert provider.calls == []


def test_extractive_fallback_never_translates_only_extracts_verbatim_source_language_text():
    """The insufficient-override fallback must copy source text verbatim
    in whatever language it is actually written in -- it must never
    synthesize a translation into the question's language, even when a
    strong cross-lingual named-anchor match (e.g. "İstanbulkart") makes
    the chunk clearly relevant."""
    provider = StubStructuredProvider(responses=[{"answer_status": "insufficient", "claims": []}])
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")

    result = svc.answer(
        "İstanbulkart nedir ve hangi ulaşımlarda geçerlidir?",
        "tr",
        {"c1": "Istanbulkart is a contactless smart card used across metro, tram, bus, and ferry."},
        {"s1": {"title": "Transport", "url": "https://example.test", "language": "en"}},
    )

    assert result.used_fallback is True
    assert result.answer_text == "Istanbulkart is a contactless smart card used across metro, tram, bus, and ferry."
    # The English source text is returned verbatim -- never rewritten into Turkish.
    assert "nedir" not in result.answer_text
    assert "ulaşım" not in result.answer_text


def test_cross_lingual_named_anchor_can_use_verbatim_grounded_fallback():
    provider = StubStructuredProvider(responses=[{"answer_status": "insufficient", "claims": []}])
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    svc = GroundedAnswerService(retrieval=retrieval, provider=provider, chunk_config="A", top_k=3, retrieval_mode="dense")

    result = svc.answer(
        "İstanbulkart nedir?",
        "tr",
        {"c1": "Istanbulkart is a contactless smart card used across all transit modes."},
        {"s1": {"title": "Transport", "url": "https://example.test", "language": "en"}},
    )

    assert result.refused is False
    assert result.used_fallback is True
    assert result.citations[0]["chunk_ids"] == ["c1"]
