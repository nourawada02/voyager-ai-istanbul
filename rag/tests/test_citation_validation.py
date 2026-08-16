"""Tests that structured claims are validated against actually-retrieved
chunk_ids, against a verbatim evidence_quote copied from the cited chunk,
and (for same-language claims) against real lexical evidence in the cited
chunk text -- a model asserting a chunk id it was never given, inventing
or paraphrasing an evidence_quote, or citing a real chunk whose text does
not actually support the claim, must never be trusted at face value."""

from __future__ import annotations

from dataclasses import dataclass

from rag.answer_service import GroundedAnswerService
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


def _service(response: dict | list[dict]) -> GroundedAnswerService:
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}])
    responses = response if isinstance(response, list) else [response]
    return GroundedAnswerService(
        retrieval=retrieval, provider=StubStructuredProvider(responses=responses),
        chunk_config="A", top_k=3, retrieval_mode="dense",
    )


_CHUNK_TEXTS = {"c1": "Hagia Sophia was built in 532 AD by Emperor Justinian the First."}
_SOURCE_META = {"s1": {"title": "Hagia Sophia", "url": "https://en.wikipedia.org/wiki/Hagia_Sophia", "language": "en"}}


_C1_TEXT = "Hagia Sophia was built in 532 AD by Emperor Justinian the First."


def test_citation_to_a_retrieved_and_evidence_supported_chunk_is_accepted():
    svc = _service({
        "answer_status": "grounded",
        "claims": [{"text": "Hagia Sophia was built in 532 AD.", "evidence_quote": _C1_TEXT, "chunk_ids": ["c1"]}],
    })
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    assert len(result.citations) == 1
    assert result.citations[0]["source_id"] == "s1"
    assert result.citations[0]["chunk_ids"] == ["c1"]
    assert result.dropped_claim_count == 0


def test_citation_to_a_never_retrieved_chunk_is_stripped():
    svc = _service({
        "answer_status": "grounded",
        "claims": [{"text": "Hagia Sophia was built in 532 AD.", "evidence_quote": _C1_TEXT, "chunk_ids": ["c999_never_retrieved"]}],
    })
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    assert result.citations == []
    assert result.refused is True  # zero valid claims remain


def test_claim_unsupported_by_cited_chunk_text_is_rejected():
    """The chunk_id is real, retrieved, and the evidence_quote is a real
    verbatim quotation from it -- but the claim text itself has no real
    lexical relationship to that chunk's content. Because the claim and
    the cited chunk share the question's language ("en"), the same-
    language lexical check must still catch this."""
    svc = _service({
        "answer_status": "grounded",
        "claims": [{
            "text": "The Grand Bazaar has four thousand shops selling carpets.",
            "evidence_quote": _C1_TEXT,
            "chunk_ids": ["c1"],
        }],
    })
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    assert result.citations == []
    assert result.refused is True


def test_claim_with_fabricated_non_verbatim_evidence_quote_is_rejected():
    """The chunk_id is real and the claim text is even lexically
    plausible, but evidence_quote is not an actual substring of the
    cited chunk (paraphrased/invented) -- the verbatim check must catch
    this independently of the lexical check."""
    svc = _service({
        "answer_status": "grounded",
        "claims": [{
            "text": "Hagia Sophia was built in 532 AD.",
            "evidence_quote": "Construction of Hagia Sophia started in the year 532.",
            "chunk_ids": ["c1"],
        }],
    })
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    assert result.citations == []
    assert result.refused is True


def test_evidence_quote_tolerates_only_whitespace_differences():
    svc = _service({
        "answer_status": "grounded",
        "claims": [{
            "text": "Hagia Sophia was built in 532 AD.",
            "evidence_quote": "  Hagia Sophia   was built\nin 532 AD by Emperor Justinian the First.  ",
            "chunk_ids": ["c1"],
        }],
    })
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    assert len(result.citations) == 1
    assert result.dropped_claim_count == 0


def test_cross_language_claim_is_accepted_via_evidence_quote_without_lexical_overlap():
    """The core repair-3 case: a Turkish claim citing an English-language
    chunk. The claim text shares essentially no tokens with the English
    source -- lexical overlap is structurally impossible here -- so
    acceptance must rely solely on the verbatim evidence_quote (copied
    from the English chunk) plus a real, retrieved chunk_id."""
    chunk_texts = {"c1": "Hagia Sophia was built between 532 and 537 AD by Emperor Justinian the First."}
    source_meta = {"s1": {"title": "Hagia Sophia", "url": "https://en.wikipedia.org/wiki/Hagia_Sophia", "language": "en"}}
    svc = _service({
        "answer_status": "grounded",
        "claims": [{
            "text": "Ayasofya 532 ile 537 yılları arasında İmparator I. Justinianus tarafından inşa edildi.",
            "evidence_quote": "Hagia Sophia was built between 532 and 537 AD by Emperor Justinian the First.",
            "chunk_ids": ["c1"],
        }],
    })
    result = svc.answer("Ayasofya ne zaman inşa edildi?", "tr", chunk_texts, source_meta)
    assert result.refused is False
    assert result.dropped_claim_count == 0
    assert result.citations[0]["chunk_ids"] == ["c1"]
    assert "İmparator" in result.answer_text or "Justinianus" in result.answer_text


def test_cross_language_claim_with_non_verbatim_evidence_quote_is_still_rejected():
    """Cross-language claims skip the lexical check, but never the
    verbatim evidence_quote check -- a translated claim cannot use its
    own untranslatable text as an excuse to skip evidence entirely."""
    chunk_texts = {"c1": "Hagia Sophia was built between 532 and 537 AD by Emperor Justinian the First."}
    source_meta = {"s1": {"title": "Hagia Sophia", "url": "https://en.wikipedia.org/wiki/Hagia_Sophia", "language": "en"}}
    svc = _service({
        "answer_status": "grounded",
        "claims": [{
            "text": "Ayasofya 532 ile 537 yılları arasında İmparator I. Justinianus tarafından inşa edildi.",
            "evidence_quote": "Bu bilgi kesinlikle doğrudur, güvenebilirsiniz.",  # invented, not from the chunk
            "chunk_ids": ["c1"],
        }],
    })
    result = svc.answer("Ayasofya ne zaman inşa edildi?", "tr", chunk_texts, source_meta)
    assert result.refused is True
    assert result.citations == []


def test_mixed_valid_and_invalid_claims_are_repaired_before_acceptance():
    svc = _service([
        {
            "answer_status": "grounded",
            "claims": [
                {"text": "Hagia Sophia was built in 532 AD by Emperor Justinian.", "evidence_quote": _C1_TEXT, "chunk_ids": ["c1"]},
                {"text": "Completely unrelated claim about tea culture.", "evidence_quote": _C1_TEXT, "chunk_ids": ["c1"]},
                {"text": "Another fact.", "evidence_quote": _C1_TEXT, "chunk_ids": ["fake_chunk_id"]},
            ],
        },
        {
            "answer_status": "grounded",
            "claims": [
                {"text": "Hagia Sophia was built in 532 AD by Emperor Justinian.", "evidence_quote": _C1_TEXT, "chunk_ids": ["c1"]},
            ],
        },
    ])
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    assert len(result.citations) == 1
    assert result.citations[0]["chunk_ids"] == ["c1"]
    assert result.dropped_claim_count == 0
    assert result.total_claim_count == 1
    assert result.refused is False


def test_citation_preserves_required_source_reference_fields():
    svc = _service({
        "answer_status": "grounded",
        "claims": [{"text": "Hagia Sophia was built in 532 AD by Justinian.", "evidence_quote": _C1_TEXT, "chunk_ids": ["c1"]}],
    })
    result = svc.answer("q", "en", _CHUNK_TEXTS, _SOURCE_META)
    citation = result.citations[0]
    for field in ("source_id", "title", "url", "chunk_ids", "language"):
        assert field in citation
    # The internal evidence_quote field is never exposed as public citation content.
    assert "evidence_quote" not in citation


def test_unknown_source_id_is_repaired_to_exact_chunk_id_before_acceptance():
    """Regression for the live gt_04_en failure: a model can copy the
    human-readable source_id instead of the opaque chunk_id. That response
    must be repaired inside the bounded loop, not reduced to an empty
    false refusal."""
    provider = StubStructuredProvider(
        responses=[
            {
                "answer_status": "grounded",
                "claims": [{
                    "text": "Beyoglu was historically known as Pera.",
                    "evidence_quote": "Beyoglu was historically known as Pera.",
                    "chunk_ids": ["wiki_en_beyoglu"],
                }],
            },
            {
                "answer_status": "grounded",
                "claims": [{
                    "text": "Beyoglu was historically known as Pera.",
                    "evidence_quote": "Beyoglu was historically known as Pera.",
                    "chunk_ids": ["c1"],
                }],
            },
        ]
    )
    retrieval = _StubRetrieval(items=[{"rank": 1, "chunk_id": "c1", "source_id": "wiki_en_beyoglu", "score": 0.9}])
    service = GroundedAnswerService(
        retrieval=retrieval, provider=provider, chunk_config="B", top_k=5, retrieval_mode="dense"
    )

    result = service.answer(
        "What was Beyoglu historically known as?",
        "en",
        {"c1": "Beyoglu was historically known as Pera."},
        {"wiki_en_beyoglu": {"title": "Beyoglu", "url": "https://example.invalid", "language": "en"}},
    )

    assert len(provider.calls) == 2
    assert result.refused is False
    assert result.answer_text == "Beyoglu was historically known as Pera."
    assert result.citations[0]["chunk_ids"] == ["c1"]
    assert result.dropped_claim_count == 0
