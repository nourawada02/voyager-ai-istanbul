"""Web search: an independent evidence-retrieval typed tool/provider
(Checkpoint Phase 4 C.0). A web result is evidence, not automatically a
trusted fact -- results are never silently converted into frozen RAG
citations (contracts/SourceReference.schema.json,
docs/adr/0005-phase3-rag-ownership.md). Never implemented inside System A
as tightly-coupled code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from providers.fingerprint import deterministic_request_id, fingerprint_request
from providers.policy import completeness_for_status, data_mode_for_status

CAPABILITY = "web_search"
SCHEMA_VERSION = "1.1.0"
RESULT_SCHEMA_VERSION = "1.0.0"

_FIXED_TEST_CLOCK = "2026-08-01T12:00:00Z"


def normalize_query(query: str) -> str:
    return " ".join(query.split()).lower()


@dataclass(frozen=True)
class WebEvidenceQuery:
    query: str
    language_hint: str = "en"

    def normalized(self) -> dict:
        return {"normalized_query": normalize_query(self.query), "language_hint": self.language_hint}

    def fingerprint(self) -> str:
        return fingerprint_request(CAPABILITY, self.normalized())


class WebEvidenceProvider(Protocol):
    """The interface a real adapter (e.g. Tavily, see
    docs/adr/0009-...md §7) and FakeWebEvidenceProvider both satisfy.
    Always returns a ProviderResponseEnvelope-shaped dict
    (contracts/ProviderResponseEnvelope.schema.json +
    contracts/WebEvidenceResult.schema.json)."""

    def search(self, query: WebEvidenceQuery) -> dict: ...


def _base_envelope(
    query: WebEvidenceQuery, provider_name: str, status: str, data_mode: str, retrieved_at: str = _FIXED_TEST_CLOCK
) -> dict:
    fingerprint = query.fingerprint()
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": deterministic_request_id(fingerprint, retrieved_at),
        "provider": provider_name,
        "capability": CAPABILITY,
        "data_mode": data_mode,
        "status": status,
        "query_fingerprint": fingerprint,
        "retrieved_at": retrieved_at,
        "source_urls": [],
        "quality": {
            "schema_version": "1.0.0",
            "completeness": completeness_for_status(status),
            "freshness": data_mode,
            "assumptions": [],
        },
        "result": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "original_query": query.query,
            "normalized_query": normalize_query(query.query),
            "items": [],
        },
    }


def build_success_envelope(
    query: WebEvidenceQuery, provider_name: str = "fake-web-search-provider", retrieved_at: str = _FIXED_TEST_CLOCK
) -> dict:
    envelope = _base_envelope(query, provider_name, status="success", data_mode="live", retrieved_at=retrieved_at)
    envelope["source_urls"] = ["https://example-synthetic-source.voyagerai.dev/fake-result"]
    envelope["result"]["items"] = [
        {
            "title": f"Result for: {query.query}",
            "canonical_url": "https://example-synthetic-source.voyagerai.dev/fake-result",
            "publisher": "Example Synthetic Publisher",
            "published_at": None,
            "retrieved_at": retrieved_at,
            "snippet": "Deterministic fake evidence snippet -- never a real fetched fact.",
            "language": query.language_hint,
            "rank": 1,
            "freshness": "live",
            "source_type": "secondary",
            "provenance": {
                "schema_version": "1.0.0",
                "provider": provider_name,
                "data_mode": "live",
                "retrieved_at": retrieved_at,
                "source_urls": ["https://example-synthetic-source.voyagerai.dev/fake-result"],
            },
        }
    ]
    return envelope


def build_degraded_envelope(
    query: WebEvidenceQuery, status: str, provider_name: str = "fake-web-search-provider",
    retrieved_at: str = _FIXED_TEST_CLOCK,
) -> dict:
    """Never returns fabricated evidence on a degraded path -- 'items'
    stays empty, and absent RAG knowledge that justified this web-search
    action in the first place remains genuinely unanswered, not silently
    backfilled. data_mode is derived honestly from status (Checkpoint
    Phase 4 C.0 correction pass) -- never hardcoded to 'estimated' when
    no evidence and no estimate exist."""
    data_mode = data_mode_for_status(status)
    return _base_envelope(query, provider_name, status=status, data_mode=data_mode, retrieved_at=retrieved_at)


@dataclass
class FakeWebEvidenceProvider:
    fixed_status: str = "success"
    provider_name: str = "fake-web-search-provider"
    clock: Callable[[], str] = field(default=lambda: _FIXED_TEST_CLOCK)
    call_log: list[WebEvidenceQuery] = field(default_factory=list)

    def search(self, query: WebEvidenceQuery) -> dict:
        self.call_log.append(query)
        now = self.clock()
        if self.fixed_status == "success":
            return build_success_envelope(query, self.provider_name, retrieved_at=now)
        return build_degraded_envelope(query, self.fixed_status, self.provider_name, retrieved_at=now)
