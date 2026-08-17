"""Hermetic tests for the web-evidence provider interface + fake
(Checkpoint Phase 4 C.0, Step 8). No real network call anywhere in this
file."""

from __future__ import annotations

import socket

import pytest

from providers.tests.conftest import make_validator
from providers.web_evidence import FakeWebEvidenceProvider, WebEvidenceQuery, build_degraded_envelope, build_success_envelope

QUERY = WebEvidenceQuery(query="Hagia Sophia current visiting hours")


def test_success_envelope_is_schema_valid(envelope_validator, registry):
    envelope = build_success_envelope(QUERY)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("WebEvidenceResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "unavailable", "provider_error", "cancelled"])
def test_degraded_envelope_is_schema_valid_for_every_failure_status(status, envelope_validator, registry):
    envelope = build_degraded_envelope(QUERY, status)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("WebEvidenceResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]
    assert envelope["result"]["items"] == []


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "unavailable", "provider_error", "cancelled"])
def test_no_data_failure_statuses_use_honest_unavailable_data_mode_never_estimated(status):
    envelope = build_degraded_envelope(QUERY, status)
    assert envelope["data_mode"] == "unavailable"
    assert envelope["data_mode"] != "estimated"
    assert envelope["quality"]["completeness"] == 0.0


def test_stale_status_reports_cached_data_mode():
    envelope = build_degraded_envelope(QUERY, "stale")
    assert envelope["data_mode"] == "cached"
    assert envelope["quality"]["completeness"] == 1.0


def test_provider_failure_is_never_represented_as_an_empty_successful_result():
    envelope = build_degraded_envelope(QUERY, "unavailable")
    assert envelope["status"] == "unavailable"
    assert envelope["result"]["items"] == []


def test_absent_evidence_never_silently_becomes_a_frozen_rag_citation():
    """Web evidence has its own schema (WebEvidenceResult), structurally
    distinct from contracts/SourceReference.schema.json (frozen RAG
    citations). A web evidence item's shape must never be mistaken for a
    SourceReference: SourceReference requires only source_id/retrieved_at
    (no snippet/source_type/rank), while a WebEvidenceItem requires
    snippet, source_type, and rank -- proving the two are not
    interchangeable by construction."""
    envelope = build_success_envelope(QUERY)
    item = envelope["result"]["items"][0]
    assert "snippet" in item
    assert "source_type" in item
    assert "rank" in item
    assert "source_id" not in item  # SourceReference's own required field, absent here on purpose


def test_source_type_distinguishes_official_from_secondary_and_unknown(registry):
    for source_type in ("official", "news", "reference", "secondary", "unknown"):
        envelope = build_success_envelope(QUERY)
        envelope["result"]["items"][0]["source_type"] = source_type
        result_validator = make_validator("WebEvidenceResult", registry)
        errors = list(result_validator.iter_errors(envelope["result"]))
        assert not errors, f"{source_type}: {[e.message for e in errors]}"


def test_language_field_supports_multilingual_results(registry):
    for language in ("en", "tr", "ar"):
        envelope = build_success_envelope(WebEvidenceQuery(query="test", language_hint=language))
        result_validator = make_validator("WebEvidenceResult", registry)
        errors = list(result_validator.iter_errors(envelope["result"]))
        assert not errors, f"{language}: {[e.message for e in errors]}"


def test_identical_input_clock_and_config_produce_byte_identical_output():
    fake_a = FakeWebEvidenceProvider(clock=lambda: "2026-08-01T12:00:00Z")
    fake_b = FakeWebEvidenceProvider(clock=lambda: "2026-08-01T12:00:00Z")
    envelope_a = fake_a.search(QUERY)
    envelope_b = fake_b.search(QUERY)
    assert envelope_a == envelope_b
    assert envelope_a["request_id"] == envelope_b["request_id"]


def test_fake_provider_never_opens_a_real_socket(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic fake-provider test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    fake = FakeWebEvidenceProvider()
    envelope = fake.search(QUERY)
    assert envelope["status"] == "success"


def test_fake_provider_tracks_call_log():
    fake = FakeWebEvidenceProvider()
    fake.search(QUERY)
    assert len(fake.call_log) == 1
    assert fake.call_log[0].query == QUERY.query
