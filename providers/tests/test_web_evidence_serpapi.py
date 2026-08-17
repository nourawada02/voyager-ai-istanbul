"""Hermetic tests for the real SerpApi web-evidence adapter (Checkpoint
Phase 4 C.2). No real network call anywhere in this file --
FakeHttpTransport never opens a socket.

Zero-secret-literal policy: every credential value used below is
generated fresh at test-run time via `_ephemeral_test_secret()` -- never
a fixed key-shaped string literal committed to this file. Only field
names and `Authorization`/`api_key` *keywords* appear as literals, which
is required to exercise redaction/secret-isolation behavior at all.
"""

from __future__ import annotations

import json
import secrets

import pytest

from providers.http_transport import FakeHttpTransport, HttpResponse, TransportError
from providers.redaction import SecretString
from providers.tests.conftest import make_validator
from providers.web_evidence import WebEvidenceQuery
from providers.web_evidence_serpapi import (
    SEARCH_URL,
    _IO_TIMEOUT_SECONDS,
    _MAX_BACKOFF_SECONDS,
    _MAX_RETRIES,
    SerpApiWebEvidenceProvider,
)

FIXED_NOW = "2026-08-17T12:00:00Z"


def _ephemeral_test_secret() -> str:
    """A fresh, random, per-run test value -- never a fixed key-shaped
    literal committed to the repository (zero-secret-literal policy)."""
    return f"ephemeral-test-token-{secrets.token_hex(16)}"


def _clock() -> str:
    return FIXED_NOW


def _provider(transport: FakeHttpTransport, **kwargs) -> SerpApiWebEvidenceProvider:
    kwargs.setdefault("api_key", SecretString(_ephemeral_test_secret()))
    return SerpApiWebEvidenceProvider(transport=transport, clock=_clock, sleep_fn=lambda seconds: None, **kwargs)


def _json_response(body: dict, status: int = 200, headers: dict | None = None) -> HttpResponse:
    return HttpResponse(status_code=status, body=json.dumps(body).encode("utf-8"), headers=headers or {})


SERPAPI_SUCCESS = {
    "search_metadata": {"id": "abc123", "status": "Success"},
    "search_parameters": {"engine": "google", "q": "Hagia Sophia visiting hours"},
    "organic_results": [
        {
            "position": 1,
            "title": "Hagia Sophia - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Hagia_Sophia",
            "snippet": "Hagia Sophia is a mosque and former church in Istanbul.",
            "date": "2026-01-15",
        },
        {
            "position": 2,
            "title": "Official museum information",
            "link": "https://muze.gov.tr/hagia-sophia",
            "snippet": "Official visiting hours and ticket information.",
        },
        {
            "position": 3,
            "title": "10 tips for visiting Hagia Sophia",
            "link": "https://example-travel-blog.com/hagia-sophia-guide",
            "snippet": "Some tips from a travel blogger who visited recently.",
            "date": "3 days ago",
        },
    ],
}

SERPAPI_EMPTY = {"search_metadata": {"id": "empty1", "status": "Success"}, "organic_results": []}


def _validate_full(envelope: dict, registry) -> None:
    envelope_validator = make_validator("ProviderResponseEnvelope", registry)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("WebEvidenceResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]


# --- secret handling --------------------------------------------------------------


def test_missing_key_is_unavailable_and_never_calls_transport(monkeypatch):
    # __post_init__ falls back to the real SERPAPI_API_KEY environment
    # variable whenever api_key=None -- it cannot distinguish "explicitly
    # no key" from "not provided" -- so a genuinely key-less environment
    # must be simulated explicitly here, regardless of what is actually
    # set in the environment this test happens to run in.
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = SerpApiWebEvidenceProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None, api_key=None)
    envelope = provider.search(WebEvidenceQuery(query="Hagia Sophia visiting hours"))
    assert envelope["status"] == "unavailable"
    assert envelope["data_mode"] == "unavailable"
    assert transport.call_log == []


def test_key_read_from_environment_when_not_injected(monkeypatch):
    secret_value = _ephemeral_test_secret()
    monkeypatch.setenv("SERPAPI_API_KEY", secret_value)
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = SerpApiWebEvidenceProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None)
    assert provider.api_key is not None
    assert provider.api_key.reveal() == secret_value


def test_key_sent_via_secret_params_never_public_params():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="Hagia Sophia visiting hours"))
    assert "api_key" not in transport.call_log[0][1]
    assert transport.secret_param_keys_log[0] == frozenset({"api_key"})


def test_api_key_never_appears_in_envelope_output():
    secret_value = _ephemeral_test_secret()
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport, api_key=SecretString(secret_value))
    envelope = provider.search(WebEvidenceQuery(query="Hagia Sophia visiting hours"))
    assert secret_value not in json.dumps(envelope)


def test_api_key_never_appears_in_transport_logs():
    secret_value = _ephemeral_test_secret()
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport, api_key=SecretString(secret_value))
    provider.search(WebEvidenceQuery(query="Hagia Sophia visiting hours"))
    assert secret_value not in str(transport.call_log)
    assert secret_value not in str(transport.secret_param_keys_log)


def test_api_key_never_appears_in_query_fingerprint():
    transport_a = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    transport_b = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider_a = _provider(transport_a, api_key=SecretString(_ephemeral_test_secret()))
    provider_b = _provider(transport_b, api_key=SecretString(_ephemeral_test_secret()))
    query = WebEvidenceQuery(query="Hagia Sophia visiting hours")
    envelope_a = provider_a.search(query)
    envelope_b = provider_b.search(query)
    assert envelope_a["query_fingerprint"] == envelope_b["query_fingerprint"]


def test_api_key_never_leaks_via_str_or_repr():
    secret_value = _ephemeral_test_secret()
    secret = SecretString(secret_value)
    assert secret_value not in str(secret)
    assert secret_value not in repr(secret)


# --- success / schema validity -----------------------------------------------------


def test_success_is_schema_valid(registry):
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="Hagia Sophia visiting hours"))
    assert envelope["status"] == "success"
    assert envelope["provider"] == "serpapi"
    _validate_full(envelope, registry)


def test_no_raw_html_or_answer_requested():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="test"))
    params = transport.call_log[0][1]
    assert params["output"] == "json"
    assert params["engine"] == "google"
    assert params["safe"] == "active"


def test_empty_result_is_schema_valid_and_honest(registry):
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_EMPTY)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="an extremely obscure query"))
    assert envelope["status"] == "success"
    assert envelope["result"]["items"] == []
    _validate_full(envelope, registry)


# --- result normalization -----------------------------------------------------------


def test_malformed_result_entries_are_skipped_not_fatal(registry):
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            "not-an-object",
            {"title": "missing link"},
            {"link": "https://example.com/a", "title": "ok", "snippet": "fine"},
        ],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "success"
    assert len(envelope["result"]["items"]) == 1
    assert any("malformed" in a for a in envelope["quality"]["assumptions"])
    assert any("missing" in a for a in envelope["quality"]["assumptions"])
    _validate_full(envelope, registry)


def test_invalid_and_credential_bearing_urls_are_rejected():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {"link": "not a url", "title": "bad", "snippet": "bad"},
            {"link": "ftp://example.com/file", "title": "bad scheme", "snippet": "bad"},
            {"link": "https://example.com/good", "title": "good", "snippet": "good"},
        ],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert len(envelope["result"]["items"]) == 1
    assert envelope["result"]["items"][0]["canonical_url"] == "https://example.com/good"


def test_result_with_credential_bearing_url_is_skipped():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {"link": "https://user:pass@example.com/secret", "title": "bad", "snippet": "bad"},
        ],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["result"]["items"] == []


def test_duplicate_canonical_urls_are_deduplicated_keeping_first():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {"link": "https://example.com/a#section1", "title": "first", "snippet": "first"},
            {"link": "https://example.com/a#section2", "title": "second", "snippet": "second"},
        ],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert len(envelope["result"]["items"]) == 1
    assert envelope["result"]["items"][0]["title"] == "first"


def test_result_order_is_deterministic_matches_serpapi_ranking():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    ranks = [item["rank"] for item in envelope["result"]["items"]]
    assert ranks == [1, 2, 3]
    assert envelope["result"]["items"][0]["canonical_url"] == "https://en.wikipedia.org/wiki/Hagia_Sophia"


def test_deterministic_source_classification():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    by_url = {item["canonical_url"]: item["source_type"] for item in envelope["result"]["items"]}
    assert by_url["https://muze.gov.tr/hagia-sophia"] == "official"
    assert by_url["https://en.wikipedia.org/wiki/Hagia_Sophia"] == "reference"
    assert by_url["https://example-travel-blog.com/hagia-sophia-guide"] == "secondary"


def test_title_claiming_official_does_not_influence_classification():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {"link": "https://random-unrecognized-domain.example/page", "title": "Official Istanbul Guide", "snippet": "s"},
        ],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["result"]["items"][0]["source_type"] == "secondary"


def test_unknown_publisher_falls_back_honestly():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [{"link": "https://totally-unrecognized-example-domain.example/x", "title": "t", "snippet": "s"}],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["result"]["items"][0]["source_type"] == "secondary"
    assert envelope["result"]["items"][0]["publisher"] == "totally-unrecognized-example-domain.example"


def test_over_length_title_and_snippet_are_truncated_with_warning():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [{"link": "https://example.com/a", "title": "x" * 300, "snippet": "y" * 900}],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    item = envelope["result"]["items"][0]
    assert len(item["title"]) <= 200
    assert len(item["snippet"]) <= 500
    assert any("truncated" in a for a in envelope["quality"]["assumptions"])


def test_result_count_is_hard_capped_regardless_of_provider_count():
    body = {
        "search_metadata": {"status": "Success"},
        "organic_results": [
            {"link": f"https://example.com/{i}", "title": f"t{i}", "snippet": f"s{i}"} for i in range(20)
        ],
    }
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test", max_results=20))
    assert len(envelope["result"]["items"]) == 8


def test_num_param_request_is_bounded_by_hard_cap():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="test", max_results=50))
    assert transport.call_log[0][1]["num"] == 8


def test_published_date_strict_iso_is_parsed_relative_date_is_not():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    by_url = {item["canonical_url"]: item["published_at"] for item in envelope["result"]["items"]}
    assert by_url["https://en.wikipedia.org/wiki/Hagia_Sophia"] == "2026-01-15T00:00:00Z"
    assert by_url["https://example-travel-blog.com/hagia-sophia-guide"] is None


# --- request validation --------------------------------------------------------------


def test_empty_query_is_invalid_request():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="   "))
    assert envelope["status"] == "invalid_request"
    assert transport.call_log == []


def test_negative_max_results_is_invalid_request():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test", max_results=0))
    assert envelope["status"] == "invalid_request"


# --- language / geographic parameters -------------------------------------------------


def test_supported_language_hint_is_honored_in_request_params():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="test", language_hint="tr"))
    assert transport.call_log[0][1]["hl"] == "tr"


def test_unrecognized_language_hint_falls_back_to_en():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="test", language_hint="fr"))
    assert transport.call_log[0][1]["hl"] == "en"


def test_istanbul_turkey_geographic_context_is_always_sent():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="test"))
    params = transport.call_log[0][1]
    assert params["location"] == "Istanbul, Turkey"
    assert params["gl"] == "tr"


def test_no_cache_param_allows_serpapi_cache_use():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport)
    provider.search(WebEvidenceQuery(query="test"))
    assert transport.call_log[0][1]["no_cache"] == "false"


# --- failure handling ------------------------------------------------------------------


def test_transport_error_is_timeout():
    transport = FakeHttpTransport(raise_transport_error=True)
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "timeout"
    assert envelope["data_mode"] == "unavailable"


def test_rate_limited_429_honors_retry_after():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response({}, status=429, headers={"Retry-After": "2"})})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "rate_limited"


def test_top_level_error_quota_message_is_rate_limited():
    body = {"error": "You have run out of searches for this month."}
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "rate_limited"


def test_top_level_error_generic_message_is_provider_error():
    body = {"error": "Invalid API key."}
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "provider_error"


def test_search_metadata_status_error_is_provider_error():
    body = {"search_metadata": {"status": "Error"}, "organic_results": []}
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "provider_error"


@pytest.mark.parametrize("status_code", [401, 403])
def test_401_and_403_are_provider_error_never_raise(status_code):
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response({}, status=status_code)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))  # must not raise
    assert envelope["status"] == "provider_error"


def test_5xx_is_provider_error():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response({}, status=503)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "provider_error"


def test_malformed_json_response_is_provider_error_never_raises():
    transport = FakeHttpTransport(responses={SEARCH_URL: HttpResponse(status_code=200, body=b"not json{{{", headers={})})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "provider_error"


def test_response_missing_organic_results_key_is_provider_error():
    body = {"search_metadata": {"status": "Success"}}
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "provider_error"


def test_successful_retry_after_one_transient_failure():
    calls = {"n": 0}

    def _responder(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return _json_response({}, status=503)
        return _json_response(SERPAPI_SUCCESS)

    transport = FakeHttpTransport(responses={SEARCH_URL: _responder})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "success"
    assert calls["n"] == 2


def test_default_timeout_and_retry_policy_is_serpapi_tuned():
    """Checkpoint Phase 4 C.2 timeout-semantics repair: SerpApi's own
    defaults, not providers.policy's shared generic defaults (which were
    too tight for this environment's observed cold-connection latency
    against serpapi.com)."""
    provider = SerpApiWebEvidenceProvider(
        transport=FakeHttpTransport(), clock=_clock, sleep_fn=lambda s: None, api_key=SecretString(_ephemeral_test_secret())
    )
    assert provider.timeout_policy.total_timeout_seconds == 25.0
    assert provider.retry_policy.max_retries == 1


def test_worst_case_execution_time_stays_under_60_second_deadline():
    worst_case_seconds = (_MAX_RETRIES + 1) * _IO_TIMEOUT_SECONDS + _MAX_RETRIES * _MAX_BACKOFF_SECONDS
    assert worst_case_seconds < 60.0


def test_retries_are_bounded_to_at_most_one_retry_two_attempts_total():
    calls = {"n": 0}

    def _always_fail(params):
        calls["n"] += 1
        return _json_response({}, status=503)

    transport = FakeHttpTransport(responses={SEARCH_URL: _always_fail})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "provider_error"
    assert calls["n"] == 2  # max_retries=1 -> exactly 2 attempts total, never 3


def test_cancellation_check_short_circuits_before_any_call():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider = _provider(transport, cancellation_check=lambda: True)
    envelope = provider.search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "cancelled"
    assert transport.call_log == []


# --- determinism / infra guarantees -----------------------------------------------------


def test_identical_input_clock_and_config_produce_byte_identical_output():
    secret_value = _ephemeral_test_secret()
    transport_a = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    transport_b = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    provider_a = _provider(transport_a, api_key=SecretString(secret_value))
    provider_b = _provider(transport_b, api_key=SecretString(secret_value))
    query = WebEvidenceQuery(query="Hagia Sophia visiting hours")
    envelope_a = provider_a.search(query)
    envelope_b = provider_b.search(query)
    assert envelope_a == envelope_b
    assert envelope_a["request_id"] == envelope_b["request_id"]


def test_cache_status_is_honestly_bypass():
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["cache_status"] == "bypass"
    assert "cache_age_seconds" not in envelope


def test_no_real_socket_is_ever_opened(monkeypatch):
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
    assert envelope["status"] == "success"


def test_provider_never_returns_an_envelope_that_fails_mandatory_validation(registry, monkeypatch):
    scenarios = [
        (SERPAPI_SUCCESS, 200, None),
        (SERPAPI_EMPTY, 200, None),
        ({}, 503, None),
        ({}, 429, None),
        ({"error": "Invalid API key."}, 200, None),
    ]
    for body, status, _ in scenarios:
        transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body, status=status)})
        envelope = _provider(transport).search(WebEvidenceQuery(query="test"))
        _validate_full(envelope, registry)

    transport_cancelled = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope_cancelled = _provider(transport_cancelled, cancellation_check=lambda: True).search(
        WebEvidenceQuery(query="test")
    )
    _validate_full(envelope_cancelled, registry)

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    transport_no_key = FakeHttpTransport(responses={SEARCH_URL: _json_response(SERPAPI_SUCCESS)})
    envelope_no_key = SerpApiWebEvidenceProvider(
        transport=transport_no_key, clock=_clock, sleep_fn=lambda s: None, api_key=None
    ).search(WebEvidenceQuery(query="test"))
    _validate_full(envelope_no_key, registry)
