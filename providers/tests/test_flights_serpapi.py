"""Hermetic tests for the real SerpApi Google Flights adapter (Checkpoint
Phase 4 C.3). No real network call anywhere in this file --
FakeHttpTransport never opens a socket.

Zero-secret-literal policy: every credential value used below is
generated fresh at test-run time via `_ephemeral_test_secret()` -- never
a fixed key-shaped string literal committed to this file.
"""

from __future__ import annotations

import json
import secrets

import pytest

from providers.flights import FlightSearchQuery
from providers.flights_serpapi import (
    FLIGHTS_URL,
    HARD_MAX_RESULTS,
    _IO_TIMEOUT_SECONDS,
    _MAX_BACKOFF_SECONDS,
    _MAX_RETRIES,
    SerpApiFlightSearchProvider,
)
from providers.http_transport import FakeHttpTransport, HttpResponse, TransportError
from providers.redaction import SecretString
from providers.tests.conftest import make_validator

FIXED_NOW = "2026-08-17T12:00:00Z"
FUTURE_DATE = "2026-09-10"


def _ephemeral_test_secret() -> str:
    return f"ephemeral-test-token-{secrets.token_hex(16)}"


def _clock() -> str:
    return FIXED_NOW


def _provider(transport: FakeHttpTransport, **kwargs) -> SerpApiFlightSearchProvider:
    kwargs.setdefault("api_key", SecretString(_ephemeral_test_secret()))
    return SerpApiFlightSearchProvider(transport=transport, clock=_clock, sleep_fn=lambda seconds: None, **kwargs)


def _json_response(body: dict, status: int = 200, headers: dict | None = None) -> HttpResponse:
    return HttpResponse(status_code=status, body=json.dumps(body).encode("utf-8"), headers=headers or {})


def _query(**overrides) -> FlightSearchQuery:
    fields = dict(
        origin="BEY", destination="IST", depart_date_from=FUTURE_DATE, depart_date_to=FUTURE_DATE, passenger_count=1
    )
    fields.update(overrides)
    return FlightSearchQuery(**fields)


DIRECT_ITINERARY = {
    "flights": [
        {
            "departure_airport": {"name": "Beirut", "id": "BEY", "time": "2026-09-10 08:30"},
            "arrival_airport": {"name": "Istanbul", "id": "IST", "time": "2026-09-10 10:15"},
            "airline": "Turkish Airlines",
            "flight_number": "TK 823",
        }
    ],
    "total_duration": 105,
    "price": 4500,
    "type": "One way",
    "booking_token": "should-never-appear-anywhere",
}

OTHER_ITINERARY = {
    "flights": [
        {
            "departure_airport": {"name": "Beirut", "id": "BEY", "time": "2026-09-10 14:00"},
            "arrival_airport": {"name": "Sabiha Gokcen", "id": "SAW", "time": "2026-09-10 16:10"},
            "airline": "Pegasus",
            "flight_number": "PC5623",
        }
    ],
    "total_duration": 130,
    "price": 3200,
}

CONNECTING_ITINERARY = {
    "flights": [
        {
            "departure_airport": {"id": "BEY", "time": "2026-09-10 06:00"},
            "arrival_airport": {"id": "IST", "time": "2026-09-10 08:00"},
            "airline": "Turkish Airlines",
            "flight_number": "TK100",
        },
        {
            "departure_airport": {"id": "IST", "time": "2026-09-10 10:00"},
            "arrival_airport": {"id": "SAW", "time": "2026-09-10 10:45"},
            "airline": "Turkish Airlines",
            "flight_number": "TK200",
        },
    ],
    "total_duration": 285,
    "price": 5000,
}

FLIGHTS_SUCCESS = {
    "search_metadata": {"id": "abc123", "status": "Success"},
    "best_flights": [DIRECT_ITINERARY],
    "other_flights": [OTHER_ITINERARY],
}

FLIGHTS_EMPTY = {"search_metadata": {"status": "Success"}, "best_flights": [], "other_flights": []}


def _validate_full(envelope: dict, registry) -> None:
    envelope_validator = make_validator("ProviderResponseEnvelope", registry)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("FlightSearchResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]
    for option in envelope["result"]["options"]:
        option_validator = make_validator("FlightOption", registry)
        option_errors = list(option_validator.iter_errors(option))
        assert not option_errors, [e.message for e in option_errors]


# --- request mapping -----------------------------------------------------------------


def test_request_uses_correct_endpoint_and_engine():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    url, params = transport.call_log[0]
    assert url == FLIGHTS_URL
    assert params["engine"] == "google_flights"


def test_exact_date_one_way_request_mapping():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    params = transport.call_log[0][1]
    assert params["outbound_date"] == FUTURE_DATE
    assert params["type"] == "2"


def test_origin_destination_mapping():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query(origin="bey", destination="ist"))
    params = transport.call_log[0][1]
    assert params["departure_id"] == "BEY"
    assert params["arrival_id"] == "IST"


def test_passenger_count_maps_to_adults():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query(passenger_count=3))
    assert transport.call_log[0][1]["adults"] == 3


def test_cabin_class_maps_to_travel_class():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query(cabin_class="business"))
    assert transport.call_log[0][1]["travel_class"] == 3


def test_no_cabin_class_omits_travel_class_param():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    assert "travel_class" not in transport.call_log[0][1]


def test_no_nonstop_param_since_query_has_no_such_field():
    """FlightSearchQuery does not define a nonstop-only preference field
    -- per the checkpoint's own instruction, a mapping is only added if
    such a field already exists. It does not, so no 'stops' request
    parameter is ever sent."""
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    assert "stops" not in transport.call_log[0][1]


def test_try_currency_is_always_used():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    envelope = provider.search_flights(_query())
    assert transport.call_log[0][1]["currency"] == "TRY"
    assert envelope["currency"] == "TRY"
    assert envelope["result"]["options"][0]["price"]["currency"] == "TRY"


def test_no_cache_param_allows_serpapi_cache_use():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    assert transport.call_log[0][1]["no_cache"] == "false"


def test_deep_search_is_disabled():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    assert transport.call_log[0][1]["deep_search"] == "false"


# --- secret handling -------------------------------------------------------------------


def test_missing_key_is_unavailable_and_never_calls_transport(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = SerpApiFlightSearchProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None, api_key=None)
    envelope = provider.search_flights(_query())
    assert envelope["status"] == "unavailable"
    assert envelope["data_mode"] == "unavailable"
    assert transport.call_log == []


def test_key_sent_via_secret_params_never_public_params():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport)
    provider.search_flights(_query())
    assert "api_key" not in transport.call_log[0][1]
    assert transport.secret_param_keys_log[0] == frozenset({"api_key"})


def test_api_key_never_appears_in_envelope_output():
    secret_value = _ephemeral_test_secret()
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport, api_key=SecretString(secret_value))
    envelope = provider.search_flights(_query())
    assert secret_value not in json.dumps(envelope)


def test_api_key_never_appears_in_transport_logs_or_errors():
    secret_value = _ephemeral_test_secret()
    transport = FakeHttpTransport(raise_transport_error=True)
    provider = _provider(transport, api_key=SecretString(secret_value))
    envelope = provider.search_flights(_query())
    assert secret_value not in str(transport.call_log)
    assert secret_value not in str(transport.secret_param_keys_log)
    assert secret_value not in json.dumps(envelope)


def test_api_key_never_appears_in_query_fingerprint():
    transport_a = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    transport_b = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider_a = _provider(transport_a, api_key=SecretString(_ephemeral_test_secret()))
    provider_b = _provider(transport_b, api_key=SecretString(_ephemeral_test_secret()))
    query = _query()
    envelope_a = provider_a.search_flights(query)
    envelope_b = provider_b.search_flights(query)
    assert envelope_a["query_fingerprint"] == envelope_b["query_fingerprint"]


def test_api_key_never_leaks_via_str_or_repr():
    secret_value = _ephemeral_test_secret()
    secret = SecretString(secret_value)
    assert secret_value not in str(secret)
    assert secret_value not in repr(secret)


# --- request validation ----------------------------------------------------------------


def test_invalid_iata_code_is_invalid_request():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(origin="BEYX"))
    assert envelope["status"] == "invalid_request"
    assert transport.call_log == []


def test_identical_origin_destination_is_invalid_request():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(origin="IST", destination="IST"))
    assert envelope["status"] == "invalid_request"


def test_date_range_is_unsupported():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(depart_date_to="2026-09-12"))
    assert envelope["status"] == "unsupported"
    assert transport.call_log == []


def test_malformed_date_is_invalid_request():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(depart_date_from="not-a-date", depart_date_to="not-a-date"))
    assert envelope["status"] == "invalid_request"


def test_past_date_is_unsupported():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(depart_date_from="2020-01-01", depart_date_to="2020-01-01"))
    assert envelope["status"] == "unsupported"


def test_round_trip_query_is_unsupported():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(
        _query(return_date_from="2026-09-15", return_date_to="2026-09-16")
    )
    assert envelope["status"] == "unsupported"
    assert transport.call_log == []


def test_zero_passenger_count_is_invalid_request():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(passenger_count=0))
    assert envelope["status"] == "invalid_request"


def test_unsupported_cabin_class_is_invalid_request():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(cabin_class="super_deluxe"))
    assert envelope["status"] == "invalid_request"


# --- normalization -----------------------------------------------------------------------


def test_success_is_schema_valid(registry):
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"
    assert envelope["provider"] == "serpapi_google_flights"
    _validate_full(envelope, registry)


def test_usd_currency_is_requested_natively_and_labels_every_price(registry):
    """Manual QA remediation Q.1 (§B): SerpApi is asked for the trip's
    own currency directly (never converted client-side after the fact),
    and every returned price -- envelope-level and per-option -- is
    labeled with that same currency, never left as a stale 'TRY'."""
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query(currency="USD"))
    assert envelope["status"] == "success"
    assert envelope["currency"] == "USD"
    assert envelope["result"]["options"]
    assert all(opt["price"]["currency"] == "USD" for opt in envelope["result"]["options"])
    _validate_full(envelope, registry)

    request_call = next(c for c in transport.call_log if c[0] == FLIGHTS_URL)
    assert request_call[1]["currency"] == "USD"


def test_try_currency_is_unchanged_by_default():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["currency"] == "TRY"
    assert all(opt["price"]["currency"] == "TRY" for opt in envelope["result"]["options"])


def test_best_flights_normalized_into_options():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    origins = {opt["carrier"] for opt in envelope["result"]["options"]}
    assert "Turkish Airlines" in origins


def test_missing_best_flights_falls_back_to_other_flights(registry):
    body = {"search_metadata": {"status": "Success"}, "other_flights": [OTHER_ITINERARY]}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"
    assert len(envelope["result"]["options"]) == 1
    assert envelope["result"]["options"][0]["carrier"] == "Pegasus"
    _validate_full(envelope, registry)


def test_direct_flight_has_zero_stops_no_legs():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    direct = next(o for o in envelope["result"]["options"] if o["carrier"] == "Turkish Airlines")
    assert direct["stops"] == 0
    assert "legs" not in direct


def test_connecting_flight_has_legs_and_correct_stops(registry):
    body = {"search_metadata": {"status": "Success"}, "best_flights": [CONNECTING_ITINERARY], "other_flights": []}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    option = envelope["result"]["options"][0]
    assert option["stops"] == 1
    assert len(option["legs"]) == 2
    assert option["origin"] == "BEY"
    assert option["destination"] == "SAW"
    _validate_full(envelope, registry)


def test_best_flights_ordered_before_other_flights():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    carriers = [opt["carrier"] for opt in envelope["result"]["options"]]
    assert carriers == ["Turkish Airlines", "Pegasus"]


def test_duplicate_itineraries_deduplicated():
    body = {
        "search_metadata": {"status": "Success"},
        "best_flights": [DIRECT_ITINERARY],
        "other_flights": [DIRECT_ITINERARY],
    }
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert len(envelope["result"]["options"]) == 1


def test_result_count_is_hard_capped_at_three():
    many = {
        "search_metadata": {"status": "Success"},
        "best_flights": [
            {
                "flights": [
                    {
                        "departure_airport": {"id": "BEY", "time": "2026-09-10 08:00"},
                        "arrival_airport": {"id": "IST", "time": "2026-09-10 10:00"},
                        "airline": "Carrier",
                        "flight_number": f"C{i}00",
                    }
                ],
                "price": 1000 + i,
            }
            for i in range(10)
        ],
        "other_flights": [],
    }
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(many)})
    envelope = _provider(transport).search_flights(_query())
    assert len(envelope["result"]["options"]) == HARD_MAX_RESULTS
    assert HARD_MAX_RESULTS <= 8


def test_price_converted_to_minor_units_exactly():
    body = {
        "search_metadata": {"status": "Success"},
        "best_flights": [{**DIRECT_ITINERARY, "price": 1234.5}],
        "other_flights": [],
    }
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["result"]["options"][0]["price"]["amount_minor_units"] == 123450


def test_flight_id_is_deterministic_and_stable():
    transport_a = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    transport_b = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope_a = _provider(transport_a).search_flights(_query())
    envelope_b = _provider(transport_b).search_flights(_query())
    id_a = envelope_a["result"]["options"][0]["flight_id"]
    id_b = envelope_b["result"]["options"][0]["flight_id"]
    assert id_a == id_b
    assert id_a.startswith("serpapi_google_flights_")


def test_bey_ist_saw_timestamps_are_offset_aware():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    for option in envelope["result"]["options"]:
        assert option["depart_at"][-6:].count(":") == 1 or option["depart_at"].endswith("Z")
        assert "+" in option["depart_at"] or "-" in option["depart_at"][10:]
        assert not option["depart_at"].endswith("+00:00Z")


def test_unknown_airport_timezone_itinerary_is_skipped():
    body = {
        "search_metadata": {"status": "Success"},
        "best_flights": [
            {
                "flights": [
                    {
                        "departure_airport": {"id": "BEY", "time": "2026-09-10 08:00"},
                        "arrival_airport": {"id": "XYZ", "time": "2026-09-10 09:00"},
                        "airline": "Test Air",
                        "flight_number": "TA100",
                    }
                ],
                "price": 1000,
            }
        ],
        "other_flights": [],
    }
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"
    assert envelope["result"]["options"] == []
    assert any("timezone" in w for w in envelope["result"]["limitations"])


def test_malformed_time_string_itinerary_is_skipped():
    body = {
        "search_metadata": {"status": "Success"},
        "best_flights": [
            {
                "flights": [
                    {
                        "departure_airport": {"id": "BEY", "time": "not-a-time"},
                        "arrival_airport": {"id": "IST", "time": "2026-09-10 09:00"},
                        "airline": "Test Air",
                    }
                ],
                "price": 1000,
            }
        ],
        "other_flights": [],
    }
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["result"]["options"] == []


def test_booking_token_never_appears_in_envelope():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    serialized = json.dumps(envelope)
    assert "should-never-appear-anywhere" not in serialized
    assert "booking_token" not in serialized
    assert "departure_token" not in serialized


def test_no_booking_or_availability_claim_in_envelope():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    forbidden_terms = ("booked", "reservation_confirmed", "payment", "purchase", "seat_available", "confirmed")
    serialized = str(envelope).lower()
    for term in forbidden_terms:
        assert term not in serialized, f"forbidden booking-adjacent term found: {term!r}"
    assert envelope["result"]["is_snapshot"] is True


def test_empty_results_is_success_with_no_options(registry):
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_EMPTY)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"
    assert envelope["result"]["options"] == []
    _validate_full(envelope, registry)


def test_malformed_itinerary_entries_are_skipped_not_fatal(registry):
    body = {
        "search_metadata": {"status": "Success"},
        "best_flights": ["not-an-object", {"flights": []}, {"flights": [{"missing": "fields"}]}, DIRECT_ITINERARY],
        "other_flights": [],
    }
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"
    assert len(envelope["result"]["options"]) == 1
    assert envelope["result"]["limitations"]
    _validate_full(envelope, registry)


# --- failure handling ------------------------------------------------------------------


def test_transport_error_is_timeout():
    transport = FakeHttpTransport(raise_transport_error=True)
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "timeout"
    assert envelope["data_mode"] == "unavailable"


def test_rate_limited_429_honors_retry_after():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response({}, status=429, headers={"Retry-After": "2"})})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "rate_limited"


def test_top_level_error_quota_message_is_rate_limited():
    body = {"error": "You have run out of searches for this month."}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "rate_limited"


def test_top_level_error_generic_message_is_provider_error():
    body = {"error": "Invalid API key."}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "provider_error"


def test_search_metadata_status_error_is_provider_error():
    body = {"search_metadata": {"status": "Error"}}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "provider_error"


@pytest.mark.parametrize("status_code", [401, 403])
def test_401_and_403_are_provider_error_never_raise(status_code):
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response({}, status=status_code)})
    envelope = _provider(transport).search_flights(_query())  # must not raise
    assert envelope["status"] == "provider_error"


def test_5xx_is_provider_error():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response({}, status=503)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "provider_error"


def test_malformed_json_response_is_provider_error_never_raises():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: HttpResponse(status_code=200, body=b"not json{{{", headers={})})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "provider_error"


def test_response_missing_both_flight_keys_is_provider_error():
    body = {"search_metadata": {"status": "Success"}}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "provider_error"


def test_successful_retry_after_one_transient_failure():
    calls = {"n": 0}

    def _responder(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return _json_response({}, status=503)
        return _json_response(FLIGHTS_SUCCESS)

    transport = FakeHttpTransport(responses={FLIGHTS_URL: _responder})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"
    assert calls["n"] == 2


def test_retries_are_bounded_to_at_most_one_retry_two_attempts_total():
    calls = {"n": 0}

    def _always_fail(params):
        calls["n"] += 1
        return _json_response({}, status=503)

    transport = FakeHttpTransport(responses={FLIGHTS_URL: _always_fail})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "provider_error"
    assert calls["n"] == 2


def test_cancellation_check_short_circuits_before_any_call():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider = _provider(transport, cancellation_check=lambda: True)
    envelope = provider.search_flights(_query())
    assert envelope["status"] == "cancelled"
    assert transport.call_log == []


# --- policy / determinism / infra guarantees --------------------------------------------


def test_default_timeout_and_retry_policy_is_serpapi_tuned():
    provider = SerpApiFlightSearchProvider(
        transport=FakeHttpTransport(), clock=_clock, sleep_fn=lambda s: None, api_key=SecretString(_ephemeral_test_secret())
    )
    assert provider.timeout_policy.total_timeout_seconds == 25.0
    assert provider.retry_policy.max_retries == 1


def test_worst_case_execution_time_stays_under_60_second_deadline():
    worst_case_seconds = (_MAX_RETRIES + 1) * _IO_TIMEOUT_SECONDS + _MAX_RETRIES * _MAX_BACKOFF_SECONDS
    assert worst_case_seconds < 60.0


def test_identical_input_clock_and_config_produce_byte_identical_output():
    secret_value = _ephemeral_test_secret()
    transport_a = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    transport_b = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    provider_a = _provider(transport_a, api_key=SecretString(secret_value))
    provider_b = _provider(transport_b, api_key=SecretString(secret_value))
    query = _query()
    envelope_a = provider_a.search_flights(query)
    envelope_b = provider_b.search_flights(query)
    assert envelope_a == envelope_b
    assert envelope_a["request_id"] == envelope_b["request_id"]


def test_cache_status_is_honestly_bypass():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["cache_status"] == "bypass"
    assert "cache_age_seconds" not in envelope


def test_no_real_socket_is_ever_opened(monkeypatch):
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope = _provider(transport).search_flights(_query())
    assert envelope["status"] == "success"


def test_provider_never_returns_an_envelope_that_fails_mandatory_validation(registry, monkeypatch):
    scenarios = [
        (FLIGHTS_SUCCESS, 200),
        (FLIGHTS_EMPTY, 200),
        ({}, 503),
        ({}, 429),
        ({"error": "Invalid API key."}, 200),
    ]
    for body, status in scenarios:
        transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body, status=status)})
        envelope = _provider(transport).search_flights(_query())
        _validate_full(envelope, registry)

    transport_cancelled = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope_cancelled = _provider(transport_cancelled, cancellation_check=lambda: True).search_flights(_query())
    _validate_full(envelope_cancelled, registry)

    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    transport_no_key = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(FLIGHTS_SUCCESS)})
    envelope_no_key = SerpApiFlightSearchProvider(
        transport=transport_no_key, clock=_clock, sleep_fn=lambda s: None, api_key=None
    ).search_flights(_query())
    _validate_full(envelope_no_key, registry)

    envelope_invalid = _provider(FakeHttpTransport()).search_flights(_query(origin="ZZZZ"))
    _validate_full(envelope_invalid, registry)

    envelope_range = _provider(FakeHttpTransport()).search_flights(_query(depart_date_to="2026-09-12"))
    _validate_full(envelope_range, registry)
