"""Hermetic tests for the flight-search provider interface + fake
(Checkpoint Phase 4 C.0, Step 8). No real network call anywhere in this
file. No booking, payment, reservation, or purchase claim is authorized
anywhere in this project -- these tests exist specifically to prove that
structurally, not just by convention."""

from __future__ import annotations

import socket

import pytest

from providers.flights import FakeFlightSearchProvider, FlightSearchQuery, build_degraded_envelope, build_success_envelope
from providers.tests.conftest import make_validator

QUERY = FlightSearchQuery(
    origin="BEY", destination="IST", depart_date_from="2026-09-10", depart_date_to="2026-09-10", passenger_count=1
)


def test_success_envelope_is_schema_valid(envelope_validator, registry):
    envelope = build_success_envelope(QUERY)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("FlightSearchResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "unavailable", "provider_error", "cancelled"])
def test_degraded_envelope_is_schema_valid_for_every_failure_status(status, envelope_validator, registry):
    envelope = build_degraded_envelope(QUERY, status)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("FlightSearchResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]
    assert envelope["result"]["options"] == []
    assert envelope["result"]["limitations"]
    assert envelope["data_mode"] == "unavailable"
    assert envelope["data_mode"] != "estimated"
    assert envelope["quality"]["completeness"] == 0.0


def test_stale_status_reports_cached_data_mode():
    envelope = build_degraded_envelope(QUERY, "stale")
    assert envelope["data_mode"] == "cached"
    assert envelope["quality"]["completeness"] == 1.0


def test_provider_failure_never_invents_offers_from_another_source():
    envelope = build_degraded_envelope(QUERY, "provider_error")
    assert envelope["result"]["options"] == []


def test_every_result_is_explicitly_a_search_snapshot():
    success = build_success_envelope(QUERY)
    failure = build_degraded_envelope(QUERY, "unavailable")
    assert success["result"]["is_snapshot"] is True
    assert failure["result"]["is_snapshot"] is True


def test_no_booking_availability_or_payment_claim_anywhere_in_the_envelope():
    envelope = build_success_envelope(QUERY)
    forbidden_terms = ("booked", "reservation_confirmed", "payment", "purchase", "seat_available", "confirmed")
    serialized = str(envelope).lower()
    for term in forbidden_terms:
        assert term not in serialized, f"forbidden booking-adjacent term found: {term!r}"


def test_flight_options_carry_timezone_aware_timestamps():
    envelope = build_success_envelope(QUERY)
    option = envelope["result"]["options"][0]
    assert option["depart_at"].endswith("Z")
    assert option["arrive_at"].endswith("Z")


def test_round_trip_query_requires_both_return_dates_together(registry):
    query = FlightSearchQuery(
        origin="BEY", destination="IST", depart_date_from="2026-09-10", depart_date_to="2026-09-10",
        passenger_count=1, return_date_from="2026-09-15", return_date_to="2026-09-16",
    )
    envelope = build_success_envelope(query)
    result_validator = make_validator("FlightSearchResult", registry)
    errors = list(result_validator.iter_errors(envelope["result"]))
    assert not errors, [e.message for e in errors]


def test_identical_input_clock_and_config_produce_byte_identical_output():
    fake_a = FakeFlightSearchProvider(clock=lambda: "2026-08-01T12:00:00Z")
    fake_b = FakeFlightSearchProvider(clock=lambda: "2026-08-01T12:00:00Z")
    envelope_a = fake_a.search_flights(QUERY)
    envelope_b = fake_b.search_flights(QUERY)
    assert envelope_a == envelope_b
    assert envelope_a["request_id"] == envelope_b["request_id"]


def test_fake_provider_never_opens_a_real_socket(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic fake-provider test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    fake = FakeFlightSearchProvider()
    envelope = fake.search_flights(QUERY)
    assert envelope["status"] == "success"


def test_fake_provider_tracks_call_log_for_duplicate_call_detection():
    fake = FakeFlightSearchProvider()
    fake.search_flights(QUERY)
    fake.search_flights(QUERY)
    assert len(fake.call_log) == 2
    assert fake.call_log[0].fingerprint() == fake.call_log[1].fingerprint()
