"""Hermetic tests for deterministic request fingerprinting (Checkpoint
Phase 4 C.0, Step 8: "deterministic request fingerprints")."""

from __future__ import annotations

from providers.fingerprint import deterministic_request_id, fingerprint_request
from providers.flights import FlightSearchQuery
from providers.weather import WeatherQuery
from providers.web_evidence import WebEvidenceQuery


def test_same_logical_request_always_fingerprints_identically():
    a = fingerprint_request("weather", {"location": "istanbul", "date_from": "2026-09-10"})
    b = fingerprint_request("weather", {"location": "istanbul", "date_from": "2026-09-10"})
    assert a == b


def test_fingerprint_is_independent_of_dict_key_order():
    a = fingerprint_request("weather", {"location": "istanbul", "date_from": "2026-09-10"})
    b = fingerprint_request("weather", {"date_from": "2026-09-10", "location": "istanbul"})
    assert a == b


def test_different_capability_never_collides_even_with_identical_fields():
    a = fingerprint_request("weather", {"x": "1"})
    b = fingerprint_request("flight_search", {"x": "1"})
    assert a != b


def test_different_request_content_produces_different_fingerprint():
    a = fingerprint_request("weather", {"location": "istanbul"})
    b = fingerprint_request("weather", {"location": "ankara"})
    assert a != b


def test_fingerprint_is_a_64_char_hex_sha256_digest():
    fp = fingerprint_request("weather", {"location": "istanbul"})
    assert len(fp) == 64
    int(fp, 16)  # raises ValueError if not valid hex


def test_weather_query_normalization_makes_whitespace_and_case_insensitive_fingerprints():
    a = WeatherQuery(location="Istanbul", timezone="Europe/Istanbul", date_from="2026-09-10", date_to="2026-09-11")
    b = WeatherQuery(location="  istanbul  ", timezone="Europe/Istanbul", date_from="2026-09-10", date_to="2026-09-11")
    assert a.fingerprint() == b.fingerprint()


def test_web_evidence_query_normalization_is_case_and_whitespace_insensitive():
    a = WebEvidenceQuery(query="Hagia Sophia Hours")
    b = WebEvidenceQuery(query="  hagia   sophia   hours  ")
    assert a.fingerprint() == b.fingerprint()


def test_flight_query_normalization_uppercases_iata_codes():
    a = FlightSearchQuery(origin="bey", destination="ist", depart_date_from="2026-09-10", depart_date_to="2026-09-10", passenger_count=1)
    b = FlightSearchQuery(origin="BEY", destination="IST", depart_date_from="2026-09-10", depart_date_to="2026-09-10", passenger_count=1)
    assert a.fingerprint() == b.fingerprint()


def test_flight_query_different_passenger_count_produces_different_fingerprint():
    a = FlightSearchQuery(origin="BEY", destination="IST", depart_date_from="2026-09-10", depart_date_to="2026-09-10", passenger_count=1)
    b = FlightSearchQuery(origin="BEY", destination="IST", depart_date_from="2026-09-10", depart_date_to="2026-09-10", passenger_count=2)
    assert a.fingerprint() != b.fingerprint()


# --- deterministic_request_id (Checkpoint Phase 4 C.0 correction pass) --------------


def test_deterministic_request_id_is_identical_for_identical_inputs():
    a = deterministic_request_id("abc123", "2026-08-01T12:00:00Z", attempt=0)
    b = deterministic_request_id("abc123", "2026-08-01T12:00:00Z", attempt=0)
    assert a == b


def test_deterministic_request_id_never_uses_random_uuid4():
    """Two independent calls with the exact same fingerprint/clock/attempt
    must never differ -- proving this is not backed by uuid4/os.urandom."""
    ids = {deterministic_request_id("fp", "2026-08-01T12:00:00Z", attempt=0) for _ in range(20)}
    assert len(ids) == 1


def test_deterministic_request_id_differs_by_fingerprint():
    a = deterministic_request_id("fp-one", "2026-08-01T12:00:00Z")
    b = deterministic_request_id("fp-two", "2026-08-01T12:00:00Z")
    assert a != b


def test_deterministic_request_id_differs_by_clock():
    a = deterministic_request_id("fp", "2026-08-01T12:00:00Z")
    b = deterministic_request_id("fp", "2026-08-01T13:00:00Z")
    assert a != b


def test_deterministic_request_id_differs_by_attempt_even_at_the_same_instant():
    """A genuine retry at the identical timestamp still gets a distinct
    id -- never collides with the first attempt's."""
    a = deterministic_request_id("fp", "2026-08-01T12:00:00Z", attempt=0)
    b = deterministic_request_id("fp", "2026-08-01T12:00:00Z", attempt=1)
    assert a != b


def test_deterministic_request_id_matches_the_envelope_uuid_pattern():
    import re

    request_id = deterministic_request_id("fp", "2026-08-01T12:00:00Z")
    assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", request_id)
