"""Hermetic tests for mandatory capability-result validation (Checkpoint
Phase 4 C.0 correction pass, item 5)."""

from __future__ import annotations

import copy
import os

import pytest

from providers.flights import FlightSearchQuery, build_success_envelope as build_flight_success
from providers.validation import EnvelopeValidationError, validate_envelope
from providers.weather import WeatherQuery, build_degraded_envelope as build_weather_degraded, build_success_envelope as build_weather_success
from providers.web_evidence import WebEvidenceQuery, build_success_envelope as build_web_success

WEATHER_QUERY = WeatherQuery(location="Istanbul", timezone="Europe/Istanbul", date_from="2026-09-10", date_to="2026-09-11")
WEB_QUERY = WebEvidenceQuery(query="Hagia Sophia hours")
FLIGHT_QUERY = FlightSearchQuery(origin="BEY", destination="IST", depart_date_from="2026-09-10", depart_date_to="2026-09-10", passenger_count=1)


# --- valid envelopes pass for every capability ---------------------------------------


def test_valid_weather_envelope_passes():
    validate_envelope(build_weather_success(WEATHER_QUERY))


def test_valid_web_evidence_envelope_passes():
    validate_envelope(build_web_success(WEB_QUERY))


def test_valid_flight_envelope_passes():
    validate_envelope(build_flight_success(FLIGHT_QUERY))


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "unavailable", "provider_error", "cancelled", "stale"])
def test_valid_degraded_weather_envelope_passes_for_every_status(status):
    validate_envelope(build_weather_degraded(WEATHER_QUERY, status))


# --- malformed envelope (violates the outer envelope shape) --------------------------


def test_malformed_envelope_missing_required_field_is_rejected():
    envelope = build_weather_success(WEATHER_QUERY)
    del envelope["retrieved_at"]
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(envelope)
    assert exc_info.value.reason == "envelope_invalid"


def test_malformed_envelope_wrong_type_is_rejected():
    envelope = build_weather_success(WEATHER_QUERY)
    envelope["source_urls"] = "not-a-list"
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(envelope)
    assert exc_info.value.reason == "envelope_invalid"


# --- unknown capability ---------------------------------------------------------------


def test_unknown_capability_string_is_rejected():
    envelope = build_weather_success(WEATHER_QUERY)
    envelope["capability"] = "time_travel_booking"
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(envelope)
    assert exc_info.value.reason == "unknown_capability"


def test_missing_capability_is_accepted_for_legacy_1_0_0_backward_compatibility():
    """A pre-1.1.0 envelope (no 'capability' field at all) must remain
    valid -- the additive-versioning contract's whole point."""
    envelope = build_weather_success(WEATHER_QUERY)
    envelope["schema_version"] = "1.0.0"
    del envelope["capability"]
    del envelope["status"]
    del envelope["query_fingerprint"]
    validate_envelope(envelope)  # must not raise


def test_missing_capability_is_rejected_when_explicitly_required():
    envelope = build_weather_success(WEATHER_QUERY)
    del envelope["capability"]
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(envelope, require_capability=True)
    assert exc_info.value.reason == "unknown_capability"


# --- mismatched capability / result shape ----------------------------------------------


def test_result_that_does_not_match_its_declared_capability_is_rejected():
    """capability='weather' but the result is actually flight-shaped --
    must be rejected, never accepted because *some* schema might match."""
    weather_envelope = build_weather_success(WEATHER_QUERY)
    flight_envelope = build_flight_success(FLIGHT_QUERY)
    mismatched = copy.deepcopy(weather_envelope)
    mismatched["result"] = flight_envelope["result"]
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(mismatched)
    assert exc_info.value.reason == "result_invalid_for_capability"


def test_malformed_result_for_a_known_capability_is_rejected():
    envelope = build_weather_success(WEATHER_QUERY)
    envelope["result"]["forecast_days"] = "not-a-list"
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(envelope)
    assert exc_info.value.reason == "result_invalid_for_capability"


def test_error_message_never_leaks_the_full_raw_jsonschema_object_only_short_messages():
    envelope = build_weather_success(WEATHER_QUERY)
    envelope["result"]["forecast_days"] = "not-a-list"
    with pytest.raises(EnvelopeValidationError) as exc_info:
        validate_envelope(envelope)
    assert "not-a-list" in str(exc_info.value) or "is not of type" in str(exc_info.value)


# --- P.1: contracts/ resolution is package-relative, never CWD-dependent -------------


def test_contracts_directory_resolves_relative_to_the_providers_package_not_cwd():
    """Production incident P.1: `_CONTRACTS_DIR` (`providers/validation.py`)
    is computed once from `__file__`, never `os.getcwd()` -- this was
    already correct before the incident (the actual bug was a Dockerfile
    packaging omission, not this resolution logic). Proven directly here
    by changing the process CWD to something with no `contracts/`
    sibling at all and confirming validation still succeeds."""
    from providers.validation import _CONTRACTS_DIR

    assert os.path.isdir(_CONTRACTS_DIR)
    assert os.path.isabs(_CONTRACTS_DIR)

    original_cwd = os.getcwd()
    scratch_dir = os.path.dirname(original_cwd)  # a real directory with no contracts/ subdirectory
    try:
        os.chdir(scratch_dir)
        # A fresh envelope build + validation, performed entirely while CWD
        # points somewhere with no contracts/ directory at all.
        validate_envelope(build_weather_success(WEATHER_QUERY))
    finally:
        os.chdir(original_cwd)
