"""Hermetic tests for the weather provider interface + fake (Checkpoint
Phase 4 C.0, Step 8). No real network call anywhere in this file --
FakeWeatherProvider is the only implementation exercised."""

from __future__ import annotations

import socket

import pytest

from providers.policy import RESULT_STATUSES
from providers.weather import FakeWeatherProvider, WeatherQuery, build_degraded_envelope, build_success_envelope

QUERY = WeatherQuery(location="Istanbul", timezone="Europe/Istanbul", date_from="2026-09-10", date_to="2026-09-11")


def test_success_envelope_is_schema_valid(envelope_validator, registry):
    from providers.tests.conftest import make_validator

    envelope = build_success_envelope(QUERY)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("WeatherResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "unavailable", "provider_error", "cancelled"])
def test_degraded_envelope_is_schema_valid_for_every_failure_status(status, envelope_validator, registry):
    from providers.tests.conftest import make_validator

    envelope = build_degraded_envelope(QUERY, status)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    result_validator = make_validator("WeatherResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]
    assert envelope["status"] == status
    assert envelope["result"]["kind"] == "unavailable"


@pytest.mark.parametrize("status", ["timeout", "rate_limited", "unavailable", "provider_error", "cancelled"])
def test_no_data_failure_statuses_use_honest_unavailable_data_mode_never_estimated(status):
    """Checkpoint Phase 4 C.0 correction: a genuine no-data failure must
    never claim data_mode='estimated' -- that value is reserved for a
    real computed fallback, which none of these statuses produce."""
    envelope = build_degraded_envelope(QUERY, status)
    assert envelope["data_mode"] == "unavailable"
    assert envelope["data_mode"] != "estimated"
    assert envelope["quality"]["freshness"] == "unavailable"
    assert envelope["quality"]["completeness"] == 0.0


def test_stale_status_reports_cached_data_mode_not_unavailable_or_estimated():
    """Stale means genuine old data exists -- 'cached', not 'unavailable'
    (no data) and not 'estimated' (a computed fallback)."""
    envelope = build_degraded_envelope(QUERY, "stale")
    assert envelope["data_mode"] == "cached"
    assert envelope["quality"]["completeness"] == 1.0  # the data is complete, just old
    assert envelope["result"]["kind"] == "forecast"  # real forecast data is present, unlike a true failure


def test_provider_failure_is_never_represented_as_an_empty_successful_result():
    envelope = build_degraded_envelope(QUERY, "provider_error")
    assert envelope["status"] != "success"
    assert envelope["result"]["forecast_days"] == []
    assert "forecast_days" in envelope["result"]["missing_fields"]


def test_weather_type_distinctions_forecast_vs_unavailable():
    success = build_success_envelope(QUERY)
    failure = build_degraded_envelope(QUERY, "unavailable")
    assert success["result"]["kind"] == "forecast"
    assert failure["result"]["kind"] == "unavailable"


def test_timestamps_are_timezone_aware_utc_z_suffix():
    envelope = build_success_envelope(QUERY)
    assert envelope["retrieved_at"].endswith("Z")


def test_result_status_is_a_member_of_the_shared_vocabulary():
    envelope = build_success_envelope(QUERY)
    assert envelope["status"] in RESULT_STATUSES


def test_identical_input_clock_and_config_produce_byte_identical_output():
    """Checkpoint Phase 4 C.0 correction: no random UUID, no wall-clock
    read -- two independently constructed fakes with the same fixed clock
    must produce the exact same envelope dict, request_id included."""
    fake_a = FakeWeatherProvider(clock=lambda: "2026-08-01T12:00:00Z")
    fake_b = FakeWeatherProvider(clock=lambda: "2026-08-01T12:00:00Z")
    envelope_a = fake_a.fetch_weather(QUERY)
    envelope_b = fake_b.fetch_weather(QUERY)
    assert envelope_a == envelope_b
    assert envelope_a["request_id"] == envelope_b["request_id"]


def test_repeated_calls_from_the_same_fake_are_also_byte_identical():
    fake = FakeWeatherProvider(clock=lambda: "2026-08-01T12:00:00Z")
    first = fake.fetch_weather(QUERY)
    second = fake.fetch_weather(QUERY)
    assert first == second


def test_different_clock_produces_a_different_request_id_but_same_shape():
    fake_a = FakeWeatherProvider(clock=lambda: "2026-08-01T12:00:00Z")
    fake_b = FakeWeatherProvider(clock=lambda: "2026-08-01T13:00:00Z")
    envelope_a = fake_a.fetch_weather(QUERY)
    envelope_b = fake_b.fetch_weather(QUERY)
    assert envelope_a["request_id"] != envelope_b["request_id"]
    assert envelope_a["query_fingerprint"] == envelope_b["query_fingerprint"]


def test_fake_provider_tracks_call_log_for_duplicate_call_detection():
    fake = FakeWeatherProvider()
    fake.fetch_weather(QUERY)
    fake.fetch_weather(QUERY)
    assert len(fake.call_log) == 2
    assert fake.call_log[0].fingerprint() == fake.call_log[1].fingerprint()


def test_fake_provider_never_opens_a_real_socket(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic fake-provider test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    fake = FakeWeatherProvider()
    envelope = fake.fetch_weather(QUERY)
    assert envelope["status"] == "success"


def test_stale_status_is_supported_and_distinct_from_unavailable(envelope_validator):
    envelope = build_degraded_envelope(QUERY, "stale")
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    assert envelope["status"] == "stale"


def test_cancelled_status_is_supported(envelope_validator):
    envelope = build_degraded_envelope(QUERY, "cancelled")
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    assert envelope["status"] == "cancelled"


def test_malformed_provider_response_is_caught_by_schema_validation(registry):
    """Simulates a malformed provider response (e.g. wrong type) reaching
    the result boundary -- schema validation must reject it, never pass
    it through silently. ProviderResponseEnvelope.result is intentionally
    unconstrained (type: object) so this must validate against
    WeatherResult specifically, not the outer envelope."""
    from providers.tests.conftest import make_validator

    malformed_result = build_success_envelope(QUERY)["result"]
    malformed_result["forecast_days"] = "not-a-list"
    result_validator = make_validator("WeatherResult", registry)
    errors = list(result_validator.iter_errors(malformed_result))
    assert errors
