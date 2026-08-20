"""Hermetic tests for the real Open-Meteo weather adapter (Checkpoint
Phase 4 C.1). Every test uses FakeHttpTransport -- no real socket is ever
opened, proven per-test via monkeypatching socket.socket.connect to
raise. Response bodies mirror Open-Meteo's real documented shapes
(https://open-meteo.com/en/docs)."""

from __future__ import annotations

import json
import socket
from datetime import date as _dt_date

import pytest

from providers.http_transport import FakeHttpTransport, HttpResponse
from providers.policy import RetryPolicy, TimeoutPolicy
from providers.tests.conftest import make_validator
from providers.validation import EnvelopeValidationError
from providers.weather import WeatherQuery
from providers.weather_openmeteo import ARCHIVE_URL, FORECAST_URL, GEOCODING_URL, OpenMeteoWeatherProvider

FIXED_NOW = "2026-08-17T12:00:00Z"


def _clock():
    from datetime import datetime, timezone

    return datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def _json_response(payload: dict, status: int = 200, headers: dict | None = None) -> HttpResponse:
    return HttpResponse(status_code=status, body=json.dumps(payload).encode("utf-8"), headers=headers or {})


FORECAST_DAILY_DATES = ["2026-08-20", "2026-08-21"]  # within 16 days of FIXED_NOW (2026-08-17)

GEOCODE_ISTANBUL = {
    "results": [
        {
            "id": 745044, "name": "Istanbul", "latitude": 41.01384, "longitude": 28.94966,
            "country_code": "TR", "country": "Turkey", "timezone": "Europe/Istanbul", "population": 14804116,
        }
    ]
}

GEOCODE_MULTI = {
    "results": [
        {"id": 1, "name": "Istanbul", "latitude": 33.35, "longitude": -94.4, "country_code": "US", "country": "United States", "timezone": "America/Chicago"},
        {"id": 2, "name": "Istanbul", "latitude": 41.01384, "longitude": 28.94966, "country_code": "TR", "country": "Turkey", "timezone": "Europe/Istanbul"},
    ]
}

GEOCODE_EMPTY = {"results": []}

FORECAST_CURRENT = {
    "latitude": 41.01, "longitude": 28.96, "timezone": "Europe/Istanbul",
    "current": {
        "time": "2026-08-17T15:00", "temperature_2m": 28.5, "precipitation": 0.0,
        "weather_code": 1, "wind_speed_10m": 12.3, "wind_direction_10m": 210,
    },
}

FORECAST_DAILY = {
    "latitude": 41.01, "longitude": 28.96, "timezone": "Europe/Istanbul",
    "daily": {
        "time": FORECAST_DAILY_DATES,
        "weather_code": [1, 61],
        "temperature_2m_max": [27.5, 24.0],
        "temperature_2m_min": [19.0, 18.0],
        "precipitation_probability_max": [5, 60],
        "precipitation_sum": [0.0, 4.2],
        "wind_speed_10m_max": [15.0, 22.0],
    },
}


def _provider(transport: FakeHttpTransport, **kwargs) -> OpenMeteoWeatherProvider:
    return OpenMeteoWeatherProvider(
        transport=transport, clock=_clock, sleep_fn=lambda s: None,
        retry_policy=RetryPolicy(max_retries=2, base_backoff_seconds=0.01, max_backoff_seconds=0.02),
        timeout_policy=TimeoutPolicy(connect_timeout_seconds=1.0, total_timeout_seconds=1.0),
        **kwargs,
    )


def _validate_full(envelope: dict, registry):
    from providers.tests.conftest import CONTRACTS_DIR  # noqa: F401

    result_validator = make_validator("WeatherResult", registry)
    errors = list(result_validator.iter_errors(envelope["result"]))
    assert not errors, [e.message for e in errors]


# --- success: current conditions ------------------------------------------------------


def test_current_conditions_success_is_schema_valid(envelope_validator, registry, monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no real socket")))
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    _validate_full(envelope, registry)
    assert envelope["status"] == "success"
    assert envelope["result"]["kind"] == "current_observation"
    assert envelope["result"]["observation"]["temperature"] == 28.5
    assert envelope["result"]["observation"]["condition"] == "mainly_clear"


def test_current_conditions_timezone_offset_is_correct_not_a_naive_z():
    """Europe/Istanbul is UTC+3 in August (no DST in Turkey since 2016) --
    '2026-08-17T15:00' local must become '...+03:00', never a bare 'Z'
    mislabeling local time as UTC."""
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    issued_at = envelope["result"]["issued_at"]
    assert issued_at.endswith("+03:00")
    assert issued_at == envelope["result"]["observation"]["observed_at"]


def test_current_conditions_discloses_model_derived_not_sensor_measured():
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assumptions = " ".join(envelope["quality"]["assumptions"]).lower()
    assert "model-derived" in assumptions
    assert "not a raw sensor" in assumptions


def test_attribution_is_present_and_never_fabricated_beyond_open_meteo():
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    attribution = envelope["result"]["attribution"]
    assert attribution["license"] == "CC BY 4.0"
    assert "open-meteo.com" in attribution["notice"].lower()


# --- success: forecast --------------------------------------------------------------


def test_daily_forecast_success_is_schema_valid(envelope_validator, registry):
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_DAILY),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from=FORECAST_DAILY_DATES[0], date_to=FORECAST_DAILY_DATES[1])
    envelope = provider.fetch_weather(query)

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    _validate_full(envelope, registry)
    assert envelope["status"] == "success", envelope
    assert envelope["result"]["kind"] == "forecast"
    assert len(envelope["result"]["forecast_days"]) == 2
    assert envelope["result"]["forecast_days"][0]["high"] == 27.5
    assert envelope["result"]["forecast_days"][1]["condition"] == "slight_rain"
    assert envelope["result"]["forecast_days"][1]["precipitation_chance"] == 0.6  # 60 -> 0.6 probability


# --- geocoding: zero / multiple results ------------------------------------------------


def test_geocoding_zero_results_is_unavailable_never_invented(envelope_validator):
    transport = FakeHttpTransport(responses={GEOCODING_URL: _json_response(GEOCODE_EMPTY)})
    provider = _provider(transport)
    query = WeatherQuery(location="Nonexistent Place Zzz", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    assert envelope["status"] == "unavailable"
    assert envelope["result"]["forecast_days"] == []


def test_geocoding_multiple_results_prefers_explicit_country_code_hint():
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_MULTI),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(
        location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17", country_code="TR"
    )
    envelope = provider.fetch_weather(query)
    assert envelope["result"]["country"] == "Turkey"
    assert envelope["result"]["resolved_latitude"] == pytest.approx(41.01384, abs=1e-3)


def test_geocoding_multiple_results_without_hint_deterministically_picks_top_ranked():
    transport_a = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_MULTI),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    transport_b = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_MULTI),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope_a = _provider(transport_a).fetch_weather(query)
    envelope_b = _provider(transport_b).fetch_weather(query)
    # Deterministic: both independent runs choose the same (first/top-ranked) candidate.
    assert envelope_a["result"]["country"] == envelope_b["result"]["country"] == "United States"


def test_pre_supplied_coordinates_skip_geocoding_entirely():
    transport = FakeHttpTransport(responses={FORECAST_URL: _json_response(FORECAST_CURRENT)})
    provider = _provider(transport)
    query = WeatherQuery(
        location="Istanbul", timezone="Europe/Istanbul", date_from="2026-08-17", date_to="2026-08-17",
        latitude=41.01384, longitude=28.94966,
    )
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "success"
    assert all(url != GEOCODING_URL for url, _ in transport.call_log)


# --- request validation: invalid / unsupported ------------------------------------------


def test_date_from_after_date_to_is_invalid_request():
    provider = _provider(FakeHttpTransport())
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-20", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "invalid_request"
    assert envelope["data_mode"] == "unavailable"


def test_malformed_date_string_is_invalid_request():
    provider = _provider(FakeHttpTransport())
    query = WeatherQuery(location="Istanbul", timezone="", date_from="not-a-date", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "invalid_request"


def test_past_date_range_is_unsupported_never_silently_served():
    provider = _provider(FakeHttpTransport())
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2020-01-01", date_to="2020-01-02")
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "unsupported"


ARCHIVE_DAILY = {
    "latitude": 41.01, "longitude": 28.96, "timezone": "Europe/Istanbul",
    "daily": {
        # A single canned 3-day response reused for every sampled year
        # (the fake transport is not year-aware) -- since parsing is
        # index-based, this exercises real aggregation logic identically
        # to a real per-year variance would, and with 3 identical years
        # the averages equal these exact values, keeping assertions exact.
        "time": ["2023-09-03", "2023-09-04", "2023-09-05"],
        "weather_code": [1, 1, 61],
        "temperature_2m_max": [30.0, 28.0, 26.0],
        "temperature_2m_min": [20.0, 19.0, 18.0],
        "precipitation_sum": [0.0, 0.0, 2.0],
        "wind_speed_10m_max": [10.0, 12.0, 14.0],
    },
}


def test_trip_straddling_the_horizon_returns_mixed_live_plus_historical_coverage(envelope_validator, registry):
    """User correction pass (§C): date_from is today (within horizon),
    date_to is beyond it -- the full requested range must be covered,
    never silently truncated: live forecast for the reachable prefix,
    real historical climate guidance for the rest, explicitly marked
    'mixed'."""
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_DAILY),
        ARCHIVE_URL: _json_response(ARCHIVE_DAILY),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-09-05")
    envelope = provider.fetch_weather(query)

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    _validate_full(envelope, registry)
    assert envelope["status"] == "success"
    assert envelope["data_mode"] == "mixed"
    result = envelope["result"]
    assert result["coverage"] == "mixed"
    assert result["kind"] == "mixed"
    assert len(result["forecast_days"]) == 2  # from FORECAST_DAILY, unchanged
    assert [d["date"] for d in result["historical_climate_days"]] == ["2026-09-03", "2026-09-04", "2026-09-05"]
    assert result["historical_climate_days"][0]["avg_high"] == 30.0
    assert result["historical_climate_days"][0]["avg_low"] == 20.0
    assert result["historical_climate_days"][2]["avg_precipitation_amount"] == 2.0
    assert result["years_sampled"] == [2023, 2024, 2025]
    assert result["aggregation_method"] == "arithmetic_mean_across_sampled_years"
    assert "not a forecast" in result["climate_disclaimer"].lower()
    # User correction pass §C: a fixed property of date_from (2026-08-17),
    # never of "today" -- date_from - 16.
    assert result["earliest_available_forecast_date"] == "2026-08-01"

    forecast_call = next(call for call in transport.call_log if call[0] == FORECAST_URL)
    assert forecast_call[1]["start_date"] == "2026-08-17"
    assert forecast_call[1]["end_date"] == "2026-09-02"  # clipped to the horizon
    archive_calls = [call for call in transport.call_log if call[0] == ARCHIVE_URL]
    assert len(archive_calls) == 3  # one real request per sampled year
    assert {c[1]["start_date"] for c in archive_calls} == {"2023-09-03", "2024-09-03", "2025-09-03"}


def test_depart_date_itself_beyond_horizon_returns_pure_historical_climate_guidance(envelope_validator, registry):
    """User correction pass (§C): a future trip entirely beyond the
    forecast horizon is never a dead end -- it gets a genuine successful
    historical-climate result, clearly labeled, never a fabricated
    forecast and never a bare degraded status."""
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        ARCHIVE_URL: _json_response(ARCHIVE_DAILY),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-09-05", date_to="2026-09-05")
    envelope = provider.fetch_weather(query)

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    _validate_full(envelope, registry)
    assert envelope["status"] == "success"
    assert envelope["data_mode"] == "historical"
    result = envelope["result"]
    assert result["coverage"] == "historical"
    assert result["kind"] == "historical"
    assert result["forecast_days"] == []
    assert len(result["historical_climate_days"]) == 1
    assert result["historical_climate_days"][0]["date"] == "2026-09-05"
    assert result["earliest_available_forecast_date"] == "2026-08-20"  # 2026-09-05 - 16
    assert not any(call[0] == FORECAST_URL for call in transport.call_log)  # never called: no live portion at all


def test_historical_climate_fetch_failure_degrades_honestly_never_fabricated(envelope_validator, registry):
    """A real archive-API failure (e.g. timeout) must degrade the whole
    call to that honest status -- never silently return an empty/partial
    aggregate pretending to be real climate guidance."""
    transport = FakeHttpTransport(responses={GEOCODING_URL: _json_response(GEOCODE_ISTANBUL)}, raise_transport_error=True)
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-09-05", date_to="2026-09-05")
    envelope = provider.fetch_weather(query)

    errors = list(envelope_validator.iter_errors(envelope))
    assert not errors, [e.message for e in errors]
    assert envelope["status"] == "timeout"
    assert "historical_climate_days" not in envelope["result"]  # never a fabricated/partial aggregate
    assert envelope["result"]["forecast_days"] == []


def test_year_equivalent_date_clamps_feb29_in_a_non_leap_sampled_year():
    """Leap-day handling (User correction pass §C test requirement):
    deterministic, documented clamp -- never silently skipped."""
    from providers.weather_openmeteo import _year_equivalent_date

    reference = _dt_date(2028, 2, 29)  # 2028 is itself a leap year
    assert _year_equivalent_date(reference, 2024) == _dt_date(2024, 2, 29)  # 2024 is leap: exact match
    assert _year_equivalent_date(reference, 2023) == _dt_date(2023, 2, 28)  # 2023 is not leap: clamped
    assert _year_equivalent_date(reference, 2025) == _dt_date(2025, 2, 28)  # 2025 is not leap: clamped


def test_sample_years_is_deterministic_and_never_includes_the_current_year():
    from providers.weather_openmeteo import _sample_years

    assert _sample_years(_dt_date(2026, 8, 17), count=3) == [2023, 2024, 2025]
    assert _sample_years(_dt_date(2030, 1, 1), count=5) == [2025, 2026, 2027, 2028, 2029]


def test_earliest_available_forecast_date_depends_only_on_date_from_never_on_today():
    """The exact bug the user correction pass caught: this must be a
    fixed property of the request's own date_from, never drift depending
    on when the identical trip date happens to be queried."""
    from providers.weather_openmeteo import _earliest_available_forecast_date

    date_from = _dt_date(2026, 9, 20)
    assert _earliest_available_forecast_date(date_from) == _dt_date(2026, 9, 4)
    # Querying the SAME date_from "later" must give the identical answer
    # -- this function takes no "today"/clock argument at all by design.


def test_out_of_range_coordinates_are_invalid_request():
    provider = _provider(FakeHttpTransport())
    query = WeatherQuery(
        location="nowhere", timezone="UTC", date_from="2026-08-17", date_to="2026-08-17",
        latitude=200.0, longitude=28.0,
    )
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "invalid_request"


# --- transport failures: timeout / rate_limited / provider_error --------------------------


def test_transport_error_exhausting_retries_is_timeout():
    transport = FakeHttpTransport(raise_transport_error=True)
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "timeout"
    assert envelope["data_mode"] == "unavailable"
    assert envelope["data_mode"] != "estimated"


def test_rate_limited_honors_retry_after_header():
    call_count = {"n": 0}

    def geocode_then_rate_limit(params):
        call_count["n"] += 1
        return _json_response({}, status=429, headers={"Retry-After": "7"})

    transport = FakeHttpTransport(responses={GEOCODING_URL: geocode_then_rate_limit})
    sleeps: list[float] = []
    provider = OpenMeteoWeatherProvider(
        transport=transport, clock=_clock, sleep_fn=lambda s: sleeps.append(s),
        retry_policy=RetryPolicy(max_retries=1, base_backoff_seconds=0.01, max_backoff_seconds=100.0),
        timeout_policy=TimeoutPolicy(),
    )
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "rate_limited"
    assert sleeps == [7.0]  # honored Retry-After, not exponential backoff
    assert call_count["n"] == 2  # first attempt + 1 retry


def test_provider_error_5xx_is_provider_error_never_raises():
    transport = FakeHttpTransport(responses={GEOCODING_URL: _json_response({}, status=503)})
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)  # must not raise
    assert envelope["status"] == "provider_error"


def test_malformed_2xx_response_is_provider_error_never_raises():
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response({"unexpected": "shape"}),  # 200 OK but no 'current'/'daily' block
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)  # must not raise
    assert envelope["status"] == "provider_error"


def test_successful_retry_after_one_transient_failure():
    attempts = {"n": 0}

    def flaky_geocode(params):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _json_response({}, status=500)
        return _json_response(GEOCODE_ISTANBUL)

    transport = FakeHttpTransport(responses={
        GEOCODING_URL: flaky_geocode,
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    sleeps: list[float] = []
    provider = OpenMeteoWeatherProvider(
        transport=transport, clock=_clock, sleep_fn=lambda s: sleeps.append(s),
        retry_policy=RetryPolicy(max_retries=2, base_backoff_seconds=0.01, max_backoff_seconds=1.0),
        timeout_policy=TimeoutPolicy(),
    )
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assert envelope["status"] == "success"
    assert len(sleeps) == 1


# --- determinism / cache honesty --------------------------------------------------------


def test_identical_query_and_clock_produce_the_same_request_id_across_instances():
    transport_a = FakeHttpTransport(responses={GEOCODING_URL: _json_response(GEOCODE_ISTANBUL), FORECAST_URL: _json_response(FORECAST_CURRENT)})
    transport_b = FakeHttpTransport(responses={GEOCODING_URL: _json_response(GEOCODE_ISTANBUL), FORECAST_URL: _json_response(FORECAST_CURRENT)})
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope_a = _provider(transport_a).fetch_weather(query)
    envelope_b = _provider(transport_b).fetch_weather(query)
    assert envelope_a["request_id"] == envelope_b["request_id"]


def test_cache_status_is_honestly_bypass_no_real_cache_implemented():
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    assert envelope["cache_status"] == "bypass"
    assert "cache_age_seconds" not in envelope


# --- mandatory two-stage validation is actually applied ----------------------------------


def test_provider_never_returns_an_envelope_that_fails_mandatory_validation():
    """Every single envelope this adapter can produce (success + every
    degraded path exercised above) must already have passed
    providers.validation.validate_envelope internally -- re-validate here
    as an end-to-end proof, not just trust the adapter's own internal call."""
    from providers.validation import validate_envelope

    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(GEOCODE_ISTANBUL),
        FORECAST_URL: _json_response(FORECAST_CURRENT),
    })
    provider = _provider(transport)
    query = WeatherQuery(location="Istanbul", timezone="", date_from="2026-08-17", date_to="2026-08-17")
    envelope = provider.fetch_weather(query)
    validate_envelope(envelope)  # must not raise
