"""Hermetic tests for the real root-provider bindings (Checkpoint Phase 4
D.1 §5). Every provider is constructed with `FakeHttpTransport` -- no real
socket is ever opened. Proves the argument mapping (D.0 arguments ->
provider Query) and that a provider's own honest `status` is reported
unmodified, never reinterpreted as success.
"""

from __future__ import annotations

import json
import socket

from providers.flights_serpapi import FLIGHTS_URL, SerpApiFlightSearchProvider
from providers.http_transport import FakeHttpTransport, HttpResponse
from providers.redaction import SecretString
from providers.weather_openmeteo import FORECAST_URL, GEOCODING_URL, OpenMeteoWeatherProvider
from providers.web_evidence_serpapi import SEARCH_URL, SerpApiWebEvidenceProvider

from orchestration.system_a.provider_bindings import fetch_weather, search_flights, search_web


def _json_response(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(status_code=status, body=json.dumps(body).encode("utf-8"), headers={})


def _clock() -> str:
    return "2026-08-17T12:00:00Z"


def _fixed_clock():
    import datetime

    return datetime.datetime(2026, 8, 17, tzinfo=datetime.timezone.utc)


def test_fetch_weather_maps_arguments_and_returns_provider_status():
    # A near-term date (3 days out) -- within Open-Meteo's 16-day
    # forecast horizon relative to the fixed clock below.
    geocoding_body = {"results": [{"latitude": 41.0, "longitude": 28.9, "name": "Istanbul", "country": "Turkey", "timezone": "Europe/Istanbul"}]}
    forecast_body = {"daily": {"time": ["2026-08-20"], "weather_code": [1], "temperature_2m_max": [27], "temperature_2m_min": [19]}}
    transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response(geocoding_body),
        FORECAST_URL: _json_response(forecast_body),
    })
    provider = OpenMeteoWeatherProvider(transport=transport, clock=_fixed_clock)
    result = fetch_weather(provider, {"location": "Istanbul", "date_from": "2026-08-20", "date_to": "2026-08-20"})
    assert result["status"] == "success"
    assert result["result"]["result"]["kind"] == "forecast"


def test_fetch_weather_unavailable_status_is_never_reinterpreted_as_success():
    transport = FakeHttpTransport(responses={GEOCODING_URL: _json_response({"results": []})})
    provider = OpenMeteoWeatherProvider(transport=transport, clock=_fixed_clock)
    result = fetch_weather(provider, {"location": "Nowhere", "date_from": "2026-08-20", "date_to": "2026-08-20"})
    assert result["status"] != "success"
    assert result["status"] == "unavailable"


def test_search_web_maps_arguments():
    body = {"search_metadata": {"status": "Success"}, "organic_results": []}
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    provider = SerpApiWebEvidenceProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None, api_key=SecretString("test"))
    result = search_web(provider, {"query": "Hagia Sophia hours", "max_results": 3})
    assert result["status"] == "success"
    assert transport.call_log[0][1]["q"] == "Hagia Sophia hours"
    assert transport.call_log[0][1]["num"] == 3


def test_search_flights_maps_exact_date_arguments():
    body = {"search_metadata": {"status": "Success"}, "best_flights": [], "other_flights": []}
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response(body)})
    provider = SerpApiFlightSearchProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None, api_key=SecretString("test"))
    result = search_flights(provider, {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1})
    assert result["status"] == "success"
    params = transport.call_log[0][1]
    assert params["departure_id"] == "BEY"
    assert params["arrival_id"] == "IST"
    assert params["outbound_date"] == "2026-09-10"
    assert params["type"] == "2"


def test_search_flights_missing_key_is_unavailable_not_success(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    transport = FakeHttpTransport(responses={})
    provider = SerpApiFlightSearchProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None, api_key=None)
    result = search_flights(provider, {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1})
    assert result["status"] == "unavailable"
    assert transport.call_log == []


def test_no_real_socket_opened_in_provider_binding_tests(monkeypatch):
    def _forbidden(*args, **kwargs):
        raise AssertionError("no network call is permitted from a hermetic test")

    monkeypatch.setattr(socket.socket, "connect", _forbidden)
    body = {"search_metadata": {"status": "Success"}, "organic_results": []}
    transport = FakeHttpTransport(responses={SEARCH_URL: _json_response(body)})
    provider = SerpApiWebEvidenceProvider(transport=transport, clock=_clock, sleep_fn=lambda s: None, api_key=SecretString("test"))
    result = search_web(provider, {"query": "test"})
    assert result["status"] == "success"
