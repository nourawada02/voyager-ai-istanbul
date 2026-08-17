"""Weather: an independent typed tool/provider (Checkpoint Phase 4 C.0).
Never implemented inside System A as tightly-coupled code, never owned by
Travel MCP or System B. `WeatherProvider` is the injectable interface a
future System A ReAct loop calls through; `FakeWeatherProvider` is the
deterministic implementation every hermetic test in this project uses --
no real network call, no real credential, ever, in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from providers.fingerprint import deterministic_request_id, fingerprint_request
from providers.policy import completeness_for_status, data_mode_for_status

CAPABILITY = "weather"
SCHEMA_VERSION = "1.1.0"
RESULT_SCHEMA_VERSION = "1.0.0"

_FIXED_TEST_CLOCK = "2026-08-01T12:00:00Z"


@dataclass(frozen=True)
class WeatherQuery:
    location: str
    timezone: str
    date_from: str  # ISO date (YYYY-MM-DD)
    date_to: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country_code: Optional[str] = None  # ISO 3166-1 alpha-2, e.g. "TR" -- disambiguates geocoding

    def normalized(self) -> dict:
        """Whitespace/case-normalized so 'Istanbul' and ' istanbul ' fingerprint identically."""
        return {
            "location": " ".join(self.location.split()).lower(),
            "timezone": self.timezone,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "country_code": self.country_code.upper() if self.country_code else None,
        }

    @property
    def has_coordinates(self) -> bool:
        return self.latitude is not None and self.longitude is not None

    def fingerprint(self) -> str:
        return fingerprint_request(CAPABILITY, self.normalized())


class WeatherProvider(Protocol):
    """The interface a real adapter (e.g. Open-Meteo, see
    docs/adr/0009-...md §7) and FakeWeatherProvider both satisfy. Never
    leaks a vendor-specific response object -- always returns a
    ProviderResponseEnvelope-shaped dict (contracts/ProviderResponseEnvelope.schema.json
    + contracts/WeatherResult.schema.json)."""

    def fetch_weather(self, query: WeatherQuery) -> dict: ...


def _base_envelope(
    query: WeatherQuery, provider_name: str, status: str, data_mode: str, retrieved_at: str = _FIXED_TEST_CLOCK
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
            "location": query.location,
            "timezone": query.timezone,
            "kind": "forecast",
            "units": {"temperature": "C", "wind_speed": "kmh", "precipitation": "mm"},
            "forecast_days": [],
            "missing_fields": [],
        },
    }


def build_success_envelope(
    query: WeatherQuery, provider_name: str = "fake-weather-provider", retrieved_at: str = _FIXED_TEST_CLOCK
) -> dict:
    envelope = _base_envelope(query, provider_name, status="success", data_mode="live", retrieved_at=retrieved_at)
    envelope["result"]["forecast_days"] = [
        {"date": query.date_from, "condition": "clear", "high": 27, "low": 19, "precipitation_chance": 0.05},
    ]
    return envelope


def build_degraded_envelope(
    query: WeatherQuery, status: str, provider_name: str = "fake-weather-provider", retrieved_at: str = _FIXED_TEST_CLOCK
) -> dict:
    """`status` must be one of the non-success values in
    providers.policy.RESULT_STATUSES (e.g. 'timeout', 'rate_limited',
    'unavailable', 'provider_error', 'cancelled', 'stale'). Never
    fabricates forecast data on a degraded path -- 'kind' becomes
    'unavailable' (or stays a real 'kind' for 'stale', where genuine old
    data exists) and every missing field is named in missing_fields.
    data_mode is derived honestly from status via
    providers.policy.data_mode_for_status -- never hardcoded to
    'estimated' when no estimate was actually computed (Checkpoint Phase
    4 C.0 correction pass)."""
    data_mode = data_mode_for_status(status)
    envelope = _base_envelope(query, provider_name, status=status, data_mode=data_mode, retrieved_at=retrieved_at)
    if data_mode != "cached":
        envelope["result"]["kind"] = "unavailable"
        envelope["result"]["missing_fields"] = ["forecast_days"]
    envelope["quality"]["assumptions"] = [
        f"Weather provider call ended with status={status!r} (data_mode={data_mode!r}); "
        f"plan built weather-neutral per architecture.md §13.5."
    ]
    return envelope


@dataclass
class FakeWeatherProvider:
    """Deterministic, injectable fake -- constructs a fixed response or a
    fixed degraded status, never touches the network. `call_log` records
    every query passed in, for hermetic tests to assert on (e.g.
    duplicate-call detection, batching)."""

    fixed_status: str = "success"
    provider_name: str = "fake-weather-provider"
    clock: Callable[[], str] = field(default=lambda: _FIXED_TEST_CLOCK)
    call_log: list[WeatherQuery] = field(default_factory=list)

    def fetch_weather(self, query: WeatherQuery) -> dict:
        self.call_log.append(query)
        now = self.clock()
        if self.fixed_status == "success":
            return build_success_envelope(query, self.provider_name, retrieved_at=now)
        return build_degraded_envelope(query, self.fixed_status, self.provider_name, retrieved_at=now)
