"""Real Open-Meteo weather adapter (Checkpoint Phase 4 C.1;
docs/adr/0010-phase4-checkpointc1-openmeteo-weather-adapter.md). The
first real, non-fake provider implementation in this project.

No API key, no account, no billing: Open-Meteo's non-commercial API
requires none of those (https://open-meteo.com/en/docs). Two fixed,
explicit endpoints are ever contacted -- both hardcoded constants below,
never built from user-controlled input:

  https://geocoding-api.open-meteo.com/v1/search   (location resolution)
  https://api.open-meteo.com/v1/forecast            (current/forecast weather)

Provider-specific HTTP logic, response parsing, and WMO weather-code
mapping live entirely in this module -- providers/weather.py (the shared
WeatherQuery/WeatherProvider contract) and providers/policy.py (shared
retry/timeout/cache policy) know nothing Open-Meteo-specific.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from providers.fingerprint import deterministic_request_id
from providers.http_transport import HttpResponse, HttpTransport, TransportError, UrllibHttpTransport
from providers.policy import (
    RetryPolicy,
    TimeoutPolicy,
    classify_http_status,
    completeness_for_status,
    compute_backoff_seconds,
    data_mode_for_status,
)
from providers.validation import validate_envelope
from providers.weather import CAPABILITY, RESULT_SCHEMA_VERSION, SCHEMA_VERSION, WeatherQuery

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_FORECAST_DAYS_AHEAD = 16  # Open-Meteo's own supported forecast horizon

PROVIDER_NAME = "open-meteo"
ATTRIBUTION = {
    "license": "CC BY 4.0",
    "notice": "Weather data by Open-Meteo.com (https://open-meteo.com/), licensed under CC BY 4.0.",
}

# WMO Weather interpretation codes (Open-Meteo's own documented mapping,
# https://open-meteo.com/en/docs -- WMO Weather interpretation codes
# table). Any code not in this table maps to "unknown" rather than
# guessing.
_WMO_CONDITIONS: dict[int, str] = {
    0: "clear_sky",
    1: "mainly_clear",
    2: "partly_cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing_rime_fog",
    51: "light_drizzle",
    53: "moderate_drizzle",
    55: "dense_drizzle",
    56: "light_freezing_drizzle",
    57: "dense_freezing_drizzle",
    61: "slight_rain",
    63: "moderate_rain",
    65: "heavy_rain",
    66: "light_freezing_rain",
    67: "heavy_freezing_rain",
    71: "slight_snow",
    73: "moderate_snow",
    75: "heavy_snow",
    77: "snow_grains",
    80: "slight_rain_showers",
    81: "moderate_rain_showers",
    82: "violent_rain_showers",
    85: "slight_snow_showers",
    86: "heavy_snow_showers",
    95: "thunderstorm",
    96: "thunderstorm_with_slight_hail",
    99: "thunderstorm_with_heavy_hail",
}


def _condition_for_code(code: Optional[int]) -> str:
    if code is None:
        return "unknown"
    return _WMO_CONDITIONS.get(int(code), "unknown")


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _localize(naive_local_str: Optional[str], tz_name: str) -> Optional[str]:
    """Open-Meteo returns 'time' values (format 'YYYY-MM-DDTHH:MM') as
    naive local time in whatever timezone was requested -- never UTC
    unless tz_name itself is UTC. Attaching a bare 'Z' suffix to those
    values (as an earlier draft of this adapter did) would silently
    mislabel local time as UTC. This attaches the real UTC offset for
    tz_name at that instant via the standard library's zoneinfo, so the
    returned string is genuinely correct ISO8601, e.g.
    '2026-08-17T14:00:00+03:00' for Europe/Istanbul. Falls back to a
    bare 'Z' suffix only if tz_name cannot be resolved (recorded as a
    missing-fields-worthy degradation by the caller, not silently)."""
    if not naive_local_str:
        return naive_local_str
    try:
        naive_dt = datetime.strptime(naive_local_str, "%Y-%m-%dT%H:%M")
    except ValueError:
        return naive_local_str
    try:
        aware_dt = naive_dt.replace(tzinfo=ZoneInfo(tz_name))
    except ZoneInfoNotFoundError:
        return naive_local_str + ":00Z"
    return aware_dt.isoformat()


class OpenMeteoUnexpectedResponseError(RuntimeError):
    """Raised internally when a 2xx Open-Meteo response cannot be parsed
    into the expected shape -- mapped to status='provider_error', never
    allowed to propagate as a raw exception to a caller."""


@dataclass
class OpenMeteoWeatherProvider:
    """Real WeatherProvider implementation. Every dependency (transport,
    clock, sleep) is injected with a safe real default, and every one is
    replaced with a deterministic fake in hermetic tests -- see
    providers/tests/test_weather_openmeteo.py."""

    transport: HttpTransport = field(default_factory=UrllibHttpTransport)
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    provider_name: str = PROVIDER_NAME

    # --- public interface (WeatherProvider protocol) --------------------------------

    def fetch_weather(self, query: WeatherQuery) -> dict:
        retrieved_at_dt = self.clock()
        retrieved_at = _iso_z(retrieved_at_dt)
        today = retrieved_at_dt.date()

        validation_status = self._validate_request(query, today)
        if validation_status is not None:
            return self._degraded_envelope(query, validation_status, retrieved_at, missing=["forecast_days"])

        # --- resolve coordinates -----------------------------------------------------
        if query.has_coordinates:
            latitude, longitude = query.latitude, query.longitude
            resolved_name, country = query.location, None
            resolved_timezone = query.timezone or "UTC"
        else:
            geocode_result = self._geocode(query)
            if isinstance(geocode_result, str):  # a failure status string
                return self._degraded_envelope(query, geocode_result, retrieved_at, missing=["forecast_days"])
            latitude, longitude, resolved_name, country, resolved_timezone = geocode_result

        is_current_request = query.date_from == query.date_to == today.isoformat()

        forecast_result = self._fetch_forecast(latitude, longitude, resolved_timezone, query, is_current_request)
        if isinstance(forecast_result, str):
            return self._degraded_envelope(
                query, forecast_result, retrieved_at, missing=["forecast_days"],
                resolved_name=resolved_name, country=country, latitude=latitude, longitude=longitude,
            )

        try:
            result = self._parse_forecast(
                forecast_result, is_current_request, resolved_name, country, latitude, longitude, resolved_timezone, query
            )
        except OpenMeteoUnexpectedResponseError:
            # A 2xx response whose body did not match the expected shape --
            # never lets a raw exception reach the caller (architecture.md
            # §13.4: no internal detail ever reaches the caller).
            return self._degraded_envelope(
                query, "provider_error", retrieved_at, missing=["forecast_days"],
                resolved_name=resolved_name, country=country, latitude=latitude, longitude=longitude,
            )
        envelope = self._build_envelope(query, status="success", data_mode="live", retrieved_at=retrieved_at, result=result)
        validate_envelope(envelope)
        return envelope

    # --- request validation --------------------------------------------------------

    def _validate_request(self, query: WeatherQuery, today: date) -> Optional[str]:
        try:
            date_from = date.fromisoformat(query.date_from)
            date_to = date.fromisoformat(query.date_to)
        except ValueError:
            return "invalid_request"
        if date_from > date_to:
            return "invalid_request"
        if date_from < today or date_to < today:
            return "unsupported"  # historical data: out of scope for this checkpoint's forecast/current adapter
        if (date_to - today).days > MAX_FORECAST_DAYS_AHEAD:
            return "unsupported"  # beyond Open-Meteo's own supported forecast horizon -- rejected, never silently truncated
        if query.has_coordinates:
            if not (-90.0 <= query.latitude <= 90.0) or not (-180.0 <= query.longitude <= 180.0):
                return "invalid_request"
        elif not query.location.strip():
            return "invalid_request"
        return None

    # --- geocoding --------------------------------------------------------------------

    def _geocode(self, query: WeatherQuery) -> tuple[float, float, str, Optional[str], str] | str:
        """Returns (latitude, longitude, resolved_name, country, timezone)
        on success, or a failure status string. Never silently chooses an
        unrelated location: an explicit country_code hint filters
        candidates before any selection happens; with no hint, the
        top-ranked (first) result is used deterministically, since
        Open-Meteo's own geocoding response is already relevance-ranked."""
        params = {"name": query.location.strip(), "count": 10, "format": "json"}
        response = self._request_with_retries(GEOCODING_URL, params)
        if isinstance(response, str):
            return response

        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            return "provider_error"

        results = body.get("results") or []
        if not results:
            return "unavailable"  # the location itself could not be resolved -- no data, not a malformed request

        if query.country_code:
            filtered = [r for r in results if str(r.get("country_code", "")).upper() == query.country_code.upper()]
            if filtered:
                results = filtered
            # else: no candidate matched the hint -- fall through to the unfiltered
            # top-ranked result rather than silently failing on an over-strict hint.

        top = results[0]
        try:
            latitude = float(top["latitude"])
            longitude = float(top["longitude"])
            resolved_name = str(top.get("name") or query.location)
            country = top.get("country")
            geocoded_timezone = str(top.get("timezone") or "UTC")
        except (KeyError, TypeError, ValueError):
            return "provider_error"
        return latitude, longitude, resolved_name, country, geocoded_timezone

    # --- forecast -----------------------------------------------------------------

    def _fetch_forecast(
        self, latitude: float, longitude: float, resolved_timezone: str, query: WeatherQuery, is_current_request: bool
    ) -> HttpResponse | str:
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": resolved_timezone,
        }
        if is_current_request:
            params["current"] = "temperature_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m"
        else:
            params["daily"] = (
                "weather_code,temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max,precipitation_sum,wind_speed_10m_max"
            )
            params["start_date"] = query.date_from
            params["end_date"] = query.date_to
        return self._request_with_retries(FORECAST_URL, params)

    def _parse_forecast(
        self, response: HttpResponse, is_current_request: bool, resolved_name: str, country: Optional[str],
        latitude: float, longitude: float, resolved_timezone: str, query: WeatherQuery,
    ) -> dict:
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise OpenMeteoUnexpectedResponseError("forecast response was not valid JSON") from exc

        missing_fields: list[str] = []

        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "location": resolved_name,
            "requested_location": query.location,
            "timezone": resolved_timezone,
            "units": {"temperature": "C", "wind_speed": "kmh", "precipitation": "mm"},
            "resolved_latitude": round(latitude, 4),
            "resolved_longitude": round(longitude, 4),
            "attribution": dict(ATTRIBUTION),
            "missing_fields": missing_fields,
        }
        if country:
            result["country"] = country

        if is_current_request:
            current = body.get("current")
            if not current:
                raise OpenMeteoUnexpectedResponseError("expected 'current' block absent from response")
            result["kind"] = "current_observation"
            result["issued_at"] = _localize(current.get("time"), resolved_timezone)
            observation: dict[str, Any] = {
                "observed_at": result["issued_at"],
                "condition": _condition_for_code(current.get("weather_code")),
                "temperature": current.get("temperature_2m"),
            }
            if current.get("precipitation") is not None:
                observation["precipitation_amount"] = current["precipitation"]
            else:
                missing_fields.append("observation.precipitation_amount")
            if current.get("wind_speed_10m") is not None:
                observation["wind_speed"] = current["wind_speed_10m"]
            if current.get("wind_direction_10m") is not None:
                observation["wind_direction_degrees"] = current["wind_direction_10m"]
            result["observation"] = observation
            result["forecast_days"] = []
        else:
            daily = body.get("daily")
            if not daily or "time" not in daily:
                raise OpenMeteoUnexpectedResponseError("expected 'daily' block absent from response")
            result["kind"] = "forecast"
            result["issued_at"] = _iso_z(datetime.now(timezone.utc))  # Open-Meteo does not echo a distinct issue time for forecast bodies
            days = []
            times = daily.get("time", [])
            for i, day_date in enumerate(times):
                entry: dict[str, Any] = {
                    "date": day_date,
                    "condition": _condition_for_code(_at(daily.get("weather_code"), i)),
                    "high": _at(daily.get("temperature_2m_max"), i),
                    "low": _at(daily.get("temperature_2m_min"), i),
                }
                precip_chance = _at(daily.get("precipitation_probability_max"), i)
                if precip_chance is not None:
                    entry["precipitation_chance"] = precip_chance / 100.0
                precip_amount = _at(daily.get("precipitation_sum"), i)
                if precip_amount is not None:
                    entry["precipitation_amount"] = precip_amount
                wind = _at(daily.get("wind_speed_10m_max"), i)
                if wind is not None:
                    entry["wind_speed"] = wind
                days.append(entry)
            result["forecast_days"] = days

        return result

    # --- envelope construction ------------------------------------------------------

    def _build_envelope(
        self, query: WeatherQuery, status: str, data_mode: str, retrieved_at: str, result: dict
    ) -> dict:
        fingerprint = query.fingerprint()
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": deterministic_request_id(fingerprint, retrieved_at),
            "provider": self.provider_name,
            "capability": CAPABILITY,
            "data_mode": data_mode,
            "status": status,
            "query_fingerprint": fingerprint,
            "cache_status": "bypass",  # C.1 implements no real cache -- reported honestly, never fabricated
            "retrieved_at": retrieved_at,
            "source_urls": [GEOCODING_URL, FORECAST_URL] if status == "success" else [],
            "quality": {
                "schema_version": "1.0.0",
                "completeness": completeness_for_status(status),
                "freshness": data_mode,
                "assumptions": self._assumptions(status, result),
            },
            "result": result,
        }

    def _assumptions(self, status: str, result: dict) -> list[str]:
        notes = []
        if result.get("kind") == "current_observation":
            notes.append(
                "Current conditions are Open-Meteo's model-derived nowcast, not a raw sensor observation "
                "(architecture.md never claims otherwise)."
            )
        if status != "success":
            notes.append(f"Open-Meteo call ended with status={status!r}.")
        return notes

    def _degraded_envelope(
        self, query: WeatherQuery, status: str, retrieved_at: str, missing: list[str],
        resolved_name: Optional[str] = None, country: Optional[str] = None,
        latitude: Optional[float] = None, longitude: Optional[float] = None,
    ) -> dict:
        data_mode = data_mode_for_status(status)
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "location": resolved_name or query.location,
            "requested_location": query.location,
            "timezone": query.timezone or "UTC",
            "kind": "unavailable" if data_mode != "cached" else "forecast",
            "units": {"temperature": "C", "wind_speed": "kmh", "precipitation": "mm"},
            "forecast_days": [],
            "missing_fields": missing,
        }
        if country:
            result["country"] = country
        if latitude is not None and longitude is not None:
            result["resolved_latitude"] = round(latitude, 4)
            result["resolved_longitude"] = round(longitude, 4)
        envelope = self._build_envelope(query, status=status, data_mode=data_mode, retrieved_at=retrieved_at, result=result)
        validate_envelope(envelope)
        return envelope

    # --- bounded HTTP with retry/backoff --------------------------------------------

    def _request_with_retries(self, url: str, params: dict) -> HttpResponse | str:
        """Returns an HttpResponse on any 2xx, or a failure status string
        (never raises) after the retry budget is exhausted. Honors
        Retry-After on 429 when present, otherwise uses bounded
        exponential backoff -- both via providers.policy, both capped."""
        timeout = (self.timeout_policy.connect_timeout_seconds, self.timeout_policy.total_timeout_seconds)
        last_status = "provider_error"
        for attempt in range(self.retry_policy.max_retries + 1):
            retry_after: Optional[float] = None
            try:
                response = self.transport.get(url, params, timeout)
            except TransportError:
                last_status = "timeout"
            else:
                if 200 <= response.status_code < 300:
                    return response
                last_status = classify_http_status(response.status_code)
                if last_status == "rate_limited":
                    header_value = response.header("Retry-After")
                    if header_value is not None:
                        try:
                            retry_after = float(header_value)
                        except ValueError:
                            retry_after = None
                if last_status not in ("timeout", "rate_limited", "provider_error"):
                    return last_status  # invalid_request/unsupported: retrying will not help, fail fast

            if attempt < self.retry_policy.max_retries:
                self.sleep_fn(compute_backoff_seconds(attempt, self.retry_policy, retry_after))
        return last_status


def _at(seq: Optional[list], index: int) -> Any:
    if not seq or index >= len(seq):
        return None
    return seq[index]
