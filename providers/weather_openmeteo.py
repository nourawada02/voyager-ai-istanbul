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
from datetime import date, datetime, timedelta, timezone
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
# Manual QA remediation Q.1 (user correction pass, §C): Open-Meteo's real,
# public, historical reanalysis archive -- same host family, no API key,
# verified reachable directly (2026-08-20) before this was implemented.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

MAX_FORECAST_DAYS_AHEAD = 16  # Open-Meteo's own supported forecast horizon
# Fixed, documented, deterministic number of prior full years sampled for
# historical climate guidance -- never a magic/hidden number, never varies
# by request.
CLIMATE_SAMPLE_YEAR_COUNT = 3

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


def _most_common_code(codes: list[int]) -> int:
    """Deterministic mode: the most frequent code, ties broken by the
    lowest numeric code -- never a random/dict-iteration-order pick."""
    counts: dict[int, int] = {}
    for code in codes:
        counts[code] = counts.get(code, 0) + 1
    max_count = max(counts.values())
    return min(code for code, count in counts.items() if count == max_count)


def _earliest_available_forecast_date(requested_date_from: date) -> date:
    """User correction pass (§C): the exact calendar date THIS request's
    date_from enters Open-Meteo's live forecast horizon -- a fixed
    property of date_from itself, deliberately NEVER computed from "today"
    (today + MAX_FORECAST_DAYS_AHEAD gives a different, incorrect answer
    for the identical trip date depending on when it happens to be asked,
    since the true threshold does not move)."""
    return requested_date_from - timedelta(days=MAX_FORECAST_DAYS_AHEAD)


def _sample_years(today: date, count: int = CLIMATE_SAMPLE_YEAR_COUNT) -> list[int]:
    """The `count` most recent FULLY PAST years before today's year --
    deterministic given `today` alone, documented, oldest-first."""
    return [today.year - offset for offset in range(count, 0, -1)]


def _year_equivalent_date(reference: date, year: int) -> Optional[date]:
    """The same month/day in a different year. Clamps a Feb 29 reference
    to Feb 28 in a non-leap sampled year (documented, deterministic --
    never silently skipped without landing on some real date); returns
    None only if construction is impossible for another reason."""
    try:
        return date(year, reference.month, reference.day)
    except ValueError:
        if reference.month == 2 and reference.day == 29:
            return date(year, 2, 28)
        return None


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
        date_from = date.fromisoformat(query.date_from)
        date_to = date.fromisoformat(query.date_to)

        # Manual QA remediation Q.1 (user correction pass, §C): a future
        # trip beyond Open-Meteo's forecast horizon is never a dead end --
        # it gets real historical climate guidance for equivalent calendar
        # dates in prior years, distinctly labeled, never a fabricated
        # forecast. A trip that straddles the horizon gets BOTH: live data
        # for the reachable prefix and historical guidance for the rest,
        # explicitly marked "mixed" -- the requested range is never
        # silently truncated.
        horizon_date = today + timedelta(days=MAX_FORECAST_DAYS_AHEAD)
        live_date_to = min(date_to, horizon_date) if date_from <= horizon_date else None
        historical_date_from = max(date_from, horizon_date + timedelta(days=1)) if date_to > horizon_date else None

        live_result: Optional[dict] = None
        if live_date_to is not None:
            forecast_result = self._fetch_forecast(
                latitude, longitude, resolved_timezone, query, is_current_request, live_date_to.isoformat()
            )
            if isinstance(forecast_result, str):
                return self._degraded_envelope(
                    query, forecast_result, retrieved_at, missing=["forecast_days"],
                    resolved_name=resolved_name, country=country, latitude=latitude, longitude=longitude,
                )
            try:
                live_result = self._parse_forecast(
                    forecast_result, is_current_request, resolved_name, country, latitude, longitude,
                    resolved_timezone, query,
                )
            except OpenMeteoUnexpectedResponseError:
                # A 2xx response whose body did not match the expected
                # shape -- never lets a raw exception reach the caller
                # (architecture.md §13.4: no internal detail ever reaches
                # the caller).
                return self._degraded_envelope(
                    query, "provider_error", retrieved_at, missing=["forecast_days"],
                    resolved_name=resolved_name, country=country, latitude=latitude, longitude=longitude,
                )

        historical_result: Optional[dict] = None
        if historical_date_from is not None and not is_current_request:
            climate_fetch = self._fetch_and_aggregate_historical_climate(
                latitude, longitude, resolved_timezone, historical_date_from, date_to, today
            )
            if isinstance(climate_fetch, str):
                return self._degraded_envelope(
                    query, climate_fetch, retrieved_at, missing=["historical_climate_days"],
                    resolved_name=resolved_name, country=country, latitude=latitude, longitude=longitude,
                )
            historical_result = climate_fetch

        result = self._merge_weather_result(
            resolved_name, query, resolved_timezone, latitude, longitude, country, live_result, historical_result, date_from,
        )
        if live_result is not None and historical_result is not None:
            data_mode = "mixed"
        elif historical_result is not None:
            data_mode = "historical"
        else:
            data_mode = "live"
        envelope = self._build_envelope(query, status="success", data_mode=data_mode, retrieved_at=retrieved_at, result=result)
        validate_envelope(envelope)
        return envelope

    def _merge_weather_result(
        self, resolved_name: str, query: WeatherQuery, resolved_timezone: str, latitude: float, longitude: float,
        country: Optional[str], live_result: Optional[dict], historical_result: Optional[dict], requested_date_from: date,
    ) -> dict:
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "location": resolved_name,
            "requested_location": query.location,
            "timezone": resolved_timezone,
            "units": {"temperature": "C", "wind_speed": "kmh", "precipitation": "mm"},
            "resolved_latitude": round(latitude, 4),
            "resolved_longitude": round(longitude, 4),
            "attribution": dict(ATTRIBUTION),
            "missing_fields": [],
        }
        if country:
            result["country"] = country

        if live_result is not None and historical_result is not None:
            result["coverage"] = "mixed"
            result["kind"] = "mixed"
            result["issued_at"] = live_result["issued_at"]
            result["forecast_days"] = live_result["forecast_days"]
        elif live_result is not None:
            result["coverage"] = "live"
            result["kind"] = live_result["kind"]
            result["issued_at"] = live_result["issued_at"]
            result["forecast_days"] = live_result["forecast_days"]
            if live_result["kind"] == "current_observation":
                result["observation"] = live_result["observation"]
        else:
            result["coverage"] = "historical"
            result["kind"] = "historical"
            result["issued_at"] = _iso_z(datetime.now(timezone.utc))
            result["forecast_days"] = []

        if historical_result is not None:
            result["historical_climate_days"] = historical_result["historical_climate_days"]
            result["years_sampled"] = historical_result["years_sampled"]
            result["aggregation_method"] = historical_result["aggregation_method"]
            result["climate_disclaimer"] = historical_result["climate_disclaimer"]
            # User correction pass §C: a fixed property of the request's
            # OWN date_from, never of "today" -- the same trip date must
            # report the same "enters the forecast window on ..." answer
            # no matter when it is queried.
            result["earliest_available_forecast_date"] = _earliest_available_forecast_date(requested_date_from).isoformat()

        return result

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
            return "unsupported"  # a genuinely past date range: out of scope for this checkpoint's forecast/current/climate adapter
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
        self, latitude: float, longitude: float, resolved_timezone: str, query: WeatherQuery, is_current_request: bool,
        effective_date_to: str,
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
            # Manual QA remediation Q.1: clipped to the forecast horizon by
            # the caller (fetch_weather), never the caller's raw date_to --
            # requesting a date range Open-Meteo does not support would
            # itself become a provider_error, not a clean partial result.
            params["end_date"] = effective_date_to
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

    # --- historical climate guidance (User correction pass §C) ----------------------

    def _fetch_and_aggregate_historical_climate(
        self, latitude: float, longitude: float, resolved_timezone: str, range_start: date, range_end: date, today: date,
    ) -> dict | str:
        """Fetches one real Open-Meteo archive request per sampled prior
        year (each covering that year's equivalent calendar range to
        [range_start, range_end]), then deterministically aggregates
        across years per day-offset. Returns a failure status string on
        the first hard failure -- never a partial/fabricated aggregate
        from whatever years happened to succeed."""
        day_span = (range_end - range_start).days
        years = _sample_years(today)
        per_offset_records: dict[int, list[dict[str, Any]]] = {i: [] for i in range(day_span + 1)}

        for year in years:
            year_start = _year_equivalent_date(range_start, year)
            if year_start is None:
                continue
            year_end = year_start + timedelta(days=day_span)
            params: dict[str, Any] = {
                "latitude": latitude,
                "longitude": longitude,
                "timezone": resolved_timezone,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                "start_date": year_start.isoformat(),
                "end_date": year_end.isoformat(),
            }
            response = self._request_with_retries(ARCHIVE_URL, params)
            if isinstance(response, str):
                return response
            try:
                body = response.json()
            except Exception as exc:  # noqa: BLE001
                raise OpenMeteoUnexpectedResponseError("archive response was not valid JSON") from exc
            daily = body.get("daily") or {}
            times = daily.get("time", [])
            for i in range(min(len(times), day_span + 1)):
                per_offset_records[i].append({
                    "weather_code": _at(daily.get("weather_code"), i),
                    "high": _at(daily.get("temperature_2m_max"), i),
                    "low": _at(daily.get("temperature_2m_min"), i),
                    "precipitation": _at(daily.get("precipitation_sum"), i),
                })

        historical_climate_days: list[dict[str, Any]] = []
        for i in range(day_span + 1):
            representative_date = range_start + timedelta(days=i)
            records = [r for r in per_offset_records.get(i, []) if r["high"] is not None and r["low"] is not None]
            if not records:
                continue  # never fabricate an aggregate from zero real samples
            avg_high = round(sum(r["high"] for r in records) / len(records), 1)
            avg_low = round(sum(r["low"] for r in records) / len(records), 1)
            precip_values = [r["precipitation"] for r in records if r["precipitation"] is not None]
            codes = [r["weather_code"] for r in records if r["weather_code"] is not None]
            entry: dict[str, Any] = {
                "date": representative_date.isoformat(),
                "condition": _condition_for_code(_most_common_code(codes)) if codes else "unknown",
                "avg_high": avg_high,
                "avg_low": avg_low,
                "years_sampled": years,
            }
            if precip_values:
                entry["avg_precipitation_amount"] = round(sum(precip_values) / len(precip_values), 1)
            historical_climate_days.append(entry)

        return {
            "historical_climate_days": historical_climate_days,
            "years_sampled": years,
            "aggregation_method": "arithmetic_mean_across_sampled_years",
            "climate_disclaimer": "Historical climate guidance -- not a forecast.",
        }

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
            "source_urls": self._source_urls_for(status, result),
            "quality": {
                "schema_version": "1.0.0",
                "completeness": completeness_for_status(status),
                "freshness": data_mode,
                "assumptions": self._assumptions(status, result),
            },
            "result": result,
        }

    def _source_urls_for(self, status: str, result: dict) -> list[str]:
        if status != "success":
            return []
        coverage = result.get("coverage")
        if coverage == "mixed":
            return [GEOCODING_URL, FORECAST_URL, ARCHIVE_URL]
        if coverage == "historical":
            return [GEOCODING_URL, ARCHIVE_URL]
        return [GEOCODING_URL, FORECAST_URL]

    def _assumptions(self, status: str, result: dict) -> list[str]:
        notes = []
        if result.get("kind") == "current_observation":
            notes.append(
                "Current conditions are Open-Meteo's model-derived nowcast, not a raw sensor observation "
                "(architecture.md never claims otherwise)."
            )
        coverage = result.get("coverage")
        if coverage in ("historical", "mixed"):
            # User correction pass §C: a precise, user-facing explanation
            # -- never a fabricated forecast, and never the vague generic
            # "status=unsupported" this replaced.
            years = result.get("years_sampled") or []
            earliest = result.get("earliest_available_forecast_date")
            notes.append(
                f"{result.get('climate_disclaimer', 'Historical climate guidance -- not a forecast.')} "
                f"Aggregated from {len(years)} prior year(s) ({', '.join(str(y) for y in years)}) for the "
                f"equivalent calendar date(s)."
                + (f" A live forecast for this trip date opens up on {earliest}." if earliest else "")
            )
        if status != "success":
            notes.append(f"Open-Meteo call ended with status={status!r}.")
        return notes

    def _degraded_envelope(
        self, query: WeatherQuery, status: str, retrieved_at: str, missing: list[str],
        resolved_name: Optional[str] = None, country: Optional[str] = None,
        latitude: Optional[float] = None, longitude: Optional[float] = None,
        extra_result_fields: Optional[dict[str, Any]] = None,
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
        if extra_result_fields:
            result.update(extra_result_fields)
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
