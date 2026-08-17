# 0010. Phase 4 Checkpoint C.1 — real Open-Meteo weather adapter

- **Status:** Accepted
- **Date:** 2026-08-17

## 1. Scope

Implements the first real, non-fake live-data provider in this project: `OpenMeteoWeatherProvider` (`providers/weather_openmeteo.py`), satisfying the `WeatherProvider` interface `providers/weather.py` and ADR 0009 already established. Current conditions and short-range daily forecast only. No System A, LangGraph, Qwen, web search, flights, frontend, Compose, booking, or System B change is part of this checkpoint.

## 2. Endpoint ownership and no-auth decision

Exactly two fixed, hardcoded endpoint constants are ever contacted — never built from user-controlled input:

- `https://geocoding-api.open-meteo.com/v1/search` — location resolution
- `https://api.open-meteo.com/v1/forecast` — current/forecast weather

`providers/http_transport.py::UrllibHttpTransport` additionally enforces an explicit host allowlist (`{geocoding-api.open-meteo.com, api.open-meteo.com}`) as defense-in-depth, independent of whatever URL the adapter itself constructs.

Open-Meteo's non-commercial API requires no API key, no account, and no billing (`https://open-meteo.com/en/docs`, confirmed by direct fetch during ADR 0009's research and re-confirmed operationally by this checkpoint's live gate, §8 below). This is why it was selected as the first real adapter: lowest-risk path to real, live evidence with zero credential-handling surface.

## 3. Attribution / licence requirement

Open-Meteo's data is CC BY 4.0 licensed and requires attribution. Every envelope this adapter produces — success or degraded — carries a structured `attribution` object (`WeatherResult.schema.json` 1.1.0, added this checkpoint):

```json
{"license": "CC BY 4.0", "notice": "Weather data by Open-Meteo.com (https://open-meteo.com/), licensed under CC BY 4.0."}
```

Never fabricated when absent — this adapter always sets it, since Open-Meteo's terms always require it for any real call.

## 4. Model-derived vs. observed wording

Open-Meteo's `current` weather block is a model/nowcast blend, not raw sensor telemetry. Every `kind="current_observation"` envelope's `quality.assumptions` explicitly states: *"Current conditions are Open-Meteo's model-derived nowcast, not a raw sensor observation"* — this project never claims otherwise (Checkpoint C.1 explicit requirement #7), proven by `providers/tests/test_weather_openmeteo.py::test_current_conditions_discloses_model_derived_not_sensor_measured`.

## 5. Timeout / retry behavior

Uses the shared `providers.policy.RetryPolicy`/`TimeoutPolicy` unchanged: bounded exponential backoff, `Retry-After` honored (and still capped) on HTTP 429, both timeout and connect bounds injectable. `classify_http_status` (already shared, C.0) maps HTTP responses to the shared status vocabulary. A transport-level failure (DNS, connection refused, socket timeout) — as opposed to a well-formed HTTP error response — is caught and mapped to `status="timeout"`, never allowed to propagate as a raw exception. Clock and sleep are both injected (`clock: Callable[[], datetime]`, `sleep_fn: Callable[[float], None]`), defaulting to real `datetime.now`/`time.sleep` but replaced with deterministic fakes in every hermetic test — no real sleep, no real clock read, ever, in `providers/tests/`.

## 6. Supported scope

- Current conditions (`kind="current_observation"`) when the requested date range is exactly today.
- Daily forecast (`kind="forecast"`) for a future date range up to `MAX_FORECAST_DAYS_AHEAD = 16` days out (Open-Meteo's own supported forecast horizon) — matching `WeatherResult.forecast_days`.
- Location resolution via geocoding when coordinates are not already supplied on `WeatherQuery`; direct lat/lon bypasses geocoding entirely.
- Deterministic geocoding disambiguation: an explicit `country_code` hint filters candidates before selection; with no hint, the top-ranked (first) result is used, since Open-Meteo's own geocoding response is already relevance-ranked — proven deterministic across independent provider instances by `test_geocoding_multiple_results_without_hint_deterministically_picks_top_ranked`.
- Genuinely timezone-aware timestamps: Open-Meteo returns naive local-time strings; this adapter attaches the real UTC offset for the resolved IANA timezone via the standard library's `zoneinfo` (e.g. `2026-08-17T16:00:00+03:00` for Europe/Istanbul in August, when Turkey observes no DST) — never a bare `Z` suffix mislabeling local time as UTC. This was a real bug caught and fixed during this checkpoint's own hermetic testing, not merely designed correctly on paper.

## 7. Unsupported / no guarantees

- **No historical data.** A request whose date range falls entirely or partly in the past is rejected with `status="unsupported"`, never silently served from a different (historical) Open-Meteo endpoint this adapter does not implement.
- **No commercial-tier guarantee.** This adapter only exercises Open-Meteo's non-commercial, no-auth surface; nothing here should be read as validating Open-Meteo's commercial SLA/rate limits.
- **No caching.** `cache_status` is always reported honestly as `"bypass"` with `cache_age_seconds` correctly absent — no real cache is implemented in C.1, and this is never dressed up as a cache hit.
- **No booking, payment, or reservation semantics of any kind** — weather has no such semantics to begin with, but this is stated for completeness with the rest of this project's invariants.

## 8. Live-gate evidence

Real network call, made once, bounded (one geocoding request + one forecast request), no credential used:

```
$ VOYAGER_LIVE_WEATHER_GATE=1 python -m pytest providers/tests/test_weather_openmeteo_live.py -v -s
providers/tests/test_weather_openmeteo_live.py::test_live_open_meteo_istanbul_current_weather
LIVE OPEN-METEO GATE: provider=open-meteo status=success location=Istanbul timezone=Europe/Istanbul
issued_at=2026-08-17T16:00:00+03:00 condition=mainly_clear temperature_c=27.3
attribution=Weather data by Open-Meteo.com (https://open-meteo.com/), licensed under CC BY 4.0.
PASSED
1 passed in 2.34s
```

Confirmed by this run: provider identity (`open-meteo`), `retrieved_at` present, `source_urls` populated with both real endpoints, `timezone` correctly resolved to `Europe/Istanbul` (not a guess — genuinely returned by Open-Meteo's geocoding response), forecast/current date correctly timezone-offset (`+03:00`, proving the `zoneinfo`-based localization works against real data, not just fixture data), `freshness`/`data_mode="live"`, and attribution present with the exact required CC BY 4.0 notice. Full two-stage schema validation (`ProviderResponseEnvelope` + `WeatherResult`) passed against this real response — not just against synthetic fixtures. The live gate passed on the first attempt; no fix-and-rerun cycle was needed.

## 9. Known limitations

- Daily forecast dates are the calendar dates Open-Meteo itself returns for the resolved location's own timezone (via the `timezone` request parameter) — correct for the common case, but a location whose IANA timezone could not be resolved falls back to a bare-`Z`-suffixed timestamp rather than a genuine local offset (see `_localize`'s documented fallback in `weather_openmeteo.py`); not exercised by the live gate since Istanbul always resolves cleanly.
- Geocoding disambiguation without an explicit `country_code` hint always takes Open-Meteo's own top-ranked result; this is deterministic but not necessarily "the place the user meant" for a genuinely ambiguous free-text query — a future checkpoint could add stronger disambiguation (e.g. surfacing multiple candidates back to the caller) if this proves to be a real problem.
- No retry-exhaustion behavior was exercised against the real API in the live gate (only against `FakeHttpTransport` in hermetic tests) — Open-Meteo did not rate-limit or error during this checkpoint's single live call, so real-world Retry-After behavior remains hermetically-proven only, not live-proven.
- This adapter is not yet wired into any orchestrator (System A does not exist yet) — it is invoked directly, by construction, only from tests.
