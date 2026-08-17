# 0012. Phase 4 Checkpoint C.3 — real SerpApi Google Flights adapter

- **Status:** Accepted.
- **Date:** 2026-08-17

## 1. Scope

Implements the first real, non-fake `FlightSearchProvider` in this project: `SerpApiFlightSearchProvider` (`providers/flights_serpapi.py`), satisfying the existing `providers/flights.py` interface and the canonical `FlightSearchResult`/`FlightOption` contracts unchanged. Reuses `providers/http_transport.py` (including its Checkpoint C.2 `secret_params` channel and timeout-semantics repair), `providers/policy.py`, `providers/redaction.py`, `providers/fingerprint.py`, and `providers/validation.py` directly — none of these were modified for this checkpoint.

## 2. Provider selection

SerpApi's Google Flights API was already named as the recommended flight-search candidate in ADR 0009 §7 ("no official Google Flights API exists; this is third-party structured access to Google Flights search-snapshot data"). Reusing SerpApi (already integrated in Checkpoint C.2 for web-search evidence) rather than introducing a second vendor keeps this project to one paid-API relationship, one host allowlist entry, and one proven secret-isolation mechanism.

## 3. V1 functional boundary: exact-date, one-way only

`FlightSearchQuery` (`providers/flights.py`) is reused unchanged — no parallel request model was introduced. For V1:

- `origin`/`destination` must match `^[A-Z]{3}$` (the same pattern `FlightOption.schema.json` itself requires) — anything else is `status="invalid_request"`, never silently passed through.
- `depart_date_from` must equal `depart_date_to`; a genuine date range (`depart_date_from != depart_date_to`) is `status="unsupported"` — never silently narrowed to one date.
- Any `return_date_from`/`return_date_to` present makes the whole request `status="unsupported"` — round-trip and multi-city are explicitly deferred (§9), never attempted with `type=1`/`type=3`.
- A malformed date string is `invalid_request`; a well-formed but past date is `unsupported` (mirrors the identical distinction already established in `providers/weather_openmeteo.py`).
- `passenger_count` maps directly to SerpApi's `adults` parameter — `FlightSearchQuery` does not distinguish passenger types, so no other passenger-count parameter is ever sent.
- `cabin_class` (`economy`/`premium_economy`/`business`/`first`, the same enum `FlightSearchResult.schema.json` already defines) maps to SerpApi's `travel_class` (`1`–`4`); a `cabin_class` outside that enum is `invalid_request`, never coerced to Economy.
- `FlightSearchQuery` has no nonstop-only preference field, so none is mapped — `stops` is never sent as a request parameter (`test_no_nonstop_param_since_query_has_no_such_field`).
- Fixed request configuration, never escalated automatically: `type=2` (one-way), `currency=TRY`, `output=json`, `hl=en`, `gl=tr`, `deep_search=false`, `no_cache=false` (allows a matching SerpApi-side cache hit, conserving quota — the same deliberate choice as Checkpoint C.2's web-evidence adapter).
- Synchronous mode only: exactly one request per `search_flights()` call. `booking_token` and `departure_token` are never read from a response and never used to issue a second request — this adapter has no code path that could do either (§8).

## 4. Snapshot, not inventory

Every successful result carries `is_snapshot=true`, `searched_at`, and a `quality.assumptions` entry stating plainly: "Search-time snapshot only; never a booking, seat-availability, or fare guarantee." `providers/tests/test_flights_serpapi.py::test_no_booking_or_availability_claim_in_envelope` asserts none of `booked`/`reservation_confirmed`/`payment`/`purchase`/`seat_available`/`confirmed` ever appears anywhere in a serialized envelope, mirroring the identical structural proof already established for the fake provider in `providers/tests/test_flights_provider.py`.

## 5. Timezone resolution — no fabricated timestamps

SerpApi's documented segment times (`departure_airport.time`/`arrival_airport.time`) are naive local strings (`"YYYY-MM-DD HH:MM"`) with no UTC offset — appending a bare `Z` would silently mislabel local time as UTC, exactly the mistake already corrected once in this project (Open-Meteo, Checkpoint C.1). This adapter uses an injectable `timezone_resolver: Callable[[str], Optional[str]]`, defaulting to a small, explicit IATA-to-timezone map covering exactly the airports this checkpoint's own live gate needs: `IST`/`SAW` → `Europe/Istanbul`, `BEY` → `Asia/Beirut`. `zoneinfo.ZoneInfo` attaches the real UTC offset for that airport's timezone at that instant, producing genuinely offset-aware RFC 3339 timestamps (e.g. `2026-09-10T08:30:00+03:00`), never a naive value.

An airport code outside this map, or a segment time that fails to parse, resolves to `None` from `_localize_segment_time` — the *entire itinerary* containing it is skipped (not just that one field), with a sanitized warning recorded in `result.limitations` (e.g. `"skipped an itinerary: unresolvable timezone or malformed time for airport BEY/XYZ"` — airport codes only, never a raw provider payload). No broad geolocation/airport-database service or additional paid API was added; this is a deliberately small, checkpoint-scoped map (§9).

## 6. Normalization

Only `best_flights` and `other_flights` are consumed, combined in that deterministic order (`test_best_flights_ordered_before_other_flights`) and capped locally at `HARD_MAX_RESULTS = 3` — never above 8, and never derived from a provider request hint (`google_flights` has no per-request result-count parameter to rely on in the first place).

Per itinerary: the first segment supplies the top-level `origin`/`depart_at`; the last segment supplies `destination`/`arrive_at`; `stops = segment_count - 1`; `legs` is populated (and schema-required `minItems: 2`) only for itineraries with more than one segment (`test_connecting_flight_has_legs_and_correct_stops`), omitted entirely for a direct flight (`test_direct_flight_has_zero_stops_no_legs`). `total_duration` is used for `duration_minutes` only when it is a valid nonnegative integer — never derived or guessed otherwise. `carrier` is always the provider-supplied `airline` string; `flight_number` is included only when SerpApi supplies one that also matches `FlightOption.schema.json`'s own pattern (spaces stripped, e.g. `"TK 823"` → `"TK823"`), otherwise omitted. `baggage_limitations`/`fare_limitations` are never populated in V1 — SerpApi's synchronous, non-`booking_token` response does not reliably supply structured data for either, and fabricating one would violate the schema's own "present only when the provider actually supplied it" contract; this is a documented, deliberate gap, not an oversight.

Price: `Decimal`-exact TRY-major-to-minor conversion (`_price_to_minor_units`, architecture.md's binary-float prohibition) — a missing, zero, negative, or unparseable price causes that one candidate to be skipped (`"skipped an itinerary missing a usable price"`), never a fabricated price. `flight_id` is generated deterministically (`sha256` over the itinerary's own normalized origin/destination/timestamps/carrier/stops/price/legs, `providers.fingerprint.sha256_hex`) — two independently-run searches with identical input produce byte-identical `flight_id`s (`test_flight_id_is_deterministic_and_stable`), and this same mechanism doubles as deduplication: equivalent itineraries collapse to one option by construction (`test_duplicate_itineraries_deduplicated`).

A malformed itinerary entry (non-object, missing segments, an incomplete segment, an unresolvable timezone, or no usable price) is skipped with a recorded `result.limitations` warning rather than failing the whole request (`test_malformed_itinerary_entries_are_skipped_not_fatal`) — the search itself still reports `status="success"` when at least the call succeeded, even if every individual candidate happened to be unusable; this mirrors the identical honest-degradation pattern in `providers/web_evidence_serpapi.py`.

## 7. Degraded-envelope schema safety (a real bug found and fixed during hermetic testing)

`FlightSearchResult.schema.json` itself validates `origin`/`destination` (pattern), `depart_date_from`/`depart_date_to` (date format), `passenger_count` (1–12), and `cabin_class` (enum) — so a degraded envelope for e.g. `status="invalid_request"` must still pass that same schema. Echoing the caller's raw, already-known-invalid values back (`origin="ZZZZ"`, `depart_date_from="not-a-date"`, `passenger_count=0`, `cabin_class="super_deluxe"`) would make the *rejection envelope itself* fail mandatory validation — found via this checkpoint's own `test_provider_never_returns_an_envelope_that_fails_mandatory_validation`. Fixed with `_sanitize_iata`/`_sanitize_date`/`_sanitize_passenger_count` (fixed, clearly-not-a-real-search placeholders: `"XXX"`, `"1970-01-01"`, `1`) and by only including `cabin_class` in the result when it is a genuinely recognized value. Every sanitizer is a no-op on the success path, since `_validate_request` only lets a query through once these same values are already well-formed — `status` and `result.limitations` already state exactly why a request was rejected, so nothing is silently masked by the placeholder.

## 8. Secret handling

Identical channel to Checkpoint C.2 (`providers/http_transport.py`'s `secret_params`, unchanged): `SERPAPI_API_KEY` is read once in `__post_init__`, wrapped immediately in `SecretString`, and `.reveal()`-ed exactly once inside `_request_with_retries` to build `secret_params = {"api_key": ...}` — never merged into the public `params` dict, never present in `FakeHttpTransport.call_log`, never in an exception message (`TransportError` names only the fixed hostname), never in `query.fingerprint()` (which only ever sees `FlightSearchQuery.normalized()`), and never in a `FlightOption.provenance` record (`source_urls: []` — no legitimate public per-itinerary URL exists to include without risking a `booking_token`-bearing link, so none is fabricated). Proven by `test_key_sent_via_secret_params_never_public_params`, `test_api_key_never_appears_in_envelope_output`, `test_api_key_never_appears_in_transport_logs_or_errors`, and `test_api_key_never_appears_in_query_fingerprint`. Every credential value in the test file is generated fresh at test-run time (`secrets.token_hex`) — no key-shaped literal is committed anywhere.

## 9. Explicitly deferred

Round-trip and multi-city search (`return_date_from`/`return_date_to` on `FlightSearchQuery` always yields `status="unsupported"`); any booking, payment, seat-selection, or fare-lock action; a `booking_token`/`departure_token` follow-up request of any kind; a broader IATA-airport-to-timezone database beyond the three airports this checkpoint's live gate needs; System A, LangGraph, Qwen, the Streamlit frontend, root Docker Compose, and MCP exposure of this capability. None of these were implemented, and none of `services/istanbul-expert-b` (System B), `rag/`, `ml/`, `data/`, or `docs/architecture-plan.pdf` were touched by this checkpoint.

## 10. Failure semantics

Every non-success status (`unavailable` for a missing key, `invalid_request`, `unsupported`, `timeout`, `rate_limited` — including SerpApi's own quota-phrasing top-level errors, mapped the same way Checkpoint C.2 already established — `provider_error`, `cancelled`) returns a schema-valid, empty-`options` envelope with `data_mode="unavailable"` (never `"estimated"`) and a `result.limitations` entry naming the status. Hermetically proven for all of the above in `providers/tests/test_flights_serpapi.py` (60 tests).

## 11. Live-gate evidence

One bounded, keyed live attempt (`VOYAGER_LIVE_FLIGHT_SEARCH_GATE=1 python -m pytest providers/tests/test_flights_serpapi_live.py -v -s`): route `BEY → IST`, one adult, economy, one-way, an exact future date, `max 3` locally-returned results, cache allowed, no `deep_search`, the adapter's own single internal retry permitted (not needed).

```
SERPAPI_API_KEY present: true
LIVE SERPAPI GOOGLE FLIGHTS GATE: provider=serpapi_google_flights status=success item_count=3
route=BEY->IST currency=TRY schema_valid=True runtime_seconds=2.89
PASSED
```

Both `ProviderResponseEnvelope` and `FlightSearchResult` validated with zero errors; every returned `FlightOption` additionally validated individually against `FlightOption.schema.json`; every timestamp offset-aware; every price in TRY minor units; no booking/payment/availability term present in the serialized envelope; no API key, complete request URL, or raw response body ever printed.
