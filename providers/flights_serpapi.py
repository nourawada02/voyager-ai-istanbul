"""Real SerpApi (Google Flights) flight-search adapter (Checkpoint Phase
4 C.3; docs/adr/0012-phase4-checkpointc3-serpapi-google-flights-adapter.md).
The first real, non-fake `FlightSearchProvider` implementation in this
project -- shares its host, secret-isolation channel, and
timeout/retry-policy design with the Checkpoint C.2 web-evidence adapter
(`providers/web_evidence_serpapi.py`), reused directly rather than
reimplemented.

Exactly one fixed endpoint, one fixed engine, is ever contacted:

  https://serpapi.com/search   (engine=google_flights)

V1 functional boundary: exact-date, one-way search only. A caller
supplying a genuine date range (`depart_date_from != depart_date_to`) or
any return-date field gets an honest `status="unsupported"` -- never a
silently-chosen single date, and never a synthesized round-trip search.
Round-trip, multi-city, booking, and payment are all explicitly out of
scope for this checkpoint (see ADR 0012).

Every result is a search-time snapshot (FlightSearchResult.is_snapshot:
const true) -- never a claim that a seat is available now, that a fare is
guaranteed, that an itinerary is booked, or that payment occurred. This
adapter never reads, stores, or forwards a provider `booking_token` or
`departure_token`, and never issues a second request to resolve one --
only the single, synchronous `best_flights`/`other_flights` search
response is ever consumed.

Authentication: identical channel to Checkpoint C.2 -- the key is read
once from `SERPAPI_API_KEY` (or an explicitly injected `api_key`),
wrapped immediately in `providers.redaction.SecretString`, and revealed
exactly once, to build the transport call's `secret_params` argument,
which `providers.http_transport.UrllibHttpTransport.get()` merges into
the outgoing URL only inside its own request construction -- never
recorded in a call log, a header log, an exception message, a request
fingerprint, a fixture, or a provenance record.
"""

from __future__ import annotations

import json as _json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from providers.fingerprint import deterministic_request_id, sha256_hex
from providers.flights import CAPABILITY, RESULT_SCHEMA_VERSION, SCHEMA_VERSION, FlightSearchQuery
from providers.http_transport import HttpResponse, HttpTransport, TransportError, UrllibHttpTransport
from providers.policy import (
    RetryPolicy,
    TimeoutPolicy,
    classify_http_status,
    completeness_for_status,
    compute_backoff_seconds,
    data_mode_for_status,
)
from providers.redaction import SecretString
from providers.validation import validate_envelope

FLIGHTS_URL = "https://serpapi.com/search"

PROVIDER_NAME = "serpapi_google_flights"

HARD_MAX_RESULTS = 3  # strict local ceiling for this checkpoint -- never a provider request hint, never above 8

# SerpApi's own documented travel_class values (google_flights engine):
# 1=Economy, 2=Premium economy, 3=Business, 4=First -- mirrors
# FlightSearchResult.schema.json's own cabin_class enum exactly. A
# cabin_class outside this map is rejected as invalid_request, never
# silently coerced to Economy.
_CABIN_CLASS_TO_TRAVEL_CLASS = {
    "economy": 1,
    "premium_economy": 2,
    "business": 3,
    "first": 4,
}

# Keyword fragments SerpApi's own documented top-level error strings use
# for quota/rate-limit conditions -- classified as 'rate_limited', never
# 'provider_error'; identical logic to the Checkpoint C.2 web-evidence
# adapter's own copy (each real adapter module stays self-contained
# rather than importing another adapter's private helpers).
_QUOTA_ERROR_KEYWORDS = ("run out of searches", "out of searches", "rate limit", "exceeded")


def _classify_error_message(message: str) -> str:
    lowered = message.lower()
    if any(keyword in lowered for keyword in _QUOTA_ERROR_KEYWORDS):
        return "rate_limited"
    return "provider_error"


_FLIGHT_NUMBER_RE = re.compile(r"^[A-Z0-9]{2,3}[0-9]{1,4}[A-Z]?$")


def _normalize_flight_number(raw: Any) -> Optional[str]:
    if not raw:
        return None
    candidate = str(raw).replace(" ", "").upper()
    return candidate if _FLIGHT_NUMBER_RE.match(candidate) else None


def _price_to_minor_units(raw_price: Any) -> Optional[int]:
    """Decimal-exact TRY-major-to-minor conversion (architecture.md:
    money is Decimal or integer minor units, never binary float). Returns
    None for anything that is not a genuine positive price -- the caller
    skips that candidate entirely rather than guessing a price."""
    if raw_price is None or isinstance(raw_price, bool):
        return None
    try:
        amount = Decimal(str(raw_price))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if amount <= 0:
        return None
    return int((amount * 100).to_integral_value(rounding=ROUND_HALF_UP))


# Istanbul-focused V1 airport-to-timezone map -- covers exactly the
# airports this checkpoint's own live gate needs (BEY, IST, SAW). An
# airport code outside this map resolves to None, and the itinerary
# containing it is skipped with a recorded warning (below) rather than
# guessing a timezone or defaulting to UTC. Deliberately not a broad
# geolocation/airport-database service -- out of scope for this
# checkpoint (ADR 0012).
_AIRPORT_TIMEZONES: dict[str, str] = {
    "IST": "Europe/Istanbul",
    "SAW": "Europe/Istanbul",
    "BEY": "Asia/Beirut",
}


def _default_timezone_resolver(airport_code: str) -> Optional[str]:
    return _AIRPORT_TIMEZONES.get(str(airport_code).upper())


def _localize_segment_time(raw_time: Any, airport_code: str, timezone_resolver: Callable[[str], Optional[str]]) -> Optional[str]:
    """SerpApi's documented segment times are local, naive strings
    ('YYYY-MM-DD HH:MM') with no UTC offset -- appending a bare 'Z' would
    silently mislabel local time as UTC (the same honest-timestamp
    principle already established for the Open-Meteo adapter, Checkpoint
    C.1). Returns None -- never a guessed or UTC-defaulted timestamp --
    when the time string is malformed or the airport's timezone cannot be
    resolved; the caller skips the whole itinerary in that case."""
    tz_name = timezone_resolver(airport_code)
    if not tz_name:
        return None
    try:
        naive_dt = datetime.strptime(str(raw_time), "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None
    try:
        aware_dt = naive_dt.replace(tzinfo=ZoneInfo(tz_name))
    except ZoneInfoNotFoundError:
        return None
    return aware_dt.isoformat()


def _default_clock() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_from_clock(retrieved_at: str) -> date:
    return datetime.strptime(retrieved_at, "%Y-%m-%dT%H:%M:%SZ").date()


# FlightSearchResult itself validates origin/destination/dates/
# passenger_count/cabin_class -- a degraded envelope for e.g.
# status="invalid_request" must still pass that same schema, so the
# caller's raw (possibly invalid) values are never echoed back verbatim.
# Each sanitizer is a no-op on every already-validated success path
# (_validate_request only lets a query through when these values are
# already well-formed) and only substitutes a fixed, clearly-not-a-real-
# search placeholder on the rejected paths -- never silently "fixing" the
# request, since status/limitations already state exactly why it failed.
def _sanitize_iata(value: str) -> str:
    candidate = str(value).strip().upper()
    return candidate if re.fullmatch(r"[A-Z]{3}", candidate) else "XXX"


def _sanitize_date(value: str) -> str:
    try:
        date.fromisoformat(value)
        return value
    except (ValueError, TypeError):
        return "1970-01-01"


def _sanitize_passenger_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 1
    return count if 1 <= count <= 12 else 1


# SerpApi-specific timeout/retry configuration -- identical values to the
# Checkpoint C.2 timeout-semantics repair (providers/web_evidence_serpapi.py):
# 25s total/I/O timeout, at most one retry (two attempts total), backoff
# capped at 5s. Worst case 2*25 + 5 = 55s, under this project's accepted
# 60-second workflow deadline.
_IO_TIMEOUT_SECONDS = 25.0
_MAX_RETRIES = 1
_BASE_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 5.0


def _default_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        max_retries=_MAX_RETRIES, base_backoff_seconds=_BASE_BACKOFF_SECONDS, max_backoff_seconds=_MAX_BACKOFF_SECONDS
    )


def _default_timeout_policy() -> TimeoutPolicy:
    return TimeoutPolicy(connect_timeout_seconds=_IO_TIMEOUT_SECONDS, total_timeout_seconds=_IO_TIMEOUT_SECONDS)


@dataclass
class SerpApiFlightSearchProvider:
    """Real FlightSearchProvider implementation. Every dependency
    (transport, clock, sleep, cancellation check, timezone resolver,
    api_key) is injected with a safe real default, and every one is
    replaced with a deterministic fake in hermetic tests -- see
    providers/tests/test_flights_serpapi.py."""

    transport: HttpTransport = field(default_factory=UrllibHttpTransport)
    clock: Callable[[], str] = field(default=_default_clock)
    sleep_fn: Callable[[float], None] = field(default=time.sleep)
    cancellation_check: Callable[[], bool] = field(default=lambda: False)
    timezone_resolver: Callable[[str], Optional[str]] = field(default=_default_timezone_resolver)
    retry_policy: RetryPolicy = field(default_factory=_default_retry_policy)
    timeout_policy: TimeoutPolicy = field(default_factory=_default_timeout_policy)
    api_key: Optional[SecretString] = field(default=None)
    provider_name: str = PROVIDER_NAME

    def __post_init__(self) -> None:
        if self.api_key is None:
            raw_key = os.environ.get("SERPAPI_API_KEY")
            if raw_key:
                self.api_key = SecretString(raw_key)

    # --- public interface (FlightSearchProvider protocol) ---------------------------

    def search_flights(self, query: FlightSearchQuery) -> dict:
        retrieved_at = self.clock()

        if self.cancellation_check():
            return self._degraded_envelope(query, "cancelled", retrieved_at, [])

        validation_status = self._validate_request(query, retrieved_at)
        if validation_status is not None:
            return self._degraded_envelope(query, validation_status, retrieved_at, [])

        response_or_status = self._request_with_retries(query)
        if isinstance(response_or_status, str):
            return self._degraded_envelope(query, response_or_status, retrieved_at, [])

        parsed = self._parse_or_status(response_or_status, retrieved_at, query.currency)
        if isinstance(parsed, str):
            return self._degraded_envelope(query, parsed, retrieved_at, [])
        options, warnings = parsed

        envelope = self._build_envelope(
            query, status="success", data_mode="live", retrieved_at=retrieved_at, options=options, warnings=warnings
        )
        validate_envelope(envelope)
        return envelope

    # --- request validation ----------------------------------------------------------

    def _validate_request(self, query: FlightSearchQuery, retrieved_at: str) -> Optional[str]:
        if self.api_key is None:
            return "unavailable"  # a deployment/config gap, not something the caller's query caused

        origin = query.origin.strip().upper()
        destination = query.destination.strip().upper()
        if not re.fullmatch(r"[A-Z]{3}", origin) or not re.fullmatch(r"[A-Z]{3}", destination):
            return "invalid_request"
        if origin == destination:
            return "invalid_request"

        if query.return_date_from is not None or query.return_date_to is not None:
            return "unsupported"  # round-trip is explicitly out of scope for this checkpoint's V1 boundary

        try:
            depart_from = date.fromisoformat(query.depart_date_from)
            depart_to = date.fromisoformat(query.depart_date_to)
        except (ValueError, TypeError):
            return "invalid_request"
        if depart_from != depart_to:
            return "unsupported"  # a genuine date range: V1 supports exact-date search only
        if depart_from < _today_from_clock(retrieved_at):
            return "unsupported"  # a past date -- out of scope, never silently served as a future search

        if query.passenger_count < 1:
            return "invalid_request"
        if query.cabin_class is not None and query.cabin_class not in _CABIN_CLASS_TO_TRAVEL_CLASS:
            return "invalid_request"
        return None

    # --- HTTP --------------------------------------------------------------------------

    def _request_params(self, query: FlightSearchQuery) -> dict[str, Any]:
        params: dict[str, Any] = {
            "engine": "google_flights",
            "departure_id": query.origin.upper(),
            "arrival_id": query.destination.upper(),
            "outbound_date": query.depart_date_from,
            "type": "2",  # one-way -- V1 never requests round-trip (type=1) or multi-city (type=3)
            "adults": query.passenger_count,
            # Manual QA remediation Q.1 (§B): requested natively in the
            # trip's own currency (SerpApi Google Flights supports this)
            # -- never converted client-side after the fact.
            "currency": query.currency,
            "output": "json",
            "hl": "en",
            "gl": "tr",
            "deep_search": "false",
            # Explicitly "false" (SerpApi's own default): allows a matching
            # cached result to be served, conserving the free-tier quota.
            "no_cache": "false",
        }
        if query.cabin_class:
            params["travel_class"] = _CABIN_CLASS_TO_TRAVEL_CLASS[query.cabin_class]
        return params

    def _request_with_retries(self, query: FlightSearchQuery) -> HttpResponse | str:
        timeout = (self.timeout_policy.connect_timeout_seconds, self.timeout_policy.total_timeout_seconds)
        params = self._request_params(query)
        # Built fresh right here, immediately before the one transport
        # call that needs it -- never stored on self, never merged into
        # `params` above, so it can never leak into a log, a fingerprint,
        # or anything derived from `params`.
        secret_params = {"api_key": self.api_key.reveal()}
        last_status = "provider_error"
        for attempt in range(self.retry_policy.max_retries + 1):
            retry_after: Optional[float] = None
            try:
                response = self.transport.get(FLIGHTS_URL, params, timeout, secret_params=secret_params)
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
                    return last_status  # invalid_request-shaped provider response: retrying will not help

            if attempt < self.retry_policy.max_retries:
                self.sleep_fn(compute_backoff_seconds(attempt, self.retry_policy, retry_after))
        return last_status

    # --- response parsing --------------------------------------------------------------

    def _parse_or_status(self, response: HttpResponse, retrieved_at: str, currency: str) -> tuple[list[dict], list[str]] | str:
        try:
            body = response.json()
        except Exception:  # noqa: BLE001
            return "provider_error"
        if not isinstance(body, dict):
            return "provider_error"

        top_level_error = body.get("error")
        if top_level_error:
            return _classify_error_message(str(top_level_error))

        metadata = body.get("search_metadata")
        if isinstance(metadata, dict) and metadata.get("status") == "Error":
            return "provider_error"

        best = body.get("best_flights")
        other = body.get("other_flights")
        if best is None and other is None:
            return "provider_error"  # neither key present at all -- unexpected shape, not a legitimate empty result

        combined: list = []
        if isinstance(best, list):
            combined.extend(best)
        if isinstance(other, list):
            combined.extend(other)

        return self._normalize_itineraries(combined, retrieved_at, currency)

    def _normalize_itineraries(self, raw_itineraries: list, retrieved_at: str, currency: str) -> tuple[list[dict], list[str]]:
        options: list[dict] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()

        for raw in raw_itineraries:
            if not isinstance(raw, dict):
                warnings.append("skipped a malformed (non-object) itinerary entry")
                continue
            segments = raw.get("flights")
            if not isinstance(segments, list) or not segments:
                warnings.append("skipped an itinerary missing usable flight segments")
                continue

            option = self._normalize_one(raw, segments, retrieved_at, warnings, currency)
            if option is None:
                continue
            if option["flight_id"] in seen_ids:
                continue  # deterministic de-duplication: equivalent itineraries share a flight_id by construction
            seen_ids.add(option["flight_id"])
            options.append(option)
            if len(options) >= HARD_MAX_RESULTS:
                break

        return options, warnings

    def _normalize_one(self, raw: dict, segments: list, retrieved_at: str, warnings: list[str], currency: str) -> Optional[dict]:
        legs: list[dict] = []
        for seg in segments:
            if not isinstance(seg, dict):
                warnings.append("skipped an itinerary with a malformed segment")
                return None
            dep = seg.get("departure_airport") or {}
            arr = seg.get("arrival_airport") or {}
            dep_id, arr_id = dep.get("id"), arr.get("id")
            dep_time, arr_time = dep.get("time"), arr.get("time")
            airline = seg.get("airline")
            if not dep_id or not arr_id or not dep_time or not arr_time or not airline:
                warnings.append("skipped an itinerary with an incomplete segment")
                return None

            depart_at = _localize_segment_time(dep_time, dep_id, self.timezone_resolver)
            arrive_at = _localize_segment_time(arr_time, arr_id, self.timezone_resolver)
            if depart_at is None or arrive_at is None:
                warnings.append(
                    f"skipped an itinerary: unresolvable timezone or malformed time for airport {str(dep_id).upper()}/{str(arr_id).upper()}"
                )
                return None

            leg: dict[str, Any] = {
                "origin": str(dep_id).upper(),
                "destination": str(arr_id).upper(),
                "depart_at": depart_at,
                "arrive_at": arrive_at,
                "carrier": str(airline),
            }
            flight_number = _normalize_flight_number(seg.get("flight_number"))
            if flight_number:
                leg["flight_number"] = flight_number
            legs.append(leg)

        if not legs:
            return None

        amount_minor_units = _price_to_minor_units(raw.get("price"))
        if amount_minor_units is None:
            warnings.append("skipped an itinerary missing a usable price")
            return None

        first_leg, last_leg = legs[0], legs[-1]
        option: dict[str, Any] = {
            "schema_version": "1.0.0",
            "origin": first_leg["origin"],
            "destination": last_leg["destination"],
            "depart_at": first_leg["depart_at"],
            "arrive_at": last_leg["arrive_at"],
            "carrier": first_leg["carrier"],
            "stops": len(legs) - 1,
            "price": {"amount_minor_units": amount_minor_units, "currency": currency},
            "provenance": {
                "schema_version": "1.0.0",
                "provider": self.provider_name,
                "data_mode": "live",
                "retrieved_at": retrieved_at,
                "source_urls": [],
            },
        }
        if "flight_number" in first_leg:
            option["flight_number"] = first_leg["flight_number"]

        total_duration = raw.get("total_duration")
        if isinstance(total_duration, int) and not isinstance(total_duration, bool) and total_duration >= 0:
            option["duration_minutes"] = total_duration

        if len(legs) > 1:
            option["legs"] = legs

        option["flight_id"] = self._deterministic_flight_id(option, legs)
        return option

    @staticmethod
    def _deterministic_flight_id(option: dict, legs: list[dict]) -> str:
        canonical = _json.dumps(
            {
                "origin": option["origin"],
                "destination": option["destination"],
                "depart_at": option["depart_at"],
                "arrive_at": option["arrive_at"],
                "carrier": option["carrier"],
                "stops": option["stops"],
                "price": option["price"],
                "legs": legs,
            },
            sort_keys=True,
            default=str,
        )
        return f"serpapi_google_flights_{sha256_hex(canonical)[:16]}"

    # --- envelope construction -----------------------------------------------------------

    def _build_envelope(
        self, query: FlightSearchQuery, status: str, data_mode: str, retrieved_at: str,
        options: list[dict], warnings: list[str],
    ) -> dict:
        fingerprint = query.fingerprint()
        assumptions = [
            "Search-time snapshot only; never a booking, seat-availability, or fare guarantee.",
            f"SerpApi request executed via Google Flights engine (type='one-way', hl='en', gl='tr', currency={query.currency!r}).",
        ]
        if status != "success":
            assumptions.append(f"SerpApi Google Flights call ended with status={status!r}.")

        limitations = list(warnings)
        if status != "success":
            limitations.append(f"Flight provider call ended with status={status!r}; no offers retrieved.")

        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "origin": _sanitize_iata(query.origin),
            "destination": _sanitize_iata(query.destination),
            "depart_date_from": _sanitize_date(query.depart_date_from),
            "depart_date_to": _sanitize_date(query.depart_date_to),
            "passenger_count": _sanitize_passenger_count(query.passenger_count),
            "is_snapshot": True,
            "searched_at": retrieved_at,
            "options": options,
            "limitations": limitations,
        }
        if query.cabin_class in _CABIN_CLASS_TO_TRAVEL_CLASS:
            result["cabin_class"] = query.cabin_class

        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": deterministic_request_id(fingerprint, retrieved_at),
            "provider": self.provider_name,
            "capability": CAPABILITY,
            "data_mode": data_mode,
            "status": status,
            "query_fingerprint": fingerprint,
            "cache_status": "bypass",  # this adapter implements no cache layer of its own -- reported honestly
            "retrieved_at": retrieved_at,
            "currency": query.currency,
            "source_urls": [],
            "quality": {
                "schema_version": "1.0.0",
                "completeness": completeness_for_status(status),
                "freshness": data_mode,
                "assumptions": assumptions,
            },
            "result": result,
        }

    def _degraded_envelope(self, query: FlightSearchQuery, status: str, retrieved_at: str, warnings: list[str]) -> dict:
        data_mode = data_mode_for_status(status)
        envelope = self._build_envelope(
            query, status=status, data_mode=data_mode, retrieved_at=retrieved_at, options=[], warnings=warnings
        )
        validate_envelope(envelope)
        return envelope
