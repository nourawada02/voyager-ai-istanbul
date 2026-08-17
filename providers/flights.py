"""Flight search: an independent typed tool/provider (Checkpoint Phase 4
C.0). Every result is a search-time snapshot (contracts/FlightSearchResult.schema.json's
is_snapshot: const true) -- never a claim that a seat is available now,
that a fare is guaranteed, that an itinerary is booked, or that payment
occurred. No booking, payment, reservation, or purchase contract is
authorized. Never implemented inside System A as tightly-coupled code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from providers.fingerprint import deterministic_request_id, fingerprint_request
from providers.policy import completeness_for_status, data_mode_for_status

CAPABILITY = "flight_search"
SCHEMA_VERSION = "1.1.0"
RESULT_SCHEMA_VERSION = "1.0.0"

_FIXED_TEST_CLOCK = "2026-08-01T12:00:00Z"


@dataclass(frozen=True)
class FlightSearchQuery:
    origin: str
    destination: str
    depart_date_from: str
    depart_date_to: str
    passenger_count: int
    return_date_from: Optional[str] = None
    return_date_to: Optional[str] = None
    cabin_class: Optional[str] = None

    def normalized(self) -> dict:
        return {
            "origin": self.origin.upper(),
            "destination": self.destination.upper(),
            "depart_date_from": self.depart_date_from,
            "depart_date_to": self.depart_date_to,
            "return_date_from": self.return_date_from,
            "return_date_to": self.return_date_to,
            "passenger_count": self.passenger_count,
            "cabin_class": self.cabin_class,
        }

    def fingerprint(self) -> str:
        return fingerprint_request(CAPABILITY, self.normalized())


class FlightSearchProvider(Protocol):
    """The interface a real adapter (e.g. SerpApi Google Flights, see
    docs/adr/0009-...md §7) and FakeFlightSearchProvider both satisfy.
    Always returns a ProviderResponseEnvelope-shaped dict
    (contracts/ProviderResponseEnvelope.schema.json +
    contracts/FlightSearchResult.schema.json)."""

    def search_flights(self, query: FlightSearchQuery) -> dict: ...


def _base_result(query: FlightSearchQuery, retrieved_at: str) -> dict:
    result: dict = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "origin": query.origin.upper(),
        "destination": query.destination.upper(),
        "depart_date_from": query.depart_date_from,
        "depart_date_to": query.depart_date_to,
        "passenger_count": query.passenger_count,
        "is_snapshot": True,
        "searched_at": retrieved_at,
        "options": [],
        "limitations": [],
    }
    if query.return_date_from:
        result["return_date_from"] = query.return_date_from
        result["return_date_to"] = query.return_date_to
    if query.cabin_class:
        result["cabin_class"] = query.cabin_class
    return result


def _base_envelope(
    query: FlightSearchQuery, provider_name: str, status: str, data_mode: str, retrieved_at: str = _FIXED_TEST_CLOCK
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
        "currency": "TRY",
        "source_urls": [],
        "quality": {
            "schema_version": "1.0.0",
            "completeness": completeness_for_status(status),
            "freshness": data_mode,
            "assumptions": ["Search-time snapshot only; never a booking or fare guarantee."],
        },
        "result": _base_result(query, retrieved_at),
    }


def build_success_envelope(
    query: FlightSearchQuery, provider_name: str = "fake-flight-search-provider", retrieved_at: str = _FIXED_TEST_CLOCK
) -> dict:
    envelope = _base_envelope(query, provider_name, status="success", data_mode="live", retrieved_at=retrieved_at)
    envelope["result"]["options"] = [
        {
            "schema_version": "1.0.0",
            "flight_id": "flight_fake_001",
            "origin": query.origin.upper(),
            "destination": query.destination.upper(),
            "depart_at": f"{query.depart_date_from}T06:30:00Z",
            "arrive_at": f"{query.depart_date_from}T08:15:00Z",
            "carrier": "FakeAir",
            "stops": 0,
            "price": {"amount_minor_units": 450000, "currency": "TRY"},
            "provenance": {
                "schema_version": "1.0.0",
                "provider": provider_name,
                "data_mode": "live",
                "retrieved_at": retrieved_at,
                "source_urls": [],
            },
        }
    ]
    return envelope


def build_degraded_envelope(
    query: FlightSearchQuery, status: str, provider_name: str = "fake-flight-search-provider",
    retrieved_at: str = _FIXED_TEST_CLOCK,
) -> dict:
    """Never invents flight offers from web snippets or any other source
    on a degraded path -- 'options' stays empty and 'limitations' names
    the reason explicitly. data_mode is derived honestly from status
    (Checkpoint Phase 4 C.0 correction pass) -- never hardcoded to
    'estimated' when no offers and no estimate exist."""
    data_mode = data_mode_for_status(status)
    envelope = _base_envelope(query, provider_name, status=status, data_mode=data_mode, retrieved_at=retrieved_at)
    envelope["result"]["limitations"] = [f"Flight provider call ended with status={status!r}; no offers retrieved."]
    return envelope


@dataclass
class FakeFlightSearchProvider:
    fixed_status: str = "success"
    provider_name: str = "fake-flight-search-provider"
    clock: Callable[[], str] = field(default=lambda: _FIXED_TEST_CLOCK)
    call_log: list[FlightSearchQuery] = field(default_factory=list)

    def search_flights(self, query: FlightSearchQuery) -> dict:
        self.call_log.append(query)
        now = self.clock()
        if self.fixed_status == "success":
            return build_success_envelope(query, self.provider_name, retrieved_at=now)
        return build_degraded_envelope(query, self.fixed_status, self.provider_name, retrieved_at=now)
