"""Narrowly scoped OPTIONAL live smoke test for the real SerpApi Google
Flights adapter (Checkpoint Phase 4 C.3). Disabled during normal `pytest`
runs -- enabled only by setting the explicit environment flag below.
Makes exactly one bounded, one-way, exact-date search (BEY -> IST, one
adult, economy, max 3 locally returned results) -- never a loop, never
repeated polling, never a departure_token/booking_token follow-up
request.

Requires SERPAPI_API_KEY to be set in the environment. Prints only
non-secret, already-public operational fields -- never the API key,
never the complete request URL, never a raw response body, and never
writes any file.

Enable with:
    VOYAGER_LIVE_FLIGHT_SEARCH_GATE=1 python -m pytest providers/tests/test_flights_serpapi_live.py -v -s
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from providers.flights import FlightSearchQuery
from providers.flights_serpapi import SerpApiFlightSearchProvider
from providers.tests.conftest import make_validator

LIVE_GATE_ENV_VAR = "VOYAGER_LIVE_FLIGHT_SEARCH_GATE"

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_GATE_ENV_VAR) != "1",
    reason=f"live SerpApi Google Flights network gate is disabled by default; set {LIVE_GATE_ENV_VAR}=1 to enable",
)


def test_live_serpapi_bey_ist_flight_search(envelope_validator, registry):
    key_present = os.environ.get("SERPAPI_API_KEY") is not None
    print(f"SERPAPI_API_KEY present: {str(key_present).lower()}")

    provider = SerpApiFlightSearchProvider()  # real transport, real clock; reads SERPAPI_API_KEY from the environment
    if provider.api_key is None:
        pytest.skip("SERPAPI_API_KEY is not set in this environment -- cannot attempt the keyed live gate")

    future_date = (datetime.now(timezone.utc) + timedelta(days=45)).date().isoformat()
    query = FlightSearchQuery(
        origin="BEY", destination="IST", depart_date_from=future_date, depart_date_to=future_date,
        passenger_count=1, cabin_class="economy",
    )

    start = time.monotonic()
    envelope = provider.search_flights(query)
    runtime_seconds = time.monotonic() - start

    envelope_errors = list(envelope_validator.iter_errors(envelope))
    assert not envelope_errors, [e.message for e in envelope_errors]
    result_validator = make_validator("FlightSearchResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]

    assert envelope["capability"] == "flight_search"
    assert envelope["provider"] == "serpapi_google_flights"
    assert envelope["status"] in ("success", "unavailable", "unsupported"), envelope.get("quality", {}).get("assumptions")
    assert envelope["result"]["is_snapshot"] is True
    assert len(envelope["result"]["options"]) <= 3

    schema_ok = True
    for option in envelope["result"]["options"]:
        option_validator = make_validator("FlightOption", registry)
        option_errors = list(option_validator.iter_errors(option))
        if option_errors:
            schema_ok = False
        assert not option_errors, [e.message for e in option_errors]
        assert option["price"]["currency"] == "TRY"
        assert "+" in option["depart_at"][10:] or "-" in option["depart_at"][10:]
        assert "+" in option["arrive_at"][10:] or "-" in option["arrive_at"][10:]

    serialized = str(envelope).lower()
    for term in ("booked", "reservation_confirmed", "payment", "purchase", "seat_available", "confirmed", "booking_token", "departure_token"):
        assert term not in serialized

    print(
        f"LIVE SERPAPI GOOGLE FLIGHTS GATE: provider={envelope['provider']} status={envelope['status']} "
        f"item_count={len(envelope['result']['options'])} route=BEY->IST currency={envelope['currency']} "
        f"schema_valid={schema_ok} runtime_seconds={runtime_seconds:.2f}"
    )
