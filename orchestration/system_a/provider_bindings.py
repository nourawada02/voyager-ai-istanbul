"""Real root-provider bindings for System A's production ToolExecutor
(Checkpoint Phase 4 D.1). Each function maps a validated `ActionDecision`
argument dict (already validated against planner-a's own closed
per-action Pydantic schema, `phase4.models.ACTION_ARGUMENT_MODELS`) into
the corresponding root provider's typed Query, calls that provider's own
`.search()`/`.fetch_weather()`/`.search_flights()` exactly once, and
returns its `ProviderResponseEnvelope`-shaped dict completely unmodified.

No HTTP call is reimplemented here -- every retry, timeout, redaction,
and provenance behavior belongs to the provider itself
(`providers.policy`, `providers.redaction`, `providers.fingerprint`),
reused exactly as Checkpoints C.1/C.2/C.3 already proved live. These
functions are pure mappers: they never let the decision provider modify
a returned result, never reinterpret an `unavailable`/failed envelope as
success, and never convert web evidence into flight inventory or frozen
RAG knowledge (the three capabilities remain three structurally distinct
result shapes, exactly as ADR 0009 §5 already established).
"""

from __future__ import annotations

from typing import Any, Optional

from phase4.context import ExecutionContext
from providers.flights import FlightSearchQuery
from providers.flights_serpapi import SerpApiFlightSearchProvider
from providers.weather import WeatherQuery
from providers.weather_openmeteo import OpenMeteoWeatherProvider
from providers.web_evidence import WebEvidenceQuery
from providers.web_evidence_serpapi import SerpApiWebEvidenceProvider

# Manual QA remediation Q.1 (§B): the one currency this system can price
# flights in when the trip's own currency is not otherwise recognized --
# never Qwen-chosen, mirrors DEFAULT_CURRENCY in tool_executor.py.
_DEFAULT_FLIGHT_CURRENCY = "TRY"
_SUPPORTED_FLIGHT_CURRENCIES = frozenset({"TRY", "USD"})


def _trip_currency(context: Optional[ExecutionContext]) -> str:
    if context is None:
        return _DEFAULT_FLIGHT_CURRENCY
    trip_request = (context.normalized_request or {}).get("trip_request") or {}
    currency = (trip_request.get("budget") or {}).get("currency")
    if currency in _SUPPORTED_FLIGHT_CURRENCIES:
        return currency
    return _DEFAULT_FLIGHT_CURRENCY


def _tool_result(envelope: dict[str, Any]) -> dict[str, Any]:
    """Every real provider envelope already carries its own honest
    `status` (1.1.0, ProviderResponseEnvelope.schema.json) -- this is the
    ONLY status `ToolExecutor`'s contract ever reports up to Observe. A
    provider that reported `timeout`/`rate_limited`/`unavailable`/... is
    never silently reinterpreted as `success` here."""
    return {"status": envelope.get("status", "provider_error"), "result": envelope}


def fetch_weather(provider: OpenMeteoWeatherProvider, arguments: dict[str, Any], context: Optional[ExecutionContext] = None) -> dict[str, Any]:
    query = WeatherQuery(
        location=arguments.get("location") or "Istanbul",
        timezone="",
        date_from=str(arguments["date_from"]),
        date_to=str(arguments["date_to"]),
    )
    return _tool_result(provider.fetch_weather(query))


def search_web(provider: SerpApiWebEvidenceProvider, arguments: dict[str, Any], context: Optional[ExecutionContext] = None) -> dict[str, Any]:
    query = WebEvidenceQuery(
        query=arguments["query"],
        language_hint="en",
        max_results=arguments.get("max_results"),
    )
    return _tool_result(provider.search(query))


def search_flights(provider: SerpApiFlightSearchProvider, arguments: dict[str, Any], context: Optional[ExecutionContext] = None) -> dict[str, Any]:
    # phase4.models.SearchFlightsArgs is exact-date, one-way only --
    # matches providers.flights_serpapi's own V1 boundary exactly, so
    # depart_date_from == depart_date_to always, never a range.
    depart_date = str(arguments["depart_date"])
    query = FlightSearchQuery(
        origin=arguments["origin"],
        destination=arguments["destination"],
        depart_date_from=depart_date,
        depart_date_to=depart_date,
        passenger_count=arguments["passenger_count"],
        cabin_class=arguments.get("cabin_class"),
        # Manual QA remediation Q.1 (§B): the trip's own currency, server-
        # injected from context -- never Qwen-chosen, exactly like
        # search_stays'/estimate_fair_price's own DEFAULT_CURRENCY.
        currency=_trip_currency(context),
    )
    return _tool_result(provider.search_flights(query))
