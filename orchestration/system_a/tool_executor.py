"""Production `ToolExecutor` (Checkpoint Phase 4 D.1): the real class
satisfying `phase4.tools.ToolExecutor`, injected into System A's existing
bounded LangGraph loop in place of `FakeToolExecutor`. Composes the real
provider bindings, the real Travel MCP client, and the real System B A2A
client -- never reimplements any of their network/validation/retry
behavior itself.

Maps `estimate_fair_price` and `call_istanbul_expert` using
already-validated prior observations from the injected `ExecutionContext`
(a `stay_id` must actually appear in a prior successful `search_stays`
observation; `call_istanbul_expert`'s stay candidates and weather context
come from prior observations and the normalized trip request) -- the
decision provider is never asked to reproduce or invent a stay id,
coordinate, or other prior tool's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from orchestration.system_a.a2a_client import IstanbulExpertA2AClient
from orchestration.system_a.mcp_client import TravelMcpClient
from orchestration.system_a.provider_bindings import fetch_weather, search_flights, search_web
from phase4.context import ExecutionContext
from phase4.models import Action

DEFAULT_SEARCH_STAYS_RESULT_LIMIT = 5
DEFAULT_CURRENCY = "TRY"
DEFAULT_DAILY_ACTIVITY_BUDGET_MINUTES = 360  # 6 hours/day -- a documented default, never an invented hard preference


def _build_search_stays_arguments(arguments: dict[str, Any], context: Optional[ExecutionContext]) -> dict[str, Any]:
    """Maps `phase4.models.SearchStaysArgs` -> the real Travel MCP
    `search_stays` contract. `session_id`/`trace_id`/`result_limit`/
    `currency`/`ranking_mode`/`schema_version` are required by the real
    tool but absent from the D.0 action schema -- filled from
    `ExecutionContext` and fixed, documented defaults, never an invented
    hard preference (a specific district/room type/budget is only ever
    forwarded when the caller's own arguments actually named one)."""
    mcp_args: dict[str, Any] = {
        "session_id": context.session_id if context else "",
        "trace_id": context.trace_id if context else "",
        "guest_count": arguments["guest_count"],
        "result_limit": DEFAULT_SEARCH_STAYS_RESULT_LIMIT,
        "currency": DEFAULT_CURRENCY,
        "ranking_mode": "preliminary_price_value_desc",
        "schema_version": "1.0.0",
    }
    if arguments.get("check_in"):
        mcp_args["check_in"] = str(arguments["check_in"])
    if arguments.get("check_out"):
        mcp_args["check_out"] = str(arguments["check_out"])
    if arguments.get("district_id"):
        mcp_args["district_ids"] = [arguments["district_id"]]
    return mcp_args


def _validated_stay_id(stay_id: Optional[str], context: Optional[ExecutionContext]) -> Optional[str]:
    """Only a `stay_id` that actually appears in a prior successful
    `search_stays` observation is ever forwarded -- never a value Qwen
    could have invented without having actually seen a real search
    result (Checkpoint D.1 §3: "never ask Qwen to invent model
    features")."""
    if not stay_id or context is None:
        return None
    for obs in context.observations_for_action(Action.SEARCH_STAYS.value):
        if obs.get("status") != "success":
            continue
        # search_stays observations are NOT ProviderResponseEnvelope-wrapped
        # (Travel MCP's own SearchStaysResult shape directly, ADR 0009 §5)
        # -- `envelope` here IS the SearchStaysResult, no nested "result" key.
        search_stays_result = obs.get("envelope") or {}
        for item in search_stays_result.get("stays", []) or []:
            if (item.get("stay") or {}).get("stay_id") == stay_id:
                return stay_id
    return None


def _build_local_plan_request(context: Optional[ExecutionContext]) -> Optional[dict[str, Any]]:
    """Builds the canonical `LocalPlanRequest` payload from the
    normalized trip request, prior `search_stays` observations (stay
    candidates), and the most recent `get_weather` observation (weather
    context) -- never from raw provider credentials or an unvalidated
    provider payload (ADR 0009 §3). Returns None when the minimum
    required evidence (a trip request and at least one prior stay
    candidate) is not yet present -- the caller treats that as
    `invalid_request`, never a fabricated call."""
    if context is None:
        return None
    trip_request = context.normalized_request.get("trip_request")
    if not trip_request:
        return None

    stay_candidates: list[dict[str, Any]] = []
    for obs in context.observations_for_action(Action.SEARCH_STAYS.value):
        if obs.get("status") != "success":
            continue
        search_stays_result = obs.get("envelope") or {}
        for item in search_stays_result.get("stays", []) or []:
            stay = item.get("stay") or {}
            coords = stay.get("coordinates") or {}
            if stay.get("stay_id") and "lat" in coords and "lon" in coords:
                stay_candidates.append({"candidate_id": stay["stay_id"], "lat": coords["lat"], "lon": coords["lon"]})
        if stay_candidates:
            break  # most recent successful search_stays observation only
    if not stay_candidates:
        return None

    weather_context = None
    weather_observations = context.observations_for_action(Action.GET_WEATHER.value)
    if weather_observations:
        latest = weather_observations[-1]
        if latest.get("status") == "success":
            envelope = latest.get("envelope") or {}
            weather_context = envelope.get("result")

    preferences = trip_request.get("preferences") or {}
    return {
        "session_id": context.session_id,
        "trace_id": context.trace_id,
        "contract_version": "1.0.0",
        "trip_start_date": trip_request["depart_date"],
        "trip_end_date": trip_request["return_date"],
        "interests": preferences.get("interests") or ["general_sightseeing"],
        "pace": preferences.get("pace", "moderate"),
        "language": preferences.get("language", "en"),
        "mobility_constraints": preferences.get("mobility_constraints", []),
        "daily_activity_budget_minutes": DEFAULT_DAILY_ACTIVITY_BUDGET_MINUTES,
        "stay_candidates": stay_candidates[:3],
        "weather_context": weather_context,
        "hard_constraints": [],
        "soft_constraints": [],
    }


@dataclass
class ProductionToolExecutor:
    """Satisfies `phase4.tools.ToolExecutor`. Every dependency is
    injected with a safe real default, and every one is replaced with a
    hermetic fake/stub in `orchestration/tests/`."""

    weather_provider: Any = None
    web_evidence_provider: Any = None
    flight_provider: Any = None
    mcp_client: TravelMcpClient = field(default_factory=TravelMcpClient)
    a2a_client: IstanbulExpertA2AClient = field(default_factory=IstanbulExpertA2AClient)

    def __post_init__(self) -> None:
        # Constructed lazily with real defaults only if not injected, so
        # importing this module never requires SERPAPI_API_KEY to be set
        # (e.g. a caller using only the weather binding in a test).
        if self.weather_provider is None:
            from providers.weather_openmeteo import OpenMeteoWeatherProvider

            self.weather_provider = OpenMeteoWeatherProvider()
        if self.web_evidence_provider is None:
            from providers.web_evidence_serpapi import SerpApiWebEvidenceProvider

            self.web_evidence_provider = SerpApiWebEvidenceProvider()
        if self.flight_provider is None:
            from providers.flights_serpapi import SerpApiFlightSearchProvider

            self.flight_provider = SerpApiFlightSearchProvider()

    def execute(self, action: Action, arguments: dict[str, Any], context: Optional[ExecutionContext] = None) -> dict[str, Any]:
        if action == Action.GET_WEATHER:
            return fetch_weather(self.weather_provider, arguments)
        if action == Action.WEB_SEARCH:
            return search_web(self.web_evidence_provider, arguments)
        if action == Action.SEARCH_FLIGHTS:
            return search_flights(self.flight_provider, arguments)
        if action == Action.SEARCH_STAYS:
            return self.mcp_client.call_tool("search_stays", _build_search_stays_arguments(arguments, context))
        if action == Action.ESTIMATE_FAIR_PRICE:
            stay_id = _validated_stay_id(arguments.get("stay_id"), context)
            if stay_id is None:
                return {"status": "invalid_request", "result": None}
            mcp_args = {
                "session_id": context.session_id if context else "",
                "trace_id": context.trace_id if context else "",
                "stay_id": stay_id,
                "currency": DEFAULT_CURRENCY,
                "schema_version": "1.0.0",
            }
            return self.mcp_client.call_tool("estimate_fair_price", mcp_args)
        if action == Action.CALL_ISTANBUL_EXPERT:
            local_plan_request = _build_local_plan_request(context)
            if local_plan_request is None:
                return {"status": "invalid_request", "result": None}
            return self.a2a_client.call_istanbul_expert(local_plan_request)
        return {"status": "unavailable", "result": None}
