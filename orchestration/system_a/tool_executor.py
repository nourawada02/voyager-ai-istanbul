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

import logging
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
MAX_LOGGED_STAY_ID_CHARS = 200  # mirrors phase4.models.EstimateFairPriceArgs.stay_id's own max_length

logger = logging.getLogger("voyager.system_a.tool_executor")


def _call_provider_binding(binding: Any, provider: Any, arguments: dict[str, Any], context: Optional[ExecutionContext] = None) -> dict[str, Any]:
    """Production real-mode incident P.1: `TravelMcpClient.call_tool()`
    and `IstanbulExpertA2AClient.call_istanbul_expert()` already never let
    a raw transport exception reach the caller (their own `except
    Exception` boundaries) -- the three DIRECT root-provider bindings
    below (`fetch_weather`/`search_web`/`search_flights`) had no
    equivalent boundary, so an unexpected provider-side exception (e.g. a
    packaging/config gap surfacing as FileNotFoundError, observed in
    production) propagated all the way to System A's own generic
    exception handler and aborted the entire run with
    `internal_execution_error`, discarding every observation already
    gathered. This restores the same boundary contract those two sibling
    clients already establish, so a genuinely unexpected exception here
    becomes an honest `provider_error` observation instead -- the bounded
    ReAct loop continues to whatever else remains eligible exactly as it
    already does for any other provider failure status."""
    try:
        return binding(provider, arguments, context)
    except Exception:  # noqa: BLE001 -- an unexpected root-provider exception must never reach the caller
        return {"status": "provider_error", "result": None}


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


def _known_stays(context: Optional[ExecutionContext]) -> list[dict[str, Any]]:
    """Every {"stay_id", "name"} pair from the MOST RECENT successful
    `search_stays` observation -- the only source of truth for which
    stay_id values genuinely exist. Used both to validate/alias-resolve
    an `estimate_fair_price` argument and to ground the specialist's own
    prompt (phase4/specialist.py) with the real values, so Qwen is never
    left to guess one (fair-price correction root-cause fix)."""
    if context is None:
        return []
    for obs in reversed(context.observations_for_action(Action.SEARCH_STAYS.value)):
        if obs.get("status") != "success":
            continue
        # search_stays observations are NOT ProviderResponseEnvelope-wrapped
        # (Travel MCP's own SearchStaysResult shape directly, ADR 0009 §5)
        # -- `envelope` here IS the SearchStaysResult, no nested "result" key.
        search_stays_result = obs.get("envelope") or {}
        known: list[dict[str, Any]] = []
        for item in search_stays_result.get("stays", []) or []:
            stay = item.get("stay") or {}
            if stay.get("stay_id"):
                known.append({"stay_id": stay["stay_id"], "name": stay.get("name")})
        if known:
            return known  # most recent successful search_stays observation only
    return []


def _validated_stay_id(stay_id: Optional[str], context: Optional[ExecutionContext]) -> Optional[str]:
    """Only a `stay_id` that actually appears in a prior successful
    `search_stays` observation is ever forwarded -- never a value Qwen
    could have invented without having actually seen a real search
    result (Checkpoint D.1 §3: "never ask Qwen to invent model
    features"). Two resolution paths, both against the SAME real,
    already-returned stay list, never a fabricated or fuzzy match:
      1. An exact `stay_id` match (the common case once
         phase4/specialist.py's prompt grounding fix is in place).
      2. A safe, explicit alias: an exact case-insensitive match against
         that same stay's own real `name` field -- a known-equivalent
         field a model could reasonably echo instead of the opaque id."""
    if not stay_id or context is None:
        return None
    known = _known_stays(context)
    for stay in known:
        if stay["stay_id"] == stay_id:
            return stay["stay_id"]
    normalized = stay_id.strip().lower()
    if normalized:
        for stay in known:
            name = stay.get("name")
            if isinstance(name, str) and name.strip().lower() == normalized:
                return stay["stay_id"]
    return None


def _log_rejected_estimate_fair_price(
    context: Optional[ExecutionContext], raw_stay_id: Any, known: list[dict[str, Any]], reason: str
) -> None:
    """Server-side-only diagnostic record, keyed by trace_id (architecture.md
    §13.4: full detail logged server-side, never returned to the caller).
    Logs only closed, bounded, structural facts -- the reason code, the
    count of genuinely known stay ids, and the raw attempted value
    (itself just a short identifier-shaped string per its own schema's
    200-char bound, never free-form exception text or a credential) --
    never anything else about internal state."""
    session_id = context.session_id if context else ""
    trace_id = context.trace_id if context else ""
    bounded = str(raw_stay_id)[:MAX_LOGGED_STAY_ID_CHARS] if raw_stay_id is not None else ""
    logger.warning(
        "estimate_fair_price rejected: reason=%s known_stay_count=%d attempted_stay_id=%r "
        "(session_id=%s trace_id=%s)",
        reason, len(known), bounded, session_id, trace_id,
    )


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
            return _call_provider_binding(fetch_weather, self.weather_provider, arguments, context)
        if action == Action.WEB_SEARCH:
            return _call_provider_binding(search_web, self.web_evidence_provider, arguments, context)
        if action == Action.SEARCH_FLIGHTS:
            return _call_provider_binding(search_flights, self.flight_provider, arguments, context)
        if action == Action.SEARCH_STAYS:
            return self.mcp_client.call_tool("search_stays", _build_search_stays_arguments(arguments, context))
        if action == Action.ESTIMATE_FAIR_PRICE:
            raw_stay_id = arguments.get("stay_id")
            stay_id = _validated_stay_id(raw_stay_id, context)
            if stay_id is None:
                known = _known_stays(context)
                reason = "no_prior_search_stays" if not known else "stay_id_not_recognized"
                _log_rejected_estimate_fair_price(context, raw_stay_id, known, reason)
                # `reason` is additive on this internal dict -- never part
                # of the Travel MCP contract, never returned raw to the
                # end user; phase4/specialist.py's observe node folds it
                # into the observation's existing `warnings` list (an
                # already-safe, already-caller-facing field) as a closed,
                # non-sensitive hint for the specialist's own bounded
                # retry, not a leaked internal detail.
                return {"status": "invalid_request", "result": None, "reason": reason}
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
