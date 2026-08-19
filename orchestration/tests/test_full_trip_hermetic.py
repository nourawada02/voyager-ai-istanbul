"""Hermetic full-trip sequence through the REAL compiled LangGraph with
the REAL `ProductionToolExecutor` (Checkpoint Phase 4 D.1 §8) -- every
underlying provider/MCP/A2A dependency is faked/stubbed (FakeHttpTransport,
stubbed MCP/A2A async calls), so no real socket is ever opened, but the
executor class, provider bindings, and mapping logic are the genuine
production code, not `FakeToolExecutor`. Measures the actual
graph-transition count for the canonical five-tool-call plan:
call_travel_search (delegating search_flights -> search_stays ->
estimate_fair_price -> get_weather to the internal Travel Search
specialist, Checkpoint Phase 4 D.3) -> call_istanbul_expert -> synthesize.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from phase4.graph import MAX_GRAPH_TRANSITIONS, build_graph, start_session
from phase4.models import PlannerRequest
from providers.flights_serpapi import FLIGHTS_URL, SerpApiFlightSearchProvider
from providers.http_transport import FakeHttpTransport, HttpResponse
from providers.redaction import SecretString
from providers.weather_openmeteo import FORECAST_URL, GEOCODING_URL, OpenMeteoWeatherProvider

from orchestration.system_a.tool_executor import ProductionToolExecutor


def _json_response(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(status_code=status, body=json.dumps(body).encode("utf-8"), headers={})


def _fixed_clock():
    import datetime

    return datetime.datetime(2026, 8, 17, tzinfo=datetime.timezone.utc)


@dataclass
class _StubMcpClient:
    calls: list = field(default_factory=list)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if tool_name == "search_stays":
            return {
                "status": "success",
                "result": {
                    "stays": [
                        {
                            "stay": {
                                "stay_id": "stay_it_001",
                                "coordinates": {"lat": 41.0086, "lon": 28.9802},
                                "nightly_price": {"amount_minor_units": 250000, "currency": "TRY"},
                            },
                            "fair_price": {"stay_id": "stay_it_001", "estimated_fair_price": {"amount_minor_units": 200000, "currency": "TRY"}},
                            "rank": 1,
                        }
                    ]
                },
            }
        if tool_name == "estimate_fair_price":
            return {
                "status": "success",
                "result": {"stay_id": arguments["stay_id"], "fair_price": {"stay_id": arguments["stay_id"], "estimated_fair_price": {"amount_minor_units": 200000, "currency": "TRY"}}},
            }
        return {"status": "unavailable", "result": None}


@dataclass
class _StubA2AClient:
    calls: list = field(default_factory=list)

    def call_istanbul_expert(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return {
            "status": "success",
            "result": {
                "schema_version": "1.0.0", "session_id": payload["session_id"], "trace_id": payload["trace_id"],
                "contract_version": "1.0.0", "recommended_base_candidate_id": "stay_it_001",
                "accessibility_scores": [{"candidate_id": "stay_it_001", "score": 0.8}],
                "selected_poi_ids": ["poi_hagia_sophia"],
                "daily_plans": [{
                    "schema_version": "1.0.0", "date": "2026-09-10", "side": "european",
                    "poi_ids": ["poi_hagia_sophia"], "legs": [], "walking_minutes": 15.0,
                    "transfer_minutes": 0.0, "activity_minutes": 60.0, "meal_minutes": 0.0,
                    "slack_minutes": 5.0, "warnings": [],
                }],
                "estimated_travel_minutes": 15.0, "expected_walking_minutes": 15.0, "side_crossings": 0,
                "citations": [], "assumptions": [], "warnings": [],
                "data_quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "live", "assumptions": []},
                "hard_constraint_validation_passed": True,
            },
        }


_REASON_CODE_BY_SCOPE = {
    "combined": "requires_both", "travel_only": "requires_travel_evidence",
    "istanbul_local_only": "requires_istanbul_local_grounding",
    "clarification_required": "insufficient_information", "out_of_scope": "outside_project_scope",
}


def _infer_capability_scope(remaining_responses) -> str:
    """Checkpoint Final Evaluation E.1S.1: infers the capability scope
    from the rest of this already-authored scripted plan so this
    scenario keeps its ground-truth action sequence unchanged while
    still exercising the real once-per-turn classification step."""
    actions = []
    for raw in remaining_responses:
        if isinstance(raw, BaseException):
            continue
        try:
            actions.append(json.loads(raw).get("action"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    has_travel, has_istanbul = "call_travel_search" in actions, "call_istanbul_expert" in actions
    if has_travel and has_istanbul:
        return "combined"
    if has_travel:
        return "travel_only"
    if has_istanbul:
        return "istanbul_local_only"
    return "out_of_scope"


class ScriptedDecisionProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self._i = 0

    def generate(self, system, user):
        from phase4.graph import CAPABILITY_SCOPE_PROMPT_MARKER

        if CAPABILITY_SCOPE_PROMPT_MARKER in system:
            scope = _infer_capability_scope(self.responses[self._i:])
            return json.dumps({"scope": scope, "reason_code": _REASON_CODE_BY_SCOPE[scope]})
        response = self.responses[self._i]
        self._i += 1
        return response


def _decision(action, arguments, reason_code="all_required_evidence_present"):
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


TRIP_REQUEST = {
    "session_id": "11111111-1111-1111-1111-111111111111",
    "trace_id": "22222222-2222-2222-2222-222222222222",
    "origin": "BEY",
    "destination": "IST",
    "depart_date": "2026-08-20",
    "return_date": "2026-08-25",
    "traveler_count": 2,
    "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


def _build_real_tool_executor() -> ProductionToolExecutor:
    flights_transport = FakeHttpTransport(responses={
        FLIGHTS_URL: _json_response({
            "search_metadata": {"status": "Success"},
            "best_flights": [{
                "flights": [{
                    "departure_airport": {"id": "BEY", "time": "2026-08-20 08:30"},
                    "arrival_airport": {"id": "IST", "time": "2026-08-20 10:15"},
                    "airline": "Turkish Airlines", "flight_number": "TK823",
                }],
                "total_duration": 105, "price": 4500,
            }],
            "other_flights": [],
        })
    })
    weather_transport = FakeHttpTransport(responses={
        GEOCODING_URL: _json_response({"results": [{"latitude": 41.0, "longitude": 28.9, "name": "Istanbul", "country": "Turkey", "timezone": "Europe/Istanbul"}]}),
        FORECAST_URL: _json_response({"daily": {"time": ["2026-08-20"], "weather_code": [1], "temperature_2m_max": [27], "temperature_2m_min": [19]}}),
    })
    return ProductionToolExecutor(
        weather_provider=OpenMeteoWeatherProvider(transport=weather_transport, clock=_fixed_clock),
        web_evidence_provider=object(),  # unused in this scenario
        flight_provider=SerpApiFlightSearchProvider(transport=flights_transport, clock=lambda: "2026-08-17T12:00:00Z", sleep_fn=lambda s: None, api_key=SecretString("test")),
        mcp_client=_StubMcpClient(),
        a2a_client=_StubA2AClient(),
    )


def test_canonical_five_tool_plan_completes_with_real_production_executor():
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_flight_info"),
        _decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-08-20", "passenger_count": 2}, "missing_flight_info"),
        _decision("search_stays", {"check_in": "2026-08-20", "check_out": "2026-08-25", "guest_count": 2}, "missing_stay_info"),
        _decision("estimate_fair_price", {"stay_id": "stay_it_001"}, "needs_fair_price"),
        _decision("get_weather", {"location": "Istanbul", "date_from": "2026-08-20", "date_to": "2026-08-20"}, "missing_weather_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("call_istanbul_expert", {"question": "What should I see near my stay?"}, "missing_local_expertise"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    tool_executor = _build_real_tool_executor()
    graph = build_graph(tool_executor, decider, decider)
    request = PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="Plan my Istanbul trip", trip_request=TRIP_REQUEST)

    result = start_session(graph, request, "t-full-trip-hermetic")

    observed_actions = [obs["action"] for obs in result["observations"]]
    print(f"FULL-TRIP HERMETIC GATE: graph_transition_count={result['graph_transition_count']} "
          f"tool_call_count={result['tool_call_count']} observed_actions={observed_actions} "
          f"final_status={result['final_result']['status']}")

    assert observed_actions == ["search_flights", "search_stays", "estimate_fair_price", "get_weather", "call_istanbul_expert"]
    assert all(obs["status"] == "success" for obs in result["observations"])
    assert result["final_result"]["status"] == "success"
    assert result["graph_transition_count"] <= MAX_GRAPH_TRANSITIONS
    assert result["tool_call_count"] == 5
    assert result.get("degraded") is not True
