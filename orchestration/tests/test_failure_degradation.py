"""Hermetic failure/degradation tests through the REAL compiled LangGraph
with the REAL `ProductionToolExecutor` (Checkpoint Phase 4 D.1 §9). Every
underlying dependency is faked/stubbed -- no real socket is ever opened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from phase4.graph import build_graph, start_session
from phase4.models import PlannerRequest
from providers.flights_serpapi import FLIGHTS_URL, SerpApiFlightSearchProvider
from providers.http_transport import FakeHttpTransport, HttpResponse
from providers.redaction import SecretString
from providers.weather_openmeteo import GEOCODING_URL, OpenMeteoWeatherProvider

from orchestration.system_a.tool_executor import ProductionToolExecutor


def _json_response(body: dict, status: int = 200) -> HttpResponse:
    return HttpResponse(status_code=status, body=json.dumps(body).encode("utf-8"), headers={})


def _fixed_clock():
    import datetime

    return datetime.datetime(2026, 8, 17, tzinfo=datetime.timezone.utc)


class ScriptedDecisionProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self._i = 0

    def generate(self, system, user):
        response = self.responses[self._i]
        self._i += 1
        return response


def _decision(action, arguments, reason_code="all_required_evidence_present"):
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


@dataclass
class _StubMcpClient:
    scenario: str = "success"
    calls: list = field(default_factory=list)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        if self.scenario == "timeout":
            return {"status": "timeout", "result": None}
        if self.scenario == "malformed":
            return {"status": "success", "result": {"unexpected": "shape"}}
        return {
            "status": "success",
            "result": {
                "stays": [{
                    "stay": {
                        "stay_id": "s1",
                        "nightly_price": {"amount_minor_units": 1, "currency": "TRY"},
                        "coordinates": {"lat": 41.0086, "lon": 28.9802},
                    },
                    "fair_price": {}, "rank": 1,
                }]
            },
        }


@dataclass
class _StubA2AClient:
    scenario: str = "success"
    calls: list = field(default_factory=list)

    def call_istanbul_expert(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if self.scenario == "timeout":
            return {"status": "timeout", "result": None}
        if self.scenario == "invalid_artifact":
            return {"status": "success", "result": {"not_a_local_itinerary": True}}
        return {"status": "success", "result": {}}


def _request(trip_request=None):
    return PlannerRequest(session_id=uuid4(), trace_id=uuid4(), user_message="Plan my trip", trip_request=trip_request)


TRIP_REQUEST = {
    "session_id": "11111111-1111-1111-1111-111111111111",
    "trace_id": "22222222-2222-2222-2222-222222222222",
    "origin": "BEY",
    "destination": "IST",
    "depart_date": "2026-08-20",
    "return_date": "2026-08-25",
    "traveler_count": 1,
    "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


def test_unavailable_weather_produces_a_partial_plan_not_a_hard_failure():
    transport = FakeHttpTransport(responses={GEOCODING_URL: _json_response({"results": []})})
    executor = ProductionToolExecutor(
        weather_provider=OpenMeteoWeatherProvider(transport=transport, clock=_fixed_clock),
        web_evidence_provider=object(), flight_provider=object(),
        mcp_client=_StubMcpClient(), a2a_client=_StubA2AClient(),
    )
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_weather_info"),
        _decision("get_weather", {"location": "Nowhere", "date_from": "2026-08-20", "date_to": "2026-08-20"}, "missing_weather_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    result = start_session(build_graph(executor, decider, decider), _request(), "t-weather-unavailable")
    assert result["observations"][0]["status"] == "unavailable"
    assert result["final_result"]["status"] == "partial"
    serialized = json.dumps(result, default=str).lower()
    assert "traceback" not in serialized
    assert "api_key" not in serialized


def test_serpapi_rate_limiting_does_not_loop_and_never_invents_flights():
    transport = FakeHttpTransport(responses={FLIGHTS_URL: _json_response({}, status=429)})
    executor = ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(),
        flight_provider=SerpApiFlightSearchProvider(transport=transport, clock=lambda: "2026-08-17T12:00:00Z", sleep_fn=lambda s: None, api_key=SecretString("test")),
        mcp_client=_StubMcpClient(), a2a_client=_StubA2AClient(),
    )
    same_args = {"origin": "BEY", "destination": "IST", "depart_date": "2026-08-20", "passenger_count": 1}
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_flight_info"),
        _decision("search_flights", same_args, "missing_flight_info"),
        _decision("search_flights", same_args, "missing_flight_info"),  # duplicate -> specialist loop breaks, returns control
        _decision("synthesize", {}, "all_required_evidence_present"),  # supervisor's next decision
    ])
    result = start_session(build_graph(executor, decider, decider), _request(), "t-rate-limited")
    assert result["observations"][0]["status"] == "rate_limited"
    assert result["observations"][0]["envelope"] is None  # no invented flight data
    assert len(result["trace"]) < 20  # bounded, never open-ended
    assert result["final_result"]["status"] in ("partial", "unavailable")


def test_mcp_timeout_produces_safe_partial_failure():
    executor = ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(), flight_provider=object(),
        mcp_client=_StubMcpClient(scenario="timeout"), a2a_client=_StubA2AClient(),
    )
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_stay_info"),
        _decision("search_stays", {"check_in": "2026-08-20", "check_out": "2026-08-25", "guest_count": 1}, "missing_stay_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    result = start_session(build_graph(executor, decider, decider), _request(), "t-mcp-timeout")
    assert result["observations"][0]["status"] == "timeout"
    assert result["final_result"]["status"] == "partial"


def test_malformed_mcp_data_is_rejected_not_inserted_as_valid():
    executor = ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(), flight_provider=object(),
        mcp_client=_StubMcpClient(scenario="malformed"), a2a_client=_StubA2AClient(),
    )
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_stay_info"),
        _decision("search_stays", {"check_in": "2026-08-20", "check_out": "2026-08-25", "guest_count": 1}, "missing_stay_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    result = start_session(build_graph(executor, decider, decider), _request(), "t-mcp-malformed")
    assert result["observations"][0]["status"] == "provider_error"
    assert result["observations"][0]["envelope"] is None
    assert any("malformed_tool_result" in w for w in result["warnings"])


def test_a2a_timeout_produces_a_safe_partial_result():
    executor = ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(), flight_provider=object(),
        mcp_client=_StubMcpClient(), a2a_client=_StubA2AClient(scenario="timeout"),
    )
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_stay_info"),
        _decision("search_stays", {"check_in": "2026-08-20", "check_out": "2026-08-25", "guest_count": 1}, "missing_stay_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("call_istanbul_expert", {"question": "what to see?"}, "missing_local_expertise"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    result = start_session(build_graph(executor, decider, decider), _request(TRIP_REQUEST), "t-a2a-timeout")
    call_istanbul_obs = [o for o in result["observations"] if o["action"] == "call_istanbul_expert"][0]
    assert call_istanbul_obs["status"] == "timeout"
    assert result["final_result"]["status"] == "partial"


def test_invalid_system_b_artifact_is_rejected():
    executor = ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(), flight_provider=object(),
        mcp_client=_StubMcpClient(), a2a_client=_StubA2AClient(scenario="invalid_artifact"),
    )
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_stay_info"),
        _decision("search_stays", {"check_in": "2026-08-20", "check_out": "2026-08-25", "guest_count": 1}, "missing_stay_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("call_istanbul_expert", {"question": "what to see?"}, "missing_local_expertise"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    result = start_session(build_graph(executor, decider, decider), _request(TRIP_REQUEST), "t-a2a-invalid-artifact")
    call_istanbul_obs = [o for o in result["observations"] if o["action"] == "call_istanbul_expert"][0]
    assert call_istanbul_obs["status"] == "provider_error"
    assert call_istanbul_obs["envelope"] is None


def test_qdrant_unavailable_warning_from_system_b_is_preserved_verbatim():
    """System B's own itinerary artifact may carry `warnings` (e.g. a
    degraded-RAG note) -- this proves that content passes through
    unmodified into the observation, never stripped or reworded."""
    warning_text = "Qdrant unavailable; degraded to catalog-only scheduling."

    @dataclass
    class _A2AWithWarning:
        def call_istanbul_expert(self, payload):
            return {
                "status": "success",
                "result": {
                    "schema_version": "1.0.0", "session_id": payload["session_id"], "trace_id": payload["trace_id"],
                    "contract_version": "1.0.0", "recommended_base_candidate_id": "s1",
                    "accessibility_scores": [{"candidate_id": "s1", "score": 0.5}],
                    "selected_poi_ids": ["poi_x"],
                    "daily_plans": [{
                        "schema_version": "1.0.0", "date": "2026-08-20", "side": "european", "poi_ids": ["poi_x"],
                        "legs": [], "walking_minutes": 0.0, "transfer_minutes": 0.0, "activity_minutes": 0.0,
                        "meal_minutes": 0.0, "slack_minutes": 0.0, "warnings": [],
                    }],
                    "estimated_travel_minutes": 0.0, "expected_walking_minutes": 0.0, "side_crossings": 0,
                    "citations": [], "assumptions": [], "warnings": [warning_text],
                    "data_quality": {"schema_version": "1.0.0", "completeness": 0.5, "freshness": "cached", "assumptions": []},
                    "hard_constraint_validation_passed": True,
                },
            }

    executor = ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(), flight_provider=object(),
        mcp_client=_StubMcpClient(), a2a_client=_A2AWithWarning(),
    )
    decider = ScriptedDecisionProvider([
        _decision("call_travel_search", {}, "missing_stay_info"),
        _decision("search_stays", {"check_in": "2026-08-20", "check_out": "2026-08-25", "guest_count": 1}, "missing_stay_info"),
        _decision("travel_search_complete", {}, "all_required_evidence_present"),
        _decision("call_istanbul_expert", {"question": "what to see?"}, "missing_local_expertise"),
        _decision("synthesize", {}, "all_required_evidence_present"),
    ])
    result = start_session(build_graph(executor, decider, decider), _request(TRIP_REQUEST), "t-qdrant-warning")
    call_istanbul_obs = [o for o in result["observations"] if o["action"] == "call_istanbul_expert"][0]
    assert call_istanbul_obs["status"] == "success"
    assert warning_text in call_istanbul_obs["envelope"]["warnings"]
