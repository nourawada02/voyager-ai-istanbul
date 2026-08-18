"""Hermetic FastAPI endpoint tests for the production System A service
(Checkpoint Phase 4 D.2A). Every app is built via `create_app()` with a
`FakeToolExecutor`/`ScriptedDecisionProvider` injected -- no real socket,
no Qwen/Groq/SerpApi/Open-Meteo/Travel MCP/System B/Qdrant call anywhere
in this file (those real boundaries were already proven in Checkpoint
D.1's own cross-process gate).
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.tests.conftest import ScriptedDecisionProvider, decision
from phase1.models import LocalItinerary
from phase4.tools import FakeToolExecutor


def _weather_then_synthesize_provider() -> ScriptedDecisionProvider:
    return ScriptedDecisionProvider([
        decision("call_travel_search", {}, "missing_weather_info"),
        decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
        decision("travel_search_complete", {}, "all_required_evidence_present"),
        decision("synthesize", {}, "all_required_evidence_present"),
    ])


def _make_app(tmp_db_path: str, decision_provider_factory=None, max_workers: int = 2):
    # A single shared provider INSTANCE serves both the supervisor and
    # specialist roles (`RunService` now calls two separate factories) --
    # each of these test scripts is one flat, sequentially-consumed
    # queue regardless of which role asks next, so both factories must
    # resolve to the exact same object, never two independent instances
    # each starting over at index 0.
    provider = (decision_provider_factory or _weather_then_synthesize_provider)()
    return create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: provider,
        specialist_decision_provider_factory=lambda: provider,
        db_path=tmp_db_path,
        max_workers=max_workers,
    )


def _poll_until_terminal(client: TestClient, run_id: str, attempts: int = 100, delay: float = 0.02) -> dict:
    data = {}
    for _ in range(attempts):
        response = client.get(f"/v1/runs/{run_id}")
        data = response.json()
        if data["status"] not in ("pending", "running"):
            return data
        time.sleep(delay)
    raise AssertionError(f"run did not reach a terminal status in time: {data}")


# --- health --------------------------------------------------------------------------


def test_health_is_deterministic_and_leaks_no_config(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    serialized = str(body).lower()
    assert tmp_db_path.lower() not in serialized
    assert "api_key" not in serialized and "secret" not in serialized


# --- run creation ----------------------------------------------------------------------


def test_create_run_with_only_user_message_returns_stable_ids(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "What's the weather in Istanbul?"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "pending"
    assert len(body["run_id"]) == 36
    assert len(body["session_id"]) == 36
    assert body["run_id"] != body["session_id"]


def test_create_run_with_full_trip_request_is_accepted(tmp_db_path):
    app = _make_app(tmp_db_path)
    trip_request = {
        "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
        "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
        "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
    }
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 201


def test_create_run_rejects_unknown_field_with_error_envelope(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "hi", "not_a_real_field": 1})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "REQUEST_VALIDATION_FAILED"
    assert "trace_id" in body and "message" in body
    assert "internal" not in body["message"].lower()


def test_create_run_rejects_malformed_date(tmp_db_path):
    app = _make_app(tmp_db_path)
    trip_request = {
        "origin": "BEY", "destination": "IST", "depart_date": "not-a-date", "return_date": "2026-09-15",
        "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
        "preferences": {"interests": [], "pace": "moderate", "language": "en", "mobility_constraints": []},
    }
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "hi", "trip_request": trip_request})
    assert response.status_code == 422


def test_create_run_rejects_malformed_passenger_count(tmp_db_path):
    app = _make_app(tmp_db_path)
    trip_request = {
        "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
        "traveler_count": 0,  # below minimum of 1
        "budget": {"amount_minor_units": 500000, "currency": "TRY"},
        "preferences": {"interests": [], "pace": "moderate", "language": "en", "mobility_constraints": []},
    }
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "hi", "trip_request": trip_request})
    assert response.status_code == 422


def test_create_run_rejects_empty_user_message(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": ""})
    assert response.status_code == 422


def test_idempotent_duplicate_submission_returns_the_same_run(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        r1 = client.post("/v1/runs", json={"user_message": "hi", "idempotency_key": "same-key"})
        r2 = client.post("/v1/runs", json={"user_message": "hi again", "idempotency_key": "same-key"})
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r1.json()["run_id"] == r2.json()["run_id"]


def test_different_idempotency_keys_create_different_runs(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        r1 = client.post("/v1/runs", json={"user_message": "hi", "idempotency_key": "key-a"})
        r2 = client.post("/v1/runs", json={"user_message": "hi", "idempotency_key": "key-b"})
    assert r1.json()["run_id"] != r2.json()["run_id"]


# --- status progression and result -----------------------------------------------------


def test_status_progresses_from_pending_to_completed_with_schema_valid_result(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "completed"
    assert final["result"]["status"] == "success"
    assert len(final["result"]["observations"]) == 1
    assert final["result"]["observations"][0]["action"] == "get_weather"


def test_get_run_never_exposes_internal_graph_state(tmp_db_path):
    """The status response must contain only run_id/session_id/status/
    timestamps/result -- never trace/pending_action/repair_count/
    executed_fingerprints or any other internal PlannerState field."""
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert set(final.keys()) == {"run_id", "session_id", "status", "created_at", "updated_at", "result"}
    forbidden_top_level_keys = {"trace", "pending_action", "repair_count", "executed_fingerprints", "tool_call_count_by_action"}
    assert forbidden_top_level_keys.isdisjoint(final.keys())
    assert forbidden_top_level_keys.isdisjoint(final["result"].keys())


def test_get_run_for_unknown_id_returns_404_error_envelope(tmp_db_path):
    app = _make_app(tmp_db_path)
    with TestClient(app) as client:
        response = client.get("/v1/runs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["error_code"] == "RUN_NOT_FOUND"


def test_full_five_tool_plan_result_validates_against_local_itinerary_where_applicable(tmp_db_path):
    provider = lambda: ScriptedDecisionProvider([
        decision("call_istanbul_expert", {"question": "what should I see?"}, "missing_local_expertise"),
        decision("synthesize", {}, "all_required_evidence_present"),
    ])
    app = _make_app(tmp_db_path, decision_provider_factory=provider)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "local expertise please"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "completed"
    obs = final["result"]["observations"][0]
    assert obs["action"] == "call_istanbul_expert"
    LocalItinerary.model_validate(obs["envelope"])
