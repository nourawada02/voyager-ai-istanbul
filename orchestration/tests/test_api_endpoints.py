"""Hermetic FastAPI endpoint tests for the production System A service
(Checkpoint Phase 4 D.2A). Every app is built via `create_app()` with a
`FakeToolExecutor`/`ScriptedDecisionProvider` injected -- no real socket,
no Qwen/Groq/SerpApi/Open-Meteo/Travel MCP/System B/Qdrant call anywhere
in this file (those real boundaries were already proven in Checkpoint
D.1's own cross-process gate).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.tests.conftest import ScriptedDecisionProvider, decision
from phase1.models import LocalItinerary
from phase4.tools import FakeToolExecutor
from providers.fx import FakeFxProvider

# Fixed test "today" -- every hardcoded trip date in this file stays safely
# in the future relative to this pinned clock forever, decoupling these
# hermetic tests from real wall-clock time (Manual QA remediation Q.1).
_TEST_WALL_CLOCK = lambda: datetime(2026, 8, 19, tzinfo=timezone.utc)


def _weather_then_synthesize_provider() -> ScriptedDecisionProvider:
    return ScriptedDecisionProvider([
        decision("call_travel_search", {}, "missing_weather_info"),
        decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
        decision("travel_search_complete", {}, "all_required_evidence_present"),
        decision("synthesize", {}, "all_required_evidence_present"),
    ])


def _make_app(tmp_db_path: str, decision_provider_factory=None, max_workers: int = 2, wall_clock=None, fx_provider_factory=None, tool_executor_factory=None):
    # A single shared provider INSTANCE serves both the supervisor and
    # specialist roles (`RunService` now calls two separate factories) --
    # each of these test scripts is one flat, sequentially-consumed
    # queue regardless of which role asks next, so both factories must
    # resolve to the exact same object, never two independent instances
    # each starting over at index 0.
    provider = (decision_provider_factory or _weather_then_synthesize_provider)()
    return create_app(
        tool_executor_factory=tool_executor_factory or FakeToolExecutor,
        decision_provider_factory=lambda: provider,
        specialist_decision_provider_factory=lambda: provider,
        db_path=tmp_db_path,
        max_workers=max_workers,
        wall_clock=wall_clock or _TEST_WALL_CLOCK,
        # Manual QA remediation Q.1 (§B): a deterministic fake by default
        # -- no test in this file makes a real FX network call just
        # because a trip's budget currency happens to be USD.
        fx_provider_factory=fx_provider_factory or (lambda: FakeFxProvider()),
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


# --- date validation (Manual QA remediation Q.1) --------------------------------------
#
# _TEST_WALL_CLOCK pins "today" to 2026-08-19: yesterday=2026-08-18,
# today=2026-08-19, tomorrow=2026-08-20. A decision provider that raises on
# any call proves Qwen is never consulted for a rejected request.


class _MustNeverBeCalledProvider:
    def generate(self, system: str, user: str) -> str:
        raise AssertionError("decision provider must never be called for a rejected trip request")


class _MustNeverBeCalledToolExecutor:
    """Pre-commit stabilization: a stronger proof for 'invalid input
    produces zero RunService execution/provider activity' than guarding
    the decision provider alone -- this guards the TOOL/provider dispatch
    layer itself (SerpApi/Travel MCP/Open-Meteo/System-B-A2A), so a
    rejected trip request is proven to never reach ANY of them, not just
    never reach Qwen."""

    def execute(self, action, arguments, context=None):
        raise AssertionError("tool executor must never be called for a rejected trip request")


def _trip_request(depart_date: str, return_date: str) -> dict:
    return {
        "origin": "BEY", "destination": "IST", "depart_date": depart_date, "return_date": return_date,
        "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
        "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
    }


def test_create_run_rejects_yesterday_depart_date_with_no_run_created_and_no_provider_call(tmp_db_path):
    app = _make_app(tmp_db_path, decision_provider_factory=_MustNeverBeCalledProvider, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-08-18", "2026-08-20")
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "TRIP_REQUEST_REJECTED"
        assert body["message"] == "depart_date_in_past"
        assert "internal" not in body["message"].lower() and "traceback" not in body["message"].lower()
        # No run was ever created: the store has zero rows for this app.
        list_response = client.get("/v1/runs/00000000-0000-0000-0000-000000000000")
        assert list_response.status_code == 404


def test_create_run_rejects_past_date_with_zero_run_service_or_provider_activity(tmp_db_path):
    """Pre-commit stabilization, required test 3+5: a genuinely past
    departure date returns the safe TRIP_REQUEST_REJECTED/
    depart_date_in_past response, with BOTH the decision provider
    (Qwen) AND the tool executor (every real provider/MCP/A2A dispatch
    path) guarded to raise if ever touched -- proves zero RunService
    execution activity, not merely zero Qwen calls."""
    app = _make_app(
        tmp_db_path, decision_provider_factory=_MustNeverBeCalledProvider,
        tool_executor_factory=_MustNeverBeCalledToolExecutor, wall_clock=_TEST_WALL_CLOCK,
    )
    trip_request = _trip_request("2026-08-18", "2026-08-20")  # yesterday relative to _TEST_WALL_CLOCK (2026-08-19)
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
        assert response.status_code == 422
        body = response.json()
        assert body["error_code"] == "TRIP_REQUEST_REJECTED"
        assert body["message"] == "depart_date_in_past"
        assert "internal" not in body["message"].lower() and "traceback" not in body["message"].lower()
        # No run row exists at all -- runs table is empty for this app.
        conn = sqlite3.connect(tmp_db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        finally:
            conn.close()
        assert count == 0


def test_create_run_accepts_today_depart_date(tmp_db_path):
    app = _make_app(tmp_db_path, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-08-19", "2026-08-22")
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 201


def test_create_run_accepts_tomorrow_depart_date(tmp_db_path):
    app = _make_app(tmp_db_path, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-08-20", "2026-08-23")
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 201


def test_create_run_rejects_return_before_depart_with_no_provider_call(tmp_db_path):
    app = _make_app(tmp_db_path, decision_provider_factory=_MustNeverBeCalledProvider, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-09-15", "2026-09-10")
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "TRIP_REQUEST_REJECTED"
    assert body["message"] == "return_date_before_depart_date"


def test_create_run_accepts_usd_currency(tmp_db_path):
    """Manual QA remediation Q.1 (user correction pass §B): USD is
    accepted now that a real FX-conversion capability exists
    (providers/fx_frankfurter.py) -- supersedes the earlier TRY-only
    restriction."""
    app = _make_app(tmp_db_path, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-09-10", "2026-09-15")
    trip_request["budget"] = {"amount_minor_units": 500000, "currency": "USD"}
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 201


def test_create_run_accepts_usd_currency_and_retains_usd_in_the_stored_trip_request(tmp_db_path):
    """Pre-commit stabilization, required test 7: a USD request must
    retain USD in the actually-submitted/stored budget -- never silently
    coerced to TRY anywhere between the HTTP boundary and persistence."""
    app = _make_app(tmp_db_path, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-09-10", "2026-09-15")
    trip_request["budget"] = {"amount_minor_units": 500000, "currency": "USD"}
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 201
    run_id = response.json()["run_id"]

    conn = sqlite3.connect(tmp_db_path)
    try:
        (request_json,) = conn.execute("SELECT request_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    stored_request = json.loads(request_json)
    assert stored_request["trip_request"]["budget"]["currency"] == "USD"
    assert stored_request["trip_request"]["budget"]["amount_minor_units"] == 500000


def test_create_run_rejects_eur_currency_no_verified_rate_source(tmp_db_path):
    app = _make_app(tmp_db_path, decision_provider_factory=_MustNeverBeCalledProvider, wall_clock=_TEST_WALL_CLOCK)
    trip_request = _trip_request("2026-09-10", "2026-09-15")
    trip_request["budget"] = {"amount_minor_units": 500000, "currency": "EUR"}
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 422
    assert response.json()["message"] == "unsupported_currency"
    body = response.json()
    assert body["error_code"] == "TRIP_REQUEST_REJECTED"
    assert body["message"] == "unsupported_currency"


def test_create_run_rejects_depart_date_that_is_in_the_past_only_under_a_later_injected_clock(tmp_db_path):
    """Proves the rejection genuinely comes from the injected wall_clock,
    not a hardcoded date: the exact same trip request that was accepted
    under _TEST_WALL_CLOCK (2026-08-19) is rejected once the app is built
    with a later clock."""
    later_clock = lambda: datetime(2026, 9, 20, tzinfo=timezone.utc)
    app = _make_app(tmp_db_path, decision_provider_factory=_MustNeverBeCalledProvider, wall_clock=later_clock)
    trip_request = _trip_request("2026-09-10", "2026-09-15")
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": trip_request})
    assert response.status_code == 422
    assert response.json()["message"] == "depart_date_in_past"


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
