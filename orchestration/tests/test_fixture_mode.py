"""Hermetic tests for System A's explicit deterministic demo/fixture mode
(Checkpoint Phase 4 D.2B). No real Qwen/SerpApi/Open-Meteo/Travel MCP/
System B/Qdrant call is ever made -- fixture mode is, by construction,
`phase4.tools.FakeToolExecutor` + `FixtureDecisionProvider`, both fully
in-process and network-free.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.system_a.config import SystemAModeConfigurationError, is_fixture_mode, system_a_mode
from orchestration.system_a.fixture_decision_provider import FixtureDecisionProvider
from phase4.tools import FakeToolExecutor

TRIP_REQUEST = {
    "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
    "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


def _poll_until_terminal(client: TestClient, run_id: str, attempts: int = 200, delay: float = 0.02) -> dict:
    data = {}
    for _ in range(attempts):
        response = client.get(f"/v1/runs/{run_id}")
        data = response.json()
        if data["status"] not in ("pending", "running"):
            return data
        time.sleep(delay)
    raise AssertionError(f"run did not reach a terminal status in time: {data}")


# --- config-level mode resolution -------------------------------------------------------


def test_unset_mode_defaults_to_real(monkeypatch):
    monkeypatch.delenv("VOYAGER_SYSTEM_A_MODE", raising=False)
    assert system_a_mode() == "real"
    assert is_fixture_mode() is False


def test_explicit_fixture_value_selects_fixture_mode(monkeypatch):
    monkeypatch.setenv("VOYAGER_SYSTEM_A_MODE", "fixture")
    assert system_a_mode() == "fixture"
    assert is_fixture_mode() is True


def test_unrecognized_mode_value_raises_loudly_never_silently_falls_back(monkeypatch):
    monkeypatch.setenv("VOYAGER_SYSTEM_A_MODE", "bogus")
    try:
        system_a_mode()
        raise AssertionError("expected SystemAModeConfigurationError")
    except SystemAModeConfigurationError:
        pass


# --- end-to-end fixture-mode run through the real API -----------------------------------


def test_fixture_mode_completes_a_real_run_through_the_public_api(tmp_db_path):
    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=FixtureDecisionProvider,
        db_path=tmp_db_path, max_workers=1, mode="fixture",
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["mode"] == "fixture"

        create_response = client.post("/v1/runs", json={"user_message": "Plan my trip", "trip_request": TRIP_REQUEST})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "completed"
    actions = [obs["action"] for obs in final["result"]["observations"]]
    assert actions == ["get_weather", "search_flights", "search_stays", "call_istanbul_expert"]
    assert all(obs["status"] == "success" for obs in final["result"]["observations"])


def test_fixture_mode_with_no_structured_trip_request_synthesizes_immediately(tmp_db_path):
    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=FixtureDecisionProvider,
        db_path=tmp_db_path, max_workers=1, mode="fixture",
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "hello"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] in ("completed", "degraded")  # an empty-evidence Synthesize is honestly reported
    assert final["result"]["observations"] == []


def test_real_mode_is_still_the_default_label_on_health(tmp_db_path):
    """Does not construct a real ProductionToolExecutor/QwenDecisionProvider
    (that would need real credentials/network) -- only proves `mode`
    defaults to "real" when explicitly passed through create_app, exactly
    as entrypoint.py does when VOYAGER_SYSTEM_A_MODE is unset."""
    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=FixtureDecisionProvider,
        db_path=tmp_db_path, max_workers=1,  # mode intentionally omitted -> default "real"
    )
    with TestClient(app) as client:
        health = client.get("/health").json()
    assert health["mode"] == "real"
