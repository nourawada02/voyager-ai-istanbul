"""Hermetic tests for degraded/failed terminal states, sanitized internal
exception handling, no secret/prompt/chain-of-thought leakage anywhere in
the API/SSE/DB surface, and persistence surviving FastAPI app recreation
(Checkpoint Phase 4 D.2A).
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.system_a.run_store import RunStatus, RunStore
from orchestration.tests.conftest import ScriptedDecisionProvider, decision
from phase4.context import ExecutionContext
from phase4.models import Action
from phase4.tools import FakeToolExecutor


def _poll_until_terminal(client: TestClient, run_id: str, attempts: int = 200, delay: float = 0.02) -> dict:
    data = {}
    for _ in range(attempts):
        response = client.get(f"/v1/runs/{run_id}")
        data = response.json()
        if data["status"] not in ("pending", "running"):
            return data
        time.sleep(delay)
    raise AssertionError(f"run did not reach a terminal status in time: {data}")


# --- degraded terminal state -------------------------------------------------------------


def test_unavailable_weather_produces_a_degraded_run_not_a_failure(tmp_db_path):
    executor = FakeToolExecutor(scenario_by_action={Action.GET_WEATHER: "unavailable"})
    app = create_app(
        tool_executor_factory=lambda: executor,
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Nowhere", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "degraded"
    assert final["result"]["status"] == "partial"


def test_malformed_tool_result_produces_a_degraded_run_never_inserted_as_valid(tmp_db_path):
    executor = FakeToolExecutor(scenario_by_action={Action.GET_WEATHER: "malformed"})
    app = create_app(
        tool_executor_factory=lambda: executor,
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "degraded"
    obs = final["result"]["observations"][0]
    assert obs["status"] == "provider_error"
    assert obs["envelope"] is None


def test_decision_format_invalid_after_repairs_exhausted_produces_degraded(tmp_db_path):
    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: ScriptedDecisionProvider(["not valid json"] * 5),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "hi"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "degraded"
    assert final["result"]["status"] == "degraded"
    assert final["result"]["reason"] in ("decision_format_invalid",)


# --- failed terminal state (genuine internal exception) -----------------------------------


def _make_raising_tool_executor(ephemeral_secret: str):
    """Builds a tool executor whose exception message embeds a runtime-
    generated, ephemeral, credential-SHAPED value -- `secrets.token_hex()`
    output, never a real-provider key prefix (no `sk-`, `AIza`, `ghp_`,
    ...; pure hex has none of those). Simulates an unforeseen internal
    defect (not a provider-reported failure status, which the graph
    already handles as `unavailable`/`provider_error` -- a genuine
    unhandled exception, e.g. a bug or a provider error response that
    happened to echo back a credential) to prove the service converts it
    to a sanitized `failed` result rather than crashing the worker thread
    or leaking any part of that exception's own text anywhere."""

    class _RaisingToolExecutor:
        def execute(self, action: Action, arguments: dict, context: Optional[ExecutionContext] = None) -> dict[str, Any]:
            raise RuntimeError(
                f"simulated internal defect with a fake local path C:\\Users\\Admin\\secret\\file.py "
                f"and API_KEY={ephemeral_secret}"
            )

    return _RaisingToolExecutor()


def test_internal_exception_produces_a_sanitized_failed_result_never_a_raw_traceback(tmp_db_path, caplog):
    # A fresh, unpredictable ephemeral value generated at test-run time --
    # never a hardcoded credential-shaped literal, never a real-provider
    # key prefix. The one and only place this exact value is deliberately
    # placed is inside the raised exception below; every assertion after
    # that proves it never escapes that one local scope.
    ephemeral_secret = secrets.token_hex(32)
    logger_name = "voyager.system_a.service"

    app = create_app(
        tool_executor_factory=lambda: _make_raising_tool_executor(ephemeral_secret),
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        with TestClient(app) as client:
            create_response = client.post("/v1/runs", json={"user_message": "weather please"})
            run_id = create_response.json()["run_id"]
            final = _poll_until_terminal(client, run_id)
            events_response = client.get(f"/v1/runs/{run_id}/events")

    assert final["status"] == "failed"

    store = RunStore(tmp_db_path)
    raw_record = store.get_run(run_id)
    raw_events = store.list_events(run_id)

    log_text = "\n".join(
        record.getMessage() + (str(record.exc_info) if record.exc_info else "") for record in caplog.records
    )

    # Every surface the approval explicitly named: API responses, SSE
    # events, logs, SQLite records, fingerprints (covered by the raw
    # event/record dumps below, which include every persisted
    # fingerprint field), and str()/repr() of the persisted record --
    # checking for the raw token itself is a strict superset of checking
    # str(exc)/repr(exc) specifically, since either of those forms would
    # still contain this exact substring if it leaked at all.
    surfaces: dict[str, str] = {
        "API status response (GET /v1/runs/{id})": json.dumps(final, default=str),
        "SSE events (GET /v1/runs/{id}/events)": events_response.text,
        "logs": log_text,
        "SQLite raw run record": json.dumps(raw_record.__dict__, default=str),
        "SQLite raw event records": json.dumps([e.__dict__ for e in raw_events], default=str),
        "str(raw SQLite record)": str(raw_record),
        "repr(raw SQLite record)": repr(raw_record),
    }
    for surface_name, content in surfaces.items():
        assert ephemeral_secret not in content, f"ephemeral secret leaked into {surface_name}"

    # These broader forbidden terms apply to the caller-facing surfaces
    # only -- the server-side log line is deliberately allowed to name
    # the exception's own class ("RuntimeError"), which is safe,
    # non-sensitive, structural debugging information (architecture.md
    # §13.4's own "logged server-side only, keyed by trace_id" allowance),
    # never applied to logs here.
    serialized_status = json.dumps(final).lower()
    serialized_events = events_response.text.lower()
    for forbidden in ("traceback", "runtimeerror", "c:\\users", "api_key", "secret"):
        assert forbidden not in serialized_status, f"{forbidden!r} leaked into GET /v1/runs/{{id}}"
        assert forbidden not in serialized_events, f"{forbidden!r} leaked into SSE events"

    # The log line is still held to the strictest, most important check:
    # the local file path and the ephemeral secret embedded in the same
    # exception message must never appear there either -- only the
    # exception's class name is expected/acceptable.
    assert "c:\\users" not in log_text.lower(), "local file path leaked into logs"


# --- persistence across app recreation ----------------------------------------------------


def test_status_and_result_survive_recreating_the_fastapi_app(tmp_db_path):
    def build():
        return create_app(
            tool_executor_factory=FakeToolExecutor,
            decision_provider_factory=lambda: ScriptedDecisionProvider([
                decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
                decision("synthesize", {}, "all_required_evidence_present"),
            ]),
            db_path=tmp_db_path, max_workers=1,
        )

    app1 = build()
    with TestClient(app1) as client1:
        create_response = client1.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        first_final = _poll_until_terminal(client1, run_id)
    assert first_final["status"] == "completed"

    # A brand-new app instance, new RunService/RunStore objects, same db file.
    app2 = build()
    with TestClient(app2) as client2:
        response = client2.get(f"/v1/runs/{run_id}")
    assert response.status_code == 200
    second_view = response.json()
    assert second_view["status"] == "completed"
    assert second_view["result"] == first_final["result"]


def test_abandoned_running_run_is_reconciled_to_failed_on_app_recreation(tmp_db_path):
    """Simulates a process crash mid-run: a row is left `running` with no
    in-process task backing it (this test never actually starts
    execution for it -- it writes the row directly via RunStore, exactly
    mimicking what a prior process's now-gone worker thread would have
    left behind). A freshly created app must not report this run as
    dishonestly still `running` forever."""
    store = RunStore(tmp_db_path)
    orphaned_run_id = "orphaned-run-id"
    store.create_run(orphaned_run_id, "s", "t", {"user_message": "x", "trip_request": None}, idempotency_key=None)
    store.mark_running(orphaned_run_id)

    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: ScriptedDecisionProvider([]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        response = client.get(f"/v1/runs/{orphaned_run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"


# --- clean database shutdown ---------------------------------------------------------------


def test_app_shutdown_closes_the_worker_pool_cleanly(tmp_db_path):
    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "hi"})
        run_id = create_response.json()["run_id"]
        _poll_until_terminal(client, run_id)
    # Exiting the `with` block ran the app's lifespan shutdown, which
    # calls RunService.shutdown(wait=True) and joins every worker thread.
    # A closed ThreadPoolExecutor refuses new work -- the standard,
    # public way to prove it was actually shut down rather than merely
    # abandoned.
    with pytest.raises(RuntimeError):
        app.state.service._pool.submit(lambda: None)
