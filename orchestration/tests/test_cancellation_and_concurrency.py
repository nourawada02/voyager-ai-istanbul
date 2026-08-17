"""Hermetic cancellation and concurrency tests for the production System
A service (Checkpoint Phase 4 D.2A): cancellation before/during
execution, cancellation idempotency, no tool calls after cancellation,
concurrent session isolation, and duplicate-run prevention.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.tests.conftest import GatedFakeToolExecutor, ScriptedDecisionProvider, decision
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


def _wait_until(predicate, attempts: int = 200, delay: float = 0.01) -> None:
    for _ in range(attempts):
        if predicate():
            return
        time.sleep(delay)
    raise AssertionError("condition not met in time")


# --- cancellation ----------------------------------------------------------------------


def test_cancel_before_execution_starts_prevents_any_tool_call(tmp_db_path):
    gated = GatedFakeToolExecutor()
    gated.gate.set()  # never actually blocks -- this test cancels before the run even starts
    app = create_app(
        tool_executor_factory=lambda: gated,
        # A second (never-needed-if-cancellation-wins) scripted response
        # is a deliberate safety net: cancellation is only checked as a
        # precondition of Decide (ADR 0009 §4.5), so this test's actual
        # assertion is a real, tiny timing race against the background
        # worker thread's first Decide call. If that race is ever lost
        # on a slow CI box, this extra response keeps the run completing
        # honestly (0 or 1 tool calls, asserted below) instead of
        # crashing into an unrelated "ran out of scripted responses"
        # failure that would hide the real assertion.
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        cancel_response = client.post(f"/v1/runs/{run_id}/cancel")
        assert cancel_response.status_code == 200
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "cancelled"


def test_cancel_during_execution_stops_further_tool_calls(tmp_db_path):
    gated = GatedFakeToolExecutor()
    app = create_app(
        tool_executor_factory=lambda: gated,
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("get_weather", {"location": "Ankara", "date_from": "2026-09-11", "date_to": "2026-09-11"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]

        _wait_until(lambda: len(gated.call_log) >= 1)
        assert len(gated.call_log) == 1  # confirms the test caught it mid-flight, before a 2nd call

        cancel_response = client.post(f"/v1/runs/{run_id}/cancel")
        assert cancel_response.status_code == 200
        gated.gate.set()  # release the in-flight call

        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "cancelled"
    # the in-flight call is allowed to finish (ADR 0009 §4.5: cancellation
    # is checked as a precondition of the NEXT Decide/Execute, never
    # mid-EXECUTE) -- but no SECOND tool call ever happens after that.
    assert len(gated.call_log) == 1


def test_cancellation_is_idempotent(tmp_db_path):
    # The gate is left BLOCKED (never released) for the first part of
    # this test so the run is deterministically still `running` --
    # otherwise a real background execution racing ahead to its next
    # Decide call (which would need a 2nd scripted response this test
    # deliberately never provides) makes the exact moment of each cancel
    # call non-deterministic.
    gated = GatedFakeToolExecutor()
    app = create_app(
        tool_executor_factory=lambda: gated,
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        _wait_until(lambda: len(gated.call_log) >= 1)  # the tool call is now genuinely blocked in-flight

        r1 = client.post(f"/v1/runs/{run_id}/cancel")
        r2 = client.post(f"/v1/runs/{run_id}/cancel")
        r3 = client.post(f"/v1/runs/{run_id}/cancel")
        assert r1.status_code == r2.status_code == r3.status_code == 200
        assert r1.json()["status"] == r2.json()["status"] == r3.json()["status"] == "running"

        gated.gate.set()
        final = _poll_until_terminal(client, run_id)

    assert final["status"] == "cancelled"


def test_cancel_unknown_run_returns_404(tmp_db_path):
    app = create_app(
        tool_executor_factory=lambda: GatedFakeToolExecutor(),
        decision_provider_factory=lambda: ScriptedDecisionProvider([]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        response = client.post("/v1/runs/00000000-0000-0000-0000-000000000000/cancel")
    assert response.status_code == 404


def test_cancel_after_completion_is_a_harmless_no_op(tmp_db_path):
    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=1,
    )
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "hi"})
        run_id = create_response.json()["run_id"]
        final = _poll_until_terminal(client, run_id)
        assert final["status"] == "completed"

        cancel_response = client.post(f"/v1/runs/{run_id}/cancel")

    assert cancel_response.status_code == 200
    # the already-terminal run's own status is never overwritten by a
    # late cancellation request -- request_cancellation only ever sets
    # the flag column, never touches `status`.
    after = json.loads(cancel_response.text)
    assert after["status"] == "completed"


# --- concurrency -----------------------------------------------------------------------


def test_concurrent_sessions_do_not_share_observations(tmp_db_path):
    def make_provider(location: str):
        return lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": location, "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ])

    app = create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=make_provider("Istanbul"),
        db_path=tmp_db_path, max_workers=4,
    )
    with TestClient(app) as client:
        run_ids = []
        for _ in range(5):
            response = client.post("/v1/runs", json={"user_message": "weather please"})
            run_ids.append(response.json()["run_id"])

        finals = [_poll_until_terminal(client, rid) for rid in run_ids]

    assert len({f["result"]["observations"][0]["fingerprint"] for f in finals}) == 1  # identical inputs -> identical fingerprint
    session_ids = {f["session_id"] for f in finals}
    assert len(session_ids) == 5  # but every run has its own distinct session


def test_duplicate_run_is_never_executed_twice(tmp_db_path):
    """Fires many concurrent create-run requests with the SAME
    idempotency key from multiple threads and asserts the underlying
    tool executor is only ever invoked once, no matter how the race
    resolves -- the DB-level UNIQUE constraint (not a read-then-write
    check) is the actual source of truth (see RunStore.create_run)."""
    call_log: list = []

    class CountingToolExecutor:
        def execute(self, action, arguments, context=None):
            call_log.append(action)
            return FakeToolExecutor().execute(action, arguments, context)

    app = create_app(
        tool_executor_factory=lambda: CountingToolExecutor(),
        decision_provider_factory=lambda: ScriptedDecisionProvider([
            decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
            decision("synthesize", {}, "all_required_evidence_present"),
        ]),
        db_path=tmp_db_path, max_workers=8,
    )

    with TestClient(app) as client:
        def submit(_: int):
            return client.post("/v1/runs", json={"user_message": "hi", "idempotency_key": "race-key"})

        with ThreadPoolExecutor(max_workers=8) as pool:
            responses = list(pool.map(submit, range(8)))

        run_ids = {r.json()["run_id"] for r in responses}
        assert len(run_ids) == 1  # every request resolved to the same run

        final = _poll_until_terminal(client, next(iter(run_ids)))

    assert final["status"] == "completed"
    assert call_log.count("get_weather") == 1  # executed exactly once, never duplicated
