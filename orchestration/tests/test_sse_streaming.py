"""Hermetic SSE streaming tests for the production System A service
(Checkpoint Phase 4 D.2A): media type, event ordering, stable IDs,
Last-Event-ID reconnect/replay, heartbeats, and no chain-of-thought/
prompt/secret leakage in any event payload.
"""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.system_a.sse import TERMINAL_STAGES
from orchestration.tests.conftest import ScriptedDecisionProvider, decision
from phase4.tools import FakeToolExecutor


def _weather_then_synthesize_provider() -> ScriptedDecisionProvider:
    return ScriptedDecisionProvider([
        decision("call_travel_search", {}, "missing_weather_info"),
        decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
        decision("travel_search_complete", {}, "all_required_evidence_present"),
        decision("synthesize", {}, "all_required_evidence_present"),
    ])


def _make_app(tmp_db_path: str, decision_provider_factory=None, sse_poll_seconds: float = 0.02, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("VOYAGER_SYSTEM_A_SSE_POLL_SECONDS", str(sse_poll_seconds))
    # A single shared provider INSTANCE serves both the supervisor and
    # specialist roles -- see the identical note in test_api_endpoints.py.
    provider = (decision_provider_factory or _weather_then_synthesize_provider)()
    return create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: provider,
        specialist_decision_provider_factory=lambda: provider,
        db_path=tmp_db_path,
        max_workers=2,
    )


def _parse_sse_stream(raw_text: str) -> list[dict]:
    """Parses raw SSE wire text into a list of {"id", "event", "data"}
    dicts, skipping comment/heartbeat lines -- a small, local, spec-literal
    parser (never a third-party SSE client) so this test verifies the
    exact wire format this service actually produces."""
    events = []
    current: dict = {}
    for line in raw_text.split("\n"):
        if line.startswith(":"):
            continue  # heartbeat/comment
        if line == "":
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith("id: "):
            current["id"] = line[len("id: "):]
        elif line.startswith("event: "):
            current["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[len("data: "):])
    return events


def test_sse_media_type_is_text_event_stream(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        response = client.get(f"/v1/runs/{run_id}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_sse_events_are_ordered_and_terminate_after_terminal_event(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        response = client.get(f"/v1/runs/{run_id}/events")

    events = _parse_sse_stream(response.text)
    stages = [e["event"] for e in events]
    assert stages == ["run_started", "action_started", "action_completed", "run_completed"]
    assert stages[-1] in TERMINAL_STAGES


def test_sse_event_ids_are_stable_and_unique(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        response = client.get(f"/v1/runs/{run_id}/events")

    events = _parse_sse_stream(response.text)
    ids = [e["id"] for e in events]
    assert len(ids) == len(set(ids))  # all unique
    for event_id in ids:
        assert len(event_id) == 36  # a UUID


def test_reconnect_with_last_event_id_replays_only_newer_events(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]

        first = client.get(f"/v1/runs/{run_id}/events")
        events = _parse_sse_stream(first.text)
        assert len(events) >= 2
        checkpoint_id = events[0]["id"]  # "run_started"

        second = client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": checkpoint_id})

    replayed = _parse_sse_stream(second.text)
    replayed_stages = [e["event"] for e in replayed]
    assert "run_started" not in replayed_stages  # already-seen event is not replayed
    assert replayed_stages == [e["event"] for e in events[1:]]


def test_reconnect_with_unknown_last_event_id_replays_from_the_beginning(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather please"})
        run_id = create_response.json()["run_id"]
        _poll_until_done(client, run_id)

        response = client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": "00000000-0000-0000-0000-000000000000"})

    events = _parse_sse_stream(response.text)
    assert events[0]["event"] == "run_started"


def test_sse_stream_for_unknown_run_id_returns_404(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        response = client.get("/v1/runs/00000000-0000-0000-0000-000000000000/events")
    assert response.status_code == 404


def test_no_chain_of_thought_prompt_or_secret_in_any_sse_payload(tmp_db_path, monkeypatch):
    provider = lambda: ScriptedDecisionProvider([
        decision("call_istanbul_expert", {"question": "what should I see?"}, "missing_local_expertise"),
        decision("synthesize", {}, "all_required_evidence_present"),
    ])
    app = _make_app(tmp_db_path, decision_provider_factory=provider, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "local expertise please"})
        run_id = create_response.json()["run_id"]
        response = client.get(f"/v1/runs/{run_id}/events")

    events = _parse_sse_stream(response.text)
    assert events, "expected at least one event"
    forbidden = ("api_key", "authorization", "bearer", "chain_of_thought", "system_prompt", "password", "secret")
    for event in events:
        serialized = json.dumps(event).lower()
        for term in forbidden:
            assert term not in serialized, f"forbidden term {term!r} leaked into SSE event {event}"


def _multi_action_delegation_provider() -> ScriptedDecisionProvider:
    return ScriptedDecisionProvider([
        decision("call_travel_search", {}, "missing_weather_info"),
        decision("get_weather", {"location": "Istanbul", "date_from": "2026-09-10", "date_to": "2026-09-10"}, "missing_weather_info"),
        decision("web_search", {"query": "Hagia Sophia hours"}, "missing_current_info"),
        decision("search_flights", {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1}),
        decision("travel_search_complete", {}, "all_required_evidence_present"),
        decision("synthesize", {}, "all_required_evidence_present"),
    ])


def test_delegated_multi_action_sse_events_are_genuinely_live_not_batched(tmp_db_path, monkeypatch):
    """Checkpoint Phase 4 D.3 correction pass: proves each specialist
    action's `action_started`/`action_completed` pair is emitted
    immediately around that ONE action -- never all three
    `action_started` events clustered together followed by all three
    `action_completed` events (the shape a post-hoc "derive events after
    the whole delegation already finished" implementation would have
    produced). `phase4.specialist.invoke_travel_search_specialist` drives
    the specialist graph with `.stream()`, and `RunService.
    _handle_specialist_event` is called synchronously as each real
    transition happens, so the wire order below is the true execution
    order, not a reconstruction."""
    app = _make_app(tmp_db_path, decision_provider_factory=_multi_action_delegation_provider, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather, hours, and flights please"})
        run_id = create_response.json()["run_id"]
        response = client.get(f"/v1/runs/{run_id}/events")
        final = client.get(f"/v1/runs/{run_id}").json()

    events = _parse_sse_stream(response.text)
    stages = [e["event"] for e in events]
    assert stages == [
        "run_started",
        "action_started", "action_completed",  # get_weather
        "action_started", "action_completed",  # web_search
        "action_started", "action_completed",  # search_flights
        "run_completed",
    ]
    # Each action_started/action_completed pair genuinely brackets ONE
    # action -- never a started event for action N+1 before the
    # completed event for action N (the "clustered/batched" shape).
    action_events = [e for e in events if e["event"] in ("action_started", "action_completed")]
    for i in range(0, len(action_events), 2):
        started, completed = action_events[i], action_events[i + 1]
        assert started["event"] == "action_started"
        assert completed["event"] == "action_completed"
        assert started["data"]["action"] == completed["data"]["action"]

    # Every action_completed/action_failed event corresponds to a real
    # observation actually present in the persisted final result --
    # never a phantom or duplicated event.
    result_actions = [obs["action"] for obs in final["result"]["observations"]]
    emitted_actions = [e["data"]["action"] for e in action_events if e["event"] == "action_started"]
    assert emitted_actions == result_actions


def test_reconnect_during_a_delegation_does_not_duplicate_specialist_events(tmp_db_path, monkeypatch):
    app = _make_app(tmp_db_path, decision_provider_factory=_multi_action_delegation_provider, monkeypatch=monkeypatch)
    with TestClient(app) as client:
        create_response = client.post("/v1/runs", json={"user_message": "weather, hours, and flights please"})
        run_id = create_response.json()["run_id"]

        first = client.get(f"/v1/runs/{run_id}/events")
        events = _parse_sse_stream(first.text)
        # Reconnect from partway through the specialist's own delegation
        # (right after the first action_completed) -- proves replay
        # resumes cleanly mid-delegation, never re-emitting or dropping
        # any specialist event.
        checkpoint_index = [e["event"] for e in events].index("action_completed")
        checkpoint_id = events[checkpoint_index]["id"]

        second = client.get(f"/v1/runs/{run_id}/events", headers={"Last-Event-ID": checkpoint_id})

    replayed = _parse_sse_stream(second.text)
    replayed_ids = [e["id"] for e in replayed]
    original_ids = [e["id"] for e in events]
    # The replayed tail is EXACTLY the events after the checkpoint --
    # neither missing one (a dropped specialist event) nor repeating one
    # already delivered before the checkpoint (a duplicated event).
    assert replayed_ids == original_ids[checkpoint_index + 1:]
    assert checkpoint_id not in replayed_ids  # the checkpoint event itself is never re-sent
    assert len(replayed_ids) == len(set(replayed_ids))  # no duplicate within the replay itself


def test_heartbeat_line_format_is_a_comment_never_a_named_event(tmp_db_path, monkeypatch):
    """Direct, isolated proof of the wire-format contract (not dependent
    on real elapsed time waiting for a heartbeat to actually fire)."""
    from orchestration.system_a.sse import format_heartbeat

    line = format_heartbeat()
    assert line.startswith(":")
    assert "event:" not in line
    assert "data:" not in line


def _poll_until_done(client: TestClient, run_id: str, attempts: int = 100, delay: float = 0.02) -> dict:
    data = {}
    for _ in range(attempts):
        response = client.get(f"/v1/runs/{run_id}")
        data = response.json()
        if data["status"] not in ("pending", "running"):
            return data
        time.sleep(delay)
    raise AssertionError(f"run did not reach a terminal status in time: {data}")
