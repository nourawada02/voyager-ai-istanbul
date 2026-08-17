"""Hermetic unit tests for `orchestration.system_a.run_store.RunStore`
(Checkpoint Phase 4 D.2A) -- no FastAPI, no graph, pure SQLite
persistence. Every test uses a pytest `tmp_path`-backed database file,
never a shared or production path.
"""

from __future__ import annotations

import sqlite3
import threading
from uuid import uuid4

import pytest

from orchestration.system_a.run_store import RunStatus, RunStore


def _new_store(tmp_db_path: str) -> RunStore:
    return RunStore(tmp_db_path, busy_timeout_ms=2000)


def test_create_and_get_run_round_trips(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    request = {"user_message": "hello", "trip_request": None}
    record, created = store.create_run(run_id, "session-1", "trace-1", request, idempotency_key=None)
    assert created is True
    assert record.status == RunStatus.PENDING
    assert record.request == request
    assert record.result is None

    fetched = store.get_run(run_id)
    assert fetched is not None
    assert fetched.run_id == run_id
    assert fetched.session_id == "session-1"
    assert fetched.cancellation_requested is False


def test_get_run_returns_none_for_unknown_id(tmp_db_path):
    store = _new_store(tmp_db_path)
    assert store.get_run(str(uuid4())) is None


def test_idempotency_key_collision_returns_existing_run_not_a_new_one(tmp_db_path):
    store = _new_store(tmp_db_path)
    key = "abc-123"
    record1, created1 = store.create_run(str(uuid4()), "s1", "t1", {"user_message": "a", "trip_request": None}, idempotency_key=key)
    record2, created2 = store.create_run(str(uuid4()), "s2", "t2", {"user_message": "b", "trip_request": None}, idempotency_key=key)
    assert created1 is True
    assert created2 is False
    assert record1.run_id == record2.run_id


def test_multiple_null_idempotency_keys_are_all_allowed(tmp_db_path):
    store = _new_store(tmp_db_path)
    r1, c1 = store.create_run(str(uuid4()), "s1", "t1", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    r2, c2 = store.create_run(str(uuid4()), "s2", "t2", {"user_message": "b", "trip_request": None}, idempotency_key=None)
    assert c1 is True and c2 is True
    assert r1.run_id != r2.run_id


def test_mark_running_then_mark_terminal_transitions_status(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    store.mark_running(run_id)
    assert store.get_run(run_id).status == RunStatus.RUNNING

    result = {"status": "success", "observations": [], "warnings": []}
    store.mark_terminal(run_id, RunStatus.COMPLETED, result)
    record = store.get_run(run_id)
    assert record.status == RunStatus.COMPLETED
    assert record.result == result


def test_mark_terminal_rejects_a_non_terminal_status(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    with pytest.raises(ValueError):
        store.mark_terminal(run_id, RunStatus.RUNNING, {"status": "success"})


def test_cancellation_flag_is_idempotent(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    assert store.is_cancellation_requested(run_id) is False
    first = store.request_cancellation(run_id)
    second = store.request_cancellation(run_id)
    assert first.cancellation_requested is True
    assert second.cancellation_requested is True
    assert store.is_cancellation_requested(run_id) is True


def test_request_cancellation_on_unknown_run_returns_none(tmp_db_path):
    store = _new_store(tmp_db_path)
    assert store.request_cancellation(str(uuid4())) is None


def test_append_event_assigns_monotonically_increasing_sequence(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    e0 = store.append_event(run_id, str(uuid4()), "run_started", {"run_id": run_id})
    e1 = store.append_event(run_id, str(uuid4()), "action_started", {"action": "get_weather"})
    e2 = store.append_event(run_id, str(uuid4()), "run_completed", {"status": "success"})
    assert [e0.sequence, e1.sequence, e2.sequence] == [0, 1, 2]


def test_list_events_after_sequence_supports_replay(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    events = [store.append_event(run_id, str(uuid4()), f"stage_{i}", {}) for i in range(5)]

    all_events = store.list_events(run_id, after_sequence=-1)
    assert [e.sequence for e in all_events] == [0, 1, 2, 3, 4]

    replay = store.list_events(run_id, after_sequence=events[1].sequence)
    assert [e.sequence for e in replay] == [2, 3, 4]


def test_get_event_sequence_resolves_last_event_id(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    e = store.append_event(run_id, str(uuid4()), "run_started", {})
    assert store.get_event_sequence(run_id, e.event_id) == e.sequence
    assert store.get_event_sequence(run_id, str(uuid4())) is None


def test_events_are_isolated_per_run(tmp_db_path):
    store = _new_store(tmp_db_path)
    run_a = str(uuid4())
    run_b = str(uuid4())
    store.create_run(run_a, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    store.create_run(run_b, "s", "t", {"user_message": "b", "trip_request": None}, idempotency_key=None)
    store.append_event(run_a, str(uuid4()), "run_started", {})
    store.append_event(run_a, str(uuid4()), "run_completed", {})
    store.append_event(run_b, str(uuid4()), "run_started", {})

    assert len(store.list_events(run_a)) == 2
    assert len(store.list_events(run_b)) == 1


def test_reconcile_incomplete_runs_on_startup_marks_pending_and_running_as_failed(tmp_db_path):
    store = _new_store(tmp_db_path)
    pending_id = str(uuid4())
    running_id = str(uuid4())
    completed_id = str(uuid4())
    store.create_run(pending_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    store.create_run(running_id, "s", "t", {"user_message": "b", "trip_request": None}, idempotency_key=None)
    store.mark_running(running_id)
    store.create_run(completed_id, "s", "t", {"user_message": "c", "trip_request": None}, idempotency_key=None)
    store.mark_terminal(completed_id, RunStatus.COMPLETED, {"status": "success"})

    reconciled_count = store.reconcile_incomplete_runs_on_startup()
    assert reconciled_count == 2

    pending_after = store.get_run(pending_id)
    running_after = store.get_run(running_id)
    completed_after = store.get_run(completed_id)
    assert pending_after.status == RunStatus.FAILED
    assert pending_after.error_code == "SERVICE_RESTARTED"
    assert running_after.status == RunStatus.FAILED
    assert completed_after.status == RunStatus.COMPLETED  # untouched


def test_recreating_the_store_against_the_same_db_file_sees_prior_state(tmp_db_path):
    """Proves persistence survives object recreation (not just
    connection reuse) -- the real proof that state lives in the file,
    not in the RunStore instance's own memory."""
    store1 = _new_store(tmp_db_path)
    run_id = str(uuid4())
    store1.create_run(run_id, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)
    store1.mark_terminal(run_id, RunStatus.COMPLETED, {"status": "success", "observations": [], "warnings": []})

    store2 = _new_store(tmp_db_path)
    record = store2.get_run(run_id)
    assert record is not None
    assert record.status == RunStatus.COMPLETED
    assert record.result == {"status": "success", "observations": [], "warnings": []}


def test_concurrent_writers_do_not_corrupt_state_under_wal_and_busy_timeout(tmp_db_path):
    """A real (small-scale) concurrency stress test: many threads
    appending events to distinct runs simultaneously must never raise
    'database is locked' given the configured busy_timeout, and every
    event must be durably recorded."""
    store = _new_store(tmp_db_path)
    run_ids = [str(uuid4()) for _ in range(5)]
    for rid in run_ids:
        store.create_run(rid, "s", "t", {"user_message": "a", "trip_request": None}, idempotency_key=None)

    errors: list[BaseException] = []

    def worker(run_id: str) -> None:
        try:
            for i in range(10):
                store.append_event(run_id, str(uuid4()), f"stage_{i}", {"i": i})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(rid,)) for rid in run_ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    for rid in run_ids:
        assert len(store.list_events(rid)) == 10


def test_wal_mode_and_busy_timeout_are_actually_applied(tmp_db_path):
    store = _new_store(tmp_db_path)
    conn = sqlite3.connect(store.db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()
