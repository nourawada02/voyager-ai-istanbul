"""SQLite-backed run/session/event persistence for the production System A
service (Checkpoint Phase 4 D.2A). A deliberately separate concern from
`langgraph.checkpoint.sqlite.SqliteSaver` (which persists the graph's own
internal per-thread state so a session can be inspected/resumed): this
store persists the API-level facts a caller actually needs -- run status,
the original (already-Pydantic-validated) request, the final schema-valid
result once terminal, and a sanitized, ordered SSE event log -- so
`GET /v1/runs/{id}`, the SSE endpoint, and reconnect/replay all work
purely by reading this store, independent of whether the FastAPI process
that started a run is still the one serving a later request.

Never stores a credential, an authorization header, a raw prompt/model
response, or chain-of-thought: every value written here is either the
caller's own already-validated request, or a `phase4.graph` `final_result`
dict, which is structurally free of those by construction (see
`phase4/graph.py`'s own module docstring and
`phase4/tests/test_graph.py::test_no_raw_chain_of_thought_anywhere_in_state_trace_or_output`).

No persistent connection is held: every operation opens a short-lived
WAL-mode connection (bounded `busy_timeout`) and closes it in a
`finally` block before returning, so there is no connection lifecycle to
manage across requests/threads and no risk of a leaked handle -- the
simplest safe pattern for a store touched concurrently from FastAPI's
event loop (via `asyncio.to_thread`) and from independent run-execution
worker threads.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset({RunStatus.COMPLETED, RunStatus.DEGRADED, RunStatus.FAILED, RunStatus.CANCELLED})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    result_json TEXT,
    error_code TEXT,
    error_message TEXT,
    cancellation_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_run_events_run_id_sequence ON run_events(run_id, sequence);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    session_id: str
    trace_id: str
    status: RunStatus
    request: dict[str, Any]
    result: Optional[dict[str, Any]]
    error_code: Optional[str]
    error_message: Optional[str]
    cancellation_requested: bool
    created_at: str
    updated_at: str

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "RunRecord":
        return cls(
            run_id=row["run_id"],
            session_id=row["session_id"],
            trace_id=row["trace_id"],
            status=RunStatus(row["status"]),
            request=json.loads(row["request_json"]),
            result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
            error_code=row["error_code"],
            error_message=row["error_message"],
            cancellation_requested=bool(row["cancellation_requested"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class RunEventRecord:
    event_id: str
    run_id: str
    sequence: int
    stage: str
    payload: dict[str, Any]
    created_at: str

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "RunEventRecord":
        return cls(
            event_id=row["event_id"],
            run_id=row["run_id"],
            sequence=row["sequence"],
            stage=row["stage"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
        )


class RunStore:
    """Owns exactly one SQLite database file, explicitly configured (never
    guessed) at construction. Safe to share across threads: every method
    opens its own connection."""

    def __init__(self, db_path: str, busy_timeout_ms: int = 5000):
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=self.busy_timeout_ms / 1000.0, check_same_thread=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    # --- run lifecycle -----------------------------------------------------------

    def create_run(
        self, run_id: str, session_id: str, trace_id: str, request: dict[str, Any], idempotency_key: Optional[str]
    ) -> tuple[RunRecord, bool]:
        """Inserts a new `pending` run. If `idempotency_key` collides with
        an existing run, returns that existing run instead (created=False)
        -- the caller must not schedule execution a second time in that
        case. The `UNIQUE` constraint (not a read-then-write race) is the
        actual source of truth here, so this is safe under concurrent
        callers submitting the same key at the same time."""
        now = _utc_now_iso()
        conn = self._connect()
        try:
            try:
                conn.execute(
                    "INSERT INTO runs (run_id, session_id, trace_id, idempotency_key, status, request_json, "
                    "result_json, error_code, error_message, cancellation_requested, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 0, ?, ?)",
                    (run_id, session_id, trace_id, idempotency_key, RunStatus.PENDING.value, json.dumps(request), now, now),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                if idempotency_key is None:
                    raise
                row = conn.execute("SELECT * FROM runs WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
                if row is None:
                    raise
                return RunRecord._from_row(row), False
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return RunRecord._from_row(row), True
        finally:
            conn.close()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return RunRecord._from_row(row) if row is not None else None
        finally:
            conn.close()

    def mark_running(self, run_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ? AND status = ?",
                (RunStatus.RUNNING.value, _utc_now_iso(), run_id, RunStatus.PENDING.value),
            )
            conn.commit()
        finally:
            conn.close()

    def mark_terminal(
        self,
        run_id: str,
        status: RunStatus,
        result: Optional[dict[str, Any]],
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Persists the terminal status + final schema-valid result. Callers
        must call this BEFORE appending the corresponding terminal SSE
        event, so a client can never observe a terminal event whose status
        is not yet reflected by `GET /v1/runs/{id}`."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"mark_terminal called with a non-terminal status: {status!r}")
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET status = ?, result_json = ?, error_code = ?, error_message = ?, updated_at = ? "
                "WHERE run_id = ?",
                (status.value, json.dumps(result) if result is not None else None, error_code, error_message, _utc_now_iso(), run_id),
            )
            conn.commit()
        finally:
            conn.close()

    def request_cancellation(self, run_id: str) -> Optional[RunRecord]:
        """Idempotent: setting the flag to 1 a second (or first) time is
        the same operation either way. Returns the current record (with
        the flag now set), or None if the run does not exist. Never
        touches `status` directly -- a running graph observes the flag on
        its own next cancellation check (`phase4.graph`'s existing,
        unmodified cancellation-check precondition) and degrades itself;
        a not-yet-started run is caught by the same check before its
        first Decide."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE runs SET cancellation_requested = 1, updated_at = ? WHERE run_id = ?",
                (_utc_now_iso(), run_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return RunRecord._from_row(row) if row is not None else None
        finally:
            conn.close()

    def is_cancellation_requested(self, run_id: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute("SELECT cancellation_requested FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return bool(row["cancellation_requested"]) if row is not None else False
        finally:
            conn.close()

    def reconcile_incomplete_runs_on_startup(self) -> int:
        """Called once at service startup. Any run left in `pending`/
        `running` reflects a prior process lifetime that ended (crash,
        redeploy) before the run reached a terminal state -- this
        process has no in-memory task for it and cannot silently resume
        mid-graph execution, so honestly reporting `failed` is the only
        truthful option (never leaving it forever `running`). Returns the
        number of runs reconciled."""
        now = _utc_now_iso()
        conn = self._connect()
        try:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, error_code = ?, error_message = ?, updated_at = ? "
                "WHERE status IN (?, ?)",
                (
                    RunStatus.FAILED.value, "SERVICE_RESTARTED",
                    "The service restarted before this run reached a terminal state.",
                    now, RunStatus.PENDING.value, RunStatus.RUNNING.value,
                ),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    # --- events --------------------------------------------------------------------

    def append_event(self, run_id: str, event_id: str, stage: str, payload: dict[str, Any]) -> RunEventRecord:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COALESCE(MAX(sequence), -1) AS max_seq FROM run_events WHERE run_id = ?", (run_id,)).fetchone()
            sequence = row["max_seq"] + 1
            now = _utc_now_iso()
            conn.execute(
                "INSERT INTO run_events (event_id, run_id, sequence, stage, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, run_id, sequence, stage, json.dumps(payload), now),
            )
            conn.commit()
            return RunEventRecord(event_id=event_id, run_id=run_id, sequence=sequence, stage=stage, payload=payload, created_at=now)
        finally:
            conn.close()

    def list_events(self, run_id: str, after_sequence: int = -1) -> list[RunEventRecord]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM run_events WHERE run_id = ? AND sequence > ? ORDER BY sequence ASC",
                (run_id, after_sequence),
            ).fetchall()
            return [RunEventRecord._from_row(r) for r in rows]
        finally:
            conn.close()

    def get_event_sequence(self, run_id: str, event_id: str) -> Optional[int]:
        """Resolves a client-supplied `Last-Event-ID` to a sequence number
        for replay. Returns None if the event is unknown for this run
        (the caller then replays from the beginning, never raises)."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT sequence FROM run_events WHERE run_id = ? AND event_id = ?", (run_id, event_id)
            ).fetchone()
            return int(row["sequence"]) if row is not None else None
        finally:
            conn.close()
