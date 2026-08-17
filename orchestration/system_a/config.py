"""Explicit, environment-only configuration for the production System A
service (Checkpoint Phase 4 D.2A). Every value has a safe, documented
default so the service starts without any required environment variable
-- matching this project's existing "no hardcoded secrets, environment
read fresh, never guessed" convention (e.g. `phase4.qwen_client`), even
though nothing here is a secret. Never reads/returns a credential; only
non-sensitive operational configuration (a local file path and small
integers).
"""

from __future__ import annotations

import os
from pathlib import Path

_ORCHESTRATION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = str(_ORCHESTRATION_DIR / "data" / "system_a.db")

DEFAULT_MAX_WORKERS = 4
DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_SSE_POLL_SECONDS = 0.1
DEFAULT_SSE_HEARTBEAT_SECONDS = 15.0


def db_path() -> str:
    """The single explicit, configurable SQLite database path -- both the
    run/session/event store and the per-run LangGraph checkpoint tables
    live in this one file (VOYAGER_SYSTEM_A_DB_PATH)."""
    return os.environ.get("VOYAGER_SYSTEM_A_DB_PATH") or DEFAULT_DB_PATH


def max_workers() -> int:
    """Bound on the run-execution thread pool (VOYAGER_SYSTEM_A_MAX_WORKERS)."""
    raw = os.environ.get("VOYAGER_SYSTEM_A_MAX_WORKERS")
    try:
        value = int(raw) if raw else DEFAULT_MAX_WORKERS
    except ValueError:
        value = DEFAULT_MAX_WORKERS
    return max(1, value)


def busy_timeout_ms() -> int:
    """SQLite `PRAGMA busy_timeout` in milliseconds -- bounds how long a
    writer waits for a lock before failing, rather than failing
    immediately (VOYAGER_SYSTEM_A_SQLITE_BUSY_TIMEOUT_MS)."""
    raw = os.environ.get("VOYAGER_SYSTEM_A_SQLITE_BUSY_TIMEOUT_MS")
    try:
        value = int(raw) if raw else DEFAULT_BUSY_TIMEOUT_MS
    except ValueError:
        value = DEFAULT_BUSY_TIMEOUT_MS
    return max(0, value)


def sse_poll_interval_seconds() -> float:
    """How often the SSE endpoint polls the persisted event log for new
    rows (VOYAGER_SYSTEM_A_SSE_POLL_SECONDS) -- kept small by default so
    hermetic tests stay fast; safe to raise in a real deployment."""
    raw = os.environ.get("VOYAGER_SYSTEM_A_SSE_POLL_SECONDS")
    try:
        value = float(raw) if raw else DEFAULT_SSE_POLL_SECONDS
    except ValueError:
        value = DEFAULT_SSE_POLL_SECONDS
    return max(0.01, value)


def sse_heartbeat_interval_seconds() -> float:
    """How often a `: heartbeat` comment is sent on an otherwise-idle SSE
    connection to keep it alive (VOYAGER_SYSTEM_A_SSE_HEARTBEAT_SECONDS)."""
    raw = os.environ.get("VOYAGER_SYSTEM_A_SSE_HEARTBEAT_SECONDS")
    try:
        value = float(raw) if raw else DEFAULT_SSE_HEARTBEAT_SECONDS
    except ValueError:
        value = DEFAULT_SSE_HEARTBEAT_SECONDS
    return max(0.5, value)
