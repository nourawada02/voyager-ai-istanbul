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


# --- deterministic demo/fixture mode (Checkpoint Phase 4 D.2B) ---------------------

VALID_SYSTEM_A_MODES = ("real", "fixture")
DEFAULT_SYSTEM_A_MODE = "real"


class SystemAModeConfigurationError(RuntimeError):
    """Raised when VOYAGER_SYSTEM_A_MODE is set to anything other than
    exactly "real" or "fixture". Fails loudly at startup rather than
    silently coercing an unrecognized value to either mode -- there is no
    guessing here, matching this project's existing "no silent fallback"
    convention for configuration (e.g. `phase4.qwen_client`'s own
    explicit-HTTPS-URL-or-fail rule)."""


def system_a_mode() -> str:
    """This function's OWN default -- what a bare, standalone process
    gets when VOYAGER_SYSTEM_A_MODE is entirely absent from its
    environment (e.g. `python -m orchestration.system_a.entrypoint` run
    directly, or any hermetic test that never sets the variable) -- is
    "real": the same real Qwen decision provider + ProductionToolExecutor
    every prior checkpoint already used, matching this project's
    "credentials/behavior never silently downgrade" rule. Fixture mode
    always requires the caller to set VOYAGER_SYSTEM_A_MODE=fixture
    explicitly; any other non-empty value is a hard configuration error,
    never a guess in either direction.

    This is a DIFFERENT layer from root docker-compose.yml's own
    `${VOYAGER_SYSTEM_A_MODE:-fixture}` substitution -- Compose supplies
    its own explicit default of "fixture" to the container's
    environment (so `docker compose up` with no `.env` file makes zero
    paid-provider calls out of the box, appropriate for this project's
    offline-by-default demo posture), which THIS function then sees as
    if the caller had typed VOYAGER_SYSTEM_A_MODE=fixture themselves --
    indistinguishable from an explicit choice, never a silent internal
    fallback. Real-provider mode under Compose requires the operator to
    set VOYAGER_SYSTEM_A_MODE=real (and supply QWEN_API_KEY/
    QWEN_BASE_URL/SERPAPI_API_KEY) in a real, never-committed `.env` --
    see docker-compose.yml's own header comment and .env.example."""
    raw = os.environ.get("VOYAGER_SYSTEM_A_MODE")
    if not raw:
        return DEFAULT_SYSTEM_A_MODE
    if raw not in VALID_SYSTEM_A_MODES:
        raise SystemAModeConfigurationError(
            f"VOYAGER_SYSTEM_A_MODE must be one of {VALID_SYSTEM_A_MODES!r} if set; got an unrecognized value"
        )
    return raw


def is_fixture_mode() -> bool:
    return system_a_mode() == "fixture"
