"""Server-Sent Events wire formatting for the production System A service
(Checkpoint Phase 4 D.2A). Pure formatting only -- no I/O, no persistence
(that is `run_store.py`'s job). Every event carries a stable `id:` (the
persisted `event_id`, a UUID) so a client's `Last-Event-ID` reconnect
header can be resolved back to an exact resume point via
`RunStore.get_event_sequence`.
"""

from __future__ import annotations

import json
from typing import Any

from orchestration.system_a.run_store import RunEventRecord

# The full closed vocabulary this checkpoint's SSE endpoint ever emits
# (architecture.md §15.4's own "only operational status and validated
# artifacts stream" principle, applied to the ReAct-shaped action loop
# rather than the original linear pipeline -- see
# phase1.models.StreamStage's Checkpoint D.2A additive values).
RUN_STARTED = "run_started"
ACTION_STARTED = "action_started"
ACTION_COMPLETED = "action_completed"
ACTION_FAILED = "action_failed"
RUN_COMPLETED = "run_completed"
RUN_DEGRADED = "run_degraded"
RUN_FAILED = "run_failed"
RUN_CANCELLED = "run_cancelled"

TERMINAL_STAGES = frozenset({RUN_COMPLETED, RUN_DEGRADED, RUN_FAILED, RUN_CANCELLED})


def format_event(record: RunEventRecord) -> str:
    """One wire-format SSE frame: stable `id:`, `event:` naming the stage,
    and a single-line JSON `data:` payload -- never multi-line raw text
    that could smuggle unstructured content past the sanitized payload
    shape."""
    data = json.dumps(record.payload, sort_keys=True, default=str)
    return f"id: {record.event_id}\nevent: {record.stage}\ndata: {data}\n\n"


def format_heartbeat() -> str:
    """A comment line (per the SSE spec, a line starting with `:`) -- kept
    out of `data:`/`event:` entirely so it can never be mistaken for a
    real event by a spec-compliant client."""
    return ": heartbeat\n\n"


def payload_contains_no_forbidden_content(payload: dict[str, Any]) -> bool:
    """Defense-in-depth check usable by tests and, cheaply, by the
    service itself: an SSE payload must never contain a credential,
    authorization header, prompt text, or chain-of-thought marker. Mirrors
    the exact scan `orchestration/tests/test_failure_degradation.py`
    already applies to full graph state."""
    serialized = json.dumps(payload, default=str).lower()
    forbidden = ("api_key", "authorization", "bearer", "chain_of_thought", "system_prompt", "password", "secret")
    return not any(term in serialized for term in forbidden)
