"""Production run-execution service for System A (Checkpoint Phase 4
D.2A) -- the composition-root layer that binds `phase4.graph`'s already-
proven, unmodified bounded LangGraph loop to real SQLite-backed run
persistence and sanitized SSE progress events, and runs it on a bounded
worker-thread pool so the FastAPI event loop is never blocked by the
graph's synchronous execution.

Execution model: `create_run` only ever inserts a `pending` row and, if
it actually won the (idempotency-key-guarded) insert, submits exactly one
job to a bounded `concurrent.futures.ThreadPoolExecutor`. That job runs
entirely on its own OS thread, independent of the asyncio event loop --
it never needs the loop "pumped" to make progress, and it writes every
status/event update directly to `RunStore` (itself safe for concurrent
callers, see `run_store.py`). FastAPI request handlers only ever *read*
from `RunStore` (or write the small cancellation flag), never touch the
graph directly.

Each run gets its own real `sqlite3.Connection` (WAL + bounded
busy_timeout) wrapping a fresh `SqliteSaver`, and its own `thread_id`
(the run_id) -- concurrent runs are structurally isolated by LangGraph's
own thread-keyed checkpoint state, never sharing a connection or a graph
instance. The connection is opened and closed within the same worker
job, so there is no cross-request connection lifetime to manage.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError

from orchestration.system_a import budget_summary as budget_summary_module
from orchestration.system_a import sse
from orchestration.system_a.run_store import RunEventRecord, RunRecord, RunStatus, RunStore
from phase4.graph import build_graph
from phase4.guards import check_input, default_wall_clock, resolve_today
from phase4.models import MAX_GRAPH_TRANSITIONS, TOOL_CALL_ACTIONS
from phase4.qwen_client import DecisionProvider
from phase4.tools import ToolExecutor


def default_fx_provider_factory() -> Any:
    from providers.fx_frankfurter import FrankfurterFxProvider

    return FrankfurterFxProvider()

logger = logging.getLogger("voyager.system_a.service")


class TripRequestRejected(Exception):
    """Raised by `RunService.create_run` when `phase4.guards.check_input`
    rejects the request (Manual QA remediation Q.1) -- e.g. a past
    departure date, a return date before departure, or an unsupported
    currency. Raised BEFORE any run row is created and BEFORE the
    execution thread pool is touched, so an invalid request never shows up
    as a run at all, and never reaches Qwen or any provider. The HTTP
    layer (api.py) catches this and returns a typed 422, never treating it
    as an internal error."""

    def __init__(self, reason_code: Optional[str], safe_error: Optional[str]) -> None:
        self.reason_code = reason_code
        self.safe_error = safe_error
        super().__init__(safe_error or "input_rejected")

_TOOL_CALL_ACTION_VALUES = frozenset(a.value for a in TOOL_CALL_ACTIONS)

_FALLBACK_RECURSION_RESULT = {
    "status": "degraded",
    "reason": "graph_transition_limit_reached",
    "observations": [],
    "warnings": [],
}


def _terminal_stage_for(status: RunStatus) -> str:
    return {
        RunStatus.COMPLETED: sse.RUN_COMPLETED,
        RunStatus.DEGRADED: sse.RUN_DEGRADED,
        RunStatus.FAILED: sse.RUN_FAILED,
        RunStatus.CANCELLED: sse.RUN_CANCELLED,
    }[status]


def _map_final_result_to_run_status(final_result: dict[str, Any], cancellation_requested: bool) -> RunStatus:
    """Translates `phase4.graph`'s own honest `final_result.status`
    vocabulary (success/partial/needs_clarification/degraded) plus this
    service's own cancellation flag into the coarser run-lifecycle
    vocabulary the public API exposes. A cancellation the graph itself
    already turned into `degrade(reason="cancelled")` (ADR 0009 §4.5,
    unmodified) is reported as `cancelled`, not a generic `degraded`, by
    checking the flag this service owns -- never by guessing from the
    reason string alone."""
    graph_status = final_result.get("status")
    if cancellation_requested and graph_status == "degraded":
        return RunStatus.CANCELLED
    if graph_status in ("success", "needs_clarification"):
        return RunStatus.COMPLETED
    # "partial" (some evidence, not all successful), "unavailable" (a
    # legitimate Synthesize outcome with zero observations -- e.g. the
    # very first decision was itself `synthesize`), and "degraded" (the
    # Degrade node's own always-used status, for any non-cancellation
    # reason: decision_format_invalid, workflow_deadline_exceeded,
    # bound_reached, input rejected by InputGuard, ...) are all honest,
    # non-exceptional graph outcomes -- never mapped to `failed`, which
    # this service reserves for a genuine internal exception it caught
    # itself (see the `except Exception` branch in `_execute`).
    if graph_status in ("partial", "unavailable", "degraded"):
        return RunStatus.DEGRADED
    return RunStatus.FAILED


class RunService:
    def __init__(
        self,
        run_store: RunStore,
        db_path: str,
        tool_executor_factory: Callable[[], ToolExecutor],
        decision_provider_factory: Callable[[], DecisionProvider],
        specialist_decision_provider_factory: Callable[[], DecisionProvider],
        max_workers: int = 4,
        busy_timeout_ms: int = 5000,
        wall_clock: Callable[[], datetime] = default_wall_clock,
        fx_provider_factory: Callable[[], Any] = default_fx_provider_factory,
    ):
        self._store = run_store
        self._db_path = db_path
        self._tool_executor_factory = tool_executor_factory
        self._decision_provider_factory = decision_provider_factory
        self._specialist_decision_provider_factory = specialist_decision_provider_factory
        self._busy_timeout_ms = busy_timeout_ms
        # Manual QA remediation Q.1 (§B): injected exactly like every
        # other provider (tool_executor_factory) -- every hermetic test
        # passes a FakeFxProvider factory, so no test ever makes a real
        # network call just because a trip's budget currency is USD.
        self._fx_provider_factory = fx_provider_factory
        # Single source of truth for "now" (Manual QA remediation Q.1) --
        # the same injected clock resolves "today" both for the
        # pre-run rejection check below AND for the graph's own InputGuard
        # (passed through to build_graph in _execute), so the two layers
        # can never disagree about what date is being validated against.
        self._wall_clock = wall_clock
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="voyager-run-worker")

    # --- public API used by the FastAPI layer ---------------------------------------

    def create_run(
        self, user_message: str, trip_request_partial: Optional[dict[str, Any]], idempotency_key: Optional[str]
    ) -> tuple[RunRecord, bool]:
        """`trip_request_partial` is the caller-supplied trip fields only
        (origin/destination/dates/traveler_count/budget/preferences) --
        never `schema_version`/`session_id`/`trace_id`, which the caller
        cannot know in advance. This method generates one session_id/
        trace_id pair for the whole run and stamps it into both the run
        record and the embedded trip_request, so the two are always the
        same session by construction, never two independently-supplied
        values that could silently diverge.

        Manual QA remediation Q.1: `phase4.guards.check_input` is
        consulted HERE, synchronously, before any run row exists and
        before the execution thread pool is touched. A rejected request
        (past departure date, return before departure, unsupported
        currency, etc.) raises `TripRequestRejected` -- no run is ever
        created, and Qwen/every provider is never called. This is in
        addition to (not a replacement for) the graph's own InputGuard,
        which still re-validates the same way for any caller that invokes
        the graph directly."""
        session_id = str(uuid4())
        trace_id = str(uuid4())
        trip_request: Optional[dict[str, Any]] = None
        if trip_request_partial is not None:
            trip_request = {"schema_version": "1.0.0", "session_id": session_id, "trace_id": trace_id, **trip_request_partial}

        today = resolve_today(self._wall_clock)
        guard_result = check_input(user_message, trip_request, today=today)
        if not guard_result.accepted:
            raise TripRequestRejected(
                guard_result.reason_code.value if guard_result.reason_code else None, guard_result.safe_error
            )

        run_id = str(uuid4())
        request_dict = {"user_message": user_message, "trip_request": trip_request}
        record, created = self._store.create_run(
            run_id=run_id, session_id=session_id, trace_id=trace_id, request=request_dict, idempotency_key=idempotency_key
        )
        if created:
            self._pool.submit(self._execute, record.run_id, record.session_id, record.trace_id, request_dict)
        return record, created

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        return self._store.get_run(run_id)

    def list_events(self, run_id: str, after_sequence: int = -1) -> list[RunEventRecord]:
        return self._store.list_events(run_id, after_sequence)

    def get_event_sequence(self, run_id: str, event_id: str) -> Optional[int]:
        return self._store.get_event_sequence(run_id, event_id)

    def request_cancellation(self, run_id: str) -> Optional[RunRecord]:
        return self._store.request_cancellation(run_id)

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    # --- worker-thread execution -----------------------------------------------------

    def _emit(self, run_id: str, stage: str, payload: dict[str, Any]) -> None:
        self._store.append_event(run_id, str(uuid4()), stage, payload)

    def _execute(self, run_id: str, session_id: str, trace_id: str, request_dict: dict[str, Any]) -> None:
        self._store.mark_running(run_id)
        self._emit(run_id, sse.RUN_STARTED, {"run_id": run_id})

        conn: Optional[sqlite3.Connection] = None
        try:
            # check_same_thread=False: LangGraph's own sync Pregel loop
            # dispatches checkpointer "put" calls to an internal helper
            # thread even during a synchronous `.stream()` call, so this
            # connection is genuinely touched from more than one thread
            # -- but only ever from within THIS one run's own execution,
            # never shared with another run's connection or graph
            # instance, so this stays safe (each run still gets its own
            # dedicated connection, closed when that run finishes).
            conn = sqlite3.connect(self._db_path, timeout=self._busy_timeout_ms / 1000.0, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            saver = SqliteSaver(conn)
            saver.setup()

            tool_executor = self._tool_executor_factory()
            decision_provider = self._decision_provider_factory()
            specialist_decision_provider = self._specialist_decision_provider_factory()
            graph = build_graph(
                tool_executor,
                decision_provider,
                specialist_decision_provider,
                cancellation_check=lambda: self._store.is_cancellation_requested(run_id),
                wall_clock=self._wall_clock,
                checkpointer=saver,
                specialist_event_callback=lambda event: self._handle_specialist_event(run_id, event),
            )

            # Mirrors phase4.graph.start_session's own initial-state
            # construction exactly (that function only exposes
            # `.invoke()`, but this service needs `.stream()` for
            # progress events without executing the graph twice -- see
            # module docstring).
            initial_state: dict[str, Any] = {
                "session_id": session_id,
                "trace_id": trace_id,
                "user_message": request_dict["user_message"],
                "trip_request": request_dict.get("trip_request"),
                "started_at_monotonic": time.monotonic(),
                "graph_transition_count": 0,
                "trace": [],
            }
            config = {"configurable": {"thread_id": run_id}, "recursion_limit": MAX_GRAPH_TRANSITIONS}

            try:
                for update in graph.stream(initial_state, config=config, stream_mode="updates"):
                    self._handle_stream_update(run_id, update)
                final_state = graph.get_state(config).values
                final_result = final_state.get("final_result") or _FALLBACK_RECURSION_RESULT
            except GraphRecursionError:
                final_result = _FALLBACK_RECURSION_RESULT

            # Manual QA remediation Q.1 (§B): a deterministic, server-side
            # post-processing step -- never part of the bounded ReAct
            # loop, never Qwen-decided. Attached only when there is a
            # real trip_request/budget to summarize; a failure here is
            # never allowed to turn an otherwise-successful run into a
            # failed one (budget_summary is a genuine best-effort
            # enrichment, not a required output).
            try:
                summary = budget_summary_module.build_budget_summary(
                    final_result, request_dict.get("trip_request"), self._fx_provider_factory
                )
                if summary is not None:
                    final_result = dict(final_result)
                    final_result["budget_summary"] = summary
            except Exception:  # noqa: BLE001 -- never lets a budget-summary bug fail the whole run
                pass

            cancellation_requested = self._store.is_cancellation_requested(run_id)
            run_status = _map_final_result_to_run_status(final_result, cancellation_requested)

            # Terminal state is persisted BEFORE the terminal SSE event is
            # appended -- a client can never observe the event without
            # GET /v1/runs/{id} already reflecting it.
            self._store.mark_terminal(run_id, run_status, final_result)
            self._emit(run_id, _terminal_stage_for(run_status), {"status": final_result.get("status")})

        except Exception as exc:  # noqa: BLE001 -- an internal exception must never reach a caller raw
            # Deliberately logs only the exception's TYPE name -- never
            # `logger.exception()`/`exc_info` and never `str(exc)`/`exc.args`.
            # The exception's own message is untrusted content (it can
            # originate from a tool executor/provider and could contain
            # anything, including a real leaked credential) -- this
            # project's own "no internal detail ever reaches the caller"
            # rule (architecture.md §13.4) is applied here even more
            # strictly than that rule requires: not even the server-side
            # log line repeats free-form exception text, only a closed,
            # structural fact (the exception's class name) plus run_id/
            # trace_id for correlation.
            logger.error(
                "run %s failed with an internal exception of type %s (trace_id=%s)",
                run_id, type(exc).__name__, trace_id,
            )
            sanitized_message = "An internal error occurred while executing this run."
            self._store.mark_terminal(
                run_id, RunStatus.FAILED,
                {"status": "failed", "observations": [], "warnings": ["internal_execution_error"]},
                error_code="INTERNAL_EXECUTION_ERROR", error_message=sanitized_message,
            )
            self._emit(run_id, sse.RUN_FAILED, {"status": "failed"})
        finally:
            if conn is not None:
                conn.close()

    def _handle_stream_update(self, run_id: str, update: dict[str, dict[str, Any]]) -> None:
        for node_name, node_output in update.items():
            if node_name == "decide":
                pending = node_output.get("pending_action") or {}
                action = pending.get("action")
                if action in _TOOL_CALL_ACTION_VALUES and not node_output.get("duplicate_skip"):
                    self._emit(run_id, sse.ACTION_STARTED, {"action": action})
            elif node_name == "observe":
                observations = node_output.get("observations") or []
                if observations:
                    obs = observations[-1]
                    stage = sse.ACTION_COMPLETED if obs.get("status") == "success" else sse.ACTION_FAILED
                    self._emit(run_id, stage, {"action": obs.get("action"), "status": obs.get("status")})

    def _handle_specialist_event(self, run_id: str, event: dict[str, Any]) -> None:
        # Checkpoint Phase 4 D.3 (correction pass): a `call_travel_search`
        # delegation runs the internal Travel Search specialist's own
        # genuinely separate, compiled LangGraph `StateGraph`
        # (`phase4/specialist.py`). `phase4.graph.build_graph`'s own
        # `specialist_event_callback` seam calls THIS method in real
        # time, DURING the specialist's own `.stream()` iteration --
        # i.e. genuinely as each specialist transition happens, not after
        # the whole delegation has already completed (the D.3 correction
        # this replaces: a prior draft derived these events from the
        # supervisor's own "execute" node update AFTER the entire
        # delegation had already finished, which looked live but was not
        # -- every event landed at once, post hoc). `event` is already
        # sanitized -- action name and status only, never a prompt, an
        # argument, or an explanation (see
        # `phase4.specialist._emit_specialist_events`).
        stage_by_name = {
            "action_started": sse.ACTION_STARTED,
            "action_completed": sse.ACTION_COMPLETED,
            "action_failed": sse.ACTION_FAILED,
        }
        stage = stage_by_name.get(event.get("stage"))
        if stage is None:
            return
        payload: dict[str, Any] = {"action": event.get("action")}
        if "status" in event:
            payload["status"] = event["status"]
        self._emit(run_id, stage, payload)
