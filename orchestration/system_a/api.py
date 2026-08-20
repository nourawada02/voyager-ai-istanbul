"""Public System A FastAPI service (Checkpoint Phase 4 D.2A) -- the real,
production-shaped HTTP/SSE surface over the already-implemented bounded
System A planner (Checkpoint D.0) wired to real capabilities (Checkpoint
D.1/D.1.1). This module owns only the HTTP boundary: request/response
schemas, error-envelope conversion, and endpoint wiring. All execution,
persistence, and event logic live in `service.py`/`run_store.py` --
kept separate so those are independently unit-testable without an ASGI
server.

`create_app()` is a factory (never a single module-level `app` singleton)
so every test builds its own isolated app pointed at its own temporary
database and its own injected fake tool executor / scripted decision
provider -- no test ever calls a paid provider, Travel MCP, System B, or
Qdrant (Checkpoint D.1 already proved those real boundaries; this
checkpoint proves the HTTP/SSE/persistence layer in front of them).

Streamlit, root Docker Compose, container networking, and browser UI
testing are explicitly out of scope here (Checkpoint D.2B).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, AsyncIterator, Callable, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestration.system_a import config, sse
from orchestration.system_a.chat_service import (
    DEFAULT_SESSION_LIST_LIMIT,
    MAX_SESSION_LIST_LIMIT,
    ChatProviderUnavailable,
    ChatSessionAccessRejected,
    ChatTurnRejected,
    ChatTurnService,
    default_chat_decision_provider_factory,
)
from orchestration.system_a.run_store import RunRecord, RunStore
from orchestration.system_a.service import RunService, TripRequestRejected, default_fx_provider_factory
from phase1.models import Money, TripPreferences
from phase4.guards import default_wall_clock
from phase4.qwen_client import DecisionProvider
from phase4.tools import ToolExecutor

# --- HTTP request/response contracts ------------------------------------------------
#
# Reuses phase1.models.Money/TripPreferences unchanged (architecture.md
# §15.1's own "core contracts... mirrored in Pydantic per service" rule).
# `TripRequestInput` mirrors phase1.models.TripRequest field-for-field
# EXCEPT schema_version/session_id/trace_id, which an HTTP caller cannot
# know in advance -- RunService.create_run stamps those in server-side so
# the constructed dict is still a byte-for-byte valid TripRequest
# instance by the time it reaches the graph's own InputGuard, which
# re-validates it unchanged (phase4/guards.py::check_input). This is the
# smallest additive HTTP-facing contract that reuses the existing
# TripRequest shape rather than inventing a competing one.


class TripRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    origin: str = Field(pattern=r"^[A-Z]{3}$")
    destination: str = Field(pattern=r"^IST$")
    depart_date: date
    return_date: date
    traveler_count: int = Field(ge=1, le=12)
    budget: Money
    preferences: TripPreferences


class PlanningRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_message: str = Field(min_length=1, max_length=4000)
    trip_request: Optional[TripRequestInput] = None
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=200)


class RunCreateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    session_id: str
    status: str


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    session_id: str
    status: str
    created_at: str
    updated_at: str
    result: Optional[dict[str, Any]] = None
    # Hybrid Chat C.2 additive field: the originally-submitted
    # `TripRequest` this run started from (`None` for a narrow-scope run
    # that never carried one) -- needed so the recent-session sidebar can
    # restore the full state a later chat modification needs (budget/
    # dates/travelers/pace/interests), not just what a fresh dashboard
    # already shows. Optional and additive: every pre-existing caller
    # that never reads this field is completely unaffected.
    trip_request: Optional[dict[str, Any]] = None


class ErrorEnvelopeResponse(BaseModel):
    """HTTP-facing mirror of contracts/ErrorEnvelope.schema.json /
    phase1.models.ErrorEnvelope -- reused shape, not a competing one."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    error_code: str
    message: str
    trace_id: str
    retriable: bool


# --- Hybrid Chat C.1: chat-turn HTTP contract -----------------------------------
#
# Follows the exact same convention as every other contract in this file:
# a small, `extra="forbid"` Pydantic model defined here (never a competing
# shape elsewhere), reusing phase1/phase4 types where they already exist.
# The frontend never sends a rewritten trip request or result -- only
# identifiers plus the new message; the authoritative prior state is
# always loaded server-side (ChatTurnService.handle_chat_turn) by
# (session_id, run_id).


class ChatTurnRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    user_message: str = Field(min_length=1, max_length=2000)
    preferred_language: str = Field(default="en", pattern=r"^(en|tr|ar)$")


class ChatTurnResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    session_id: str
    run_id: str
    intent: str
    assistant_message: str
    response_language: str
    trip_patch: Optional[dict[str, Any]] = None
    requires_new_run: bool
    new_run_id: Optional[str] = None
    new_run_status: Optional[str] = None
    clarification_required: bool
    warnings: list[str] = Field(default_factory=list)


class ChatHistoryTurnModel(BaseModel):
    """Hybrid Chat C.1 persistent-history correction §5: a closed,
    already-safe transcript row -- never a database path, credential,
    system prompt, or raw provider payload (`ChatTurnService.
    get_session_history` only ever builds this from
    `orchestration.system_a.run_store.ChatTurnRecord`, which is itself
    structurally incapable of holding any of those, see its own
    docstring)."""

    model_config = ConfigDict(extra="forbid")
    turn_id: str
    role: str
    content: str
    intent: Optional[str] = None
    response_language: Optional[str] = None
    status: str
    created_at: str


class ChatHistoryResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    session_id: str
    run_id: str
    turns: list[ChatHistoryTurnModel]


class SessionSummaryModel(BaseModel):
    """Hybrid Chat C.2: one recent-session sidebar row -- a closed,
    already-safe set of fields only. Never the transcript, a prompt, a
    provider payload, or the raw stored request/result."""

    model_config = ConfigDict(extra="forbid")
    session_id: str
    latest_run_id: str
    latest_run_status: str
    title: str
    origin: Optional[str] = None
    destination: Optional[str] = None
    depart_date: Optional[str] = None
    return_date: Optional[str] = None
    preferred_language: Optional[str] = None
    chat_turn_count: int
    created_at: str
    updated_at: str


class SessionListResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    sessions: list[SessionSummaryModel]
    total: int
    limit: int
    offset: int


def _error_response(status_code: int, error_code: str, message: str, retriable: bool = False, trace_id: Optional[str] = None) -> JSONResponse:
    envelope = ErrorEnvelopeResponse(
        error_code=error_code, message=message, trace_id=trace_id or str(uuid4()), retriable=retriable
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


def _record_to_status_response(record: RunRecord) -> RunStatusResponse:
    # `record.result` is None for every non-terminal status by
    # construction (RunStore.mark_terminal is the only writer of
    # result_json) -- passed through as-is, never guessed here.
    trip_request = record.request.get("trip_request") if isinstance(record.request, dict) else None
    return RunStatusResponse(
        run_id=record.run_id, session_id=record.session_id, status=record.status.value,
        created_at=record.created_at, updated_at=record.updated_at, result=record.result,
        trip_request=trip_request,
    )


def create_app(
    tool_executor_factory: Callable[[], ToolExecutor],
    decision_provider_factory: Callable[[], DecisionProvider],
    specialist_decision_provider_factory: Callable[[], DecisionProvider],
    db_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    mode: str = "real",
    wall_clock: Callable[[], datetime] = default_wall_clock,
    fx_provider_factory: Callable[[], Any] = default_fx_provider_factory,
    chat_decision_provider_factory: Callable[[], DecisionProvider] = default_chat_decision_provider_factory,
) -> FastAPI:
    """Builds one independent FastAPI app instance, owning one `RunStore`
    (one SQLite database file) and one bounded run-execution thread pool.
    `tool_executor_factory`/`decision_provider_factory` are called once
    per run (never shared mutable state across concurrent runs) -- the
    real production entrypoint passes factories that build a
    `ProductionToolExecutor`/`QwenDecisionProvider`; every test passes a
    factory that builds a fake/scripted/stubbed equivalent instead, so no
    test ever depends on network access.

    `specialist_decision_provider_factory` (Checkpoint Phase 4 D.3
    correction pass) is a SEPARATE, explicitly supplied factory for the
    internal Travel Search specialist's own `DecisionProvider` -- never
    the same object as `decision_provider_factory`'s result, and never
    inferred from prompt content at runtime (ADR 0017 §3). Production
    wiring (`entrypoint.py`) passes a second, independent
    `QwenDecisionProvider`/`SpecialistFixtureDecisionProvider` factory.

    `mode` is a non-secret operational label only (`"real"` or
    `"fixture"`, Checkpoint D.2B) -- it never changes which factories run,
    it only lets `/health` honestly report which pair the caller already
    chose, so the frontend can label demo mode without guessing from
    behavior."""
    resolved_db_path = db_path or config.db_path()
    resolved_max_workers = max_workers if max_workers is not None else config.max_workers()

    store = RunStore(resolved_db_path, busy_timeout_ms=config.busy_timeout_ms())
    service = RunService(
        run_store=store,
        db_path=resolved_db_path,
        tool_executor_factory=tool_executor_factory,
        decision_provider_factory=decision_provider_factory,
        specialist_decision_provider_factory=specialist_decision_provider_factory,
        max_workers=resolved_max_workers,
        busy_timeout_ms=config.busy_timeout_ms(),
        wall_clock=wall_clock,
        fx_provider_factory=fx_provider_factory,
    )
    # Hybrid Chat C.1: a genuinely separate DecisionProvider instance from
    # decision_provider_factory/specialist_decision_provider_factory --
    # same "each role gets its own explicitly constructed provider"
    # precedent as the supervisor/specialist split (ADR 0017 §3). Never
    # constructed per-request: this service is stateless per call
    # (chat_service.py never caches anything across turns), so one
    # instance for the app's lifetime is safe, matching how `store`
    # itself is a single shared, thread-safe instance.
    chat_service = ChatTurnService(
        run_store=store, run_service=service,
        chat_decision_provider_factory=chat_decision_provider_factory, wall_clock=wall_clock,
    )

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Startup: any run left `pending`/`running` belongs to a prior
        # process lifetime -- reconciled to `failed` so status reporting
        # stays honest after the app is recreated (never left dishonestly
        # `running` forever).
        await asyncio.to_thread(store.reconcile_incomplete_runs_on_startup)
        yield
        # Shutdown: the bounded worker pool is drained/closed cleanly --
        # no leaked threads or dangling SQLite connections.
        await asyncio.to_thread(service.shutdown, True)

    app = FastAPI(
        title="agent-system-a (planner-a) -- System A public API (Checkpoint Phase 4 D.2A)",
        lifespan=_lifespan,
    )
    app.state.service = service
    app.state.store = store

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        return _error_response(
            422, "REQUEST_VALIDATION_FAILED", f"Invalid request field: {field}", retriable=False
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        code = {404: "RUN_NOT_FOUND", 409: "CONFLICT"}.get(exc.status_code, "REQUEST_FAILED")
        return _error_response(exc.status_code, code, str(exc.detail), retriable=False)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "agent-system-a", "checkpointer": "sqlite", "mode": mode}

    @app.post("/v1/runs", response_model=RunCreateResponse, status_code=201)
    async def create_run(body: PlanningRunRequest) -> Any:
        trip_request_partial = body.trip_request.model_dump(mode="json") if body.trip_request is not None else None
        try:
            record, created = await asyncio.to_thread(
                service.create_run, body.user_message, trip_request_partial, body.idempotency_key
            )
        except TripRequestRejected as exc:
            # Manual QA remediation Q.1: a typed client validation error --
            # no run row was created (RunService.create_run raises this
            # BEFORE inserting one), and no Qwen/provider call was ever
            # made. Never surfaced as a 500/internal error.
            return _error_response(422, "TRIP_REQUEST_REJECTED", exc.safe_error or "input_rejected", retriable=False)
        status_code = 201 if created else 200
        payload = RunCreateResponse(run_id=record.run_id, session_id=record.session_id, status=record.status.value)
        return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))

    @app.get("/v1/runs/{run_id}", response_model=RunStatusResponse)
    async def get_run(run_id: str) -> RunStatusResponse:
        record = await asyncio.to_thread(service.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run_not_found")
        return _record_to_status_response(record)

    @app.post("/v1/runs/{run_id}/cancel", response_model=RunStatusResponse)
    async def cancel_run(run_id: str) -> RunStatusResponse:
        record = await asyncio.to_thread(service.request_cancellation, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run_not_found")
        return _record_to_status_response(record)

    @app.get("/v1/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request) -> StreamingResponse:
        record = await asyncio.to_thread(service.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run_not_found")

        after_sequence = -1
        last_event_id = request.headers.get("last-event-id")
        if last_event_id:
            resolved = await asyncio.to_thread(service.get_event_sequence, run_id, last_event_id)
            if resolved is not None:
                after_sequence = resolved

        poll_interval = config.sse_poll_interval_seconds()
        heartbeat_interval = config.sse_heartbeat_interval_seconds()

        async def event_generator():
            nonlocal after_sequence
            last_heartbeat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    return
                events = await asyncio.to_thread(service.list_events, run_id, after_sequence)
                for event in events:
                    after_sequence = event.sequence
                    yield sse.format_event(event)
                    if event.stage in sse.TERMINAL_STAGES:
                        return
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    yield sse.format_heartbeat()
                    last_heartbeat = now
                await asyncio.sleep(poll_interval)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/turns", response_model=ChatTurnResponseModel)
    async def create_chat_turn(body: ChatTurnRequestModel) -> Any:
        try:
            result = await asyncio.to_thread(
                chat_service.handle_chat_turn, body.session_id, body.run_id, body.user_message, body.preferred_language
            )
        except ChatTurnRejected as exc:
            # Same convention as TripRequestRejected above: a typed client
            # error, never an internal 500, and never leaks which part of
            # the lookup failed beyond a closed safe_error string.
            return _error_response(422, "CHAT_TURN_REJECTED", exc.safe_error or "chat_turn_rejected", retriable=False)
        except ChatProviderUnavailable as exc:
            # Persistent-history/grounding correction §10: a genuine
            # provider outage (permanent auth/quota failure, or a
            # transient one that exhausted its bounded retry) is surfaced
            # honestly as a typed 503 -- never disguised as a 422 client
            # error/user-ambiguity result, and never leaking a raw
            # provider status code, body, or credential.
            return _error_response(
                503, "PROVIDER_UNAVAILABLE", "The chat assistant is temporarily unavailable.", retriable=exc.retriable
            )
        payload = ChatTurnResponseModel(
            session_id=result.session_id, run_id=result.run_id, intent=result.intent,
            assistant_message=result.assistant_message, response_language=result.response_language,
            trip_patch=result.trip_patch, requires_new_run=result.requires_new_run,
            new_run_id=result.new_run_id, new_run_status=result.new_run_status,
            clarification_required=result.clarification_required, warnings=result.warnings,
        )
        return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))

    @app.get("/v1/chat/sessions/{session_id}/turns", response_model=ChatHistoryResponseModel)
    async def get_chat_session_turns(session_id: str, run_id: str) -> Any:
        try:
            turns = await asyncio.to_thread(chat_service.get_session_history, session_id, run_id)
        except ChatSessionAccessRejected as exc:
            return _error_response(422, "CHAT_SESSION_ACCESS_REJECTED", exc.safe_error, retriable=False)
        payload = ChatHistoryResponseModel(
            session_id=session_id, run_id=run_id,
            turns=[
                ChatHistoryTurnModel(
                    turn_id=t.turn_id, role=t.role, content=t.content, intent=t.intent,
                    response_language=t.response_language, status=t.status, created_at=t.created_at,
                )
                for t in turns
            ],
        )
        return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))

    @app.get("/v1/chat/sessions", response_model=SessionListResponseModel)
    async def list_chat_sessions(
        limit: int = Query(default=DEFAULT_SESSION_LIST_LIMIT, ge=1, le=MAX_SESSION_LIST_LIMIT),
        offset: int = Query(default=0, ge=0),
    ) -> Any:
        # Hybrid Chat C.2: the recent-session sidebar's one bounded
        # listing call. `Query(ge=1, le=MAX_SESSION_LIST_LIMIT)` rejects
        # an out-of-range value with the existing 422 RequestValidationError
        # handler BEFORE this handler body ever runs; `ChatTurnService.
        # list_recent_sessions` re-clamps defensively too (never trusts a
        # single validation layer, matching this project's established
        # precedent).
        summaries, total = await asyncio.to_thread(chat_service.list_recent_sessions, limit, offset)
        payload = SessionListResponseModel(
            sessions=[
                SessionSummaryModel(
                    session_id=s.session_id, latest_run_id=s.latest_run_id, latest_run_status=s.latest_run_status,
                    title=s.title, origin=s.origin, destination=s.destination, depart_date=s.depart_date,
                    return_date=s.return_date, preferred_language=s.preferred_language,
                    chat_turn_count=s.chat_turn_count, created_at=s.created_at, updated_at=s.updated_at,
                )
                for s in summaries
            ],
            total=total, limit=limit, offset=offset,
        )
        return JSONResponse(status_code=200, content=payload.model_dump(mode="json"))

    return app
