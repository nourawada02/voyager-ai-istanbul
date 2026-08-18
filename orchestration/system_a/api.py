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
from datetime import date
from typing import Any, AsyncIterator, Callable, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from orchestration.system_a import config, sse
from orchestration.system_a.run_store import RunRecord, RunStore
from orchestration.system_a.service import RunService
from phase1.models import Money, TripPreferences
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


class ErrorEnvelopeResponse(BaseModel):
    """HTTP-facing mirror of contracts/ErrorEnvelope.schema.json /
    phase1.models.ErrorEnvelope -- reused shape, not a competing one."""

    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0.0"
    error_code: str
    message: str
    trace_id: str
    retriable: bool


def _error_response(status_code: int, error_code: str, message: str, retriable: bool = False, trace_id: Optional[str] = None) -> JSONResponse:
    envelope = ErrorEnvelopeResponse(
        error_code=error_code, message=message, trace_id=trace_id or str(uuid4()), retriable=retriable
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))


def _record_to_status_response(record: RunRecord) -> RunStatusResponse:
    # `record.result` is None for every non-terminal status by
    # construction (RunStore.mark_terminal is the only writer of
    # result_json) -- passed through as-is, never guessed here.
    return RunStatusResponse(
        run_id=record.run_id, session_id=record.session_id, status=record.status.value,
        created_at=record.created_at, updated_at=record.updated_at, result=record.result,
    )


def create_app(
    tool_executor_factory: Callable[[], ToolExecutor],
    decision_provider_factory: Callable[[], DecisionProvider],
    specialist_decision_provider_factory: Callable[[], DecisionProvider],
    db_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    mode: str = "real",
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
        record, created = await asyncio.to_thread(
            service.create_run, body.user_message, trip_request_partial, body.idempotency_key
        )
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

    return app
