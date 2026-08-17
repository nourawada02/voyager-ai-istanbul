# 0015. Phase 4 Checkpoint D.2A — public System A API, SSE progress, and SQLite persistence

- **Status:** Accepted.
- **Date:** 2026-08-17

## 1. Scope

Exposes the already-implemented, bounded System A planner (Checkpoint D.0's LangGraph ReAct core, Checkpoint D.1/D.1.1's real provider/MCP/A2A wiring) through a production-shaped FastAPI service with sanitized Server-Sent Events and durable SQLite-backed run/session/event persistence. Streamlit, root Docker Compose, container networking, production deployment, browser UI testing, and visual polish are explicitly deferred to **Checkpoint D.2B** — this checkpoint proves the service boundary in isolation first.

Every D.0/D.1 bound is preserved unmodified: `recursion_limit=25`, 8 external-tool-call budget, 2-calls-per-tool cap, 2 decision-format repairs, 60s workflow deadline, duplicate-action fingerprinting, cancellation checks, the closed action allowlist, and every strict schema. No planner-a bound, guard, or graph node topology was loosened, weakened, or bypassed to make this checkpoint pass.

## 2. Public API ownership and package layout

Per this checkpoint's own architectural rule ("keep planner-a vendor-neutral; keep cross-system composition root-owned under `orchestration/system_a/`"), the entire HTTP/SSE/persistence layer is new, root-owned code:

- `orchestration/system_a/config.py` — environment-only, non-secret configuration (DB path, worker-pool size, SQLite busy-timeout, SSE poll/heartbeat intervals), every value defaulted.
- `orchestration/system_a/run_store.py` — `RunStore`, the SQLite-backed run/session/event persistence layer (§5).
- `orchestration/system_a/sse.py` — pure SSE wire-format helpers and the closed stage-name vocabulary this endpoint emits.
- `orchestration/system_a/service.py` — `RunService`, the composition-root execution layer: binds `phase4.graph.build_graph` (unmodified) to real persistence and a bounded worker-thread pool (§6).
- `orchestration/system_a/api.py` — `create_app()`, a FastAPI application **factory** (never a single module-level singleton) so every test builds an isolated app against its own temporary database and injected fake/scripted dependencies.
- `orchestration/system_a/entrypoint.py` — the one module that wires `create_app()` to the REAL `ProductionToolExecutor` (Checkpoint D.1) and REAL `QwenDecisionProvider` (Checkpoint D.0); no test imports this module.

`services/planner-a` gained no FastAPI/SSE/SQLite-persistence code at all — it still owns only the graph, action models, `ToolExecutor` protocol, Qwen client, guards, and state, exactly as D.1 established. The one planner-a change this checkpoint made is a small, additive contract fix (§4).

### 2.1 Packaging: a real declared dependency path, not PYTHONPATH

Checkpoint D.1 (ADR 0014 §2) explicitly deferred replacing its temporary `PYTHONPATH=services/planner-a` test-invocation convention with "a real dependency install" to D.2. This checkpoint does that: a new root `pyproject.toml` declares an editable-installable package (`voyager-orchestration`) covering `providers`, `orchestration`, `orchestration.system_a`, and — via `[tool.setuptools.package-dir]` — `phase1`/`phase4` mapped directly at `services/planner-a/phase1` and `services/planner-a/phase4`. `pip install -e .` from the repo root makes every one of these packages importable from any working directory with **no `PYTHONPATH` set and no `sys.path` mutation anywhere**, verified directly:

```
$ cd /tmp && unset PYTHONPATH && python -c "import phase1.models, phase4.graph, providers, orchestration.system_a.tool_executor; print('OK')"
OK
```

This maps directories, not files — it copies nothing into `services/planner-a` and adds no dependency of that submodule on this file or on being installed this way; the submodule's own hermetic test suite keeps running exactly as before, unaffected. `services/planner-a` remains free to be built/tested entirely on its own, matching CLAUDE.md's submodule-independence rule. Third-party dependencies stay out of `pyproject.toml` by design (`dependencies = []`) — declared instead in a new `orchestration/requirements.in`/`requirements.lock` pair, matching this repository's **existing** per-package `requirements.in`→`requirements.lock` convention exactly (mirroring `services/planner-a/phase0/requirements.in`'s own pattern) rather than inventing a second dependency-declaration mechanism. `requirements.lock` was generated programmatically — not hand-typed — by walking the real installed transitive dependency graph of the ten direct dependencies via `importlib.metadata`, scoped to only what this package actually needs (74 pinned lines), not a raw `pip freeze` of this shared devbox's entire, much larger, unrelated global environment.

## 3. Request/result contracts — reused, not invented

Per this checkpoint's own instruction to search existing contracts before inventing new ones: the planning-run creation body's `trip_request` field is `TripRequestInput`, a field-for-field mirror of `phase1.models.TripRequest` / `contracts/TripRequest.schema.json` **minus** `schema_version`/`session_id`/`trace_id` (which an HTTP caller cannot know in advance) — `RunService.create_run` stamps those in server-side, generating one `session_id`/`trace_id` pair per run and using it for both the run record and the embedded trip request, so the two can never silently diverge. It reuses `phase1.models.Money`/`TripPreferences` unchanged. By the time this constructed dict reaches the graph's own `InputGuard`, it re-validates as a byte-for-byte `TripRequest` instance, unchanged from Checkpoint D.0.

Error responses reuse the exact `ErrorEnvelope` shape (`contracts/ErrorEnvelope.schema.json` / `phase1.models.ErrorEnvelope`) for both request-validation failures (422) and not-found (404) — a dedicated `RequestValidationError` exception handler converts FastAPI/Pydantic's default validation error format into this envelope, naming only the offending field, never echoing raw pydantic error text.

`GET /v1/runs/{id}`'s `result` field is exactly `phase4.graph`'s own `final_result` dict, unmodified — the same `{"status", "observations", "warnings", "narrative"}` (or `{"status", "reason", "observations", "warnings"}` for a Degrade outcome) shape `phase4/graph.py::_synthesize_node`/`_degrade_node` already produce and the existing D.0/D.1 test suites already prove is free of chain-of-thought/prompts/credentials by construction. No new "final result" schema was invented.

## 4. The one additive planner-a change: `StreamStage` widening

`architecture.md` §15.4's ten-value `StreamStage` vocabulary (`request.accepted`, `search.started`, ...) describes §5.3's original linear pipeline. System A's real shape since Checkpoint D.0 is the bounded ReAct action loop (ADR 0009 §4), which this checkpoint's SSE endpoint streams progress for instead. Rather than inventing a second, competing event-stage contract, `phase1.models.StreamStage` (and `contracts/StreamEvent.schema.json`, bumped `1.0.0` → `1.1.0` following the exact same additive-widening precedent `ErrorEnvelope.schema.json` already established) gained eight new values: `run_started`, `action_started`, `action_completed`, `action_failed`, `run_completed`, `run_degraded`, `run_failed`, `run_cancelled`. Purely additive — every pre-existing `1.0.0` `StreamEvent` instance remains valid unchanged, proven directly in `contracts/tests/test_stream_event_react_stages.py` and `services/planner-a/phase1/tests/test_models.py`. This is the same class of small, additive, contract-gap fix Checkpoint D.1 already made twice (`ProviderResponseEnvelope`/`FlightOption` 1.1.0 fields) — not a redesign of D.0/D.1's architecture.

## 5. SQLite persistence

One explicit, configurable database file (`VOYAGER_SYSTEM_A_DB_PATH`, default `orchestration/data/system_a.db`) holds two independent concerns:

1. **`RunStore`'s own tables** (`runs`, `run_events`) — API-level facts: run status, the original validated request, the final schema-valid result once terminal, cancellation flag, and an ordered, sanitized SSE event log. `runs.idempotency_key` is `UNIQUE` (nullable — SQLite allows unlimited `NULL`s in a unique column, so idempotency stays optional); a colliding insert is caught via `sqlite3.IntegrityError`, never a read-then-write race. `RunStore` holds no persistent connection — every operation opens a short-lived WAL-mode connection (`PRAGMA busy_timeout` from config, default 5000ms) and closes it in a `finally` block, so there is no cross-request connection lifetime to manage and no leaked handle to reason about.
2. **The LangGraph checkpointer's own tables**, created by `langgraph-checkpoint-sqlite`'s real `SqliteSaver` (installed this checkpoint; `architecture.md` §5.1/§16 name "LangGraph SQLite checkpointer" as the intended choice for System A, so this is the architecturally-specified choice, not a fallback substitute) — replacing `MemorySaver` for every run executed through the public API. Each run's worker thread opens its own dedicated `sqlite3.Connection` to the same file (`check_same_thread=False`, documented: LangGraph's own sync Pregel loop genuinely dispatches checkpointer `put` calls to an internal helper thread even during a synchronous `.stream()` call — verified directly; the first implementation attempt crashed with `sqlite3.ProgrammingError` under the default `check_same_thread=True` for exactly this reason), sets WAL + the same bounded `busy_timeout`, and closes that connection when the run finishes. Concurrent runs never share a connection or a graph instance — each gets its own `thread_id` (the run's own `run_id`), so LangGraph's own thread-keyed state isolation structurally prevents cross-run bleed (verified in `test_cancellation_and_concurrency.py::test_concurrent_sessions_do_not_share_observations`).

Nothing beyond the two categories above is ever written: no credential, no authorization header, no raw prompt/model response, no chain-of-thought — every value persisted is either the caller's own already-Pydantic-validated request or a `final_result`/`ToolObservation` dict, both structurally free of those by construction (inherited unchanged from Checkpoint D.0's own proof).

## 6. Execution model, concurrency, and cancellation

`RunService.create_run` only ever inserts one `pending` row and, if it actually won the (idempotency-guarded) insert, submits **exactly one** job to a bounded `concurrent.futures.ThreadPoolExecutor` (`VOYAGER_SYSTEM_A_MAX_WORKERS`, default 4). That job runs entirely on its own OS thread — real threads, not `asyncio.to_thread`/`run_in_executor` awaited from a request handler — so it structurally cannot block the FastAPI event loop and needs no event-loop cooperation to make progress; it writes every status/event update directly to `RunStore` (itself safe for concurrent callers). FastAPI request handlers only ever *read* from `RunStore` (or write the small cancellation flag), wrapped in `asyncio.to_thread` so even those tiny synchronous SQLite calls never touch the event loop's own thread.

Each run's worker thread calls `graph.stream(..., stream_mode="updates")` — not `.invoke()` — exactly once, deriving sanitized SSE events from each yielded `{node_name: partial_state_update}` pair: a `"decide"` update whose selected action is a real tool call (not `duplicate_skip`) emits `action_started`; an `"observe"` update's newest observation emits `action_completed`/`action_failed`. This required duplicating `phase4.graph.start_session`'s ~6-line initial-state construction locally (documented in `service.py`), since that function only exposes `.invoke()` semantics and executing the graph a second time via `.stream()` to get progress events was never an option. No other part of `phase4/graph.py` was touched.

**Terminal state is always persisted before the terminal SSE event is appended** (`RunStore.mark_terminal` runs, then `_emit(...)` — never the reverse), so a client can never observe a terminal SSE event whose status `GET /v1/runs/{id}` does not yet reflect.

**Cancellation** reuses the graph's own existing, unmodified mechanism: `cancellation_check=lambda: self._store.is_cancellation_requested(run_id)` is passed straight into `build_graph()`, the same closure-based precondition Checkpoint D.0 already proved (checked before Decide/Execute, never mid-Execute — ADR 0009 §4.5). `POST /v1/runs/{id}/cancel` only ever sets a flag column (idempotent by construction: setting `1` to `1` again is the same operation) and never touches `status` directly. Verified end-to-end with a gated fake tool executor: an in-flight call is allowed to finish (matching the frozen "never mid-EXECUTE" design), but the very next Decide call observes the flag and degrades — exactly one tool call total, never a second.

**Client SSE disconnection never cancels the underlying run** — the SSE generator is a read-only poller of `RunStore`, entirely decoupled from the independent worker-thread execution; disconnecting just stops that one generator.

**Startup reconciliation:** a run left `pending`/`running` when the app restarts belongs to a prior process lifetime with no in-memory task backing it — this service cannot silently resume mid-graph execution, so `RunStore.reconcile_incomplete_runs_on_startup()` (called from the FastAPI `lifespan` context manager) honestly marks it `failed` (`error_code=SERVICE_RESTARTED`) rather than leaving it dishonestly `running` forever.

## 7. SSE event model

| Stage | When |
|---|---|
| `run_started` | Immediately when a run's worker thread begins execution. |
| `action_started` | A `decide` step selects a real tool-call action (not a duplicate-skip). |
| `action_completed` | The matching `observe` step's newest observation has `status="success"`. |
| `action_failed` | ...any other status (`timeout`, `rate_limited`, `provider_error`, `unavailable`, `invalid_request`, `cancelled`). |
| `run_completed` / `run_degraded` / `run_failed` / `run_cancelled` | Exactly one, always last, chosen by mapping the graph's own honest `final_result.status` (`success`/`needs_clarification` → completed; `partial`/`unavailable`/`degraded` → degraded, unless this service's own cancellation flag was set, in which case → cancelled) plus this service's own caught-exception path → failed. |

Every event carries a stable `event_id` (persisted UUID, used as the SSE `id:` field), so `Last-Event-ID` reconnect resolves to an exact resume point via `RunStore.get_event_sequence` — an unknown/stale ID safely replays from the beginning rather than erroring. A `: heartbeat` comment line (never a named `event:`) is emitted on an otherwise-idle connection. The SSE endpoint is a poller over the persisted event log, not an in-memory pub/sub — replay works correctly even across process restarts.

## 8. Why internal ReAct reasoning is never exposed

Nothing new was built to enforce this — it is inherited, unchanged, from Checkpoint D.0's own structural guarantee: `PlannerState` (`phase4/graph.py`) has no chain-of-thought/raw-prompt/raw-response/credential field by construction, and every value this checkpoint's SSE/API/DB layer ever touches (`final_result`, `ToolObservation`, the caller's own validated request) is drawn from that same structurally-clean state. `orchestration/system_a/sse.py::payload_contains_no_forbidden_content` and a dedicated test (`test_degradation_and_security.py::test_internal_exception_produces_a_sanitized_failed_result_never_a_raw_traceback`, which deliberately raises an exception containing a fake local path and a fake API key from inside a tool executor) prove this holds even under a genuine internal exception — the exception's own text goes only to `logging.exception(...)` (server-side), never into the DB, the API response, or an SSE event.

## 9. Concurrency and cancellation model — summary

See §6. In short: bounded real-thread pool (never the asyncio event loop) for graph execution; SQLite (WAL + bounded busy_timeout) for all cross-thread/cross-request state; idempotency-key `UNIQUE` constraint (not a race-prone read-then-write check) for duplicate-run prevention; per-run `thread_id` for LangGraph-level session isolation; the graph's own existing cancellation-check closure, unmodified, for cancellation.

## 10. Persistence limitations (explicit, not hidden)

- **No mid-run resume across a process restart.** A run interrupted by a crash/redeploy is honestly reported `failed` (§6), never silently resumed — genuine multi-process resume of an in-flight graph execution is out of scope for this checkpoint (and was never promised by D.0/D.1's own design, which persists a *session's completed* evidence for `resume_session`-style follow-up turns, not an in-flight execution).
- **No SQLite connection pooling/tuning beyond WAL + busy_timeout** — appropriate for this checkpoint's hermetic-test and small-deployment scope; a high-throughput production deployment might warrant a real pool or a different database engine (Postgres, matching `architecture.md` §5.1's own "swappable for Postgres later" note) — an explicit, out-of-scope-for-D.2A decision, not an oversight.
- **The SSE endpoint polls rather than pushing via an in-memory event bus** — the simplest design that is correct under multi-process/restart scenarios (§7); a future checkpoint could add a push-based fast path without changing the persisted-event-log contract this one establishes.

## 11. Explicitly deferred to Checkpoint D.2B

Streamlit frontend; root Docker Compose; container networking; production deployment; browser UI testing; visual polish. None of these were implemented, started, or modified in this checkpoint.

## 12. Test evidence

Every test in `orchestration/tests/{test_run_store,test_api_endpoints,test_sse_streaming,test_cancellation_and_concurrency,test_degradation_and_security}.py` is hermetic: a `FakeToolExecutor`/gated variant and a `ScriptedDecisionProvider` are injected through `create_app()`'s factory parameters in every single test — no test in this checkpoint calls Qwen, Groq, SerpApi, Open-Meteo, Travel MCP, System B, or Qdrant (those real boundaries were already proven end-to-end in Checkpoint D.1's cross-process gate, ADR 0014 §10/§10.1). 58 new tests added across the five new files (16 + 13 + 8 + 7 + 7 + 7 — persistence, API, SSE, cancellation/concurrency, degradation/security — the last file also covering persistence-across-app-recreation and clean shutdown), plus 9 new contract-mirror tests in `services/planner-a/phase1/tests/test_models.py` and 9 new JSON-Schema conformance tests in `contracts/tests/test_stream_event_react_stages.py`. All pre-existing suites (orchestration hermetic: 38; planner-a phase4: 100/1 skipped; planner-a phase1: 52 baseline; root contracts: 420 baseline) remain green, unmodified in behavior.
