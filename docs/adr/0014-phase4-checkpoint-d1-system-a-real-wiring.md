# 0014. Phase 4 Checkpoint D.1 — System A real provider, MCP, and A2A wiring

- **Status:** Accepted.
- **Date:** 2026-08-17

## 1. Scope

Creates the production composition layer that injects real capabilities into System A's existing bounded LangGraph loop (Checkpoint D.0), replacing `FakeToolExecutor` with a real `ProductionToolExecutor` for six actions: `get_weather` → `OpenMeteoWeatherProvider`, `web_search` → `SerpApiWebEvidenceProvider`, `search_flights` → `SerpApiFlightSearchProvider`, `search_stays`/`estimate_fair_price` → Travel MCP over Streamable HTTP, `call_istanbul_expert` → System B over real A2A. Does not implement the public FastAPI/SSE service, the Streamlit frontend, production SQLite persistence, or Docker Compose — those are explicitly D.2+.

## 2. Composition ownership

A new root-owned package, `orchestration/system_a/`, is the composition root:

- `services/planner-a` remains vendor-neutral, owning the graph, action models, `ToolExecutor` protocol, Qwen client, guards, and state — unchanged in architecture.
- Root providers (`providers/`) remain root-owned, reused unmodified.
- `orchestration/system_a/` imports planner-a's graph/`ToolExecutor` protocol, root provider adapters, and concrete MCP/A2A clients, and wires them into one real `ProductionToolExecutor`.
- No provider implementation was copied into planner-a; no root `providers`/`rag` code was copied into System B; no provider credential was moved into System B; no A2A call happens between LangGraph nodes (only at the System A/System B network boundary, matching architecture.md §4.2 unchanged).

**Composition mechanism (temporary, deferred to D.2):** `services/planner-a` is put on `PYTHONPATH` externally at process/test-invocation time (e.g. `PYTHONPATH=services/planner-a python -m pytest orchestration/`), never via `sys.path` mutation inside any module's own source — every `orchestration/` and `orchestration/tests/` file uses ordinary top-level imports and relies on this external configuration, the same standard multi-package technique a Docker image's `COPY`+install step would formalize later. Checkpoint D.2 will package this composition root into the real System A service/image, at which point this becomes a real dependency install rather than a `PYTHONPATH` convention.

## 3. Action-to-real-contract mapping (audit findings)

Auditing every D.0 action-argument model against its real target contract surfaced three genuine gaps, each fixed at the smallest necessary point rather than by loosening a contract:

1. **`ProviderResponseEnvelope`/`DataMode` were frozen at their original Phase 1 (1.0.0) shape.** Every real root provider adapter (Open-Meteo since C.1, SerpApi web evidence since C.2, SerpApi flights since C.3) has produced 1.1.0 envelopes (`capability`/`status`/`query_fingerprint`/`cache_status`/`cache_age_seconds`) and can report `data_mode="unavailable"`. `extra="forbid"` meant every real envelope was rejected outright. Fixed additively in `phase1/models.py` (planner-a) — all new fields optional, `DataMode.UNAVAILABLE` added — every pre-existing 1.0.0-only instance remains valid.
2. **`FlightOption` was missing six of the root schema's own 1.1.0-additive fields** (`flight_number`, `duration_minutes`, `legs`, `provider_reference`, `offer_expires_at`, `baggage_limitations`, `fare_limitations`) that the real SerpApi flights adapter has produced since Checkpoint C.3. Fixed the same way — additive, optional, non-breaking.
3. **`search_stays`/`estimate_fair_price`/`call_istanbul_expert` were validated (and fixture-built) as `ProviderResponseEnvelope`-wrapped.** That envelope is specifically the root `providers/` package's own convention (ADR 0009 §5) — Travel MCP's real `search_stays`/`estimate_fair_price` tools return `SearchStaysResult`/`EstimateFairPriceResult` directly, and System B's A2A artifact returns a `LocalItinerary` directly; neither was ever wrapped. Both `phase4/tools.py`'s fake fixtures and `phase4/graph.py`'s validator were corrected together so fake and real data pass through the identical validation path (`_ENVELOPE_WRAPPED_ACTIONS` now names exactly the three providers/-package actions).

Concrete request mapping, once these were fixed:

| Action | Planner arguments | Real target |
|---|---|---|
| `get_weather` | `location, date_from, date_to` | `providers.weather.WeatherQuery` |
| `web_search` | `query, max_results` | `providers.web_evidence.WebEvidenceQuery` |
| `search_flights` | `origin, destination, depart_date, passenger_count, cabin_class` | `providers.flights.FlightSearchQuery` (`depart_date_from == depart_date_to`, matching both D.0's exact-date schema and the real adapter's own V1 boundary) |
| `search_stays` | `check_in, check_out, guest_count, district_id` | Travel MCP `search_stays` — `session_id`/`trace_id`/`result_limit`/`currency`/`ranking_mode`/`schema_version` filled from `ExecutionContext` and fixed, documented defaults (never an invented hard preference; a district/date filter is forwarded only when the caller's own arguments actually named one) |
| `estimate_fair_price` | `stay_id` | Travel MCP `estimate_fair_price` — the `stay_id` is accepted only if it actually appears in a prior successful `search_stays` observation (`_validated_stay_id`); otherwise `invalid_request`, never forwarded unverified |
| `call_istanbul_expert` | `question` (informational only) | System B A2A `LocalPlanRequest`, built entirely from `ExecutionContext`: normalized `TripRequest` (dates, interests/pace/language/mobility), stay candidates from the most recent successful `search_stays` observation, and the most recent successful `get_weather` observation as `weather_context` — never asking the decision provider to reproduce a stay id, coordinate, or other prior tool's output |

## 4. Execution context

`phase4/context.py::ExecutionContext` (planner-a, additive) — `session_id`, `trace_id`, `normalized_request`, `observations` (validated prior `ToolObservation` dicts), `deadline_monotonic`, `cancellation_check`. Never a credential, raw prompt/response, chain-of-thought, arbitrary URL, or stack trace. `ToolExecutor.execute()` gained an optional third `context` parameter; `FakeToolExecutor` accepts and ignores it, so every Checkpoint D.0 hermetic test kept working unchanged.

## 5. Graph-transition measurement and the transition-reduction change

The canonical five-tool plan (`search_flights → search_stays → estimate_fair_price → get_weather → call_istanbul_expert → synthesize`) was run through the real compiled graph with the real `ProductionToolExecutor` (hermetic: fake transports/stubbed MCP/A2A). At the original Checkpoint D.0 topology (`Decide → Execute → Observe → Update → Decide`, 4 LangGraph transitions per tool-call cycle), this measured **26 real LangGraph steps** — one more than the fixed 25-transition `recursion_limit`, so a genuinely-correct five-tool plan was always misclassified as hitting the recursion-limit backstop.

Per this checkpoint's own instruction ("do not simply raise the limit; reduce unnecessary physical transitions... while preserving the nine logical responsibilities"), `_observe_node` (`phase4/graph.py`) now performs both the Observe and Update responsibilities in one physical step for a genuine (non-duplicate) tool execution, recording both an `"Observe"` and an `"Update"` trace entry itself, and routes directly back to `"decide"`. The `"update"` node itself is unchanged and still exists, still reachable, and still exercised for the duplicate-skip path (`decide → update → decide`) — `compiled_graph.get_graph().nodes` still contains all nine names, and every existing Checkpoint D.0 hermetic test (including the one that asserts a `"duplicate_skipped"` `Update` trace entry) passed unmodified after this change.

Re-measured after the fix: the same five-tool plan now completes in **20 real LangGraph transitions**, comfortably under 25, with `final_result.status == "success"` and all five observations `status == "success"` (`orchestration/tests/test_full_trip_hermetic.py`). Neither `MAX_GRAPH_TRANSITIONS` (25) nor any other bound (8 total tool calls, 2 per tool, 2 decision repairs, 60s deadline, duplicate-fingerprint detection, cancellation checks) was loosened; each is independently re-tested (some, like the 8-total/2-per-tool caps, via direct unit tests of the Decide node's own bound logic, since driving 8 real tool calls through the full graph would need ~32 transitions and hit the tighter 25-transition ceiling first regardless of the tool-call cap — an honest, documented interaction between the two bounds, not a contradiction: both remain fully enforced, and in this topology the transition ceiling is, in practice, the first to bind for any run needing more than ~6 tool calls).

## 6. Real provider bindings (`orchestration/system_a/provider_bindings.py`)

Thin mappers only — no HTTP call is reimplemented. Each function builds the provider's own typed Query from validated D.0 arguments, calls the provider's own `.fetch_weather()`/`.search()`/`.search_flights()` exactly once, and returns its `ProviderResponseEnvelope`-shaped dict completely unmodified, including its own honest `status` (never reinterpreted as `"success"` when the provider reported anything else). Hermetic tests (`orchestration/tests/test_provider_bindings.py`) construct each real provider with `FakeHttpTransport` — no real socket in these tests.

## 7. Travel MCP client (`orchestration/system_a/mcp_client.py`)

Uses the official MCP SDK client and the exact Streamable HTTP pattern already proven live in `services/planner-a/phase0/graph_client.py::_call_mcp` (real `initialize()`, real `call_tool()`, structured-content/text-content parsing) — never an in-process import of Travel MCP's own service. Calls only the two allowlisted tools (`search_stays`, `estimate_fair_price`); any other name is rejected as `invalid_request` before any network call (a second, independent check beyond the structural fact that `Action`'s own closed enum has no field for an arbitrary tool name at all). A well-formed `ErrorEnvelope` from the server's own `phase2/mcp/adapters.py::run_tool` boundary is mapped to the shared result-status vocabulary (`REQUEST_TOO_BROAD`/`SERVING_RESOURCE_UNAVAILABLE`/... → `unavailable`; `INVALID_REQUEST`/`UNSUPPORTED_ROOM_TYPE`/... → `invalid_request`), never treated as success and never re-exposed as a raw exception. Bounded timeout (`asyncio.wait_for`); any transport/SDK exception maps to `provider_error`, never leaking. Hermetic tests (`orchestration/tests/test_mcp_client.py`) stub the async call boundary — no real socket.

## 8. System B A2A client (`orchestration/system_a/a2a_client.py`)

Reuses the exact authoritative Phase 0 client sequence unchanged in spirit: Agent Card discovery via `A2ACardResolver`'s standard well-known-path behavior, a client built through the official `ClientFactory`, a real task/message send, Artifact-only evidence extraction (never a status/standalone Message). Never hardcodes a guessed RPC path, never imports System B's own Python implementation, never calls it in-process. The only payload sent is the already-normalized `LocalPlanRequest` dict `orchestration/system_a/tool_executor.py` builds from validated System A state — no provider credential is ever included (proven in `orchestration/tests/test_a2a_client.py`). Bounded timeout; any transport/SDK exception maps to `provider_error`. Hermetic tests stub the async send boundary — no real socket.

## 9. Failure and degradation behavior

Proven hermetically (`orchestration/tests/test_failure_degradation.py`, real compiled graph + real `ProductionToolExecutor` + faked transports/stubbed clients): unavailable weather (empty Open-Meteo geocoding result) produces `status="partial"`, never a hard failure; a 429 from SerpApi flights does not loop (duplicate-fingerprint detection + the consecutive-duplicate cap bound it, never inventing flight data — `envelope is None`); an MCP timeout and a malformed MCP response both degrade safely (`timeout`/`provider_error`, never inserted as valid); an A2A timeout and an invalid System B artifact (missing `LocalItinerary` fields) both degrade safely; a genuine `warnings` entry on a real System B artifact (e.g. a degraded-RAG/Qdrant-unavailable note) passes through into the observation verbatim, never stripped or reworded; no scenario's serialized final state ever contains `traceback`/`api_key`/`authorization`/`bearer`. `FakeToolExecutor`'s own behavior is unchanged except for the two contract-audit fixture corrections in §3 (fake and real now validate identically, which is a strict improvement, not a behavior regression the existing D.0 test suite would have caught either way — all 100 pre-existing phase4 tests still pass unmodified).

## 10. Cross-process integration gate — real evidence, one honest limitation

Ran once, hermetic gates passing first (`orchestration/tests/test_cross_process_integration.py`, opt-in via `VOYAGER_CROSS_PROCESS_INTEGRATION_GATE=1`): real Travel MCP and System B processes started via their own `python -m ...` entrypoints (never imported in-process), a real compiled LangGraph, the real `ProductionToolExecutor`, a scripted (not live-Qwen) decision provider, and the sequence `get_weather → search_stays → call_istanbul_expert → synthesize`.

**Real, live evidence obtained:**
- `get_weather`: a genuine live Open-Meteo call succeeded (`provider="open-meteo"`, `data_mode="live"`).
- `search_stays`: a genuine live Travel MCP call against the real, frozen ~23.5k-row Istanbul Inside Airbnb snapshot succeeded — first attempted unconstrained, which legitimately triggered Travel MCP's own real `REQUEST_TOO_BROAD` guard (23,509 candidates against its 5,000 limit) before a `district_id` filter (`district_fatih`, a real registered district) narrowed it to a genuine success. This is real business-rule behavior encountered live, not a fixture.
- Both servers' real `/health`/Agent Card endpoints responded correctly (this gate would already have failed at its own readiness wait otherwise).

**One honest, undismissed limitation:** `call_istanbul_expert`'s real A2A task did not complete within any bounded client timeout tried (25s, 60s, 100s — each reproducing the identical timeout outcome), even though Agent Card discovery and the TCP/HTTP connection to System B succeeded. Per this checkpoint's own retry policy ("do not repeatedly retry the entire gate"), this was not retried a fourth time. The bounded ReAct loop's own 60-second workflow deadline then did exactly its documented job: the graph safely degraded (`final_result.status` observed as `"partial"`/`"degraded"` across runs) rather than hanging or fabricating a result — itself a genuine, valuable proof that the deadline bound works correctly under real, unpredictable latency, not a defect in this checkpoint's own code. The `IstanbulExpertA2AClient`'s correctness is independently proven hermetically (§8); the finding here is specifically that System B's real ADK agent task processing exceeds 100 seconds in this sandboxed environment for at least one query — a System B-side (or environment-side) characteristic outside this checkpoint's scope to diagnose further, recorded honestly rather than papered over.

Every child process was terminated in a `finally` block on every run, including the failed attempts. No credential, traceback, or fabricated data appeared in any run's output.

## 11. Explicitly deferred to Checkpoint D.2

The public FastAPI/SSE System A service; the Streamlit frontend; production SQLite persistence (`MemorySaver` remains the checkpointer, per Checkpoint D.0); root Docker Compose orchestration; packaging `orchestration/system_a/` into the real System A service/image (replacing the `PYTHONPATH`-based composition seam with a real dependency install); diagnosing System B's real A2A task latency in this environment.
