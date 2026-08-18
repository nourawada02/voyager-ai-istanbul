# 0017. Phase 4 Checkpoint D.3 — Internal Travel Search ReAct specialist

- **Status:** Accepted.
- **Date:** 2026-08-18
- **Corrected in place (same date):** the first draft of this checkpoint implemented the internal specialist as a plain Python `for` loop running inside the supervisor's own Execute node — real tool calls, real bounds, but never a second LangGraph graph, and its SSE progress events were derived after the whole delegation had already finished (looked live, was not). A review before commit rejected that as "not strong enough to defend as a genuine LangGraph ReAct specialist" and required a focused correction pass, recorded here. Nothing in §§1–9 below describes the rejected draft; it describes only the corrected, final implementation.

## 1. Scope

Introduces exactly one new internal ReAct specialist as a genuinely separate, compiled `langgraph.graph.StateGraph` (`services/planner-a/phase4/specialist.py`), invoked by System A's existing bounded LangGraph loop (`phase4/graph.py`, Checkpoint D.0/D.1) from inside its own Execute node — never a second network hop, never A2A (that boundary is reserved for the System A/System B edge, architecture.md §4.2). The specialist owns the five travel-search tool actions (`get_weather`, `web_search`, `search_flights`, `search_stays`, `estimate_fair_price`). System A's existing graph becomes the **supervisor** — it can no longer call any of those five tools directly; it may only select the closed action `call_travel_search` to delegate the whole batch, or `call_istanbul_expert`/`ask_clarification`/`synthesize`/`degrade` as before. System B remains the independent, external, A2A-only Istanbul specialist, entirely unaffected by this checkpoint.

No new container, service, or public API. No change to the public System A API/SSE surface's request/response shape, to `phase6`'s frontend contract, or to any provider/MCP/A2A client. The five delegated tools still call the exact same real providers, Travel MCP, and `ToolExecutor` protocol Checkpoints D.1/D.2 already wired — the specialist reuses them unmodified.

## 2. The specialist is a genuine, separately compiled LangGraph `StateGraph`

`phase4/specialist.py::build_specialist_graph()` compiles a real, three-node cycle:

```
specialist_decide → specialist_execute_observe → specialist_decide → ... → specialist_end
```

Every arrow above is a genuine LangGraph transition, subject to this graph's own `recursion_limit` (`MAX_SPECIALIST_GRAPH_TRANSITIONS = 15`) — never a plain Python loop hiding tool calls from LangGraph's own engine. `invoke_travel_search_specialist()` (the one boundary function `phase4/graph.py`'s Execute node calls) drives it with `.stream(..., stream_mode="updates")` against a fresh, function-call-scoped `MemorySaver` checkpointer and a fresh `thread_id` per delegation — never `.invoke()`, specifically so real-time progress events can be derived (§6) and so a genuine `GraphRecursionError` still leaves the last successfully-completed superstep's real progress recoverable via `compiled_graph.get_state(config).values` (LangGraph checkpoints after every completed superstep, before the next one's recursion-limit check runs) rather than silently discarding real tool calls already made — mirroring the exact recovery shape `phase4/graph.py::_safe_recursion_limit_result` already established for the supervisor's own 25-transition ceiling.

A delegation still costs the supervisor exactly one physical `Decide → Execute → Observe` cycle at the SUPERVISOR level, regardless of how many of the specialist's own real transitions happen inside that one `Execute` call — `MAX_GRAPH_TRANSITIONS` (25, supervisor) and `MAX_SPECIALIST_GRAPH_TRANSITIONS` (15, specialist) are two independent counters over two independent graphs, never conflated.

## 3. Typed state and a validated result artifact

`TravelSearchState` (a `TypedDict`) is the specialist's own graph state: `session_id`/`trace_id`/`normalized_request` (context), `inherited_observations` (read-only, the supervisor's own prior evidence) and `specialist_observations` (this delegation's own, growing evidence) kept as two separate fields, the SHARED `tool_call_count`/`tool_call_count_by_action`/`executed_fingerprints` seeded from the supervisor's current values at invocation time, `warnings`, the shared `started_at_monotonic` clock origin, and the specialist's own `graph_transition_count`/`repair_count`.

`TravelSearchResult` (a `pydantic.BaseModel`, `extra="forbid"`) is the ONLY shape `invoke_travel_search_specialist()` ever returns: `schema_version`, `status` (`success`/`partial`/`degraded`/`cancelled`), `observations` (a `list[ToolObservation]` — the exact same reused model the supervisor's own Observe node produces, so the supervisor can flatten them into its own public observation list unchanged, structurally, not by convention), `completed_capabilities`/`degraded_capabilities`, `warnings`, `calls_consumed`, `transitions_consumed`, `tool_call_count_by_action`, `new_fingerprints`. Both `_specialist_end_node` (the normal-completion path) and `_degraded_result_from_partial_state` (the recursion-limit-recovery path) construct this model and let Pydantic validate it — `invoke_travel_search_specialist()` cannot return an untyped dict even by accident. `phase4/graph.py`'s Execute node flattens `result.observations` (each already a validated `ToolObservation`) into the supervisor's own public `observations` list.

## 4. Separate agent boundaries — construction, never prompt-sniffing

`phase4/graph.py::build_graph()` now takes two explicit, separate `DecisionProvider` parameters: `decision_provider` (the supervisor's own) and `specialist_decision_provider` (a SEPARATE instance for the internal specialist). It builds the specialist's own compiled graph once, at construction time, via `build_specialist_graph(tool_executor, specialist_decision_provider, ...)` — the two roles are distinguished purely by which object the caller constructed and passed to which parameter, never by inspecting `system` prompt text at runtime. (The rejected first draft's `FixtureDecisionProvider` told the two roles apart by checking for the substring `'travel_search_complete'` in the prompt; that class no longer exists.)

- **Fixture mode** (`orchestration/system_a/fixture_decision_provider.py`): two classes, `SupervisorFixtureDecisionProvider` and `SpecialistFixtureDecisionProvider`, each deterministic, each ignoring `system` entirely — role is fixed by class identity. `entrypoint.py` constructs one of each.
- **Real mode**: `entrypoint.py` constructs two separate `QwenDecisionProvider()` instances (a frozen, stateless-per-call dataclass — the two are behaviorally interchangeable, but constructing two keeps "role" a real constructor-time fact for both modes symmetrically, per this checkpoint's own requirement, rather than relying on that interchangeability).

Each prompt keeps its own closed action allowlist and JSON-Schema argument contract, generated from the shared `ACTION_ARGUMENT_MODELS` registry via `phase4/prompt_contract.py::action_argument_contract()` (extracted to its own module so both prompt builders draw from exactly the same canonical source, never two hand-maintained copies). `_decide_node` (supervisor) structurally rejects any of the five specialist-only tool names; `_specialist_decide` (specialist) structurally rejects `call_travel_search`/`synthesize`/`ask_clarification`/`degrade`/`call_istanbul_expert` — both via `Action` enum/schema membership, not prompt wording.

## 5. Bounds

| Bound | Value | Scope |
|---|---|---|
| `MAX_GRAPH_TRANSITIONS` | 25 | supervisor's own graph, unchanged |
| `MAX_EXTERNAL_TOOL_CALLS` | 8 | **shared** — one counter, seeded into the specialist's `TravelSearchState.tool_call_count` at invocation, read back out via `TravelSearchResult.tool_call_count_by_action` |
| `MAX_CALLS_PER_TOOL` | 2 | shared, per-action-name, checked independently inside `_specialist_decide` (specialist) and `_decide_node` (supervisor, `call_istanbul_expert` only) |
| `MAX_DECISION_REPAIRS` | 2 | supervisor's own decision-repair budget |
| `MAX_SPECIALIST_DECISION_REPAIRS` | 2 | specialist's own, separate decision-repair budget |
| `MAX_SPECIALIST_GRAPH_TRANSITIONS` | **15** | specialist's own LangGraph `recursion_limit` |
| `MAX_SPECIALIST_EXTERNAL_CALLS` | **5** | specialist's own, smaller, per-delegation real-tool-call backstop, distinct from the shared 8 |
| `TOTAL_WORKFLOW_DEADLINE_SECONDS` | 60.0 | shared, one clock origin (`started_at_monotonic`), read inside `_specialist_decide` every iteration |

No existing bound was loosened; `MAX_SPECIALIST_GRAPH_TRANSITIONS`/`MAX_SPECIALIST_EXTERNAL_CALLS` are new, additive, and strictly tighter than their shared counterparts.

**A real bug found and fixed during this correction's own test-writing:** `state.get("started_at_monotonic") or monotonic_clock()` (both in `specialist.py`'s own decide node and in `graph.py`'s call into `invoke_travel_search_specialist`) silently discarded a legitimate `started_at_monotonic` value of exactly `0.0` — Python's `or` treats `0.0` as falsy, so it fell through to a FRESH clock read instead, silently resetting the deadline origin. Caught by `test_deadline_propagates_into_the_specialist_graph` (a real seeded value of `0.0` combined with a clock stub that always returns `61.0`): the deadline should have fired immediately but didn't, because `started_at` was silently recomputed as `61.0`. Fixed to an explicit `is None` check in both places (matching the pattern the supervisor's own pre-existing `_decide_node` already used correctly). `0.0` is not a realistic real-`time.monotonic()` value, but it is a realistic *test* value, and the bug would have silently reset the deadline for any caller that legitimately passed it.

## 6. Honest, real-time SSE — not a post-hoc batch

The first draft's `RunService._handle_stream_update` derived `action_started`/`action_completed` events for a delegation from the supervisor's own "execute" node update, AFTER `_run_travel_search_specialist`'s plain loop had already run to completion inside that one node call — every event for every specialist action landed on the wire at once, back to back, looking live but reporting entirely past-tense information.

The corrected design drives the specialist's own compiled graph with `.stream(..., stream_mode="updates")` (§2) and calls an injected `on_event` callback — `specialist_event_callback`, threaded from `build_graph()` down to `invoke_travel_search_specialist` — once per REAL specialist transition, DURING that iteration: `specialist_decide` naming a real tool fires `action_started` immediately (before that tool actually runs); the following `specialist_execute_observe` update fires `action_completed`/`action_failed` immediately after. `orchestration/system_a/service.py::RunService._handle_specialist_event` is that callback — called synchronously, on the same worker thread, during the supervisor's own `graph.stream()` iteration (the specialist's sub-stream runs entirely inside the supervisor's own Execute node call), so `self._emit()` writes each event to `RunStore` in true execution order, not reconstructed afterward. The old post-hoc `node_name == "execute"` branch in `_handle_stream_update` was deleted, not merely supplemented.

Verified (`orchestration/tests/test_sse_streaming.py`):
- `test_delegated_multi_action_sse_events_are_genuinely_live_not_batched` — a 3-action delegation (`get_weather`, `web_search`, `search_flights`) produces the wire sequence `run_started, action_started(get_weather), action_completed(get_weather), action_started(web_search), action_completed(web_search), action_started(search_flights), action_completed(search_flights), run_completed` — never three `action_started` events clustered before any `action_completed` (the shape the rejected batched design would have produced) — and every emitted action name matches the persisted final result's own observation list exactly.
- `test_reconnect_during_a_delegation_does_not_duplicate_specialist_events` — reconnecting with `Last-Event-ID` set to a point mid-delegation replays exactly the tail of events after that point, no fewer, no more, no duplicates.
- `test_sse_events_are_ordered_and_terminate_after_terminal_event`/`test_no_chain_of_thought_prompt_or_secret_in_any_sse_payload` (pre-existing, re-verified) — event shape and content sanitization are unaffected.

Live proof against real running containers (§8): a real `docker compose` fixture-mode full-trip run streamed `run_started → action_started/action_completed(get_weather) → action_started/action_completed(search_flights) → action_started/action_completed(search_stays) → action_started/action_completed(call_istanbul_expert) → run_completed` over the real HTTP/SSE endpoint.

## 7. Regression repair and new coverage

Every test that scripted one of the five now-delegated tool names as a direct supervisor decision, or called `build_graph()`/`create_app()` with only one decision provider, needed updating for the new two-provider construction — mechanical in most files, substantive in two:

- `test_maximum_25_graph_transitions_enforced_by_recursion_limit`'s original all-six-tools-cycling provider was replaced with one that always proposes `call_travel_search`; since that action is structurally invalid for the specialist's own prompt, each delegation exhausts the specialist's repair budget and falls back to an immediate `travel_search_complete`, consuming zero real external tool calls — isolating the supervisor's 25-transition ceiling as the only bound that can fire.
- A secret-isolation test (`test_internal_exception_produces_a_sanitized_failed_result_never_a_raw_traceback`) was found, on inspection, to have been passing for the wrong reason after the first D.3 draft's graph changes (an `IndexError` from a starved fixture, not the `RuntimeError` it claims to prove is sanitized) — fixed to genuinely route through the specialist and reach the raising executor.

New tests added specifically for this correction pass (`services/planner-a/phase4/tests/test_graph.py` unless noted):

- `test_specialist_graph_recursion_limit_is_a_real_langgraph_bound` — drives the compiled specialist graph directly with an artificially tiny `recursion_limit`, proving a real `GraphRecursionError` fires and real partial progress is recoverable from the checkpointer.
- `test_production_specialist_wrapper_recovers_cleanly_from_a_recursion_limit` — the production wrapper (`invoke_travel_search_specialist`, real `MAX_SPECIALIST_GRAPH_TRANSITIONS`) proven end to end via `monkeypatch` shrinking that one bound; asserts a valid, schema-compliant `TravelSearchResult` with real partial progress, never a crash or an untyped dict.
- `test_specialist_external_call_limit_of_five_fires_before_the_shared_eight_call_budget` — a decision provider that never voluntarily stops is cut off at exactly 5 real calls (never 8), against a shared budget still mostly empty.
- `test_global_eight_call_budget_is_shared_and_cannot_be_reset_by_a_second_delegation` — a first delegation (5 calls) + a direct supervisor `call_istanbul_expert` (6th) + a second delegation is only ever allowed 2 more (7th, 8th) before the SAME shared ceiling stops it — the budget is never reset by starting a new delegation.
- `test_cancellation_propagates_into_the_specialist_graph` / `test_deadline_propagates_into_the_specialist_graph` — direct proof (not routed only through the public-API gated-executor tests) that cancellation and the shared deadline are honored inside the specialist's own `specialist_decide` node.
- `test_specialist_duplicate_call_breaks_the_delegation_loop_without_re_executing` (carried over from the first draft, still valid against the new graph) and `test_duplicate_action_is_blocked_not_re_executed` (rewritten to use `call_istanbul_expert`, the one remaining supervisor-level `TOOL_CALL_ACTION`, since `get_weather` moved to the specialist).
- `test_tool_call_actions_partition_is_exactly_the_six_original_actions` / `test_call_istanbul_expert_is_the_one_tool_call_action_the_supervisor_still_owns_directly` / `test_specialist_and_supervisor_actions_are_a_strict_partition_of_every_action` (`phase4/tests/test_models.py`) — a direct structural audit of `TOOL_CALL_ACTIONS`/`SPECIALIST_ACTIONS`/`SUPERVISOR_ACTIONS`, proving `call_istanbul_expert` is the only tool-call action still supervisor-owned. Combined with `orchestration/tests/test_full_trip_hermetic.py` (the REAL `ProductionToolExecutor`, not `FakeToolExecutor`) asserting `tool_call_count == 5` for a plan that includes `call_istanbul_expert`, this proves the claim numerically, not only via final-result success.

## 8. Structural conformance evaluation (`phase4/tests/test_d3_evaluation.py`)

A twelve-case ground-truth suite run through the real compiled supervisor+specialist graph with `FakeToolExecutor` (hermetic, no paid call). Every case's decision sequence is **scripted** — a hand-written ground-truth plan, never a live Qwen call or any model inference.

**What these metrics prove, and what they explicitly do not:** they prove the supervisor and specialist GRAPHS enforce the expected delegation routes, tool-execution boundaries, the shared call budget, and degradation behavior EXACTLY as designed, when handed a plan already known to be correct (or, for the adversarial cases, an input already known to be invalid/malicious/failing). They do **not** measure the live Qwen model's own ability to CHOOSE those routes from free text — that is model routing/intent-recognition accuracy, a distinct and still-open question that belongs to this project's final evaluation phase (not yet started) and requires real, paid model calls this hermetic suite is forbidden from making. No metric name below should be read, quoted, or cited as a live-model accuracy figure.

Cases: full trip, flight-only, stay-only, weather-only, current web evidence, local-itinerary-only, local-knowledge-only (no delegation needed), invalid input (InputGuard-rejected), malicious/prompt-injection input (InputGuard-rejected), provider failure (`get_weather` → `unavailable`), MCP failure (`search_stays` → `timeout`), and a duplicate-follow-up (a second, resumed turn) case.

Measured (real run, `pytest -s phase4/tests/test_d3_evaluation.py`; numerators/denominators unchanged from the first measurement — only the metric names were corrected):

| Metric (structural conformance under scripted ground truth) | Result |
|---|---|
| Supervisor delegation conformance | 11/11 = 100.0% |
| Specialist tool-execution conformance | 11/11 = 100.0% |
| Unnecessary scripted-delegation rate | 0/11 = 0.0% |
| Unnecessary executed-tool rate | 0/14 = 0.0% |
| Schema validity (`check_output`, zero violations) | 11/11 = 100.0% |
| Degradation conformance (the 4 adversarial/failure cases) | 4/4 = 100.0% |
| Duplicate follow-up: real executions | 1/1 across 2 turns = 100.0% (never repeated) |

None of the rows above are, or should be read as, live Qwen model routing/intent-recognition accuracy — see the caption above the table.

Every case is also asserted individually (delegation flag, exact specialist/supervisor action sets, exact final status, zero `check_output` violations) — the aggregate percentages above cannot hide one broken case behind an average.

## 9. Verification

Full regression, all green:

| Suite | Result |
|---|---|
| `orchestration/tests` (excl. opt-in cross-process gate) | 97 passed |
| `services/planner-a/phase4` | 114 passed, 1 skipped |
| `services/planner-a/phase1` | 61 passed |
| `contracts/tests` | 429 passed |
| `services/frontend/phase6/tests` | 60 passed |

`git diff --check`: clean in both the superproject and the `services/planner-a` submodule (only harmless LF→CRLF conversion notices). Credential-literal scan across the full diff boundary and every new file: nothing found.

**Live Docker Compose gate** (`docker compose build agent-system-a` then `docker compose up -d`, fixture mode, zero paid-provider calls): all five services (`agent-system-a`, `agent-system-b`, `mcp-server`, `vector-db`, `chatbot-ui`) reached `healthy`; exactly five containers, confirmed by count. A real trip submitted through `POST /v1/runs` streamed the honest, real-time SSE sequence quoted in §6 over the real HTTP/SSE endpoint and reached a schema-valid `completed` result (`get_weather`, `search_flights`, `search_stays` via the specialist; `call_istanbul_expert` via the supervisor directly to System B over real A2A on the Compose network). `docker compose restart agent-system-a` preserved the completed run AND every specialist-produced observation, fetched again afterward byte-for-byte. `chatbot-ui`'s own health endpoint responded `200` throughout. `docker compose down` stopped all five cleanly; zero containers remained.

## 10. Explicitly deferred / unaffected

System B and the A2A boundary are untouched — `call_istanbul_expert` remains a single, direct, supervisor-level action exactly as Checkpoint D.1 wired it. The public System A API/SSE/SQLite surface, the Streamlit frontend, and Docker Compose topology are unchanged in shape (`phase6/results.py`'s extraction functions key off flattened `obs["action"]` string values, which remain exactly the original six tool names — never `call_travel_search`/`travel_search_complete` — so no frontend change was needed or made). Fixture mode performs no paid call. §8's structural-conformance suite does not, and is not claimed to, exercise a live LLM's own intent recognition or routing choice — real model routing accuracy is deferred, unstarted, to this project's final evaluation phase, and will require real paid model calls this hermetic suite cannot make.

## 11. Persistence behavior — exact scope

- Each specialist delegation runs against a **fresh, function-call-scoped `MemorySaver`** (`phase4/specialist.py::build_specialist_graph`, a new checkpointer instance per `build_specialist_graph()` call, a fresh `thread_id` per delegation within `invoke_travel_search_specialist`) — this checkpointer exists only for the duration of that one `invoke_travel_search_specialist()` call and is never written to disk.
- The specialist's own validated `TravelSearchResult` — and, once flattened, every one of its `observations` — is merged into the SUPERVISOR's own `PlannerState` by `phase4/graph.py`'s Execute node, and it is the SUPERVISOR's graph (not the specialist's) whose state is checkpointed into the real, on-disk System A `SqliteSaver`/`RunStore` (Checkpoint D.2A, unchanged by this checkpoint). This is why a completed run's specialist-produced observations persist and survive a System A restart (verified live in §9's Docker gate: `docker compose restart agent-system-a` preserved `get_weather`/`search_flights`/`search_stays` unchanged) — they were already copied into the supervisor's own persisted state before that restart, not recovered from the specialist's own (already-discarded) in-memory checkpointer.
- **An in-progress specialist delegation is NOT resumed mid-delegation after a process crash.** If System A's process dies while a delegation is in progress, the specialist's own `MemorySaver` (in-memory, function-call-scoped) is gone with it — there is no on-disk record of "the specialist had completed 2 of its planned 4 tool calls." This is an accepted limitation, not hidden and not worked around by this checkpoint: Checkpoint D.2A's existing startup reconciliation (`RunStore.reconcile_incomplete_runs_on_startup`, unchanged) already marks any run left `running`/`pending` at a prior process's death as `failed` — honest, not a false "resumed successfully" — and that is exactly what happens to a run interrupted mid-delegation too. No new persistence system was built to narrow this gap; D.2A's own existing, coarser "the whole run reconciles to failed" behavior already covers it.
