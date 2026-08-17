# 0013. Phase 4 Checkpoint D.0 — System A bounded ReAct core and Qwen action planner

- **Status:** Accepted.
- **Date:** 2026-08-17

## 1. Scope

Implements System A's real bounded LangGraph ReAct control loop and a real Qwen structured-decision provider in `services/planner-a/phase4/`, with deterministic fake tool executors for every allowed capability. Real provider/MCP/A2A wiring is explicitly deferred to Checkpoint D.1 (§11) — this checkpoint proves reasoning, routing, bounds, state, validation, persistence, degradation, and Qwen compatibility in isolation, never hiding a real network call inside a D.0 fake executor.

## 2. A genuine LangGraph StateGraph, nine logical states

`phase4/graph.py` builds and compiles a real `langgraph.graph.StateGraph` — not a plain Python loop imported alongside `langgraph` (proven structurally: `build_graph()` returns a `langgraph.graph.state.CompiledStateGraph`, `compiled_graph.get_graph().nodes` lists all nine node names, and `.invoke()` drives execution through LangGraph's own engine, verified end-to-end by every hermetic graph test).

Nine nodes — this checkpoint's concrete realization of ADR 0009 §4.1's conceptual `Assess → Select → Validate → Execute → Observe → Update → (repeat) → Synthesize`, with an explicit `Degrade` exit:

```
InputGuard → LoadSession → Decide ⇄ (Execute → Observe → Update) → Synthesize/Degrade → End
```

`Decide` folds ADR 0009's separate Assess/Select/Validate steps into one node: producing, then Pydantic-validating, a structured `ActionDecision` is one atomic operation, since Pydantic validation *is* the "Validate" step and there is no useful intermediate state between proposing and validating a decision worth its own node/edge. `InputGuard` and `End` are this checkpoint's own additions to ADR 0009's conceptual loop — concrete pre-loop input safety and an explicit LangGraph terminal, both normal, expected refinements when turning a conceptual design into a real graph.

## 3. Structured decisions, Qwen's limited authority

`phase4/models.py::ActionDecision` (`extra="forbid"`) is the only shape a decision provider may ever produce: a closed `Action` enum (`search_flights, search_stays, estimate_fair_price, get_weather, web_search, call_istanbul_expert, ask_clarification, synthesize, degrade`), a closed `ReasonCode` enum, an `arguments` dict validated against exactly that action's own closed Pydantic model (`ACTION_ARGUMENT_MODELS`), and a ≤280-character `explanation` — never a reasoning transcript. An unknown action, an arbitrary URL, code execution, or a booking/payment request is a validation error (`ActionDecisionValidationError`), never a value that reaches the graph (`phase4/tests/test_models.py`). The Decide-node system prompt explicitly states that a user message can never redefine the action list or override these rules, even if it claims to be a system instruction (`phase4/tests/test_graph.py::test_prompt_injection_cannot_add_a_tool` — though that specific case is actually caught earlier, by InputGuard, §7).

Qwen decides which action to take next and may contribute the final synthesis's short narrative string. Deterministic Python retains sole authority over: money (fake fixtures use `Money`/integer minor units, never a float — reused unchanged from `phase1/models.py`), the tool allowlist (Pydantic enum, not a prompt-only rule), iteration/time bounds (§5), two-stage schema validation of every tool result (§6), the booking/payment prohibition (`check_output`, §7), and provenance. Qwen never fabricates a price, schedule, forecast, or citation in place of a real typed tool call.

## 4. Prompt repair: the action-argument contract is generated, not hand-duplicated

**The first live-gate attempt failed** (sanitized evidence preserved below, §10) because the original prompt told Qwen the outer `ActionDecision` shape but never the exact per-action `arguments` field names/constraints. Qwen returned an understandable but invalid response: city names (`"Beirut"`, `"Istanbul"`) instead of IATA codes, and `date`/`passengers`/`trip_type` instead of `depart_date`/`passenger_count`/(no trip-type field at all). Pydantic correctly rejected it and no tool executed — the failure was fully contained.

The fix (`phase4/graph.py::_action_argument_contract`) generates the prompt's per-action schema section **directly from `ACTION_ARGUMENT_MODELS`** — the exact same canonical registry `parse_action_decision` validates against — via each model's own `model_json_schema()` (Pydantic's built-in JSON Schema export, carrying `required`, enum/pattern/format/min/max constraints, and `additionalProperties: false` automatically from each model's `ConfigDict(extra="forbid")`). Only the purely-cosmetic `title` keys are stripped for compactness; nothing is hand-transcribed. `phase4/tests/test_decision_prompt.py::test_contract_is_generated_from_canonical_models_not_duplicated` proves the prompt's schema section is byte-for-byte what the canonical models themselves produce. The prompt additionally states explicitly: use IATA codes not city names (with "Beirut→BEY"/"Istanbul→IST" as this project's own default mapping unless another airport is named), never invent a field name, omit an optional field rather than invent a value, and return only the JSON object with no reasoning/analysis/prompt text/extra keys — plus one concrete valid example for exactly the BEY→IST scenario. Serialization is deterministic (`json.dumps(..., sort_keys=True)`), so the same state always produces the same prompt (`test_prompt_uses_deterministic_ordering`).

**Pydantic validation itself was never weakened** to accept `date`/`passengers`/`trip_type`/`Beirut`/`Istanbul` — the model must conform to the contract, not the reverse (`test_known_malformed_live_gate_response_is_still_rejected`, reconstructing the exact previously-observed malformed payload and asserting it is still rejected after the fix).

## 5. Hard bounds — enforced in code, and an honest interaction between two of them

| Bound | Value | Enforced |
|---|---|---|
| LangGraph `recursion_limit` (whole graph, every node type) | 25 | Passed to every `.invoke()` call; `GraphRecursionError` is caught by `start_session`/`resume_session` and converted to a safe `status="degraded"` result — never a raw exception reaching a caller. |
| External tool calls (this ReAct loop) | 8 | Checked in `Decide` before proposing a new tool call; once reached, any further tool-call decision is overridden to `synthesize`. |
| Calls per individual tool | 2 | Same override, checked independently of the total-8 bound. |
| Decision-format repairs | 2 | `Decide`'s own bounded retry loop (1 initial attempt + up to 2 repairs = 3 total); exhausted repairs force `degrade` with `reason_code=decision_format_invalid`. |
| Total workflow deadline | 60s | Checked at the top of every `Decide` call against an injectable monotonic clock; exceeding it forces `degrade` before the decision provider is even called. |
| Duplicate action | fingerprinted, 1 graceful skip then forced synthesize | `fingerprint_action` (sha256 of canonical action+arguments JSON) checked against `executed_fingerprints`; a second *consecutive* duplicate forces `synthesize` rather than looping — defense in depth alongside the recursion_limit backstop. |
| Cancellation | checked before Decide and before Execute | Immediately routes to `degrade` with `reason_code=cancelled`, no tool call made. |

**Honest, documented finding:** with this checkpoint's required 9-node topology, one tool-call cycle (`Decide → Execute → Observe → Update`) costs 4 LangGraph transitions. `InputGuard`+`LoadSession` cost 2 more. So `k` completed tool calls cost `2 + 4k` transitions — at `k=6`, that is already 26, past the 25-transition ceiling. **In practice, for any run needing more than ~5 non-duplicate tool calls, the 25-transition structural ceiling binds before the 8-total-tool-call bound ever could.** Both bounds are still fully implemented and independently correct — `phase4/tests/test_graph.py::test_maximum_eight_tool_calls_enforced` and `test_per_tool_call_cap_of_two_is_enforced_independently_of_the_total_cap` unit-test the Decide node's own bound logic directly (given a state that already recorded the relevant call counts), precisely because driving 8 *real* tool calls through the full graph would hit the transition ceiling first and never exercise the 8-call bound in isolation; `test_maximum_25_graph_transitions_enforced_by_recursion_limit` separately proves the transition ceiling itself, using a provider that cycles through 6 distinct, valid, never-duplicate tool-call decisions specifically so neither the per-tool nor the duplicate-detection bound intervenes first. This is not a contradiction requiring either number to change — both bounds mean exactly what they say; this is simply which one binds first, in this topology, for a long run. A future tuning pass (not this checkpoint) could revisit either number against real measured latency, exactly as ADR 0009 §4.3 already anticipated for the deadline bound specifically.

## 6. Deterministic fake-tool boundary

`phase4/tools.py::FakeToolExecutor` implements the `ToolExecutor` protocol for D.0: no socket, no SerpApi/Open-Meteo/Groq/Qwen/MCP/A2A quota consumed, deterministic fixtures built directly from the unmodified `phase1/models.py` mirrors (`FlightOption`, `StayOption`, `FairPriceEstimate`, `LocalItinerary`) with `uuid5`-derived (never `uuid4`/random) ids, so identical fixture inputs produce byte-identical output. Supports `success`/`unavailable`/`timeout`/`rate_limited`/`malformed` scenarios per action. `Observe` (`phase4/graph.py::_observe_node`/`_validate_tool_result`) is solely responsible for validating a raw tool result against the appropriate canonical model before it is ever inserted into state — a malformed result is rejected with `status="provider_error"` and a recorded warning, never inserted as if valid, whether it came from the fake or (in D.1) a real adapter. Never imports the root `providers`/`rag` packages (ADR 0009 §6.1) — this submodule has no path-hack dependency on the superproject root.

**Production composition rule for D.1**, recorded here and not implemented: System A owns orchestration; root live-provider adapters are injected at exactly this `ToolExecutor` boundary in place of `FakeToolExecutor`; Travel MCP remains accommodation owner; System B is invoked only through real A2A and never receives provider credentials; A2A is never used between LangGraph nodes (only at the System A/System B network boundary, architecture.md §4.2).

## 7. Input and output guards

`phase4/guards.py::check_input` — deterministic, regex/allowlist-based, never an LLM classification (the same "small explicit allowlist, never inferred from free text" precedent as `providers.web_evidence_serpapi.classify_source_type`). Rejects: excessive text (>2000 chars), a fixed prompt-injection pattern list (attempts to expose secrets or request booking/payment/code execution), and — when a full `trip_request` is supplied — reuses the exact, unmodified `phase1.models.TripRequest` for validation (non-`IST` destination, malformed IATA, invalid traveler count, past/inverted dates, unsupported currency all map to a small set of fixed safe-error codes, never the raw pydantic error text). `check_output` scans the final serialized result for a fixed set of booking/payment/availability terms and forces `status="degraded"` if any appear — the same structural proof already established for the fake flight/stay providers, applied to whatever Qwen's synthesis narrative might otherwise claim.

## 8. Checkpointing and trace privacy

`MemorySaver` (`langgraph.checkpoint.memory`) — the fallback path this checkpoint's own instructions authorized, since `langgraph-checkpoint-sqlite` is not installed/pinned in this environment and no new, unvetted dependency was added to obtain it. `PlannerState` (`phase4/graph.py`) has no chain-of-thought/raw-prompt/raw-response/credential field by construction — structurally asserted in `test_no_raw_chain_of_thought_anywhere_in_state_trace_or_output` (checks `PlannerState.__annotations__` directly) and independently in `test_no_prompt_or_secret_leaks_into_graph_state_trace_or_checkpoint` / `test_no_secret_or_prompt_persisted_in_checkpoint` (serializes the actual checkpointed `channel_values` and asserts the constructed system prompt, `Authorization`, and `Bearer` never appear). `test_resumed_session_does_not_repeat_completed_actions` proves a second `resume_session()` call on the same `thread_id` — deliberately omitting counters/trace/observations from its input so the checkpointer supplies them — does not re-execute an already-completed, fingerprint-identical tool call.

## 9. Qwen decision provider

`phase4/qwen_client.py::QwenDecisionProvider` reuses the already-proven configuration conventions from `rag/llm_providers.py::QwenProvider` (OpenAI-compatible Chat Completions, `QWEN_API_KEY`/`DASHSCOPE_API_KEY`, an explicit-HTTPS `QWEN_BASE_URL` never a guessed region, `enable_thinking: false`, `response_format: json_object`, `temperature=0`) **without importing `rag` at runtime** — reimplemented stdlib-only (`urllib.request`), needing no new third-party HTTP dependency. Default model `qwen3.7-flash` (matching `rag/llm_providers.py::QWEN_GENERATOR_MODEL`'s own default), overridable via `QWEN_MODEL`/`QWEN_GENERATOR_MODEL`. Neither the key nor the base URL is ever stored as a dataclass field — both are read fresh from the environment inside `generate()`, so the object can never leak either through `repr()`/a log line/a checkpoint. Bounded 25s timeout; no internal retry loop of its own (the caller — `Decide` — owns the bounded decision-repair budget, a different concern from a transport retry).

## 10. Live-gate evidence

### 10.1 First attempt — failed (prompt defect, diagnosed and fixed, §4)

```
QWEN_API_KEY present: true
```
One real request; Qwen reachable and authenticated (no `QwenConfigurationError`/`QwenTransportError`). Response failed `SearchFlightsArgs` validation: `origin`/`destination` were city names, not IATA codes; `depart_date`/`passenger_count` were missing; `date`/`passengers`/`trip_type` were rejected as unknown fields (`additionalProperties: false`). No tool executed. No secret, header, prompt text, or full request URL was ever printed — only the model's own returned (wrong) argument shape.

### 10.2 Second attempt — passed, after the prompt repair

```
QWEN_API_KEY present: true
model=qwen3.7-flash
selected_action=search_flights
schema_valid=true
runtime_seconds=2.62
exit_result=pass
```
`decision.action == Action.SEARCH_FLIGHTS`; `arguments == {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1, ...}` (schema-valid, no unknown/extra field); the test additionally asserts `runtime_seconds < 30.0` and scans the serialized decision for `chain_of_thought`/`api_key`/`authorization`/`bearer` (absent). Exactly one live HTTP request was made (the client has no internal retry loop); no `FakeToolExecutor`/tool execution, no MCP, no A2A, no other provider was ever invoked by this test.

## 11. Explicitly deferred to Checkpoint D.1

Real provider adapters (Open-Meteo, SerpApi web/flights) wired in at the `ToolExecutor` boundary; real Travel MCP calls; real A2A calls to System B; the Streamlit frontend; root Docker Compose changes; any booking/payment action. None of these were implemented or attempted in this checkpoint.
