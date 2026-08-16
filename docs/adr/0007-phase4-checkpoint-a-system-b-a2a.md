# 0007. Phase 4 Checkpoint A — System B / A2A Vertical Slice

- **Status:** Accepted
- **Date:** 2026-08-17
- **Superproject commit at closure:** `41506f8901e624367eccefc5132864767226a27f` (branch `phase-4/istanbul-expert`, forked from the closed `phase-3/multilingual-rag` branch)
- **`services/istanbul-expert-b` commit at closure:** `4f65ac7971368044adf79350c086dc0982c2d248` (branch `phase-4/istanbul-expert`)

## 1. Checkpoint scope

Checkpoint A delivers a complete, independently runnable, independently testable vertical slice of System B (`agent-system-b`, `services/istanbul-expert-b`): a Google ADK agent exposed over the official A2A protocol, composed with a genuine FastAPI application, implementing a deterministic side-aware and ferry-aware day-by-day Istanbul itinerary engine with hard-constraint validation and stay-accessibility scoring. It is the first working slice of System B; it is not the completion of Phase 4 (see §11).

## 2. Why deterministic Python — not an LLM — owns arithmetic, geography, scheduling, and hard constraints

Itinerary construction in this checkpoint requires: summing walking/transfer minutes against a budget, computing great-circle distance between coordinates (Haversine), counting Bosphorus side-crossings, enforcing a small hard-constraint DSL (`max_daily_walking_minutes<=N`, `must_visit:*`, `avoid_side:*`, `no_ferry`), and repairing violations by dropping the lowest-ranked non-must-visit POI. Every one of these is a closed-form computation with a single correct answer. Per architecture.md's cross-system invariant that money is never binary float and per the project's zero-hard-itinerary-violation target (§14.10), none of this is delegated to an LLM: `phase4/itinerary.py`, `phase4/scoring.py`, and `phase4/travel_time.py` are pure, deterministic, unit-testable Python with no model call in the loop. The ADK agent (`phase4/agent.py`) is a thin, deterministic `BaseAgent` — it parses the request, calls the deterministic core (`phase4/service_core.py`), and re-emits the validated result. No generation step exists in this checkpoint; there is nothing for an LLM to hallucinate.

## 3. FastAPI / A2A composition design

`phase4/a2a_server.py::build_app()` returns a genuine `fastapi.FastAPI` instance, not the raw Starlette app produced by `to_a2a()`. Composition:

1. `to_a2a(agent, host=..., port=..., protocol=...)` builds the official A2A Starlette app unmodified — Agent Card discovery and A2A RPC routing are never hand-rolled or reimplemented.
2. A `FastAPI(title=..., lifespan=lifespan)` instance is constructed, where `lifespan` is a dedicated async context manager that does `async with a2a_app.router.lifespan_context(a2a_app): yield` — Starlette's own public mechanism for the ASGI lifespan protocol, cascading the A2A app's startup/shutdown into the outer app's. Verified necessary by direct repro: without this cascade, Agent Card discovery 404s because `to_a2a()`'s own startup-time route/executor wiring never runs (`phase4/tests/test_fastapi_composition.py::test_a2a_app_startup_genuinely_requires_the_cascaded_outer_lifespan`).
3. `/health` and `/v1/local-plan` are registered on the outer `FastAPI` app as real `APIRoute`s.
4. Only after both routes are registered is the A2A app mounted at `/` via `app.mount("/", a2a_app)`. Starlette/FastAPI route matching is registration-order-sensitive, so the two specific routes always win and the mount can never shadow them (`test_the_a2a_app_is_mounted_after_the_two_fastapi_routes`).

One process, one port, two cooperating lifespans — never a hand-rolled substitute for the SDK's own A2A transport or Agent Card generation.

## 4. Separate-process A2A evidence

`phase4/tests/test_a2a_integration.py::test_real_separate_process_a2a_discovery_task_submission_and_artifact` launches `python -m phase4.run_server` as a genuine OS subprocess (not an in-process import), polls readiness via real Agent Card resolution over real HTTP, submits a real A2A task over real HTTP using the official `a2a-sdk` client (`A2ACardResolver`, `ClientFactory`, `client.send_message()`), asserts the task reaches `TASK_STATE_COMPLETED` with a schema-valid `LocalItinerary` artifact (`LocalItinerary.model_validate(data)`), then asserts clean shutdown: `proc.terminate()` followed by `proc.wait(timeout=5)` succeeds without ever needing `SIGKILL`. This is the System A↔System B network-boundary contract exercised for real, not simulated.

## 5. Container evidence

The submodule's `Dockerfile` builds independently: build context is `services/istanbul-expert-b` only, `COPY phase0 phase1 phase4`, no dependency on the root-owned `rag/` package. This was verified by an actual `docker build` + `docker run` + health check + all 4 endpoints (`/health`, `/v1/local-plan`, `/.well-known/agent-card.json`, the A2A task RPC) + clean shutdown, not merely inspected. Because `rag/` is structurally unreachable inside the container, `phase4/rag_client.py::get_citation_provider()` always returns `None` there and the service degrades to the built-in structured POI catalog with no descriptive RAG claims (architecture.md §13.5) — by design, not a gap.

Three real bugs were caught only by actually running the container, never by local dev testing:

1. `Path(__file__).resolve().parents[3]` IndexErrors in `poi_catalog.py` and `rag_client.py`, because the container filesystem is shallower than the local monorepo checkout; fixed by guarding `len(file_parents) <= 3` before indexing.
2. A malformed (non-UUID) inbound `trace_id` crashed the `/v1/local-plan` error-handling safety net itself: `build_error_envelope()` called `UUID(str(trace_id))` unguarded while building the very envelope meant to report the original failure safely, so the second exception escaped uncaught and Starlette returned a bare "Internal Server Error" instead of the intended structured JSON envelope — a direct violation of architecture.md §13.4. Fixed with a `_resolve_trace_id()` helper in `phase4/errors.py` that falls back to a freshly generated UUID on any unparseable input; covered by a new regression test, `test_malformed_trace_id_never_crashes_the_error_envelope_path`.
3. `docker stop` needed `SIGKILL` (exit code 137) instead of shutting down cleanly on `SIGTERM` within the grace period. Root cause: the Dockerfile's `CMD ["sh", "-c", "python -m phase4.run_server ..."]` runs uvicorn as a child of `sh`, which does not forward `SIGTERM` to its child by default, so the process never saw the signal. Fixed by adding `exec` to the shell command so the Python process replaces `sh` as PID 1 and receives signals directly; verified afterward with `docker stop` completing in 2s at exit code 0.

## 6. Stay-accessibility formula

`phase4/scoring.py::score_stay_accessibility()`, weights asserted to sum to exactly 1.00 at call time:

- Proximity: **0.60**
- Side alignment: **0.25**
- Crossing avoidance: **0.15**

Side is inferred from a stay's coordinates via a documented, simplified fixed-longitude threshold (`BOSPHORUS_LONGITUDE_THRESHOLD = 28.99`), explicitly surfaced as an assumption in `LocalItinerary.assumptions` rather than presented as a geodesic boundary.

## 7. Hard-constraint repair limit

`phase4.config.MAX_REPAIR_ITERATIONS = 2`. `_build_day`/`_repair_day` in `phase4/itinerary.py` attempt up to `MAX_REPAIR_ITERATIONS + 1` total passes, dropping the lowest-ranked non-must-visit POI on each repair; any violation still unresolved after the limit is surfaced as a warning rather than silently dropped or silently left in the output.

## 8. Baseline comparison — 22 cases

`services/istanbul-expert-b/phase4/evaluation/run_baseline_comparison.py` (reproducibility command: `python -m phase4.evaluation.run_baseline_comparison`), written to `evaluation/datasets/istanbul_expert_baseline_{comparison_report,raw_results}.json`. All 22 hand-authored fixture cases were feasible under both strategies, with **zero hard violations in either strategy**:

| Metric (mean over 22 cases) | Naive baseline | Side-aware |
|---|---:|---:|
| Side crossings | 0.591 | **0.091** |
| Interest coverage | 0.121 | 0.121 |
| POIs scheduled | 7.045 | 7.136 |
| Walking minutes | 25.318 | 32.159 |
| Transfer minutes | 69.623 | 68.800 |
| Daily slack minutes | 280.941 | 273.032 |
| Runtime (ms) | 0.168 | 0.190 |
| Hard violations (total) | 0 | 0 |

## 9. Honest interpretation

The side-aware engine's clear, load-bearing win is side-crossing reduction: **0.591 → 0.091 mean crossings**, roughly an 85% reduction, which is exactly the property this checkpoint was built to deliver (a side-aware, ferry-aware itinerary). Interest coverage is identical between strategies (0.121 both) — side-awareness did not come at the cost of matching fewer stated interests. Transfer minutes and daily slack both improved marginally. However, **walking minutes are higher under the side-aware strategy** (32.159 vs 25.318) — clustering POIs by side trades some transfer efficiency for more same-side walking, a real tradeoff, not a universal improvement. Runtime is marginally higher (0.19ms vs 0.168ms) and both are effectively instantaneous. There is no fabricated global superiority claim here: the side-aware engine is better specifically at the metric it targets (crossings) and neutral-to-mixed on the others, over a 22-case hand-authored fixture set, not a large or adversarially sampled benchmark. These numbers describe this fixture set under this implementation, not a general claim about itinerary quality.

## 10. What Checkpoint A does not include (deferred to Checkpoint B or later)

Explicitly out of scope for this checkpoint and not implemented:

- Operational Qdrant/RAG integration inside System B (the container structurally cannot reach `rag/`; `rag_client.py` always degrades to the POI catalog there)
- Web search
- Live flight data
- Live weather data
- System A (LangGraph/planner-a)
- Frontend (Streamlit chatbot-ui)
- Root Docker Compose orchestration
- Additional POI catalog expansion beyond the 8 built-in fixtures
- Migration to a native ADK session service (current persistence is a bespoke SQLite `SessionStore`)
- Non-root container user
- Any new LLM provider integration

## 11. Status

Phase 4 Checkpoint A is complete and closed as of this ADR. Phase 4 overall is **not** closed — Checkpoint B (operational Qdrant/RAG integration into System B, among the items in §10) is the next gate and has not started.
