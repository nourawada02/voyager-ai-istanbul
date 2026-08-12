# VoyagerAI Istanbul

Decision-support system for trip planning to Istanbul: ranked flights, ranked stays with an explainable fair-price deal score, a side-aware and ferry-aware day-by-day itinerary, and a deterministic budget breakdown, all with full data-provenance. It never books, reserves, or pays for anything. Full spec: [`docs/architecture.md`](docs/architecture.md).

## Repository / submodule boundaries

This root repo is an orchestrator. Each `services/*` directory is an **independent git submodule with its own remote and commit history**:

| Submodule | Architecture role | Stack |
|---|---|---|
| `services/planner-a` | `agent-system-a` | LangGraph, FastAPI, SSE |
| `services/istanbul-expert-b` | `agent-system-b` | Google ADK, FastAPI, A2A |
| `services/travel-mcp` | `mcp-server` | Official MCP Python SDK, Streamable HTTP |
| `services/frontend` | `chatbot-ui` | Streamlit |
| *(no submodule — managed image)* | `vector-db` | Qdrant |

A change inside a submodule and the corresponding pointer bump in this root repo are **two separate commits, in two separate repos**. Committing a submodule pointer bump here does not commit whatever you changed inside that submodule, and vice versa.

## Cross-system invariants

- A2A is used only across the System A / System B network boundary — never between LangGraph nodes or plain function calls. ([architecture.md §4.2](docs/architecture.md#42-design-principles))
- MCP is used only for shared tools reachable over the network. ([§4.2](docs/architecture.md#42-design-principles))
- RAG (Qdrant) holds only stable knowledge; live/dynamic facts always go through MCP, never retrieval. ([§9.1](docs/architecture.md#91-corpus-boundary))
- Money is `Decimal` or integer minor units — never binary float. ([§12](docs/architecture.md#12-budget--currency-engine))
- No stack traces, exception text, file paths, or credentials ever reach a caller; full detail is logged server-side keyed by `trace_id`. ([§13.4](docs/architecture.md#134-error-handling--no-internal-detail-ever-reaches-the-caller))
- The system never books, reserves, or pays for anything, and never claims a fixture/historical listing is currently available. ([§1](docs/architecture.md#1-executive-summary), [§13.2](docs/architecture.md#132-output-guard))
- Every MCP tool / provider-adapter response carries the `data_mode`/provenance envelope. ([§8.2](docs/architecture.md#82-required-response-envelope)) A2A requests and exchanges separately carry their own required fields — `session_id`/`trace_id`/contract version on `LocalPlanRequest`, `task_id`/`context_id`/protocol version/artifact schema version on the exchange. ([§7.1–7.3](docs/architecture.md#7-agent-to-agent-a2a-collaboration-contract)) These are two distinct requirements — do not merge them into one universal envelope.

## Phase-gating rules

One gated phase at a time. No whole-repository code generation in a single pass. Each phase report lists files changed, commands run, test results, unresolved risks, and the next gate. A broken gate is reported, never hidden or worked around silently. Full phase table: [§17](docs/architecture.md#17-implementation-timeline--gated-phases).

**Current state as of 2026-08-12: Phase 0 (compatibility spike) closed. Phase 1 (contracts & fixtures) closed. Phase 2 (data & accommodation ML) Checkpoint A (data and architecture foundation) complete. Checkpoint B has not started — no model trained, no MCP tool implemented yet.** Update this line only when a phase gate is formally passed.

## Testing / reporting expectations

Every later phase is measured against the targets in [§14.10](docs/architecture.md#1410-targets): routing accuracy ≥90%, tool-selection correctness ≥90%, schema validity 100%, budget arithmetic errors 0, hard itinerary violations 0, grounded-claim citation coverage 100%, external-call timeout test coverage 100%, ML MAE improvement over strong baseline ≥10% to enable a learned model.

## Operational warnings

- Submodule commits and root pointer-bump commits are never combined into one commit.
- Do not commit or push without explicit approval.
