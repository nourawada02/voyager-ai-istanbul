# ADR 0001: Phase 1 contract, ownership, and scope decisions

Status: accepted. Date: 2026-08-11.

This is a design note, not a rewrite of `docs/architecture.md` (canonical) or
`docs/architecture-plan.pdf` (immutable original) — it records the specific
implementation decisions Phase 1 made where the architecture document
specifies intent but not file-level mechanics.

## Decisions

1. **A2A communication uses the official resolver and A2A client.** All
   agent-to-agent calls go through `A2ACardResolver` (Agent Card discovery)
   and a client built by the official A2A SDK's `ClientFactory`, exactly as
   proven in the Phase 0 spike. No hand-rolled A2A transport code.
2. **MCP communication uses the official MCP client and `ClientSession`.**
   All MCP tool calls go through the official `mcp` package's
   `ClientSession` over Streamable HTTP, exactly as proven in Phase 0. No
   hand-rolled MCP transport code.
3. **`httpx` is reserved for provider adapters, readiness probes, or SDK
   internals** — never as a substitute for the official A2A or MCP clients
   for protocol traffic between System A, System B, and the MCP server.
4. **System B (Istanbul Expert) owns Istanbul knowledge retrieval and
   accesses Qdrant directly.** Per architecture §9.1, RAG holds only stable
   knowledge; System B is the only component with a Qdrant client.
5. **MCP owns dynamic, shared travel capabilities — not Qdrant retrieval.**
   The MCP server fronts provider adapters and fixtures (flights, stays,
   weather, routing, FX, fair-price, operational status); it never queries
   the vector database.
6. **Canonical contracts are JSON Schema files in the superproject**
   (`contracts/*.schema.json`, Draft 2020-12), per architecture §15.1. They
   are the single source of truth for shape; each service mirrors only what
   it needs as Pydantic v2 models — contracts are never imported as Python
   across repositories.
7. **Each service mirrors only the models it consumes or produces** — not
   the full contract catalog. See the per-service ownership list below.
8. **Explicitly out of scope for the foreseeable roadmap**: booking,
   payment, autonomous purchases, production authentication, guaranteed
   live availability, and multi-city planning. Nothing in Phase 1 (or any
   phase since) implements these; the system remains decision-support only,
   per architecture §1.

## Per-service model ownership (Phase 1)

- **`planner-a`**: `TripRequest`, `TripPreferences`, `TripPlan`,
  `CandidateCombination`, `BudgetBreakdown`, `ConstraintConflict`,
  `PartialFailure`, `StreamEvent`, `ErrorEnvelope`, `SourceReference`,
  `DataProvenance`, `DataQuality` (orchestration, trip, stream, aggregation,
  and error contracts).
- **`istanbul-expert-b`**: `LocalPlanRequest`, `LocalItinerary`, `POI`,
  `TravelLeg`, `DailyPlan`, `SourceReference`, `DataProvenance`,
  `DataQuality`, `ErrorEnvelope` (local-plan request/response, itinerary,
  POI, route, provenance, and error contracts).
- **`travel-mcp`**: `ProviderResponseEnvelope`, `FlightOption`,
  `StayOption`, `FairPriceEstimate`, `DataProvenance`, `DataQuality`,
  `ErrorEnvelope` (dynamic tool input/output, provider result, provenance,
  quality, status, and error contracts).

No repository mirrors the full 21-contract catalog; each mirrors only the
subset above.
