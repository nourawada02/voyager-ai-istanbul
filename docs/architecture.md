<!--
Source: docs/architecture-plan.pdf
Converted: 2026-08-10
Status: Canonical, version-controlled implementation specification.

Conversion scope: The proposal's architectural content (Sections 1-17
below) receives no additions, omissions, or scope changes in this
conversion. The metadata block you are reading, the source-precedence
notes, and the table of contents are editorial scaffolding introduced
by this conversion, not part of the original proposal content.

Source precedence:
- docs/architecture-plan.pdf is the immutable original proposal.
- This file becomes canonical only once the conversion below has been
  manually reviewed and accepted (see Status above).
- Once accepted, future architecture changes are made through reviewed
  diffs to this Markdown file, not by re-exporting the PDF.
- Any discrepancy discovered between this file and the PDF must be
  reported, never silently resolved in either direction.
-->

# VoyagerAI Istanbul — Project Proposal & Architecture Plan

*Production AI Agent System — Final Project*
*Nour Awada*

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Topic Proposal](#2-problem-statement--topic-proposal)
3. [Scope Boundaries](#3-scope-boundaries)
4. [Architecture Overview](#4-architecture-overview)
5. [System A — Voyager Trip Planner](#5-system-a--voyager-trip-planner)
6. [System B — Istanbul Local Expert](#6-system-b--istanbul-local-expert)
7. [Agent-to-Agent (A2A) Collaboration Contract](#7-agent-to-agent-a2a-collaboration-contract)
8. [MCP Server & Tool Inventory](#8-mcp-server--tool-inventory)
9. [Multilingual RAG Pipeline](#9-multilingual-rag-pipeline)
10. [Accommodation Fair-Price ML Pipeline](#10-accommodation-fair-price-ml-pipeline)
11. [Istanbul Itinerary Optimizer](#11-istanbul-itinerary-optimizer)
12. [Budget & Currency Engine](#12-budget--currency-engine)
13. [Guardrails, Resilience & Error Handling](#13-guardrails-resilience--error-handling)
14. [Evaluation Plan](#14-evaluation-plan)
15. [API & Contract Surface](#15-api--contract-surface)
16. [Technology Stack](#16-technology-stack)
17. [Implementation Timeline & Gated Phases](#17-implementation-timeline--gated-phases)

---

## 1. Executive Summary

VoyagerAI Istanbul is a decision-support system for travelers planning trips from anywhere (demonstrated: Beirut) to Istanbul. Given a budget, dates, traveler count, and interest profile, it returns a typed, source-aware plan: ranked flights, ranked stays with an explainable fair-price deal score, a side-aware and ferry-aware day-by-day itinerary, a deterministic budget breakdown, and full data-provenance and confidence indicators. It does not book, reserve, or pay for anything.

The system satisfies every mandatory architecture requirement of the Production AI Agent System brief — two independently deployed agent systems on different stacks communicating over a real network boundary, a multilingual RAG pipeline, a domain MCP server, a five-container Docker Compose deployment, a FastAPI layer with streaming and sessions, input/output guardrails, and an evaluation suite that meets or exceeds every required metric. Istanbul was chosen deliberately: it is a genuine domain boundary with cross-continent travel logistics, three operative languages (English, Turkish, Arabic), and a real, reproducible open dataset (Inside Airbnb) that supports an honest machine-learning experiment.

## 2. Problem Statement & Topic Proposal

Trip planning for Istanbul is unusually hard to automate well because the city spans two continents, has a dynamic, multilingual information ecosystem (ferries, museum hours, transliterated place names), and has no single authoritative source for "is this listing actually a good price." Existing planning tools either ignore geography (treating Istanbul as one undifferentiated area) or ignore price fairness (showing raw listings with no baseline). VoyagerAI Istanbul addresses both: it is accessibility-aware (which stay lets you reach today's plan without excessive transfers) and price-aware (is this listing actually a deal, backed by a model, not a marketing claim).

Topic scope for Version 1: Istanbul only. The architecture is provider-interface-based so future cities are a configuration and data problem, not a redesign — but the project does not claim or build multi-city support now.

## 3. Scope Boundaries

### 3.1 In scope

- Flight and stay search, ranking, and an explainable fair-price / deal score
- Side-aware (European / Asian), ferry-aware multi-day itinerary construction
- Multilingual (EN / TR / AR) grounded retrieval for stable destination knowledge
- Deterministic budget and currency computation with full provenance
- Two independently deployed, network-separated agent systems collaborating over A2A
- Honest degradation when any provider, model, or service is unavailable

## 4. Architecture Overview

Exactly two independently deployed agent systems, one shared MCP tool service, one vector database, and one frontend — five containers total, matching the brief's required container roles one-to-one.

```
UI (Chatbot Frontend)
      |
      v
agent-system-a  --A2A tasks-->  agent-system-b
(LangGraph, Trip Planner)       (Google ADK, Istanbul Expert)
      |                                |
      +-----------> mcp-server <-------+
                        |
                        v
              vector-db (Qdrant, multilingual RAG)
```

mcp-server also fronts provider adapters, fixtures, and the ML artifact.

### 4.1 Container map (aligned to required naming)

| Compose service key (required) | What it runs | Internal folder |
|---|---|---|
| agent-system-a | LangGraph primary trip planner, FastAPI, SSE | services/planner-a |
| agent-system-b | Google ADK Istanbul local expert, FastAPI, A2A | services/istanbul-expert-b |
| mcp-server | Travel MCP server (Streamable HTTP, 7 tools) | services/travel-mcp |
| vector-db | Qdrant — multilingual dense + sparse RAG index | (managed image) |
| chatbot-ui | Streamlit frontend | services/frontend |

### 4.2 Design principles

- Agents interpret, plan, delegate, and resolve conflicts. Python does arithmetic, filtering, scoring, schema validation, and policy enforcement.
- A2A is used only across the System A / System B network boundary — never between LangGraph nodes or plain function calls.
- MCP is used only for shared domain tools reachable over the network.
- RAG is used only for relatively stable knowledge; live/dynamic facts are always fetched fresh through MCP.
- Live, cached, historical, and fixture data are visibly and permanently distinguished in every payload.
- A usable partial plan is always produced when one provider fails.
- Every intelligent claim (ML price estimate, itinerary heuristic, retrieval) is evaluated against a stated baseline.

## 5. System A — Voyager Trip Planner

### 5.1 Stack

- LangGraph orchestration
- FastAPI + Uvicorn
- Pydantic v2 for all request/response models
- httpx for outbound calls to System B and MCP
- LangGraph SQLite checkpointer (Docker-volume backed; swappable for Postgres later)
- Server-Sent Events on the public streaming endpoint

### 5.2 Responsibilities

- Validate and normalize the incoming trip request
- Classify intent and required capabilities so irrelevant branches are skipped
- Maintain session and trip state across turns
- Search flights and stays through MCP; run deterministic candidate filtering and preliminary budget feasibility
- Delegate local Istanbul planning to System B over A2A
- Combine global price/value with System B's accessibility assessment, select a feasible combination
- Deterministically compute and repair the budget, validate the final TripPlan, and stream progress plus the final result

### 5.3 LangGraph workflow

```
input_guard -> load_session -> classify_intent -> parse_trip_request
   -> [optional clarification_interrupt]
   -> parallel: search_flights_tool | search_stays_mcp | fetch_weather_tool
   -> rank_global_candidates -> budget_feasibility
   -> call_istanbul_expert_a2a -> combine_global_and_local_scores
   -> constraint_repair -> output_guard -> format_trip_plan
```

`search_flights_tool` and `fetch_weather_tool` are the independent typed provider tools described in §8's superseded-row note (Phase 4 Checkpoint C.0, ADR 0009) — invoked directly by System A's bounded ReAct orchestration, not `_mcp`-suffixed. `search_stays_mcp` is unchanged: accommodation remains a genuine Travel MCP tool.

The router skips irrelevant branches: a museum-hours question never triggers flight or stay search; a flight-only request never invokes System B.

### 5.4 Human-in-the-loop

A LangGraph interrupt fires only for genuinely material ambiguity: no plan satisfies both budget and hard lodging constraints, a real tradeoff exists between two alternatives (central/expensive vs. distant/cheap), or essential dates/traveler counts are missing. Optional preferences use documented defaults instead of interrupting.

## 6. System B — Istanbul Local Expert

### 6.1 Stack

- Google ADK, independent FastAPI application
- Official A2A SDK (current 1.x line, confirmed in the Phase 0 spike)
- ADK persistent session service, SQLite-backed
- qdrant-client for grounded retrieval

### 6.2 Responsibilities

- Interpret interests, pace, mobility, language, and local constraints
- Normalize Istanbul place aliases to canonical POI IDs
- Retrieve grounded Istanbul knowledge from Qdrant; fetch operational status, weather, and travel-time data through MCP
- Score accessibility of System A's candidate stay locations, select POIs, and build a side-aware, ferry-aware itinerary
- Enforce time windows, transfer buffers, walking limits, daily duration; return a typed LocalItinerary with citations and warnings

System B is one specialist application. Its internal capabilities (scoring, optimization) are tools and deterministic modules, not additional networked agents.

### 6.3 Internal workflow

```
1. Validate LocalPlanRequest
2. Normalize language, aliases, districts, POI names
3. Retrieve stable destination knowledge (Qdrant)
4. Build candidate POI set from the structured catalog
5. Fetch dynamic weather / operational status only when relevant
6. Score each candidate stay for access to chosen POIs
7. Optimize the itinerary around the best feasible base
8. Repair hard violations, max 2 iterations
9. Generate explanations only after the schedule validates
10. Return a schema-valid A2A artifact
```

## 7. Agent-to-Agent (A2A) Collaboration Contract

A2A is used only between System A and System B, carrying real decision-relevant payloads in both directions — this is the collaboration the brief is testing, not a formality.

### 7.1 A → B: LocalPlanRequest

- Trip dates and available local time; interests, pace, language, mobility constraints; daily activity budget
- Top three feasible stay candidates with stable IDs and coordinates
- Weather context if already fetched; hard and soft constraints
- session_id, trace_id, and contract version

### 7.2 B → A: LocalItinerary artifact

- Accessibility score per supplied stay candidate; recommended base candidate ID
- Selected POIs with canonical IDs; daily schedules and transfer legs
- Estimated travel/waiting time, expected walking, side crossings, weather substitutions
- Citations, assumptions, warnings, data quality, and hard-constraint validation results

### 7.3 Final decision ownership

System A remains the final decision owner: it combines global stay value, fair-price deal score, System B's accessibility score, and preference fit, minus total-trip budget pressure. If budget repair forces a different base than the one System B used, System A is allowed exactly one additional A2A replan — collaborative, but bounded, with no open-ended negotiation loop. task_id, context_id, protocol version, and artifact schema version are persisted on every exchange for debugging.

## 8. MCP Server & Tool Inventory

One independent MCP server, official Python SDK, Streamable HTTP transport.

| Tool | Primary caller | Purpose |
|---|---|---|
| search_stays | System A | Find snapshot, sandbox, or live stay candidates |
| estimate_fair_price | System A | Run the persisted Istanbul ML pipeline |
| get_travel_times | System B | Route, transit, walking, ferry estimates |
| get_operational_status | System B | Hours, closures, timetable notices from allowlisted sources |
| get_fx_rate | System A | Normalize currencies from a pinned rate snapshot |

**Superseded rows (Phase 4 Checkpoint C.0, accepted; [ADR 0009](adr/0009-phase4-checkpointc0-live-data-and-react-design.md)):** `search_flights` and `get_weather` are no longer Travel MCP tools. Weather, web-search evidence, and flight search are instead independent typed provider/tool capabilities (root `providers/`) that System A's bounded ReAct orchestration invokes directly — never routed through Travel MCP, and never handed to System B, which receives only pre-normalized context from System A and never gains provider credentials. Every other row above is unchanged: Travel MCP remains responsible for accommodation search, fair-price estimation, travel-time, operational-status, and FX-rate lookups exactly as before. This supersedes only these two rows/wording, not the rest of this document.

### 8.1 Provider adapters (fixture-first, live optional)

```
FlightProvider            : FixtureFlightProvider (mandatory), LiveFlightProvider (optional)
StayProvider               : InsideAirbnbSnapshotProvider + FixtureStayProvider (mandatory),
                              LiveStayProvider (optional)
WeatherProvider             : FixtureWeatherProvider (mandatory), OpenMeteoProvider (preferred live)
TravelTimeProvider          : CachedIstanbulMatrixProvider + HaversineFallbackProvider (mandatory),
                               OSRMOrTransitProvider (optional)
OperationalStatusProvider   : FixtureStatusProvider (mandatory), OfficialAllowlistProvider (optional)
```

### 8.2 Required response envelope

*Scope: this envelope shape applies to MCP tool / provider-adapter responses (this section sits directly under the provider adapter list, §8.1, within "MCP Server & Tool Inventory"). The PDF does not state that this exact envelope is also required for A2A payloads — A2A carries its own, separately specified required fields (§7.1–7.3). These are two distinct requirements, not merged here.*

```json
{
  "request_id": "uuid",
  "provider": "inside-airbnb-snapshot",
  "data_mode": "live|cached|historical|fixture|estimated",
  "retrieved_at": "2026-08-09T12:00:00Z",
  "valid_for": "optional ISO-8601 duration",
  "currency": "TRY|null",
  "source_urls": [],
  "quality": { "completeness": 0.94, "freshness": "historical", "assumptions": [] },
  "result": {}
}
```

## 9. Multilingual RAG Pipeline

### 9.1 Corpus boundary

Qdrant holds only relatively stable information: history/culture/heritage, attraction and neighborhood descriptions, stable transport guidance, etiquette, accessibility notes, and structured facts with a known source and review date. It never answers current prices, availability, museum hours, weather, current events, disruptions, or anything requiring live official verification — those go through MCP.

### 9.2 Retrieval stack

- Dense embeddings: intfloat/multilingual-e5-small (exact revision pinned in the ingestion manifest, with query:/passage: prefixes applied)
- Sparse retrieval: BM25-compatible sparse vectors
- Reciprocal Rank Fusion for the hybrid experiment
- Heading-aware recursive token chunking, with metadata filters; no neural reranker in Version 1

### 9.3 Grounded generation prompt (explicit template)

```
SYSTEM: You are the Istanbul knowledge assistant for VoyagerAI.
Answer ONLY using the CONTEXT block below. Do not use outside knowledge.
For every factual claim, attach the chunk id(s) it came from.
If the answer is not present in CONTEXT, respond exactly:
  "I don't have grounded information on this in my current sources."
Never state or imply current prices, availability, or opening hours from CONTEXT;
those must come from a live tool call, not retrieval.

CONTEXT:
{retrieved_chunks}

QUESTION:
{user_question}
```

### 9.4 Entity normalization

Aliases are resolved to canonical POI/district IDs before retrieval (e.g., Hagia Sophia / Ayasofya → poi_ayasofya; Uskudar / Üsküdar → district_uskudar). Original user wording is preserved for the reply; canonical IDs are used for joins, route matrices, evaluation, and citations. Coordinates always come from the verified catalog, never from LLM generation.

### 9.5 Retrieval experiment

| Config | Chunk tokens | Overlap | Top-K |
|---|---|---|---|
| A | 350 | 50 | 3 |
| B | 350 | 50 | 5 |
| C | 700 | 100 | 3 |
| D | 700 | 100 | 5 |

45 ground-truth questions (15 English, 15 Turkish, 15 Arabic) drive a two-stage experiment: first select chunk size and Top-K on dense retrieval, then compare the winning dense configuration against dense+sparse RRF hybrid. Reported: Precision@K, Recall@K, MRR, latency, zero-result rate, per-language breakdown; NDCG only if a reranker or graded relevance judgments are actually introduced.

## 10. Accommodation Fair-Price ML Pipeline

Framed honestly as a fair-price estimator under the dated Istanbul Inside Airbnb snapshot conditions — not a live price predictor, and never implying current availability.

### 10.1 Data pipeline

- Record source URL, snapshot date, retrieval time, license, file checksum, row count, schema
- Explicit currency/numeric parsing; documented outlier rules fitted only on training data
- Normalize amenities/categorical fields; stable feature and target schemas
- Split train/test by host_id where available, to reduce multi-listing host leakage
- All transforms inside a scikit-learn Pipeline/ColumnTransformer
- Persist model, preprocessing, feature schema, dataset manifest, and metrics together

### 10.2 Target, features, models

Target: y = log1p(cleaned nightly listing price). Features: geospatial cluster, neighborhood/side, room/property type, capacity, amenities, review score/count, host-quality fields not derived from price, minimum nights where defensible. Price itself, any price derivative, or a full-dataset target-encoded statistic is never a feature.

| Model | Role |
|---|---|
| Global median | Weak baseline |
| Neighborhood + room-type median | Strong baseline — the promotion bar |
| Ridge regression | Linear learned model |
| HistGradientBoostingRegressor / Random Forest | Non-linear learned model |

Primary metric MAE, plus RMSE, median AE, and R², reported overall and by room type / neighborhood. The learned model is only persisted for production use if it beats the strong baseline by a predeclared margin (10% MAE reduction); otherwise the baseline ships and the negative result is reported honestly.

### 10.3 Deal score

The ML model estimates fair price only; Python computes the recommendation score from stated, capped components:

```
deal_score =
    0.35 * capped_price_value
  + 0.25 * itinerary_accessibility
  + 0.15 * rating_quality
  + 0.10 * preference_match
  + 0.10 * amenities_match
  + 0.05 * review_confidence
```

Every component, weight, raw value, and normalization rule is returned with the score, and the price-advantage contribution is capped so one erroneous prediction cannot dominate ranking.

**Invariant**: the six weights above always sum to exactly `1.00`. Any implementation of this formula must assert this at call time and fail loudly if it does not hold (see `docs/adr/0002-phase2-checkpoint-a.md`).

## 11. Istanbul Itinerary Optimizer

The LLM selects preferences and explains tradeoffs; Python constructs and validates the schedule.

### 11.1 Algorithm

```
1. Filter candidates by hard constraints
2. Score interest and preference fit
3. Cluster by district and European/Asian side
4. Assign coherent clusters to days
5. Evaluate supplied stay candidates against POI access cost
6. Order stops using route-matrix / cached travel-time cost
7. Add congestion/ferry-wait buffers only from explicit data or documented assumptions
8. Calculate walking, transfer, activity, meal, and slack time
9. Detect hard violations
10. Repair, max 2 iterations
11. Generate the explanation after validation
```

### 11.2 Istanbul-specific rules

- Prefer one side per day unless a crossing is itself part of the plan; penalize repeated crossings
- Model ferries as scheduled transfer legs with waiting buffers when schedule evidence exists; label estimated transit/ferry time when only road routing is available
- Keep Sultanahmet/Fatih, Beyoğlu/Galata, Beşiktaş, Kadıköy/Moda, and Üsküdar as separate clusters; a Princes' Islands day is its own cluster, never an inserted stop

### 11.3 Optimization evaluation

Compared against naive popularity ordering on at least 20 trip cases, measuring total travel minutes, side crossings, interest coverage, POIs scheduled, walking/time-window/budget violations, and mean daily slack. The headline claim is numerical and not pre-written — for example, "the side-aware heuristic reduced estimated transfer time by X% and eliminated Y hard violations relative to popularity ordering."

## 12. Budget & Currency Engine

- Money represented with Decimal or integer minor units — never binary floats
- Each source currency preserved; normalized to display currency via one pinned FX snapshot (provider, rate, timestamp, inverse-rate method stored)
- Known, estimated, and contingency costs kept separate; totals recomputed after every repair
- Small candidate sets: enumerate the Cartesian product of top flights × stays, score only feasible combinations — simpler and more explainable than an LLM budget negotiation
- If nothing is feasible: return a structured ConstraintConflict with the minimum budget gap and ranked relaxations; ask the user before relaxing any hard constraint

## 13. Guardrails, Resilience & Error Handling

### 13.1 Input guard

- Istanbul is the only fully supported destination; dates must be valid and non-past
- Trip length, traveler count, and text size are bounded; budget positive, currency supported
- IATA codes, coordinates, language, and mobility fields validate
- User content cannot override tool allowlists, data-mode labels, schemas, or the booking prohibition
- Ambiguous critical fields trigger clarification rather than silent guessing

### 13.2 Output guard

- Every inter-service response and final output validates against its schema
- All monetary totals recomputed in Python; every price declares currency and provenance
- Every dynamic result declares data mode and retrieval time; every RAG-derived claim maps to a source reference
- No fixture or historical listing is ever called currently available; no booking confirmation is invented
- Every itinerary item is within requested dates and passes hard constraints

### 13.3 Iteration limits — every agent loop, explicitly

| Loop | Limit | Enforced by |
|---|---|---|
| LangGraph main graph execution (System A) | recursion_limit = 25 supersteps | LangGraph runtime config |
| ADK agent reasoning loop (System B) | max_iterations = 10 tool-call rounds | ADK run config |
| Itinerary hard-violation repair | max 2 repair iterations | itinerary optimizer |
| A2A replan after budget repair | max 1 additional replan | System A orchestration |
| MCP / A2A / provider call retries | max 2 retries, bounded exponential backoff | httpx retry wrapper |

### 13.4 Error handling — no internal detail ever reaches the caller

*Note: the block below is illustrative/pseudocode as it appears in the source proposal (it mixes JSON-like structure with `//` comments and a `|`-separated placeholder enum), not strict parseable JSON — it is reproduced as-is, not converted into valid JSON, per the source-precedence rule against silently strengthening the source.*

```
ErrorEnvelope {
  "error_code": "MCP_TIMEOUT | A2A_TIMEOUT | SCHEMA_INVALID | ...",
  "message": "human-readable, safe-to-display summary",
  "trace_id": "uuid",
  "retriable": true
}
// Rule: stack traces, internal exception text, file paths, and provider
// credentials NEVER appear in message or anywhere in the response body.
// Full exception detail is logged server-side only, keyed by trace_id.
```

### 13.5 Failure policy

| Failure | Degraded behavior |
|---|---|
| Live flight provider unavailable | Use versioned flight fixtures, labeled |
| Stay source unavailable | Use pinned Istanbul snapshot or fixtures; never claim availability |
| System B / A2A timeout | Return global candidates; state local itinerary is unavailable |
| Qdrant unavailable | Use the structured POI catalog without descriptive RAG claims |
| Route provider unavailable | Cached matrix, then Haversine fallback, labeled as estimate |
| Weather unavailable | Build a weather-neutral plan; disclose missing forecast |
| Operational status unavailable | Omit current-open claims; direct user to verify official sources |
| ML artifact invalid/missing | Fall back to neighborhood + room-type median baseline |

- No hardcoded secrets anywhere; .env.example committed with placeholders, real .env never committed

## 14. Evaluation Plan

### 14.1 Retrieval

- 45 ground-truth questions (15 EN / 15 TR / 15 AR) with expected document, section, content type, POI IDs, and relevant chunk IDs
- Four chunk/Top-K configurations compared, then dense vs. hybrid
- Precision@K, Recall@K, MRR, latency, zero-result rate, per-language breakdown
- Generated-answer rubric: citation coverage, faithfulness, correctness, relevance

### 14.2 Routing and tool selection

At least 50 cases spanning full plan, flight-only, stay-only, price-estimate-only, local knowledge, itinerary-only, follow-up modification, EN/TR/AR, invalid/ambiguous input, non-Istanbul destination, current-hours/status questions, prompt-injection attempts, and provider failure. Metrics: route accuracy, expected tools, unnecessary calls, schema validity, clarification correctness, graceful degradation.

### 14.3 Accommodation ML

Two baselines and at least two learned models compared on an untouched, host-group-separated held-out set. MAE, RMSE, median AE, R², latency, and subgroup metrics reported, with leakage checks and feature ablation recorded. Product use is gated on the predeclared improvement threshold.

### 14.4 Deal ranking

At least 25 pairwise or listwise cases with explicit preference profiles, verifying hard filters, score calculation, weight application, stable ordering, and counterfactual explanations (why A beats B).

### 14.5 Itinerary optimization

At least 20 requests comparing the side-aware heuristic against naive popularity ordering on travel minutes, side crossings, interest coverage, scheduled POIs, daily slack, and hard violations.

### 14.6 A2A / MCP contracts

Contract tests for Agent Card discovery, A2A task/artifact exchange, MCP tool discovery, typed results, timeouts, version mismatch, malformed payloads, and retry limits.

### 14.7 End-to-end and resilience

At least 12 full scenarios plus injected failures for A2A timeout, MCP timeout, Qdrant unavailable, no RAG hits, malformed provider data, route failure, weather failure, and invalid ML artifact.

### 14.8 Configuration comparisons (≥2, with numbers)

- Comparison 1: chunk size / Top-K sweep across configs A–D (dense retrieval)
- Comparison 2: winning dense configuration vs. dense+sparse RRF hybrid

Both reported with the actual measured winner and the measured margin — never asserted in advance.

### 14.9 Failure case documentation — explicit taxonomy

| Field | Requirement |
|---|---|
| Symptom | What the user observed / what broke |
| Failure class | Exactly one of: Model failure \| Prompt failure \| Design failure |
| Root cause | The actual mechanism, traced to a specific component |
| Fix applied | What changed |
| Post-fix result | Measured outcome after the fix, not asserted |

### 14.10 Targets

| Metric | Target |
|---|---|
| Routing accuracy | ≥ 90% |
| Tool-selection correctness | ≥ 90% |
| Final and inter-service schema validity | 100% |
| Budget arithmetic errors | 0 |
| Hard itinerary violations | 0 |
| Grounded-claim citation coverage | 100% |
| External-call timeout test coverage | 100% |
| ML MAE improvement over strong baseline to enable learned model | ≥ 10% |

## 15. API & Contract Surface

### 15.1 Core contracts (contracts/*.schema.json, mirrored in Pydantic per service — never shared as imported Python)

TripRequest, TripPreferences, FlightOption, StayOption, FairPriceEstimate, CandidateCombination, LocalPlanRequest, POI, TravelLeg, DailyPlan, LocalItinerary, BudgetBreakdown, SourceReference, DataProvenance, DataQuality, ConstraintConflict, PartialFailure, TripPlan, ErrorEnvelope, StreamEvent. Every contract carries schema_version; every request carries session_id and trace_id; every externally sourced object carries provenance.

### 15.2 System A endpoints

```
GET  /health
POST /v1/chat
POST /v1/chat/stream        (SSE)
GET  /v1/sessions/{id}
GET  /v1/plans/{id}
```

### 15.3 System B endpoints

```
GET  /health
POST /v1/local-plan              (direct contract-test endpoint)
GET  /.well-known/agent-card.json  (or SDK-standard path)
POST /a2a/...                     (SDK-managed A2A endpoint)
```

Exact Agent Card and A2A endpoint paths are taken from whatever the pinned A2A/ADK SDK versions generate — never hard-coded from a guessed draft, confirmed in the Phase 0 spike.

### 15.4 Stream events

```
request.accepted, search.started, search.completed, local_plan.started,
local_plan.completed, budget.validated, warning, partial_failure,
plan.completed, error
```

Each event carries event_id, session_id, trace_id, sequence, timestamp, stage, and a typed payload. Only operational status and validated artifacts stream — never private chain-of-thought.

## 16. Technology Stack

| Concern | Choice |
|---|---|
| Primary orchestration | LangGraph |
| Independent specialist | Google ADK |
| Agent interoperability | Official A2A SDK |
| Shared tools | Official MCP Python SDK, Streamable HTTP |
| APIs | FastAPI + Uvicorn |
| Async HTTP | httpx |
| Validation / config | Pydantic v2 + pydantic-settings |
| Vector DB | Qdrant |
| Dense embeddings | multilingual-e5-small |
| Sparse retrieval | BM25-compatible sparse vectors + RRF |
| ML | scikit-learn |
| Session A | LangGraph SQLite checkpointer |
| Session B | ADK SQLite-backed session service |
| Frontend | Streamlit |
| Map / charts | Folium + Plotly |
| Tests | pytest + pytest-asyncio + respx |
| Retries | Tenacity or a small explicit retry wrapper |
| Local deployment | Docker Compose |

The LLM sits behind a provider interface; a fast hosted tool-capable model drives the demo. A local 2B Qwen model may be benchmarked as a latency/cost baseline but is never required for correct routing or acceptable demo speed.

## 17. Implementation Timeline & Gated Phases

One gated phase at a time; no whole-repository generation in a single pass; the two-system boundary and container count are fixed for the duration of the project.

| Phase | Deliverable | Gate to pass |
|---|---|---|
| 0 — Compatibility spike | Pinned versions; minimal ADK↔A2A call across processes; minimal MCP tool call over Streamable HTTP | Both network protocol spikes pass before any repository-wide code generation |
| 1 — Contracts & fixtures | JSON Schemas, example payloads, eval case formats, deterministic fixtures, 5 service skeletons + health checks | All containers start; schemas validate examples; contract tests pass |
| 2 — Data & accommodation ML | Dataset manifest, cleaning pipeline, baselines, learned models, held-out eval, MCP stay/price tools | Reproducible metrics; no leakage-test failure; MCP contract tests pass |
| 3 — RAG | Source manifests, ingestion, alias catalog, Qdrant indexes, dense-then-hybrid experiment, citation API | Saved raw evaluation results and chosen configuration |
| 4 — Istanbul expert | POI catalog, accessibility scoring, itinerary heuristic, hard-validator, ADK agent, A2A Agent Card + artifact | Optimizer beats or clearly characterizes itself vs. naive baseline; A2A tests pass |
| 5 — Planner | LangGraph workflow, routing, candidate search, budget engine, A2A client, sessions, guards, structured SSE | Routing suite, arithmetic tests, follow-up session test, failure injections pass |
| 6 — End-to-end & UI | Full evaluation + failure analysis first; Streamlit UI, map, chart, provenance badges, warnings; README, EVALUATION.md, diagram, backup recording | Final smoke test works offline with fixtures, optionally with live adapters |

Every phase report lists files changed, commands run, test results, measured outputs, unresolved risks, and the next gate. A broken gate is reported, never hidden or worked around silently.
