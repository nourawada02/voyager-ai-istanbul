# 0009. Phase 4 Checkpoint C.0 — live-data contracts, tool ownership, and bounded ReAct orchestration design

- **Status:** Accepted
- **Date:** 2026-08-17 (correction pass and formal acceptance same day)

## 1. Scope

This checkpoint establishes the provider-neutral contract and architecture foundation for three future capabilities System A will eventually orchestrate: live weather, web-search evidence, and flight-search information, plus the bounded ReAct-style control loop that will select among them. It is a **design and contract checkpoint**, not an implementation of System A. No LangGraph execution, no Streamlit, no root Compose, no booking/purchasing, and no production provider call are authorized or performed here. See §9 for what is deliberately not done.

## 2. Requirements-to-owner traceability

| Capability | Owner | Caller | Protocol/boundary | Data mode | Status this checkpoint | Future checkpoint |
|---|---|---|---|---|---|---|
| Trip-level orchestration | System A (`services/planner-a`) | End user / Streamlit | HTTP + SSE (architecture.md §15.2) | n/a | Not started (Phase 0/1 skeleton only) | Phase 5 |
| Bounded ReAct planning loop | System A | Internal to System A | n/a (in-process control loop) | n/a | **Designed this checkpoint (§4)**, not implemented | Phase 5 |
| Weather | `providers.weather` (new, root-owned) | System A (future), optionally System B per architecture.md §6.2 | Typed Python provider interface; real adapter TBD (§7) | live/cached/historical/estimated | **Contract + interface + fake implemented this checkpoint**; no real adapter | Phase 5 (real adapter), or a dedicated future checkpoint |
| Web-search evidence | `providers.web_evidence` (new, root-owned) | System A (future) | Typed Python provider interface; real adapter TBD (§7) | live/estimated | **Contract + interface + fake implemented this checkpoint**; no real adapter | Phase 5 |
| Flight search | `providers.flights` (new, root-owned) | System A (future) | Typed Python provider interface; real adapter TBD (§7) | live/estimated | **Contract + interface + fake implemented this checkpoint**; no real adapter | Phase 5 |
| Accommodation search + fair price | `services/travel-mcp` | System A | MCP, Streamable HTTP | historical (snapshot) | Complete and frozen (Phase 2) | — |
| Local Istanbul expertise / itinerary | `services/istanbul-expert-b` | System A | A2A | live (frozen RAG) / fixture (catalog) | Complete (Phase 4 Checkpoints A/B) | Phase 4 Checkpoint C.1+ (if any) |
| Frozen RAG retrieval | `rag/` (root-owned) → consumed read-only by System B | System B | In-process query (System B's own `phase4/knowledge/`) | historical/frozen | Complete and frozen (Phase 3 + Checkpoint B) | — |
| Citations/provenance (RAG) | `rag/` corpus + System B | System B → System A (via A2A artifact) | `SourceReference` (frozen contract) | historical | Complete | — |
| Citations/provenance (web evidence) | `providers.web_evidence` | System A | `WebEvidenceResult` (new contract, §3) | live/estimated | **Contract designed this checkpoint** | Phase 5 |
| Frontend | `services/frontend` (Streamlit) | End user | HTTP | n/a | Not started (Phase 0/1 skeleton only) | Phase 6 |
| Deployment (root Compose) | root `docker-compose.yml` | Operator | n/a | n/a | Exists but not extended this checkpoint | Phase 6 |

**Conflicts found and preserved, not silently resolved either direction:**

The canonical `docs/architecture.md` was not broadly rewritten to match this checkpoint's new code. Now that this ADR is formally Accepted, exactly the two conflicting rows named below were updated in architecture.md §8, each pointing back to this ADR by number — not a silent, undocumented rewrite, and not a rewrite of anything beyond those two rows.

1. **architecture.md §5.3 and §8** originally described weather (and flights) as MCP tools (`fetch_weather_mcp`, `get_weather`, `search_flights`) served by the *same* single Travel MCP server that also owns accommodation. This checkpoint's explicit architectural rule instead scopes Travel MCP to accommodation only, with weather/web-search/flight-search as independent typed provider modules outside MCP. **This ADR is now Accepted, and formally supersedes *only* the two specific weather/flight ownership rows of architecture.md §8's tool-inventory table** (the `get_weather` and `search_flights` rows) — not the document as a whole, not §5.3's other nodes, and not silently: the corresponding minimal edit to architecture.md §8 was made at acceptance time (see architecture.md §8's own superseded-row note, which references this ADR by number), not merely implied by this ADR's existence.
2. **Web-search evidence and the "bounded ReAct-style state loop" framing do not appear anywhere in architecture.md.** §5.3's LangGraph workflow is a mostly-linear DAG with one conditional human-in-the-loop interrupt (§5.4), not a described action-selection loop, and no web-search node exists in it at all. This checkpoint's ReAct design (§4 below) is therefore new scope layered on top of the canonical architecture, justified by this checkpoint's own explicit instructions, not derived from the PDF. This ADR proposes it as an addition to §5.3, not a replacement of it.
3. **Amadeus Self-Service was decommissioned** (portal shutdown 2026-07-17), **but Amadeus Enterprise continues to operate** as a separate offering (`https://developers.amadeus.com/enterprise`) — see the corrected §7 comparison. The earlier draft of this ADR overstated this as "Amadeus is dead"; that claim is retracted below.

## 3. Architecture and ownership decision

- **System A** owns trip-level orchestration and the bounded ReAct tool-selection loop (§4). It is the only component that decides *which* tool to call and *when*.
- **Weather, web-search evidence, and flight search are independent typed tools/providers** (`providers.weather`, `providers.web_evidence`, `providers.flights`), never implemented as tightly-coupled code inside System A. System A calls them through the `WeatherProvider`/`WebEvidenceProvider`/`FlightSearchProvider` Protocol interfaces (§6); a real adapter is swapped in without touching orchestration code.
- **Travel MCP** (`services/travel-mcp`) owns accommodation search and fair-price estimation only, per this checkpoint's explicit scoping (see conflict #1 above).
- **System B** (`services/istanbul-expert-b`) owns Istanbul-local knowledge, the frozen RAG retrieval path, and itinerary construction. It does not gain direct credentials or independently call every external provider — its only outbound calls remain the frozen, read-only Qdrant path (Checkpoint B) and, per architecture.md §6.2, MCP calls for operational status/travel-time data it already owned in the canonical design.
- **System A invokes these capabilities through explicit boundaries**: MCP for accommodation, A2A for System B, and the new typed provider interfaces for weather/web/flights — never an ad hoc HTTP call buried in orchestration logic.
- **System A may pass normalized weather, flight, accommodation, and current-web context to System B** when local scheduling requires it (e.g. a weather-substituted indoor-day itinerary) — always as already-normalized, already-validated typed data, never as raw provider credentials or unvalidated provider payloads.
- System B **never** gains direct credentials for weather/web-search/flight providers. This is structural: `providers/` is a root-owned package System B's own submodule does not import (mirroring Checkpoint B's own "System B never imports `rag` at runtime" precedent) — any weather/web/flight context reaching System B arrives only pre-normalized, inside a `LocalPlanRequest`-shaped payload System A constructs.

### Why System B remains the Istanbul specialist, not a general orchestrator

System B's deterministic itinerary engine (Checkpoint A) and RAG integration (Checkpoint B) are already complete and scoped narrowly on purpose: side-aware/ferry-aware scheduling and Istanbul-knowledge retrieval. Giving it its own ReAct loop or direct provider credentials would duplicate System A's orchestration responsibility, create two independent places that could call the same paid API, and blur which system owns a given failure — exactly the "nested autonomous agents" anti-pattern this checkpoint's instructions explicitly reject (§4.6).

## 4. Bounded ReAct design

### 4.1 State machine

```
        ┌─────────────┐
        │   ASSESS    │◄────────────────────────────┐
        │ (trip state,│                              │
        │  missing    │                              │
        │  info)      │                              │
        └──────┬──────┘                              │
               │ missing info remains                │
               ▼                                      │
        ┌─────────────┐                               │
        │   SELECT    │  no justified tool            │
        │ (one tool,  │─────────────────┐             │
        │  justified) │                 │             │
        └──────┬──────┘                 │             │
               │                        ▼             │
               ▼                 ┌─────────────┐      │
        ┌─────────────┐          │  DEGRADE    │      │
        │  VALIDATE   │  invalid │ (explicit,  │      │
        │ (tool args) │─────────►│  labeled)   │      │
        └──────┬──────┘  args    └──────┬──────┘      │
               │ valid                  │              │
               ▼                        │              │
        ┌─────────────┐                 │              │
        │  EXECUTE    │                 │              │
        │ (bounded    │                 │              │
        │  call)      │                 │              │
        └──────┬──────┘                 │              │
               │                        │              │
               ▼                        │              │
        ┌─────────────┐  failure/       │              │
        │  OBSERVE    │  bound reached  │              │
        │ (validate   │─────────────────┘              │
        │  result)    │                                │
        └──────┬──────┘                                │
               │ success                                │
               ▼                                        │
        ┌─────────────┐                                 │
        │   UPDATE    │  info still missing,             │
        │ (state +    │  bounds not reached ─────────────┘
        │  evidence   │
        │  registry)  │
        └──────┬──────┘
               │ completion criteria met
               ▼
        ┌─────────────┐
        │  SYNTHESIZE │
        └─────────────┘
```

This is the 9-step loop from this checkpoint's instructions, rendered as states: **Assess → Select → Validate → Execute → Observe → Update → (repeat while required info is missing) → Synthesize**, with an explicit **Degrade** exit reachable from Select (no justified tool), Validate (invalid arguments after repair attempts exhausted), or Observe (tool unavailable / a bound reached).

### 4.2 Structured decision trace (no raw chain-of-thought)

Every ReAct step emits exactly the fields this checkpoint's instructions enumerate — nothing else. This mirrors the precedent already set by `contracts/StreamEvent.schema.json` ("Only operational status and validated artifacts stream — never private chain-of-thought", architecture.md §15.4) and `contracts/PartialFailure.schema.json`'s honest-degradation record:

| Field | Example |
|---|---|
| step_number | 3 |
| decision_summary | "weather missing for trip dates; selecting weather tool" (short, operational — never the model's full reasoning text) |
| selected_tool | "weather" |
| validated_arguments_fingerprint | the sha256 hex from `providers.fingerprint.fingerprint_request` — the *fingerprint*, not necessarily the raw arguments, when they might contain anything sensitive |
| result_status | one of `providers.policy.RESULT_STATUSES` |
| evidence_ids | list of `request_id`s / `SourceReference.chunk_ids` this step contributed to the evidence registry |
| retry_count | 0..max_retries |
| degradation_reason | null, or a `PartialFailure.component`-style short code |
| next_state | "UPDATE" / "DEGRADE" / "SYNTHESIZE" |

A future implementation should shape this trace as its own small contract (e.g. `contracts/ReActStep.schema.json`) when System A is actually built (Phase 5) — not created in this checkpoint, since no consumer exists yet and an unused contract with no real producer/consumer is exactly the premature-abstraction pattern this project avoids elsewhere (see docs/adr/0005 §"Why root, not a submodule" for the same reasoning applied to RAG).

### 4.3 Bounds

**Correction:** the prior draft conflated two different things under one "maximum total iterations = 25" bound — the LangGraph *graph-transition* ceiling (how many supersteps the whole System A graph may execute, including non-ReAct nodes like input validation and output formatting) and an *external tool-call* budget (how many times this ReAct loop specifically may call out to weather/web-search/flight-search). 25 is not, and was never intended to be, an allowance for 25 provider calls — that reading is corrected below with its own, much smaller, dedicated bound.

| Bound | Default | Rationale |
|---|---|---|
| Maximum graph transitions (whole System A graph, all node types) | 25 | Matches the existing `recursion_limit = 25` already set for System A's LangGraph graph (architecture.md §13.3). This ceiling covers every superstep of the entire graph — input guard, session load, intent classification, ReAct steps, budget repair, output formatting — **not** a per-tool-call budget. |
| **Maximum external tool calls (this ReAct loop specifically, per workflow)** | **8** | A new, smaller, dedicated bound this checkpoint introduces: roughly 2–3 attempts (first call + `providers.policy.RetryPolicy.max_retries=2` retries) across up to 3 distinct capabilities (weather, web-search, flights) in a typical trip, with headroom — never anywhere near the 25-transition graph ceiling. Chosen to keep worst-case external latency and cost bounded independently of how large the surrounding graph grows. |
| Maximum repair iterations (invalid tool arguments) | 2 | Matches the existing project-wide repair-iteration convention (itinerary hard-violation repair, architecture.md §13.3). Counted against the external-tool-call budget above when a repair itself re-executes a call; argument-only repair (no re-call) is not. |
| Maximum calls per tool/provider (per workflow) | 2 | Matches `providers.policy.RetryPolicy.max_retries` default — no single tool is called more than the same bound already governs retries. Always ≤ the total external-tool-call budget above; three tools each hitting this per-tool ceiling once (3 calls) is the common case, not the worst case (up to 8 is reserved for retries). |
| Per-call timeout | 10s total / 3s connect | `providers.policy.TimeoutPolicy` defaults. |
| Total workflow deadline | 60s | New bound this checkpoint introduces (not previously named in architecture.md), sized to comfortably fit the 8-call external-tool-call budget × a bounded per-call timeout with room for retries — not the 25-transition graph ceiling, which is a step count, not a time bound. A future implementation should tune this against real measured latency, not this checkpoint's estimate. |
| Duplicate-call detection | via `providers.fingerprint.fingerprint_request` | A step whose selected tool + fingerprint exactly matches an already-recorded evidence-registry entry is never re-executed — the existing result is reused and the step is recorded as a no-op UPDATE, never counted against the per-tool call bound a second time. |
| Explicit completion condition | all required fields for the current trip-state category are present in the evidence registry with `status in ("success", "partial", "stale")` | — |
| Explicit degraded completion | any required field remains missing after bounds are reached, or `status in FAILURE_STATUSES` for every attempt on a required tool | Synthesis proceeds anyway, with the missing/degraded fields explicitly disclosed — never silently omitted, mirroring `contracts/PartialFailure.schema.json`'s existing "a usable partial plan is always produced" principle. |
| Cancellation | checked before every SELECT transition | A cancelled workflow degrades through the same DEGRADE state with `degradation_reason="cancelled"` and `result_status="cancelled"` on any in-flight call, rather than leaving state inconsistent. |

### 4.4 Duplicate-call prevention and evidence aggregation

The evidence registry is keyed by `query_fingerprint` (per `ProviderResponseEnvelope.schema.json` 1.1.0). Before SELECT chooses a tool, ASSESS checks the registry for an existing entry with a matching fingerprint; if found and not stale (`providers.policy.cache_status_for`), that step is skipped entirely — this is the same mechanism that will back real caching later, reused directly for duplicate-call prevention rather than inventing a second, parallel bookkeeping structure.

### 4.5 Cancellation and degraded completion

Cancellation is checked as a precondition of SELECT (never mid-EXECUTE, since an in-flight call already has its own bounded timeout as the backstop). A cancelled workflow synthesizes a degraded result using whatever evidence the registry already holds, exactly the same code path degraded-but-bound-reached completion uses — cancellation is not a special case requiring separate synthesis logic.

### 4.6 Why no ReAct loop inside System B or individual provider adapters

Nesting an autonomous loop inside System B or inside a provider adapter would let two independent loops decide to call the same paid API (duplicate calls, unpredictable cost), would make it unclear which loop's bound actually governs a given call, and would violate the single-decision-owner principle §3 establishes. System B's own internal workflow (architecture.md §6.3) is already a fixed, deterministic 10-step pipeline, not a loop — it stays that way. Provider adapters are pure functions from typed request to typed response; the retry/backoff/circuit-breaker behavior in `providers.policy` is applied by the *caller* (eventually System A), never self-invoked by the adapter.

## 5. Provenance and evidence design

Every live-data result carries, via `ProviderResponseEnvelope` (1.1.0) + its capability-specific result schema:

- `schema_version`, `provider`, `capability`, `request_id`, `query_fingerprint`
- `retrieved_at` (UTC), `valid_for`, `cache_status`/`cache_age_seconds` when cached
- `data_mode` (kind of data) and `status` (operational outcome) — kept orthogonal on purpose (§3 of the schema's own description)
- `source_urls` and `quality.assumptions` (explicit limitations)
- capability-specific provenance: `WeatherResult.issued_at`, `WebEvidenceResult`'s per-item `provenance` (`DataProvenance`, reused unchanged), `FlightSearchResult.options[].provenance`

Web evidence is never silently converted into a frozen RAG `SourceReference` — the two schemas are structurally incompatible by construction (see `providers/tests/test_web_evidence_provider.py::test_absent_evidence_never_silently_becomes_a_frozen_rag_citation`), so a caller cannot accidentally pass one where the other is expected without both a type error and a schema-validation failure.

## 6. Provider-interface design

`providers.weather.WeatherProvider`, `providers.web_evidence.WebEvidenceProvider`, and `providers.flights.FlightSearchProvider` are `typing.Protocol` interfaces — structural typing, no inheritance required, so a real adapter and `FakeWeatherProvider`/`FakeWebEvidenceProvider`/`FakeFlightSearchProvider` are interchangeable without a shared base class. Every method returns a plain `dict` shaped exactly like `ProviderResponseEnvelope` + the capability's result schema — never a vendor-specific response object (an `httpx.Response`, a vendor SDK model) leaking into shared contracts. This mirrors `phase4/knowledge/qdrant_client.py`'s own "never leak vendor-specific response objects" precedent from Checkpoint B.

### 6.1 Deployment ownership — clarified, not decided here

Root `providers/` is, **as of this checkpoint, a canonical contract/policy/reference layer plus deterministic fakes only** — not a deployed service, and not yet bound to any specific consumer's import mechanism. This is deliberately unresolved for one reason: **`services/planner-a` (the future System A) must never import it the way Checkpoint A's original `phase4/rag_client.py` imported the root `rag/` package** — via `sys.path.insert()` root-path manipulation from inside a submodule. That anti-pattern was identified and explicitly reversed in Checkpoint B (`docs/adr/0008-phase4-checkpoint-b-rag-integration.md` §2) precisely because it breaks a submodule's independent buildability and blurs ownership; `providers/` must not reintroduce it from the other direction.

**What C.1/C.2 (or the equivalent future checkpoint) must decide, and this correction pass does not:** whether `providers/` becomes (a) a published/vendored Python package `planner-a` depends on via its own `requirements.lock` (mirroring how `services/istanbul-expert-b` now has its own canonical lock per ADR 0008 §4, rather than reaching across the repository boundary at runtime), (b) a dedicated small HTTP/MCP-fronted adapter service with its own container, independently deployable like Travel MCP, or (c) some other explicit boundary. **No such service is created in this correction pass** — `providers/` remains exactly what it was: importable and testable from the superproject root, with no submodule depending on it yet.

## 7. Provider-selection analysis (corrected)

Researched via official primary documentation where fetchable, corroborated by current search results where the official page could not be fetched directly (403/empty-render responses, noted per row below); re-verified 2026-08-17 after the correction pass. No account created, no key requested, no live call made.

### Weather

| Provider | Free tier | Auth | Coverage | Freshness distinction | Official source |
|---|---|---|---|---|---|
| **Open-Meteo** (architecture.md's own pre-named "preferred live" choice) | No auth required for non-commercial use | None (commercial requires a `customer-` prefixed key) | Global, multi-model, regional high-resolution where available | Explicitly separates current conditions (15-min model data), forecast (up to 16 days), and historical (separate endpoint) | `https://open-meteo.com/en/docs` |

ISO8601 timestamps, per-model provenance (DWD, NOAA, Météo-France, etc.). **Recommended — confirms the architecturally pre-named choice remains sound.**

### Web search / evidence

| Provider | Free tier | Card required | Official source |
|---|---|---|---|
| **Tavily** | 1,000 credits/month | **No** — confirmed via official docs: "No credit card required" | `https://docs.tavily.com/documentation/api-credits` |
| Brave Search API | **$5/month in free credits, auto-applied to all accounts** (≈1,000 search requests/month at $5 per 1,000 requests) | **Yes** — a card is required for identity/anti-fraud verification even on the free allowance, though it is not charged while usage stays within the credit | `https://brave.com/search/api/` |
| DuckDuckGo Instant Answer API | Free, no key, no formally published limit (community-observed throttling reported near 30 req/min) | No | `https://duckduckgo.com/api` (secondary sources only — no fetchable official rate-limit page found; treat this row as unverified until checked again at implementation time) |
| SerpApi (Google Search-adjacent) | 250 searches/month (official pricing page) | Not stated on the pricing page itself | `https://serpapi.com/pricing` |

**Correction:** the prior draft incorrectly stated Brave's free tier had been "removed"; it has not — Brave provides an ongoing $5/month credit, the card requirement is for verification rather than an outright paywall. **Tavily remains the recommended primary candidate** (no card, largest genuinely free monthly allowance, and its structured extracted-content response shape is the best fit for `WebEvidenceResult`'s snippet-based schema); **SerpApi and DuckDuckGo Instant Answer remain in the comparison** as a paid-adjacent alternative and a zero-cost but narrow fallback, respectively, and Brave is retained as a real, viable, currently-priced option rather than an excluded one.

### Flight search

| Provider | Status as of 2026-08-17 | Official source |
|---|---|---|
| Amadeus for Developers **Self-Service** | Portal decommissioned 2026-07-17 — registration paused since spring 2026, existing test-environment keys no longer work | Decommission timeline confirmed via trade-press reporting on Amadeus's own user letter (`https://www.phocuswire.com/amadeus-shut-down-self-service-apis-portal-developers`); the official Amadeus announcement page itself could not be fetched directly (403/empty render) — this row should be re-confirmed against an official Amadeus source before any implementation decision. |
| **Amadeus Enterprise** | **Still operating**, unaffected by the Self-Service shutdown | `https://developers.amadeus.com/enterprise` (fetchable, confirms the offering exists). Access requires a request-based approval process and, per third-party integration guides, likely IATA/ARC-adjacent accreditation and a multi-week negotiation — not a quick-start option for an academic project, but a real, existing offering, not a dead one. **Correction: the prior draft's "Amadeus is dead" framing is retracted** — Self-Service is gone, Amadeus as a company/API provider is not. |
| Duffel | **Test mode** (`duffel_test_`-prefixed token) provides free, unlimited sandbox access to a synthetic "Duffel Airways" test airline — "you won't see realistic flight schedules or prices" per Duffel's own docs. Useful for integration testing the request/response *shape*, but explicitly **does not prove real production/live-inventory access or pricing** — that requires a live token and account, whose cost/approval terms were not found in the fetched test-mode documentation and must be verified separately. | `https://duffel.com/docs/api/overview/test-mode/duffel-airways` |
| Kiwi.com Tequila API | Invitation-only for new partners as of 2026 | Secondary sources only (no official self-service registration page found); excluded pending a viable open registration path. |
| SerpApi (Google Flights API) | 100–250 free searches/month depending on which SerpApi product page is consulted; no official Google Flights API exists, this is third-party structured access to Google Flights search-snapshot data | `https://serpapi.com/pricing`, `https://serpapi.com/google-flights-api` |

**No single flight provider is recommended as "the only realistic candidate."** Duffel's test mode is the lowest-friction path to *integration-test* the flight-search shape end-to-end without any cost, but its sandbox data must never be mistaken for proof that real production offers are freely available. SerpApi's Google Flights API remains a genuine free-tier candidate for real (if third-party-sourced) snapshot data. Amadeus Enterprise remains the most authoritative source but the least accessible for a fast academic-project timeline. **Which one (if any) becomes the real adapter is deliberately left open for a future checkpoint**, not decided here.

**To be verified later, not guessed here:** exact current Tavily/SerpApi contractual terms for academic/non-commercial use; exact current per-minute rate limits for Tavily, SerpApi, and DuckDuckGo Instant Answer; whether Open-Meteo's "Non-Commercial" license terms (linked but not fully inlined on the docs page fetched) impose any attribution requirement; Duffel's live-mode account/cost/approval requirements (not present in the fetched test-mode documentation); and Amadeus Enterprise's exact accreditation/negotiation requirements for a project of this scale.

No vendor is hard-coded into any shared contract in this checkpoint — `ProviderResponseEnvelope.provider` and `capability` are open strings precisely so a future adapter swap never requires a schema change.

## 8. Failure, retry, cache, and circuit-breaker policy

Implemented in `providers/policy.py` (pure, deterministic, no real clock/network — see `providers/tests/test_policy.py`):

- `RetryPolicy`: `max_retries=2` (architecture.md §13.3's existing project-wide default), bounded exponential backoff (`compute_backoff_seconds`), `Retry-After` honored and still capped.
- `classify_http_status`: total function mapping any HTTP status to the shared `status` vocabulary (429→`rate_limited`, 408/504→`timeout`, 5xx→`provider_error`, etc.).
- `CacheEntry`/`cache_status_for`: age-aware, produces exactly the `(cache_status, cache_age_seconds)` pair `ProviderResponseEnvelope` 1.1.0 expects; a stale entry is reported as `stale`, never silently as `live`.
- `CircuitBreaker`: closed/open/half-open state machine, explicit `now` parameter throughout.
- `providers/redaction.py`: `redact_secrets()` for log lines, `SecretString` for in-memory credential isolation (never reveals via `str()`/`repr()`).

Fallback honesty rules (enforced by the schemas + tests in §5/§6, not merely documented): unavailable weather is never replaced by RAG and called live weather (structurally impossible — they are different schemas with different `capability` values); unavailable flights are never invented from web snippets (`FlightSearchResult` has no field a web snippet could populate); web evidence remains labeled `capability="web_search"` even when it is the only thing available; a stale cache entry is always reported with its age; provider failure always produces a non-`success` `status`, never an empty `result` with no signal.

## 9. Future LLM choice for System A (recorded as future scope only — not authorized in this checkpoint)

**Qwen API is the proposed primary LLM for System A's bounded ReAct action selection (SELECT) and final synthesis (SYNTHESIZE)** once System A is actually built (Phase 5). This is a scope record for that future work, not an integration decision executed here.

- Qwen is proposed **only** for the two LLM-shaped steps of the ReAct loop itself — deciding which tool to call next, and composing the final natural-language synthesis from already-validated structured evidence. **It must never replace any of the deterministic/typed capabilities this checkpoint defines**: weather, web-search evidence, and flight search remain typed provider calls (§6); accommodation remains Travel MCP's deterministic serving pipeline; RAG retrieval remains System B's frozen, deterministic Checkpoint B path. An LLM never fabricates a price, a schedule, a forecast, or a citation in place of calling the real typed tool.
- **Qwen integration is not authorized in Checkpoint C.0.** No Qwen call, key, or SDK dependency is introduced by this checkpoint or this correction pass — `providers/` remains fake-only, as proven by the entire `providers/tests/` suite (every test asserts no real socket is opened).
- Credentials for Qwen (and any future LLM provider) must remain **environment-only** — read from an environment variable at call time, never hardcoded, never committed, matching this project's existing "no hardcoded secrets anywhere" rule (architecture.md §13.5) and `providers.redaction.SecretString`'s existing isolation pattern (§8), which a future Qwen client should reuse rather than reinvent.
- **A different model/provider than Qwen should be used for LLM-as-judge evaluation**, to reduce correlated bias between the system generating an action/synthesis and the system judging it. This is not a new idea for this project — it is the exact separation already established and proven in Phase 3 (`docs/adr/0006-phase3-generation-acceptance.md`: `generator_provider: qwen` paired with `judge_provider: groq`, a different vendor entirely). A future System A evaluation harness should follow that same precedent rather than re-deriving it.

## 10. Explicitly not done in this checkpoint (and this correction pass)

Real weather/web-search/flight provider adapter calls; System A implementation; LangGraph execution; Qwen or any other LLM integration; Streamlit; root Compose changes; booking or payment actions; any account creation, API key request, or billing change; any change to `rag/**`, Phase 3's frozen retrieval winner, `data/**`, `ml/**`, or `services/travel-mcp/**`; creation of a deployable `providers/` adapter service (§6.1).
