# 0008. Phase 4 Checkpoint B — operational System B / Qdrant RAG integration

- **Status:** Accepted
- **Date:** 2026-08-17

## 1. Scope

Checkpoint B connects System B's independent container operationally to
the frozen Phase 3 Qdrant collection (`istanbul_rag_B`), so citations
attached to a `LocalItinerary` can come from real retrieval rather than
being permanently empty. It does not add web search, flights, weather,
System A, the frontend, root Docker Compose, new RAG experiments, hybrid
retrieval, reranking, or the Phase 3 judge-context repair.

## 2. Ownership boundary

**The Phase 3 root pipeline (`rag/ingest.py` + `rag/qdrant_store.py`) owns
writing the `istanbul_rag_B` collection.** It runs entirely outside
System B, ahead of time, exactly as it did for Phase 3's own experiment
and generation-eval reports.

**System B owns read-only query-time access only**, implemented in
`services/istanbul-expert-b/phase4/knowledge/`:

- `encoder.py` — reproduces the frozen query-embedding contract
  (`intfloat/multilingual-e5-small` @ `614241f622f53c4eeff9890bdc4f31cfecc418b3`,
  `"query: "` prefix, `normalize_embeddings=True`, dim 384) independently,
  in System B's own process.
- `qdrant_client.py` — a read-only `qdrant-client` wrapper that only ever
  calls `get_collections`, `get_collection`, `retrieve`, and
  `query_points`. It never calls `create_collection`, `upsert`, or any
  delete method.
- `citation_registry.py` — a small, static, frozen copy of the
  bibliographic fields (`title`/`url`/`retrieved_at`/`language`/`license`)
  from `rag/manifests/corpus_manifest.json` for the 20 frozen source
  documents. Qdrant's own stored chunk payload deliberately never carries
  `title`/`url`/`retrieved_at` (see `rag/ingest.py`'s `chunk_payloads`
  shape); rather than importing the root `rag` package at runtime to get
  them, or leaving citations without a title/URL, this module bundles
  that provenance table once. It is not corpus text, not chunking, not
  embeddings, and not evaluation logic — a one-time copy of 20
  bibliographic records, explicitly documented as needing a manual
  refresh only if the frozen corpus itself ever changed (it must not).

**System B never imports the root `rag` package at runtime.** This is a
deliberate change from Checkpoint A's `phase4/rag_client.py`, which
(when the monorepo `rag` package happened to be importable) called
`rag.run_generation_eval.build_retrieval_context()` — a function that
itself **ingests the frozen corpus into an in-memory Qdrant collection**.
That was correct for a first vertical slice but is exactly the pattern
this checkpoint replaces: System B must never perform ingestion, and the
independent container must never depend on the monorepo `rag/` package
being on its filesystem at all. `phase4/rag_client.py` now only
constructs `phase4.knowledge`'s encoder and read-only Qdrant client and
orchestrates them behind the same `get_citation_provider()`/
`rag_is_available()` interface `phase4.service_core` and
`phase4.a2a_server` already depended on.

## 3. Frozen query contract reproduced

| Item | Value |
|---|---|
| Collection | `istanbul_rag_B` |
| Vector name | `dense` |
| Vector dim | 384 |
| Distance | COSINE |
| Expected fingerprint | `34830f8e42e681ce7fdedc5a36fbd33675ae4800505a06768bc783d6057d7b84` |
| Embedding model | `intfloat/multilingual-e5-small` @ `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| Query prefix | `"query: "` |
| Top-K | 5 (frozen dense config B) |

## 4. Dependency layout

ADR 0002 §8 already decided (for `travel-mcp`, not yet applied there) the
canonical shape: a service-root `requirements.in`/`requirements-test.in`/
`requirements.lock` replacing a phase-scoped lock as what the Dockerfile
installs, with the phase-scoped file kept only as historical scope
documentation. This checkpoint applies that same pattern to
`services/istanbul-expert-b` for the first time: `requirements.in`
(canonical runtime source) and `requirements.lock` (fully pinned,
including the new `qdrant-client==1.15.1`, `sentence-transformers==5.1.2`,
`transformers==4.57.6`, `torch==2.13.0` — the exact versions
`rag/requirements.in` already pins, so query vectors are byte-identical
to what Phase 3 measured) now live at the service root. The Dockerfile
installs `requirements.lock` instead of `phase0/requirements.lock`, which
is left untouched as historical/scope documentation for what Phase 0
itself needed, per ADR 0002 §8's own plan.

## 5. Degradation policy

`RAG_REQUIRED=false` (default): the core service stays ready
(`service_ready=true`) even when Qdrant/the encoder is unavailable or
incompatible; `/health` reports `rag_operational=false` and a
`degraded_reason` code, and the itinerary engine schedules from the
structured POI catalog only, with an explicit warning
(`rag_unavailable_structured_catalog_only` or
`rag_no_relevant_evidence_retrieved`) and `data_quality.completeness`
dropped to 0.6 — never a fabricated descriptive claim.

`RAG_REQUIRED=true`: the same failure instead makes `service_ready=false`
(the Docker `HEALTHCHECK` already keys off exactly that field), so the
degradation is visible and fails closed rather than silently serving
catalog-only responses under an operator's assumption that RAG is live.

## 6. What this does not decide

- Reranking, hybrid retrieval, or any new retrieval-configuration
  experiment — the frozen dense config B winner is unconditionally
  reused, never re-evaluated.
- Root Docker Compose wiring for a production Qdrant deployment.
- Any change to `rag/`'s own ingestion, chunking, embeddings, or
  evaluation code — all read-only from System B's perspective.
