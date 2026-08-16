# ADR 0005: Phase 3 — RAG ownership boundary

Status: accepted. Date: 2026-08-15.

Short design note, not a rewrite of `docs/architecture.md`. Records where
Phase 3's ingestion/retrieval/grounded-answer code and evaluation assets
live, and why.

## Decision

All Phase 3 code and assets live at the superproject root:

- `rag/` — corpus, ingestion, chunking, embeddings, Qdrant access,
  retrieval, grounded-answer service, LLM provider abstraction, ground
  truth, experiment/evaluation runners.
- `evaluation/datasets/` — the 45-question ground-truth set and retrieval
  experiment results (already scaffolded, unused until now).

Nothing is added to `services/istanbul-expert-b` or any other submodule.

## Why root, not a submodule

`docs/architecture.md` §6.1 names `qdrant-client` as part of **System B's**
stack, and §6.2/§6.3 describe System B as the eventual RAG consumer. It
would therefore be natural, long-term, for retrieval code to live inside
`services/istanbul-expert-b`. But this phase is explicitly scoped to
exclude ADK, A2A, and any Istanbul-agent behavior — building the
ingestion/retrieval/generation stack directly inside that submodule now
would either (a) sit unused and untested by the submodule's own eventual
agent code, or (b) tempt scope creep into implementing agent plumbing
just to exercise it, both undesirable.

Root ownership instead mirrors the precedent Phase 2 already set:
`data/`, `ml/`, and `contracts/` are root-owned, framework-independent
shared assets that `services/travel-mcp` (a submodule) consumes only
through explicit configuration (`AccommodationResourceConfig` paths) —
never by being physically inside the submodule. RAG follows the same
pattern: `rag/` is root-owned, importable, and testable entirely on its
own; System B will later consume it (via `qdrant-client` and explicit
config, per its own architecture) without Phase 3 having pre-decided any
ADK/A2A wiring for it.

## What this does not decide

- How `services/istanbul-expert-b` will actually import or deploy `rag/`
  once ADK/A2A work begins — deferred to that phase.
- Docker/Compose wiring for a production Qdrant service — deferred;
  Phase 3 only requires hermetic local-mode tests and one real-server
  integration test.

## Real-server integration pin

The non-skippable real-server integration gate uses Qdrant server
version 1.15.4, aligned with the canonically pinned
`qdrant-client==1.15.1`.

The server image is launched through the immutable reference:

`qdrant/qdrant@sha256:6ac4807063bbecddca0250bfbcff52acf18c22263b904d12919349e6d0a408f1`

Reproduction command:

```powershell
docker run --rm -d `
  --name voyager-phase3-qdrant `
  -p 6333:6333 `
  qdrant/qdrant@sha256:6ac4807063bbecddca0250bfbcff52acf18c22263b904d12919349e6d0a408f1

## Minimal new contracts

Two new contracts were added, both the smallest addition that closes a
real gap:

- `contracts/Chunk.schema.json` — the ingested unit indexed in Qdrant.
- `contracts/RetrievalResult.schema.json` — one ranked retrieval
  response, used by both the experiment runner and the grounded-answer
  service.

Citations reuse the existing `contracts/SourceReference.schema.json`
unchanged — no new citation contract was needed. Entity IDs reuse the
existing `contracts/POI.schema.json` and the existing 39-district
`IstanbulDistrictRegistry` unchanged — corpus documents reference these
canonical IDs directly; district truth is never duplicated.
