"""Phase 3 multilingual RAG package (root-owned shared infrastructure).

Ownership decision (Checkpoint Phase 3, see docs/adr/0005-phase3-rag-ownership.md):
this package lives at the superproject root, NOT inside any `services/*`
submodule. System B (services/istanbul-expert-b) is the eventual consumer
per architecture.md §6.1 ("qdrant-client for grounded retrieval"), but ADK/
A2A/the Istanbul agent are explicitly out of scope for this phase. Root
ownership mirrors how `data/`/`ml/`/`contracts/` already work for the
Phase 2 accommodation pipeline: shared, framework-independent assets and
code that a not-yet-implemented service will later consume via explicit
configuration, never by import-time coupling to a submodule.

Corpus boundary (architecture.md §9.1): only stable knowledge -- history,
heritage, neighborhoods, culture, etiquette, stable transport guidance,
accessibility, stable attraction descriptions. Never current prices,
hours, availability, weather, or anything requiring live verification.
"""
