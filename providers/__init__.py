"""Root-owned, framework-independent live-data provider layer (Checkpoint
Phase 4 C.0; docs/adr/0009-phase4-checkpointc0-live-data-and-react-design.md).

Mirrors the precedent already established by `rag/` (docs/adr/0005-phase3-rag-ownership.md)
and `data/`/`ml/` (docs/adr/0002-phase2-checkpoint-a.md §4): a capability
that a not-yet-built consumer (System A) will later wire in lives at the
superproject root, importable and testable entirely on its own, never
physically inside a submodule.

Weather (`providers.weather`), web-search evidence (`providers.web_evidence`),
and flight search (`providers.flights`) are each independent, injectable
provider interfaces -- never implemented as tightly-coupled code inside
System A, and never owned by Travel MCP (scoped to accommodation only) or
System B (Istanbul-local knowledge only). `providers.policy` holds the
provider-neutral timeout/retry/cache/circuit-breaker policy every adapter
shares; `providers.fingerprint` computes the deterministic request
fingerprints used for caching and duplicate-call detection.

No real network call, no real credential, and no paid/quota-consuming
API call happens anywhere in this package as of Checkpoint C.0 -- every
provider here is a typed interface plus a deterministic fake
implementation for hermetic testing.

Deployment ownership (Checkpoint C.0 correction pass; ADR 0009 §6.1):
this package is, as of this checkpoint, a canonical contract/policy/
reference layer plus deterministic fakes only -- not yet a deployed
service, and not yet bound to any specific consumer's import mechanism.
A future `services/planner-a` (System A) MUST NOT import this package
via sys.path root-manipulation from inside its own submodule -- that is
the exact anti-pattern Checkpoint B identified and reversed in
`services/istanbul-expert-b/phase4/rag_client.py` (see
docs/adr/0008-phase4-checkpoint-b-rag-integration.md §2), and it must
not be reintroduced here in the opposite direction. Which explicit
deployable boundary `providers/` gets (a vendored/published package, a
dedicated adapter service, or something else) is a decision for a future
checkpoint (C.1/C.2 or equivalent), not this one.
"""

from __future__ import annotations
