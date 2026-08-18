"""Production entrypoint for the public System A service (Checkpoint
Phase 4 D.2A, extended by D.2B with explicit deterministic demo/fixture
mode). Wires `create_app()` to one of two capability pairs, chosen
**only** by `VOYAGER_SYSTEM_A_MODE` (`orchestration.system_a.config`):

- `real` (the default -- absence of the variable means this): the REAL
  `ProductionToolExecutor` (Checkpoint D.1) and the REAL
  `QwenDecisionProvider` (Checkpoint D.0).
- `fixture` (requires the explicit value `VOYAGER_SYSTEM_A_MODE=fixture`):
  `phase4.tools.FakeToolExecutor` (already-existing, production-owned
  deterministic fixture executor) and the new, deterministic, rule-based
  `FixtureDecisionProvider` (Checkpoint D.2B) -- no network call, no paid
  quota, ever.

There is no third path and no silent fallback in either direction: an
unrecognized `VOYAGER_SYSTEM_A_MODE` value raises
`SystemAModeConfigurationError` before the app is even built (see
`config.system_a_mode`), never quietly defaulting to fixture data
labeled as real, and never quietly defaulting to a real network call
when the caller asked for fixture mode.

This module is the only place in this checkpoint that constructs the
real (or fixture) capability objects for the HTTP service; every test in
`orchestration/tests/` builds its own app via `create_app()` with an
injected fake/stub instead and never imports this module.

Run with: `python -m orchestration.system_a.entrypoint` (reads PORT,
defaults to 8010) or, in a real deployment, any ASGI server pointed at
`orchestration.system_a.entrypoint:app`.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from orchestration.system_a import config
from orchestration.system_a.api import create_app
from orchestration.system_a.fixture_decision_provider import (
    SpecialistFixtureDecisionProvider,
    SupervisorFixtureDecisionProvider,
)
from orchestration.system_a.tool_executor import ProductionToolExecutor
from phase4.qwen_client import QwenDecisionProvider
from phase4.tools import FakeToolExecutor


def _build_production_app() -> FastAPI:
    mode = config.system_a_mode()  # raises SystemAModeConfigurationError on an unrecognized value
    if mode == "fixture":
        tool_executor_factory = FakeToolExecutor
        decision_provider_factory = SupervisorFixtureDecisionProvider
        specialist_decision_provider_factory = SpecialistFixtureDecisionProvider
    else:
        tool_executor_factory = ProductionToolExecutor
        # Two separate instances -- the internal Travel Search
        # specialist gets its own `DecisionProvider` object, never the
        # supervisor's own (ADR 0017 §3). `QwenDecisionProvider` is a
        # frozen, stateless-per-call dataclass, so the two behave
        # identically; the separate construction is what makes "role" a
        # constructor-time fact rather than something inferred later.
        decision_provider_factory = QwenDecisionProvider
        specialist_decision_provider_factory = QwenDecisionProvider
    return create_app(
        tool_executor_factory=tool_executor_factory,
        decision_provider_factory=decision_provider_factory,
        specialist_decision_provider_factory=specialist_decision_provider_factory,
        mode=mode,
    )


app = _build_production_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
