"""Production entrypoint for the public System A service (Checkpoint
Phase 4 D.2A). Wires `create_app()` to the REAL `ProductionToolExecutor`
(Checkpoint D.1) and the REAL `QwenDecisionProvider` (Checkpoint D.0) --
never a fake/scripted substitute. This module is the only place in this
checkpoint that constructs those real objects for the HTTP service; every
test in `orchestration/tests/` builds its own app via `create_app()`
with an injected fake/stub instead and never imports this module.

Run with: `python -m orchestration.system_a.entrypoint` (reads PORT,
defaults to 8010) or, in a real deployment, any ASGI server pointed at
`orchestration.system_a.entrypoint:app`.
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from orchestration.system_a.api import create_app
from orchestration.system_a.tool_executor import ProductionToolExecutor
from phase4.qwen_client import QwenDecisionProvider


def _build_production_app() -> FastAPI:
    return create_app(
        tool_executor_factory=ProductionToolExecutor,
        decision_provider_factory=QwenDecisionProvider,
    )


app = _build_production_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8010"))
    uvicorn.run(app, host="0.0.0.0", port=port)
