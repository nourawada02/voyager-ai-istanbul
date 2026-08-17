"""Shared hermetic test fixtures for orchestration/tests/ (Checkpoint
Phase 4 D.2A additions). Existing test files each define their own local
`ScriptedDecisionProvider`/`_decision` helper (established pattern since
Checkpoint D.1); this conftest adds shared equivalents for the new D.2A
test files only, without touching any pre-existing test file.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from phase4.context import ExecutionContext
from phase4.models import Action
from phase4.tools import FakeToolExecutor


class ScriptedDecisionProvider:
    """A `phase4.qwen_client.DecisionProvider` that returns pre-scripted
    JSON responses in order -- never a live model call."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._i = 0

    def generate(self, system: str, user: str) -> str:
        response = self.responses[self._i]
        self._i += 1
        return response


def decision(action: str, arguments: dict, reason_code: str = "all_required_evidence_present") -> str:
    return json.dumps({"action": action, "arguments": arguments, "reason_code": reason_code, "explanation": "ok"})


@dataclass
class GatedFakeToolExecutor:
    """Wraps `FakeToolExecutor`, blocking on a `threading.Event` before
    returning -- lets a test pause execution mid-tool-call (e.g. to issue
    a cancellation while a call is genuinely in flight) without any real
    network latency. `call_log` records every action actually executed,
    in order, for asserting "no further tool call happened"."""

    inner: FakeToolExecutor = field(default_factory=FakeToolExecutor)
    gate: threading.Event = field(default_factory=threading.Event)
    call_log: list[Action] = field(default_factory=list)
    block_on_first_call_only: bool = True

    def execute(self, action: Action, arguments: dict[str, Any], context: Optional[ExecutionContext] = None) -> dict[str, Any]:
        should_block = self.block_on_first_call_only and len(self.call_log) == 0
        self.call_log.append(action)
        if should_block:
            self.gate.wait(timeout=5.0)
        return self.inner.execute(action, arguments, context)


@pytest.fixture()
def tmp_db_path(tmp_path) -> str:
    return str(tmp_path / "system_a_test.db")
