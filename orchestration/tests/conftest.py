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
from phase4.graph import CAPABILITY_SCOPE_PROMPT_MARKER
from phase4.models import Action
from phase4.tools import FakeToolExecutor

_REASON_CODE_BY_SCOPE = {
    "combined": "requires_both",
    "travel_only": "requires_travel_evidence",
    "istanbul_local_only": "requires_istanbul_local_grounding",
    "clarification_required": "insufficient_information",
    "out_of_scope": "outside_project_scope",
}


def _infer_capability_scope(remaining_responses: list) -> str:
    """Checkpoint Final Evaluation E.1S.1: infers a
    `phase4.models.CapabilityScope` value from the REST of an already-
    authored scripted decision plan, so every existing hermetic test in
    this suite keeps its ground-truth action sequence completely
    unchanged while still exercising the real once-per-turn
    classification step the supervisor's Decide node now performs. Never
    used in production (a real classification is a genuine Qwen call) --
    this reads the same ground truth the test author already encoded in
    `responses`, rather than guessing from free text."""
    actions = []
    for raw in remaining_responses:
        if isinstance(raw, BaseException):
            continue
        try:
            actions.append(json.loads(raw).get("action"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    has_travel = "call_travel_search" in actions
    has_istanbul = "call_istanbul_expert" in actions
    if has_travel and has_istanbul:
        return "combined"
    if has_travel:
        return "travel_only"
    if has_istanbul:
        return "istanbul_local_only"
    if "ask_clarification" in actions:
        return "clarification_required"
    return "out_of_scope"


class ScriptedDecisionProvider:
    """A `phase4.qwen_client.DecisionProvider` that returns pre-scripted
    JSON responses in order -- never a live model call. Transparently
    answers the supervisor's once-per-turn capability-scope
    classification call (Checkpoint Final Evaluation E.1S.1) by peeking
    ahead at its own still-queued responses, WITHOUT consuming one --
    existing scripted action plans need no changes."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self._i = 0

    def generate(self, system: str, user: str) -> str:
        if CAPABILITY_SCOPE_PROMPT_MARKER in system:
            scope = _infer_capability_scope(self.responses[self._i:])
            return json.dumps({"scope": scope, "reason_code": _REASON_CODE_BY_SCOPE[scope]})
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
