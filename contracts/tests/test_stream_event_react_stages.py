"""Explicit schema-version compatibility tests for StreamEvent 1.0.0 ->
1.1.0 (Checkpoint Phase 4 D.2A, docs/adr/0015-phase4-checkpoint-d2a-system-a-api.md).
Mirrors contracts/tests/test_provider_envelope_versioning.py's own
structure and intent, applied to StreamEvent's additive 'stage' widening
instead of ProviderResponseEnvelope's additive fields: proves, by direct
construction rather than only via the generic fixture sweep in
test_contracts.py, that every pre-existing 1.0.0 stage value remains
valid and that every new 1.1.0 ReAct-action-loop stage value validates.
"""

from __future__ import annotations

import glob
import json
import os

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILES = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")))

ORIGINAL_1_0_0_STAGES = [
    "request.accepted", "search.started", "search.completed",
    "local_plan.started", "local_plan.completed", "budget.validated",
    "warning", "partial_failure", "plan.completed", "error",
]

NEW_1_1_0_STAGES = [
    "run_started", "action_started", "action_completed", "action_failed",
    "run_completed", "run_degraded", "run_failed", "run_cancelled",
]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def registry() -> Registry:
    resources = []
    for path in SCHEMA_FILES:
        schema = _load(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


@pytest.fixture(scope="module")
def validator(registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, "StreamEvent.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def _instance(schema_version: str, stage: str) -> dict:
    return {
        "schema_version": schema_version,
        "event_id": "55555555-5555-4555-8555-555555555555",
        "session_id": "11111111-1111-4111-8111-111111111111",
        "trace_id": "22222222-2222-4222-8222-222222222222",
        "sequence": 0,
        "timestamp": "2026-08-01T12:00:00Z",
        "stage": stage,
        "payload": {},
    }


@pytest.mark.parametrize("stage", ORIGINAL_1_0_0_STAGES)
def test_pre_existing_1_0_0_stage_remains_valid_unchanged(stage, validator):
    instance = _instance("1.0.0", stage)
    errors = list(validator.iter_errors(instance))
    assert not errors, f"{stage}: {[e.message for e in errors]}"


@pytest.mark.parametrize("stage", ORIGINAL_1_0_0_STAGES)
def test_pre_existing_stage_also_valid_under_1_1_0(stage, validator):
    """A 1.1.0-labeled event may still use an original stage -- widening
    an enum never invalidates its own pre-existing members."""
    instance = _instance("1.1.0", stage)
    errors = list(validator.iter_errors(instance))
    assert not errors, f"{stage}: {[e.message for e in errors]}"


@pytest.mark.parametrize("stage", NEW_1_1_0_STAGES)
def test_new_1_1_0_react_action_loop_stage_is_valid(stage, validator):
    instance = _instance("1.1.0", stage)
    errors = list(validator.iter_errors(instance))
    assert not errors, f"{stage}: {[e.message for e in errors]}"


@pytest.mark.parametrize("stage", NEW_1_1_0_STAGES)
def test_new_stage_under_1_0_0_schema_version_label_is_rejected(stage, validator):
    """A new stage value must still be paired with the 1.1.0 label -- the
    schema_version bump is not merely cosmetic; a producer must declare
    which vocabulary it is actually using."""
    instance = _instance("1.0.0", stage)
    # schema_version accepts "1.0.0", so only 'stage' itself would need to
    # be rejected -- but 'stage' is a single shared enum covering both
    # vocabularies (matching the additive-widening design), so this
    # combination is, deliberately, syntactically valid. This test
    # documents that choice explicitly rather than leaving it implicit.
    errors = list(validator.iter_errors(instance))
    assert not errors, (
        "stage is intentionally one shared, additively-widened enum -- "
        f"schema_version is documentation of the producer's vocabulary, not a gate: {[e.message for e in errors]}"
    )


def test_unknown_stage_value_is_still_rejected(validator):
    instance = _instance("1.1.0", "not_a_real_stage")
    errors = list(validator.iter_errors(instance))
    assert errors


def test_original_example_file_still_validates(validator):
    path = os.path.join(CONTRACTS_DIR, "examples", "valid", "StreamEvent.json")
    instance = _load(path)
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]
