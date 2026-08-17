"""Explicit schema-version compatibility tests for ProviderResponseEnvelope
1.0.0 -> 1.1.0 (Checkpoint Phase 4 C.0 correction pass, item 9). Proves,
by direct construction rather than only via the generic fixture sweep in
test_contracts.py, that:

  1. every pre-existing 1.0.0 fixture (predating this checkpoint) remains
     valid unchanged;
  2. a hand-built minimal 1.0.0 instance with none of the new fields is
     valid;
  3. the 1.1.0 conditional requirement (cache_status in {hit, stale}
     requires cache_age_seconds) is actually enforced, not just declared;
  4. the new 'unavailable' data_mode/freshness enum value validates.
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
FIXTURES_DIR = os.path.join(CONTRACTS_DIR, "fixtures")

# Fixtures that predate Checkpoint Phase 4 C.0 (schema_version "1.0.0",
# no 'capability'/'status'/'cache_status' fields) -- these must remain
# valid exactly as they were before ProviderResponseEnvelope grew its
# 1.1.0 fields.
PRE_EXISTING_1_0_0_FIXTURES = [
    "flights_success.json",
    "flights_provider_unavailable.json",
    "fx_rates_success.json",
    "provider_status_success.json",
    "routes_success.json",
    "routes_estimated_fallback.json",
    "stays_success.json",
    "stays_partial_result.json",
    "weather_success.json",
    "weather_unavailable.json",
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
def envelope_validator(registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, "ProviderResponseEnvelope.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


@pytest.mark.parametrize("filename", PRE_EXISTING_1_0_0_FIXTURES)
def test_pre_existing_1_0_0_fixture_remains_valid_unchanged(filename, envelope_validator):
    instance = _load(os.path.join(FIXTURES_DIR, filename))
    assert instance["schema_version"] == "1.0.0"
    assert "capability" not in instance
    assert "status" not in instance
    errors = list(envelope_validator.iter_errors(instance))
    assert not errors, f"{filename}: {[e.message for e in errors]}"


def test_minimal_hand_built_1_0_0_instance_with_no_new_fields_is_valid(envelope_validator):
    instance = {
        "schema_version": "1.0.0",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "provider": "some-provider",
        "data_mode": "live",
        "retrieved_at": "2026-08-01T12:00:00Z",
        "source_urls": [],
        "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "live", "assumptions": []},
        "result": {},
    }
    errors = list(envelope_validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_1_1_0_instance_with_every_new_field_set_is_valid(envelope_validator):
    instance = {
        "schema_version": "1.1.0",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "provider": "some-provider",
        "capability": "weather",
        "data_mode": "cached",
        "status": "success",
        "query_fingerprint": "a" * 64,
        "cache_status": "hit",
        "cache_age_seconds": 42,
        "retrieved_at": "2026-08-01T12:00:00Z",
        "source_urls": [],
        "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "cached", "assumptions": []},
        "result": {},
    }
    errors = list(envelope_validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_cache_status_hit_without_cache_age_seconds_is_rejected(envelope_validator):
    """The 1.1.0 conditional requirement must actually be enforced, not
    just declared in the schema's prose."""
    instance = {
        "schema_version": "1.1.0",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "provider": "some-provider",
        "data_mode": "cached",
        "cache_status": "hit",
        "retrieved_at": "2026-08-01T12:00:00Z",
        "source_urls": [],
        "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "cached", "assumptions": []},
        "result": {},
    }
    errors = list(envelope_validator.iter_errors(instance))
    assert errors


def test_cache_status_stale_without_cache_age_seconds_is_also_rejected(envelope_validator):
    instance = {
        "schema_version": "1.1.0",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "provider": "some-provider",
        "data_mode": "cached",
        "cache_status": "stale",
        "retrieved_at": "2026-08-01T12:00:00Z",
        "source_urls": [],
        "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "cached", "assumptions": []},
        "result": {},
    }
    errors = list(envelope_validator.iter_errors(instance))
    assert errors


def test_cache_status_miss_never_requires_cache_age_seconds(envelope_validator):
    instance = {
        "schema_version": "1.1.0",
        "request_id": "11111111-1111-4111-8111-111111111111",
        "provider": "some-provider",
        "data_mode": "live",
        "cache_status": "miss",
        "retrieved_at": "2026-08-01T12:00:00Z",
        "source_urls": [],
        "quality": {"schema_version": "1.0.0", "completeness": 1.0, "freshness": "live", "assumptions": []},
        "result": {},
    }
    errors = list(envelope_validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


@pytest.mark.parametrize("field_schema_name", ["ProviderResponseEnvelope", "DataProvenance", "DataQuality"])
def test_unavailable_is_a_valid_data_mode_freshness_value(field_schema_name, registry):
    """Checkpoint Phase 4 C.0 correction: 'unavailable' is a purely
    additive new enum value on all three schemas that carry a
    data_mode/freshness-shaped field."""
    schema = _load(os.path.join(CONTRACTS_DIR, f"{field_schema_name}.schema.json"))
    validator = Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
    if field_schema_name == "ProviderResponseEnvelope":
        instance = {
            "schema_version": "1.1.0",
            "request_id": "11111111-1111-4111-8111-111111111111",
            "provider": "some-provider",
            "data_mode": "unavailable",
            "status": "timeout",
            "retrieved_at": "2026-08-01T12:00:00Z",
            "source_urls": [],
            "quality": {"schema_version": "1.0.0", "completeness": 0.0, "freshness": "unavailable", "assumptions": []},
            "result": {},
        }
    elif field_schema_name == "DataProvenance":
        instance = {
            "schema_version": "1.0.0",
            "provider": "some-provider",
            "data_mode": "unavailable",
            "retrieved_at": "2026-08-01T12:00:00Z",
        }
    else:  # DataQuality
        instance = {"schema_version": "1.0.0", "completeness": 0.0, "freshness": "unavailable", "assumptions": []}
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]
