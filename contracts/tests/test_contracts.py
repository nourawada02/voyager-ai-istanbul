"""Parametrized contract test suite for the canonical JSON Schema catalog.

Proves: every schema is valid Draft 2020-12; every valid example passes its
schema; every invalid example fails for a real reason; provider-response
fixtures conform to ProviderResponseEnvelope (except the one deliberately
malformed fixture); and every fixture is deterministic (fixed timestamps,
no network, no local machine paths, no secrets).

Run with: python -m pytest contracts/tests -v
"""

from __future__ import annotations

import glob
import json
import os
import re

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILES = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")))
VALID_DIR = os.path.join(CONTRACTS_DIR, "examples", "valid")
INVALID_DIR = os.path.join(CONTRACTS_DIR, "examples", "invalid")
FIXTURES_DIR = os.path.join(CONTRACTS_DIR, "fixtures")

CONTRACT_NAMES = sorted(
    os.path.basename(p)[: -len(".schema.json")] for p in SCHEMA_FILES
)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def registry() -> Registry:
    resources = []
    for path in SCHEMA_FILES:
        schema = _load(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validator(name: str, registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


# ---------------------------------------------------------------------------
# Schema structural correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: os.path.basename(p))
def test_schema_is_valid_draft_2020_12(schema_path: str):
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: os.path.basename(p))
def test_schema_has_stable_id_and_schema_version_field(schema_path: str):
    schema = _load(schema_path)
    assert schema["$id"].startswith((
        "https://schemas.voyagerai.dev/phase1/",
        "https://schemas.voyagerai.dev/phase2/",
        "https://schemas.voyagerai.dev/phase3/",
        "https://schemas.voyagerai.dev/phase4/",
    ))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    props = schema.get("properties", {})
    assert "schema_version" in props, f"{schema_path} missing schema_version property"


@pytest.mark.parametrize("schema_path", SCHEMA_FILES, ids=lambda p: os.path.basename(p))
def test_closed_protocol_objects_reject_additional_properties(schema_path: str):
    schema = _load(schema_path)
    assert schema.get("additionalProperties") is False, (
        f"{schema_path} must be a closed object (additionalProperties: false)"
    )


# ---------------------------------------------------------------------------
# Valid examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_every_contract_has_a_valid_example(name: str):
    path = os.path.join(VALID_DIR, f"{name}.json")
    assert os.path.exists(path), f"missing valid example for {name}"


@pytest.mark.parametrize("name", CONTRACT_NAMES)
def test_valid_example_passes_its_canonical_schema(name: str, registry: Registry):
    validator = _validator(name, registry)
    instance = _load(os.path.join(VALID_DIR, f"{name}.json"))
    errors = list(validator.iter_errors(instance))
    assert not errors, f"{name}: {[e.message for e in errors]}"


# ---------------------------------------------------------------------------
# Invalid examples
# ---------------------------------------------------------------------------


INVALID_FILES = sorted(glob.glob(os.path.join(INVALID_DIR, "*.json")))


@pytest.mark.parametrize("path", INVALID_FILES, ids=lambda p: os.path.basename(p))
def test_invalid_example_fails_its_canonical_schema(path: str, registry: Registry):
    contract_name = os.path.basename(path).split("_")[0]
    # Handle two-word contract prefixes (none currently start with an
    # underscore-splitting ambiguity, but resolve by exact schema-file match).
    matches = [n for n in CONTRACT_NAMES if os.path.basename(path).startswith(n)]
    assert matches, f"cannot map {path} to a known contract"
    contract_name = max(matches, key=len)
    validator = _validator(contract_name, registry)
    instance = _load(path)
    errors = list(validator.iter_errors(instance))
    assert errors, f"{path} was expected to fail schema validation but passed"


# ---------------------------------------------------------------------------
# Fixtures: envelope conformance + determinism
# ---------------------------------------------------------------------------


FIXTURE_FILES = sorted(glob.glob(os.path.join(FIXTURES_DIR, "*.json")))
CONFORMING_FIXTURES = [f for f in FIXTURE_FILES if "malformed" not in os.path.basename(f)]
MALFORMED_FIXTURES = [f for f in FIXTURE_FILES if "malformed" in os.path.basename(f)]


@pytest.mark.parametrize("path", CONFORMING_FIXTURES, ids=lambda p: os.path.basename(p))
def test_conforming_fixture_matches_provider_response_envelope(path: str, registry: Registry):
    validator = _validator("ProviderResponseEnvelope", registry)
    instance = _load(path)
    errors = list(validator.iter_errors(instance))
    assert not errors, f"{path}: {[e.message for e in errors]}"


@pytest.mark.parametrize("path", MALFORMED_FIXTURES, ids=lambda p: os.path.basename(p))
def test_malformed_fixture_is_deliberately_invalid(path: str, registry: Registry):
    validator = _validator("ProviderResponseEnvelope", registry)
    instance = _load(path)
    errors = list(validator.iter_errors(instance))
    assert errors, f"{path} was supposed to be deliberately malformed but validated cleanly"


_FIXED_TIMESTAMP_RE = re.compile(r"^2026-08-01T\d{2}:00:00Z$")
_FORBIDDEN_PATTERNS = [
    re.compile(r"[Cc]:\\\\Users"),
    re.compile(r"AppData"),
    re.compile(r"localhost"),
    re.compile(r"127\.0\.0\.1"),
    re.compile(r"(?i)api[_-]?key"),
    re.compile(r"(?i)secret"),
    re.compile(r"(?i)password"),
]


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: os.path.basename(p))
def test_fixture_is_deterministic_and_contains_no_local_or_secret_data(path: str):
    with open(path, encoding="utf-8") as f:
        raw_text = f.read()
    for pattern in _FORBIDDEN_PATTERNS:
        assert not pattern.search(raw_text), f"{path} contains forbidden pattern {pattern.pattern}"

    instance = json.loads(raw_text)
    retrieved_at = instance.get("retrieved_at")
    if retrieved_at is not None:
        assert _FIXED_TIMESTAMP_RE.match(retrieved_at), (
            f"{path} retrieved_at={retrieved_at!r} is not one of the fixed synthetic timestamps"
        )
    provider = instance.get("provider", "")
    if "malformed" not in os.path.basename(path):
        assert "synthetic" in provider or provider in (
            "inside-airbnb-snapshot",
            "pinned-fx-snapshot",
        ), f"{path} provider {provider!r} does not clearly identify as synthetic/fixture"
