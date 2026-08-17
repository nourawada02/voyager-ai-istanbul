"""Shared schema-validation fixtures for providers/tests (Checkpoint
Phase 4 C.0). Mirrors contracts/tests/test_contracts.py's own
registry-building pattern so provider fake output is validated against
the exact same canonical schemas the contract-conformance suite uses --
no separate/duplicated schema copy."""

from __future__ import annotations

import glob
import json
import os

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "contracts")
SCHEMA_FILES = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")))


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


@pytest.fixture(scope="session")
def envelope_validator(registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, "ProviderResponseEnvelope.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def make_validator(name: str, registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
