"""Checkpoint C.2.1 contract tests for the estimate_fair_price design
(smallest standalone contract for a stable stay_id lookup -- never
arbitrary caller-supplied model features). No production tool wiring is
exercised here -- these tests lock the JSON Schema contracts only, the
same role test_search_stays_contract.py plays for search_stays.

Baseline structural checks (valid/invalid example presence, schema
validity, additionalProperties:false, schema_version property) are already
covered generically for every *.schema.json file by
contracts/tests/test_contracts.py's parametrized suite -- not repeated
here.

Run with: python -m pytest contracts/tests -v
"""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import sys

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILES = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")))
VALID_DIR = os.path.join(CONTRACTS_DIR, "examples", "valid")

_TRAVEL_MCP_PATH = os.path.abspath(os.path.join(CONTRACTS_DIR, "..", "services", "travel-mcp"))
if _TRAVEL_MCP_PATH not in sys.path:
    sys.path.insert(0, _TRAVEL_MCP_PATH)


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


def _validator(name: str, reg: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=reg, format_checker=FormatChecker())


def _valid(name: str) -> dict:
    return _load(os.path.join(VALID_DIR, f"{name}.json"))


# ---------------------------------------------------------------------------
# EstimateFairPriceRequest: stable-id lookup only, never raw features
# ---------------------------------------------------------------------------


def test_currency_other_than_try_is_rejected(registry: Registry):
    validator = _validator("EstimateFairPriceRequest", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceRequest"))
    instance["currency"] = "USD"
    assert list(validator.iter_errors(instance))


def test_stay_id_is_required(registry: Registry):
    validator = _validator("EstimateFairPriceRequest", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceRequest"))
    del instance["stay_id"]
    assert list(validator.iter_errors(instance))


def test_empty_stay_id_is_rejected(registry: Registry):
    validator = _validator("EstimateFairPriceRequest", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceRequest"))
    instance["stay_id"] = ""
    assert list(validator.iter_errors(instance))


@pytest.mark.parametrize(
    "forbidden_field",
    ["accommodates", "latitude", "longitude", "bedrooms", "room_type", "price"],
)
def test_raw_feature_fields_are_rejected_not_just_undocumented(registry: Registry, forbidden_field: str):
    """The request contract must never accept arbitrary model features --
    additionalProperties:false makes this a hard schema rejection, not
    merely an implementation choice to ignore extra fields."""
    validator = _validator("EstimateFairPriceRequest", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceRequest"))
    instance[forbidden_field] = 1
    assert list(validator.iter_errors(instance))


def test_null_session_id_is_rejected(registry: Registry):
    validator = _validator("EstimateFairPriceRequest", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceRequest"))
    instance["session_id"] = None
    assert list(validator.iter_errors(instance))


# ---------------------------------------------------------------------------
# EstimateFairPriceResult: reuses FairPriceEstimate by $ref, never duplicated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "required_field",
    ["stay_id", "snapshot_date", "dataset_sha256", "bundle_sha256", "model_version", "currency", "provenance", "disclaimer", "fair_price"],
)
def test_estimate_fair_price_result_requires_every_top_level_field(registry: Registry, required_field: str):
    validator = _validator("EstimateFairPriceResult", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceResult"))
    del instance[required_field]
    assert list(validator.iter_errors(instance)), f"{required_field} should be required"


def test_result_currency_must_be_exactly_try(registry: Registry):
    validator = _validator("EstimateFairPriceResult", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceResult"))
    instance["currency"] = "USD"
    assert list(validator.iter_errors(instance))


def test_nested_fair_price_must_be_exactly_schema_version_1_1_0(registry: Registry):
    """Mirrors test_search_stays_contract.py's identical assertion on
    SearchStaysResult -- the same tightening rule, reused, never a separate
    parallel definition."""
    validator = _validator("EstimateFairPriceResult", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceResult"))
    instance["fair_price"]["schema_version"] = "1.0.0"
    assert list(validator.iter_errors(instance))


def test_nested_fair_price_requires_interval(registry: Registry):
    validator = _validator("EstimateFairPriceResult", registry)
    instance = copy.deepcopy(_valid("EstimateFairPriceResult"))
    del instance["fair_price"]["interval"]
    assert list(validator.iter_errors(instance))


def test_fair_price_ref_resolves_to_the_real_shared_schema_not_a_duplicate(registry: Registry):
    """Proves the $ref actually points at the one canonical
    FairPriceEstimate.schema.json (the same file StayOption/SearchStaysResult
    use) -- not a hand-copied inline redefinition that could silently drift."""
    schema = _load(os.path.join(CONTRACTS_DIR, "EstimateFairPriceResult.schema.json"))
    fair_price_schema = schema["properties"]["fair_price"]
    assert any(
        "FairPriceEstimate.schema.json" in json.dumps(branch) for branch in fair_price_schema["allOf"]
    ), "fair_price must $ref the shared FairPriceEstimate.schema.json, not redefine its shape"
    assert "estimated_fair_price" not in fair_price_schema.get("properties", {}), (
        "fair_price must not duplicate FairPriceEstimate's own properties inline"
    )


def test_provenance_ref_resolves_to_the_real_data_provenance_schema(registry: Registry):
    schema = _load(os.path.join(CONTRACTS_DIR, "EstimateFairPriceResult.schema.json"))
    assert schema["properties"]["provenance"]["$ref"] == "https://schemas.voyagerai.dev/phase1/DataProvenance.schema.json"


def test_valid_result_example_validates_end_to_end_through_every_ref(registry: Registry):
    validator = _validator("EstimateFairPriceResult", registry)
    instance = _valid("EstimateFairPriceResult")
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_disclaimer_is_truthful_in_the_canonical_valid_example():
    from phase2.serving_contract_reference import assert_disclaimer_is_truthful

    instance = _valid("EstimateFairPriceResult")
    assert_disclaimer_is_truthful(instance["disclaimer"])  # must not raise


# ---------------------------------------------------------------------------
# STAY_NOT_FOUND error envelope: safe, structured, no internal detail
# ---------------------------------------------------------------------------


_FORBIDDEN_PATTERNS = [
    re.compile(r"[Cc]:\\\\Users"),
    re.compile(r"Traceback"),
    re.compile(r"\.py\", line \d+"),
    re.compile(r"(?i)api[_-]?key"),
    re.compile(r"(?i)secret"),
]


def test_stay_not_found_error_example_validates_and_contains_no_internal_detail(registry: Registry):
    validator = _validator("ErrorEnvelope", registry)
    instance = _load(os.path.join(VALID_DIR, "ErrorEnvelope_stay_not_found.json"))
    assert not list(validator.iter_errors(instance))
    raw_text = json.dumps(instance)
    for pattern in _FORBIDDEN_PATTERNS:
        assert not pattern.search(raw_text), f"forbidden pattern {pattern.pattern} found"
    assert instance["retriable"] is False
    assert instance["error_code"] == "STAY_NOT_FOUND"


def test_stay_not_found_error_has_structured_stay_id_detail(registry: Registry):
    instance = _load(os.path.join(VALID_DIR, "ErrorEnvelope_stay_not_found.json"))
    assert instance["details"]["stay_id"] == "stay_does_not_exist"
