"""Cross-repository schema-conformance evidence for Travel MCP's
Checkpoint C.1 pure search service (Checkpoint C.1.1 §5).

This validation belongs here, not in Travel MCP's own suite: the
superproject is where both the accepted JSON Schema contracts AND the
pinned Travel MCP gitlink exist together, so it is the only place a
same-conversation, always-in-sync cross-repo check can run without forcing
travel-mcp's own hermetic suite to depend on the superproject's filesystem
layout. This test FAILS -- it never skips -- if Travel MCP cannot be
imported/executed, if the serializer shape diverges from the accepted
schemas, if schema `$ref` resolution fails, if required fields are
missing, or if a prohibited field (host_id, a local path) appears.

Reuses the pinned submodule's own synthetic-resource-building code
(phase2.tests.serving.conftest, imported directly, not reimplemented) so
this is exactly the same resource construction travel-mcp's own offline
tests already exercise -- not a second, hand-maintained fixture.

Run with: python -m pytest contracts/tests -v
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_FILES = sorted(glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")))
SUPERPROJECT_ROOT = os.path.dirname(CONTRACTS_DIR)
TRAVEL_MCP_PATH = os.path.join(SUPERPROJECT_ROOT, "services", "travel-mcp")

# No try/except here: if the pinned submodule cannot be imported, every
# test in this module must fail loudly at collection time, not skip.
if TRAVEL_MCP_PATH not in sys.path:
    sys.path.insert(0, TRAVEL_MCP_PATH)

from phase2.serving.config import AccommodationResourceConfig  # noqa: E402
from phase2.serving.errors import NoMatchingStaysError, UnknownDistrictError  # noqa: E402
from phase2.serving.loader import load_accommodation_resources  # noqa: E402
from phase2.serving.models import SearchStaysRequest  # noqa: E402
from phase2.serving.search_service import AccommodationSearchService  # noqa: E402
from phase2.serving_contract_reference import UnsupportedRoomTypeError  # noqa: E402
from phase2.tests.serving.conftest import (  # noqa: E402
    write_bundle,
    write_manifest_and_dataset,
    write_registry,
    _synthetic_raw_rows,
)


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


def _validator(name: str, reg: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=reg, format_checker=FormatChecker())


@pytest.fixture(scope="module")
def real_service_and_result():
    """Builds real synthetic resources through the pinned submodule's own
    code (loader + search service, unmodified), using its own conftest
    builders -- not a duplicate fixture."""
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        df = _synthetic_raw_rows(n=60, seed=42)
        manifest_path, dataset_path = write_manifest_and_dataset(tmp_path, df)
        import hashlib

        dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
        bundle_path, bundle_sha256 = write_bundle(tmp_path, dataset_sha256=dataset_sha256)
        registry_path = write_registry(tmp_path / "registry.json")
        config = AccommodationResourceConfig(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            bundle_path=bundle_path,
            district_registry_path=registry_path,
            expected_bundle_sha256=bundle_sha256,
        )
        resources = load_accommodation_resources(config)
        service = AccommodationSearchService(resources)
        request = SearchStaysRequest(
            session_id="cccccccc-cccc-4ccc-8ccc-cccccccccc01",
            trace_id="cccccccc-cccc-4ccc-8ccc-cccccccccc02",
            guest_count=1,
            currency="TRY",
            result_limit=5,
        )
        result = service.search(request)
        yield service, result, config


def test_travel_mcp_serializer_module_is_importable_and_executable(real_service_and_result):
    """If phase2.serving could not be imported or search() could not run,
    this fixture itself would already have raised -- this test exists so
    that failure is attributed to an explicit, named assertion."""
    service, result, _ = real_service_and_result
    assert service is not None
    assert result is not None
    assert len(result.stays) >= 1


def test_success_result_conforms_to_search_stays_result_schema(real_service_and_result, registry):
    _, result, _ = real_service_and_result
    validator = _validator("SearchStaysResult", registry)
    instance = result.to_dict()
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_success_result_required_fields_are_all_present(real_service_and_result):
    _, result, _ = real_service_and_result
    instance = result.to_dict()
    required = {
        "schema_version", "snapshot_date", "dataset_sha256", "bundle_sha256", "model_version",
        "currency", "data_mode", "disclaimer", "predictions_produced", "ranking_basis",
        "excluded_row_count", "stays",
    }
    assert required.issubset(instance.keys())
    item = instance["stays"][0]
    assert {"stay", "fair_price", "preliminary_capped_price_value", "rank"}.issubset(item.keys())


def test_success_result_contains_no_prohibited_fields_or_local_paths(real_service_and_result):
    service, result, config = real_service_and_result
    raw = json.dumps(result.to_dict())
    assert '"host_id"' not in raw
    assert str(config.manifest_path) not in raw
    assert str(config.dataset_path) not in raw
    assert config.manifest_path.name not in raw
    assert config.dataset_path.name not in raw


def test_unsupported_room_type_error_details_conform_to_error_envelope_schema(registry):
    validator = _validator("ErrorEnvelope", registry)
    exc = UnsupportedRoomTypeError(["Hotel room"])
    instance = {
        "schema_version": "1.1.0",
        "error_code": exc.error_code,
        "message": "One or more requested room types are not supported.",
        "trace_id": "cccccccc-cccc-4ccc-8ccc-cccccccccc03",
        "retriable": exc.retriable,
        "details": exc.to_error_details(),
    }
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_unknown_district_error_details_conform_to_error_envelope_schema(registry):
    validator = _validator("ErrorEnvelope", registry)
    exc = UnknownDistrictError(["district_fake"], ["district_fake"])
    instance = {
        "schema_version": "1.1.0",
        "error_code": exc.error_code,
        "message": "Unknown district.",
        "trace_id": "cccccccc-cccc-4ccc-8ccc-cccccccccc04",
        "retriable": exc.retriable,
        "details": exc.to_error_details(),
    }
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_no_matching_stays_error_details_conform_to_error_envelope_schema(registry):
    validator = _validator("ErrorEnvelope", registry)
    exc = NoMatchingStaysError(["room_type", "min_bedrooms"])
    instance = {
        "schema_version": "1.1.0",
        "error_code": exc.error_code,
        "message": "No stays matched every requested filter.",
        "trace_id": "cccccccc-cccc-4ccc-8ccc-cccccccccc05",
        "retriable": exc.retriable,
        "details": exc.to_error_details(),
    }
    errors = list(validator.iter_errors(instance))
    assert not errors, [e.message for e in errors]


def test_schema_resolution_succeeds_for_nested_stay_option_and_fair_price_refs(real_service_and_result, registry):
    """Explicitly exercises the nested StayOption/FairPriceEstimate $ref
    resolution inside SearchStaysResult -- a schema-resolution failure here
    (e.g. a missing $id in the registry) must fail this test, not silently
    validate an under-constrained instance."""
    _, result, _ = real_service_and_result
    validator = _validator("SearchStaysResult", registry)
    schema = _load(os.path.join(CONTRACTS_DIR, "SearchStaysResult.schema.json"))
    item_schema = schema["$defs"]["resultItem"]
    assert any("StayOption.schema.json" in json.dumps(branch) for branch in item_schema["properties"]["stay"]["allOf"])
    assert any(
        "FairPriceEstimate.schema.json" in json.dumps(branch) for branch in item_schema["properties"]["fair_price"]["allOf"]
    )
    # And the real instance still validates end-to-end through those refs.
    errors = list(validator.iter_errors(result.to_dict()))
    assert not errors, [e.message for e in errors]
