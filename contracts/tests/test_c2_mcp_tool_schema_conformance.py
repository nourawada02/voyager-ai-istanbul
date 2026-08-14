"""Cross-repository, real-MCP-protocol schema-conformance evidence for
Travel MCP's Checkpoint C.2.1 production accommodation tools
(search_stays, estimate_fair_price).

Belongs here, not in Travel MCP's own suite, for exactly the reason
test_c1_serializer_schema_conformance.py already established: this is
where both the accepted JSON Schema contracts AND the pinned Travel MCP
gitlink exist together. This test FAILS -- it never skips -- if Travel MCP
cannot be imported/executed, if a real tool response diverges from the
accepted schemas, or if a prohibited value (a local path, a stack trace)
appears in an error payload.

Unlike test_c1_serializer_schema_conformance.py (which exercises the pure
serializer directly), this test drives the FULL production MCP surface: a
real `mcp.client.session.ClientSession` performing a real initialize
handshake, a real `tools/list`, and real `tools/call` requests against the
real `phase2.mcp.server.build_server()` MCPServer instance (over the
official SDK's own in-memory test transport) -- proving the protocol
boundary itself, not just the dataclasses underneath it.

Requires the `mcp==2.0.0` line from the canonical
`services/travel-mcp/requirements.lock` (the same lock the production
Dockerfile is intended to install per docs/adr/0002-phase2-checkpoint-a.md
§8) -- run this file with an interpreter that has it installed, exactly as
`services/travel-mcp/phase0`'s own tests already require a pinned `mcp`
environment. An ambient interpreter with an incompatible `mcp` major
version fails this test at collection time (a real, informative failure),
never a silent skip.

Run with: python -m pytest contracts/tests -v
(using an interpreter satisfying services/travel-mcp/requirements.lock)
"""

from __future__ import annotations

import asyncio
import glob
import hashlib
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

# No try/except here: if the pinned submodule (or its mcp==2.0.0
# dependency) cannot be imported, every test in this module must fail
# loudly at collection time, not skip.
if TRAVEL_MCP_PATH not in sys.path:
    sys.path.insert(0, TRAVEL_MCP_PATH)

from mcp.client._memory import InMemoryTransport  # noqa: E402
from mcp.client.session import ClientSession  # noqa: E402

from phase2.mcp.server import build_server  # noqa: E402
from phase2.serving.config import AccommodationResourceConfig  # noqa: E402
from phase2.tests.serving.conftest import (  # noqa: E402
    _synthetic_raw_rows,
    write_bundle,
    write_manifest_and_dataset,
    write_registry,
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


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def tool_input_schemas() -> dict:
    """The real, registered `tools/list` input schema (`Tool.parameters`)
    for each production tool, keyed by name -- built once, no resources
    need to actually load for this (schema registration happens at
    `build_server()` time, independent of the lifespan/config)."""
    config = AccommodationResourceConfig(
        manifest_path=Path("unused"),
        dataset_path=Path("unused"),
        bundle_path=Path("unused"),
        district_registry_path=Path("unused"),
        expected_bundle_sha256="0" * 64,
    )
    server = build_server(config)
    return {t.name: t.parameters for t in server._tool_manager.list_tools()}


def _resolve_ref(schema: dict, node: dict) -> dict:
    """Resolves a single local `$ref` (e.g. '#/$defs/BudgetArgs') against
    the enclosing schema's own $defs -- exactly the one level of nesting
    `search_stays`'s `budget` property uses."""
    if "$ref" not in node:
        return node
    pointer = node["$ref"]
    assert pointer.startswith("#/$defs/"), pointer
    return schema["$defs"][pointer[len("#/$defs/") :]]


_SEMANTIC_KEYS = (
    "type", "pattern", "format", "enum", "const", "minimum", "maximum",
    "minItems", "maxItems", "uniqueItems", "minLength", "maxLength",
)


def _semantic_subset(node: dict) -> dict:
    """Strips cosmetic-only keys (title, default, description) so two
    schema fragments can be compared on meaning, not presentation. `type`
    is also dropped whenever `const` is present: `const` alone already
    fully determines the accepted value (and therefore its type), so a
    hand-written contract omitting a redundant sibling `type` and an
    auto-generated schema including one are not a real disagreement."""
    subset = {k: v for k, v in node.items() if k in _SEMANTIC_KEYS}
    if "const" in subset:
        subset.pop("type", None)
    return subset


def test_search_stays_required_set_exactly_matches_canonical_contract(tool_input_schemas: dict):
    canonical = _load(os.path.join(CONTRACTS_DIR, "SearchStaysRequest.schema.json"))
    actual = tool_input_schemas["search_stays"]
    assert set(actual["required"]) == set(canonical["required"]), (
        f"MCP-required {sorted(actual['required'])} != contract-required {sorted(canonical['required'])}"
    )


def test_estimate_fair_price_required_set_exactly_matches_canonical_contract(tool_input_schemas: dict):
    canonical = _load(os.path.join(CONTRACTS_DIR, "EstimateFairPriceRequest.schema.json"))
    actual = tool_input_schemas["estimate_fair_price"]
    assert set(actual["required"]) == set(canonical["required"]), (
        f"MCP-required {sorted(actual['required'])} != contract-required {sorted(canonical['required'])}"
    )


def test_search_stays_property_set_exactly_matches_canonical_contract(tool_input_schemas: dict):
    canonical = _load(os.path.join(CONTRACTS_DIR, "SearchStaysRequest.schema.json"))
    actual = tool_input_schemas["search_stays"]
    assert set(actual["properties"].keys()) == set(canonical["properties"].keys())


def test_estimate_fair_price_property_set_exactly_matches_canonical_contract(tool_input_schemas: dict):
    canonical = _load(os.path.join(CONTRACTS_DIR, "EstimateFairPriceRequest.schema.json"))
    actual = tool_input_schemas["estimate_fair_price"]
    assert set(actual["properties"].keys()) == set(canonical["properties"].keys())


def test_search_stays_additional_properties_is_false(tool_input_schemas: dict):
    assert tool_input_schemas["search_stays"]["additionalProperties"] is False


def test_estimate_fair_price_additional_properties_is_false(tool_input_schemas: dict):
    assert tool_input_schemas["estimate_fair_price"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "field",
    [
        "session_id", "trace_id", "guest_count", "result_limit", "currency",
        "ranking_mode", "schema_version", "check_in", "check_out", "side",
        "min_bedrooms", "min_beds", "min_bathrooms",
    ],
)
def test_search_stays_scalar_field_semantic_constraints_match_contract(tool_input_schemas: dict, field: str):
    """Exact semantic-constraint equality (type/pattern/format/enum/const/
    bounds) for every scalar SearchStaysRequest field -- not merely
    property-name equality."""
    canonical = _load(os.path.join(CONTRACTS_DIR, "SearchStaysRequest.schema.json"))
    expected = _semantic_subset(canonical["properties"][field])
    actual = _semantic_subset(tool_input_schemas["search_stays"]["properties"][field])
    assert actual == expected, f"{field}: MCP schema {actual} != contract {expected}"


@pytest.mark.parametrize("field", ["district_ids", "room_types", "required_amenities"])
def test_search_stays_array_field_semantic_constraints_match_contract(tool_input_schemas: dict, field: str):
    """Array-level bounds/uniqueItems AND item-level pattern/length bounds
    both compared -- an array filter that matched on outer bounds alone
    but silently dropped its item-level pattern would still fail here."""
    canonical = _load(os.path.join(CONTRACTS_DIR, "SearchStaysRequest.schema.json"))
    expected_outer = canonical["properties"][field]
    actual_outer = tool_input_schemas["search_stays"]["properties"][field]

    assert actual_outer["type"] == "array" == expected_outer["type"]
    assert actual_outer["minItems"] == expected_outer["minItems"]
    assert actual_outer["maxItems"] == expected_outer["maxItems"]
    assert actual_outer["uniqueItems"] == expected_outer["uniqueItems"] is True

    expected_items = _semantic_subset(expected_outer["items"])
    actual_items = _semantic_subset(actual_outer["items"])
    assert actual_items == expected_items, f"{field}.items: MCP schema {actual_items} != contract {expected_items}"


def test_search_stays_budget_nested_structure_matches_contract(tool_input_schemas: dict):
    """Resolves the MCP schema's $ref for `budget` and compares it against
    the contract's own inline budget object -- required set, per-field
    types/consts/enums/bounds, and additionalProperties: false."""
    canonical = _load(os.path.join(CONTRACTS_DIR, "SearchStaysRequest.schema.json"))
    canonical_budget = canonical["properties"]["budget"]

    schema = tool_input_schemas["search_stays"]
    actual_budget = _resolve_ref(schema, schema["properties"]["budget"])

    assert actual_budget["additionalProperties"] is False == canonical_budget["additionalProperties"]
    assert set(actual_budget["required"]) == set(canonical_budget["required"])
    assert set(actual_budget["properties"].keys()) == set(canonical_budget["properties"].keys())
    for field in canonical_budget["properties"]:
        expected = _semantic_subset(canonical_budget["properties"][field])
        actual = _semantic_subset(actual_budget["properties"][field])
        assert actual == expected, f"budget.{field}: MCP schema {actual} != contract {expected}"


@pytest.mark.parametrize("field", ["session_id", "trace_id", "stay_id", "currency", "schema_version"])
def test_estimate_fair_price_field_semantic_constraints_match_contract(tool_input_schemas: dict, field: str):
    canonical = _load(os.path.join(CONTRACTS_DIR, "EstimateFairPriceRequest.schema.json"))
    expected = _semantic_subset(canonical["properties"][field])
    actual = _semantic_subset(tool_input_schemas["estimate_fair_price"]["properties"][field])
    assert actual == expected, f"{field}: MCP schema {actual} != contract {expected}"


def test_explicit_null_on_optional_field_is_rejected_not_treated_as_no_filter(tool_input_schemas: dict):
    """Contract rule (SearchStaysRequest §6): omitting a key means 'no
    filter'; an explicit JSON null must fail -- never be silently treated
    as 'no filter'. A field typed with an `anyOf: [T, null]` branch would
    accept null; the MCP schema must NOT have that branch on any optional
    field, exactly like the canonical contract."""
    schema = tool_input_schemas["search_stays"]
    for field in ("check_in", "check_out", "district_ids", "side", "room_types",
                  "min_bedrooms", "min_beds", "min_bathrooms", "required_amenities", "budget"):
        prop = schema["properties"][field]
        assert "anyOf" not in prop, f"{field} accepts an anyOf-null branch, contradicting the contract"
        assert prop.get("type") != "null"


def test_room_types_has_no_enum_restriction_matching_deliberate_open_design(tool_input_schemas: dict):
    """Checkpoint C.0.1 §4: room_types is deliberately NOT a closed enum --
    every value, known-unsupported or unrecognized, must reach the
    semantic UnsupportedRoomTypeError path uniformly. If the MCP schema
    added an enum here, an unrecognized room type would be rejected as a
    generic schema error instead, contradicting that design."""
    prop = tool_input_schemas["search_stays"]["properties"]["room_types"]["items"]
    assert "enum" not in prop


@pytest.fixture(scope="module")
def mcp_round_trip():
    """Builds the real production server (phase2.mcp.server.build_server,
    unmodified) over synthetic resources built through Travel MCP's own
    conftest builders, then performs real MCP protocol calls: one
    successful search_stays, one successful estimate_fair_price (for a
    stay returned by that search), and one deliberate UNSUPPORTED_ROOM_TYPE
    error -- all through a real ClientSession/InMemoryTransport round
    trip, never a direct Python function call."""
    with tempfile.TemporaryDirectory() as d:
        tmp_path = Path(d)
        df = _synthetic_raw_rows(n=60, seed=42)
        manifest_path, dataset_path = write_manifest_and_dataset(tmp_path, df)
        dataset_sha256 = _sha256_of(dataset_path)
        bundle_path, bundle_sha256 = write_bundle(tmp_path, dataset_sha256=dataset_sha256)
        registry_path = write_registry(tmp_path / "registry.json")
        config = AccommodationResourceConfig(
            manifest_path=manifest_path,
            dataset_path=dataset_path,
            bundle_path=bundle_path,
            district_registry_path=registry_path,
            expected_bundle_sha256=bundle_sha256,
        )
        server = build_server(config)

        session_id = "dddddddd-dddd-4ddd-8ddd-dddddddddd01"

        async def _run():
            async with InMemoryTransport(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    search_result = await session.call_tool(
                        "search_stays",
                        {
                            "session_id": session_id,
                            "trace_id": "dddddddd-dddd-4ddd-8ddd-dddddddddd02",
                            "guest_count": 1,
                            "result_limit": 5,
                            "currency": "TRY",
                            "ranking_mode": "preliminary_price_value_desc",
                            "schema_version": "1.0.0",
                        },
                    )
                    stay_id = search_result.structured_content["stays"][0]["stay"]["stay_id"]

                    estimate_result = await session.call_tool(
                        "estimate_fair_price",
                        {
                            "session_id": session_id,
                            "trace_id": "dddddddd-dddd-4ddd-8ddd-dddddddddd03",
                            "stay_id": stay_id,
                            "currency": "TRY",
                            "schema_version": "1.0.0",
                        },
                    )

                    error_result = await session.call_tool(
                        "search_stays",
                        {
                            "session_id": session_id,
                            "trace_id": "dddddddd-dddd-4ddd-8ddd-dddddddddd04",
                            "guest_count": 1,
                            "result_limit": 5,
                            "currency": "TRY",
                            "ranking_mode": "preliminary_price_value_desc",
                            "schema_version": "1.0.0",
                            "room_types": ["Hotel room"],
                        },
                    )

                    return search_result, estimate_result, error_result

        yield asyncio.run(_run())


def test_travel_mcp_server_is_importable_and_reachable_over_real_protocol(mcp_round_trip):
    """If phase2.mcp.server could not be imported, or the real protocol
    round trip failed, this fixture itself would already have raised --
    this test exists so that failure is attributed to an explicit, named
    assertion, matching test_c1_serializer_schema_conformance.py's pattern."""
    search_result, estimate_result, error_result = mcp_round_trip
    assert search_result.is_error is False
    assert estimate_result.is_error is False
    assert error_result.is_error is False


def test_real_search_stays_response_conforms_to_search_stays_result_schema(mcp_round_trip, registry: Registry):
    search_result, _, _ = mcp_round_trip
    validator = _validator("SearchStaysResult", registry)
    errors = list(validator.iter_errors(search_result.structured_content))
    assert not errors, [e.message for e in errors]


def test_real_estimate_fair_price_response_conforms_to_its_result_schema(mcp_round_trip, registry: Registry):
    _, estimate_result, _ = mcp_round_trip
    validator = _validator("EstimateFairPriceResult", registry)
    errors = list(validator.iter_errors(estimate_result.structured_content))
    assert not errors, [e.message for e in errors]


def test_real_error_response_conforms_to_error_envelope_schema(mcp_round_trip, registry: Registry):
    _, _, error_result = mcp_round_trip
    validator = _validator("ErrorEnvelope", registry)
    errors = list(validator.iter_errors(error_result.structured_content))
    assert not errors, [e.message for e in errors]
    assert error_result.structured_content["error_code"] == "UNSUPPORTED_ROOM_TYPE"


def test_real_responses_contain_no_prohibited_fields_or_local_paths(mcp_round_trip):
    search_result, estimate_result, error_result = mcp_round_trip
    for result in (search_result, estimate_result, error_result):
        raw = json.dumps(result.structured_content)
        assert '"host_id"' not in raw
        assert "C:\\Users" not in raw
        assert "Traceback" not in raw
        assert "site-packages" not in raw


def test_schema_resolution_succeeds_for_nested_estimate_fair_price_ref(mcp_round_trip, registry: Registry):
    """Explicitly exercises EstimateFairPriceResult's nested FairPriceEstimate
    $ref resolution -- a missing/broken $id in the registry must fail this
    test, never silently validate an under-constrained instance."""
    _, estimate_result, _ = mcp_round_trip
    schema = _load(os.path.join(CONTRACTS_DIR, "EstimateFairPriceResult.schema.json"))
    fair_price_schema = schema["properties"]["fair_price"]
    assert any("FairPriceEstimate.schema.json" in json.dumps(branch) for branch in fair_price_schema["allOf"])
    validator = _validator("EstimateFairPriceResult", registry)
    errors = list(validator.iter_errors(estimate_result.structured_content))
    assert not errors, [e.message for e in errors]
