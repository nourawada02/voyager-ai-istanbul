"""Checkpoint C.0.1 contract tests for the search_stays design
(docs/adr/0003-phase2-checkpoint-c0-serving-contract.md). No production
tool exists yet -- these tests lock the JSON Schema contracts only. Room-
type/date/budget/ranking/rounding semantic rules are tested separately in
services/travel-mcp/phase2/tests/test_serving_contract_reference.py.

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


def _validator(name: str, registry: Registry) -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, f"{name}.schema.json"))
    return Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())


def _valid(name: str) -> dict:
    return _load(os.path.join(VALID_DIR, f"{name}.json"))


def _fair_price_incomplete_base() -> dict:
    return _load(os.path.join(VALID_DIR, "FairPriceEstimate_scoring_incomplete.json"))


def _fair_price_complete_base() -> dict:
    return _valid("FairPriceEstimate")  # schema_version 1.0.0, unconditionally complete


# ---------------------------------------------------------------------------
# Request limits enforced
# ---------------------------------------------------------------------------


def test_result_limit_of_50_is_accepted(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["result_limit"] = 50
    assert not list(validator.iter_errors(instance))


def test_result_limit_over_50_is_rejected(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["result_limit"] = 51
    assert list(validator.iter_errors(instance))


def test_result_limit_of_zero_is_rejected(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["result_limit"] = 0
    assert list(validator.iter_errors(instance))


def test_unknown_top_level_field_is_rejected(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["arbitrary_extra_field"] = "should not be accepted"
    assert list(validator.iter_errors(instance))


def test_null_on_a_non_nullable_optional_field_is_rejected(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["side"] = None
    assert list(validator.iter_errors(instance))


def test_old_deal_score_desc_ranking_mode_is_now_rejected(registry: Registry):
    """Checkpoint C.0.1 §1: the dishonest 'deal_score_desc' request mode no
    longer exists -- a request naming it must fail."""
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["ranking_mode"] = "deal_score_desc"
    assert list(validator.iter_errors(instance))


# --- Room-type: schema stays open, everything reaches the semantic layer ----


@pytest.mark.parametrize("room_type", ["Hotel room", "Shared room"])
def test_known_but_unsupported_room_type_is_schema_valid(registry: Registry, room_type: str):
    """Schema-valid on purpose -- rejection happens at the semantic layer
    (UNSUPPORTED_ROOM_TYPE), not as a generic schema error."""
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["room_types"] = [room_type]
    assert not list(validator.iter_errors(instance))


def test_genuinely_unknown_room_type_string_is_ALSO_schema_valid(registry: Registry):
    """Checkpoint C.0.1 §4 correction: room_types is no longer a closed
    enum, so an unrecognized string is schema-valid too -- it must reach
    the semantic validator (services/travel-mcp/phase2/serving_contract_reference.py::resolve_effective_room_types)
    and fail there as UnsupportedRoomTypeError, never as a generic schema
    error that would hide the structured supported/unsupported detail."""
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["room_types"] = ["Yurt"]
    assert not list(validator.iter_errors(instance))

    from phase2.serving_contract_reference import UnsupportedRoomTypeError, resolve_effective_room_types

    with pytest.raises(UnsupportedRoomTypeError):
        resolve_effective_room_types(instance["room_types"])


def test_room_types_string_length_is_bounded(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["room_types"] = ["x" * 65]
    assert list(validator.iter_errors(instance))


def test_room_types_list_size_is_bounded(registry: Registry):
    validator = _validator("SearchStaysRequest", registry)
    instance = copy.deepcopy(_valid("SearchStaysRequest"))
    instance["room_types"] = [f"type_{i}" for i in range(9)]
    assert list(validator.iter_errors(instance))


# ---------------------------------------------------------------------------
# Success fields required (Checkpoint C.0.1 §9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "required_field",
    [
        "snapshot_date", "dataset_sha256", "bundle_sha256", "model_version", "currency",
        "data_mode", "disclaimer", "predictions_produced", "ranking_basis", "stays",
    ],
)
def test_search_stays_result_requires_success_fields(registry: Registry, required_field: str):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    del instance[required_field]
    assert list(validator.iter_errors(instance)), f"{required_field} should be required"


def test_currency_must_be_exactly_try(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["currency"] = "USD"
    assert list(validator.iter_errors(instance))


def test_data_mode_must_be_exactly_historical(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["data_mode"] = "live"
    assert list(validator.iter_errors(instance))


def test_predictions_produced_must_be_exactly_true(registry: Registry):
    """Checkpoint C.0.1 §3/§9: a success with predictions_produced=false can
    no longer be represented -- that state is now NO_MATCHING_STAYS."""
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["predictions_produced"] = False
    assert list(validator.iter_errors(instance))


def test_ranking_basis_must_be_exactly_preliminary(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    for bad_value in ("deal_score", "fair_price", "deal_score_desc"):
        instance = copy.deepcopy(_valid("SearchStaysResult"))
        instance["ranking_basis"] = bad_value
        assert list(validator.iter_errors(instance)), bad_value


def test_search_stays_result_item_requires_availability_status_unknown(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    del instance["stays"][0]["stay"]["availability_status"]
    assert list(validator.iter_errors(instance))


def test_availability_status_cannot_be_anything_other_than_unknown(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["stays"][0]["stay"]["availability_status"] = "available"
    assert list(validator.iter_errors(instance))


def test_result_item_room_type_is_restricted_to_the_supported_set(registry: Registry):
    """Even at the schema level, a result row cannot claim to be a Hotel
    room or Shared room -- the type system itself enforces the policy."""
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["stays"][0]["stay"]["room_type"] = "Hotel room"
    assert list(validator.iter_errors(instance))


def test_result_item_fair_price_requires_interval(registry: Registry):
    """Checkpoint C.0.1 §9: every returned stay must carry both a
    prediction and a calibrated interval."""
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    del instance["stays"][0]["fair_price"]["interval"]
    assert list(validator.iter_errors(instance))


def test_empty_stays_array_is_no_longer_a_valid_success(registry: Registry):
    """Checkpoint C.0.1 §3: NO_MATCHING_STAYS is the sole no-result
    behavior -- a success can never have zero stays."""
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["stays"] = []
    assert list(validator.iter_errors(instance))


def test_disclaimer_is_truthful_in_the_canonical_valid_example():
    from phase2.serving_contract_reference import assert_disclaimer_is_truthful

    instance = _valid("SearchStaysResult")
    assert_disclaimer_is_truthful(instance["disclaimer"])  # must not raise


# ---------------------------------------------------------------------------
# Nested schema-version pinning: a 1.0 payload cannot masquerade as C.0.1
# ---------------------------------------------------------------------------


def test_nested_stay_must_be_exactly_schema_version_1_1_0(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["stays"][0]["stay"]["schema_version"] = "1.0.0"
    # 1.0.0 also forbids room_type/availability_status/snapshot_date (present here),
    # and the pinning branch independently requires exactly "1.1.0" -- either
    # reason is sufficient, both must fire.
    assert list(validator.iter_errors(instance))


def test_nested_fair_price_must_be_exactly_schema_version_1_1_0(registry: Registry):
    validator = _validator("SearchStaysResult", registry)
    instance = copy.deepcopy(_valid("SearchStaysResult"))
    instance["stays"][0]["fair_price"]["schema_version"] = "1.0.0"
    assert list(validator.iter_errors(instance))


# ---------------------------------------------------------------------------
# FairPriceEstimate scoring_status truth table -- adversarial coverage
# ---------------------------------------------------------------------------


def test_fair_price_estimate_incomplete_state_omits_deal_score(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_incomplete_base()
    assert not list(validator.iter_errors(instance))
    assert "deal_score" not in instance
    assert "components" not in instance
    assert instance["missing_components"] == ["itinerary_accessibility"]


def test_incomplete_with_only_deal_score_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_incomplete_base()
    instance["deal_score"] = 0.5
    assert list(validator.iter_errors(instance))


def test_incomplete_with_only_components_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    complete = _fair_price_complete_base()
    instance = _fair_price_incomplete_base()
    instance["components"] = complete["components"]
    assert list(validator.iter_errors(instance))


def test_incomplete_with_both_deal_score_and_components_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    complete = _fair_price_complete_base()
    instance = _fair_price_incomplete_base()
    instance["deal_score"] = complete["deal_score"]
    instance["components"] = complete["components"]
    assert list(validator.iter_errors(instance))


def test_complete_with_missing_components_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_complete_base()
    instance["schema_version"] = "1.1.0"
    instance["scoring_status"] = "complete"
    del instance["components"]
    assert list(validator.iter_errors(instance))


def test_complete_with_missing_deal_score_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_complete_base()
    instance["schema_version"] = "1.1.0"
    instance["scoring_status"] = "complete"
    del instance["deal_score"]
    assert list(validator.iter_errors(instance))


def test_complete_with_incomplete_only_fields_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_complete_base()
    instance["schema_version"] = "1.1.0"
    instance["scoring_status"] = "complete"
    instance["missing_components"] = ["itinerary_accessibility"]
    instance["incomplete_reason"] = "should not be here"
    assert list(validator.iter_errors(instance))


def test_version_1_0_claiming_incomplete_status_is_rejected(registry: Registry):
    """Checkpoint C.0.1 §2: a 1.0.0 payload cannot set any 1.1-only state
    field, including scoring_status='incomplete'."""
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_complete_base()
    assert instance["schema_version"] == "1.0.0"
    instance["scoring_status"] = "incomplete"
    instance["missing_components"] = ["itinerary_accessibility"]
    instance["incomplete_reason"] = "test"
    del instance["deal_score"]
    del instance["components"]
    assert list(validator.iter_errors(instance))


def test_version_1_0_setting_scoring_status_at_all_is_rejected(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _fair_price_complete_base()
    instance["scoring_status"] = "complete"
    assert list(validator.iter_errors(instance))


def test_fair_price_estimate_component_weights_match_deal_score_reference():
    from phase2.deal_score_reference import DEAL_SCORE_WEIGHTS

    schema = _load(os.path.join(CONTRACTS_DIR, "FairPriceEstimate.schema.json"))
    component_keys = set(schema["properties"]["components"]["required"])
    assert component_keys == set(DEAL_SCORE_WEIGHTS.keys())
    assert abs(sum(DEAL_SCORE_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Backward compatibility: every pre-existing Phase 1 fixture/example
# referencing the migrated schemas still validates unchanged.
# ---------------------------------------------------------------------------


def test_original_1_0_0_stay_option_example_still_validates(registry: Registry):
    validator = _validator("StayOption", registry)
    instance = _valid("StayOption")
    assert instance["schema_version"] == "1.0.0"
    assert "room_type" not in instance  # unchanged, never backfilled
    assert not list(validator.iter_errors(instance))


def test_original_1_0_0_fair_price_estimate_example_still_validates(registry: Registry):
    validator = _validator("FairPriceEstimate", registry)
    instance = _valid("FairPriceEstimate")
    assert instance["schema_version"] == "1.0.0"
    assert "scoring_status" not in instance  # unchanged, never backfilled
    assert not list(validator.iter_errors(instance))


def test_pre_existing_stays_fixtures_still_conform_to_provider_envelope(registry: Registry):
    validator = _validator("ProviderResponseEnvelope", registry)
    for name in ("stays_success", "stays_partial_result"):
        instance = _load(os.path.join(CONTRACTS_DIR, "fixtures", f"{name}.json"))
        assert not list(validator.iter_errors(instance)), name


# ---------------------------------------------------------------------------
# Error envelopes: no stack traces, no local paths, bounded structured details
# ---------------------------------------------------------------------------


_FORBIDDEN_PATTERNS = [
    re.compile(r"[Cc]:\\\\Users"),
    re.compile(r"Traceback"),
    re.compile(r"\.py\", line \d+"),
    re.compile(r"(?i)api[_-]?key"),
    re.compile(r"(?i)secret"),
]

_NEW_ERROR_EXAMPLES = [
    os.path.join(CONTRACTS_DIR, "examples", "valid", "ErrorEnvelope_unsupported_room_type.json"),
    os.path.join(CONTRACTS_DIR, "examples", "valid", "ErrorEnvelope_no_matching_stays.json"),
    os.path.join(CONTRACTS_DIR, "examples", "valid", "ErrorEnvelope_request_too_broad.json"),
    os.path.join(CONTRACTS_DIR, "examples", "valid", "ErrorEnvelope_unknown_district.json"),
]


@pytest.mark.parametrize("path", _NEW_ERROR_EXAMPLES, ids=lambda p: os.path.basename(p))
def test_new_error_examples_validate_and_contain_no_internal_detail(path: str, registry: Registry):
    validator = _validator("ErrorEnvelope", registry)
    instance = _load(path)
    assert not list(validator.iter_errors(instance))
    raw_text = json.dumps(instance)
    for pattern in _FORBIDDEN_PATTERNS:
        assert not pattern.search(raw_text), f"{path} contains forbidden pattern {pattern.pattern}"
    assert instance["retriable"] is False


def test_unsupported_room_type_error_has_structured_machine_readable_details(registry: Registry):
    """The machine-readable support information must not live only in the
    human message string."""
    instance = _load(os.path.join(VALID_DIR, "ErrorEnvelope_unsupported_room_type.json"))
    assert instance["details"]["supported_room_types"] == ["Entire home/apt", "Private room"]
    assert "Hotel room" in instance["details"]["unsupported_room_types"]
    assert "Hotel room" in instance["details"]["requested_room_types"]


def test_error_details_rejects_a_nested_object(registry: Registry):
    validator = _validator("ErrorEnvelope", registry)
    instance = copy.deepcopy(_load(os.path.join(VALID_DIR, "ErrorEnvelope_unsupported_room_type.json")))
    instance["details"]["nested"] = {"a": {"b": "c"}}
    assert list(validator.iter_errors(instance))


def test_error_details_rejects_too_many_properties(registry: Registry):
    validator = _validator("ErrorEnvelope", registry)
    instance = copy.deepcopy(_load(os.path.join(VALID_DIR, "ErrorEnvelope_unsupported_room_type.json")))
    instance["details"] = {f"key_{i}": "value" for i in range(11)}
    assert list(validator.iter_errors(instance))


def test_error_details_rejects_an_overlong_string_value(registry: Registry):
    validator = _validator("ErrorEnvelope", registry)
    instance = copy.deepcopy(_load(os.path.join(VALID_DIR, "ErrorEnvelope_unsupported_room_type.json")))
    instance["details"] = {"message_dup": "x" * 129}
    assert list(validator.iter_errors(instance))


def test_1_0_0_error_envelope_without_details_still_validates(registry: Registry):
    validator = _validator("ErrorEnvelope", registry)
    instance = _valid("ErrorEnvelope")
    assert instance["schema_version"] == "1.0.0"
    assert "details" not in instance
    assert not list(validator.iter_errors(instance))


# ---------------------------------------------------------------------------
# Canonical Istanbul district registry (Checkpoint C.0.1 §5)
# ---------------------------------------------------------------------------


def _registry_instance() -> dict:
    return _load(os.path.join(VALID_DIR, "IstanbulDistrictRegistry.json"))


def test_district_registry_validates_against_its_schema(registry: Registry):
    validator = _validator("IstanbulDistrictRegistry", registry)
    instance = _registry_instance()
    assert not list(validator.iter_errors(instance))


def test_district_registry_has_exactly_39_entries():
    instance = _registry_instance()
    assert len(instance["districts"]) == 39


def test_district_registry_ids_are_unique():
    instance = _registry_instance()
    ids = [d["district_id"] for d in instance["districts"]]
    assert len(ids) == len(set(ids))


def test_district_registry_side_values_are_uppercase_canonical():
    instance = _registry_instance()
    for d in instance["districts"]:
        assert d["side"] in ("EUROPEAN", "ASIAN")


def test_district_registry_matches_side_mapping_py_exactly():
    """Cross-checks against the accepted 39-district side mapping this
    registry was derived from -- one authoritative source, no drift."""
    from phase2.data_pipeline.side_mapping import _DISTRICTS_BY_SIDE, normalize_district_name

    expected: dict[str, str] = {}
    for side, names in _DISTRICTS_BY_SIDE.items():
        for name in names:
            expected[normalize_district_name(name)] = side.upper()

    instance = _registry_instance()
    reg_keys = {d["district_id"][len("district_"):]: d["side"] for d in instance["districts"]}

    # side_mapping.py has 40 lookup keys (Eyup + the Eyupsultan alias for
    # the same real district); the registry correctly has one row per real
    # district (39), so "eyupsultan" is the one expected extra alias key.
    missing = set(expected.keys()) - set(reg_keys.keys()) - {"eyupsultan"}
    assert not missing, f"registry is missing real districts: {missing}"
    assert not (set(reg_keys.keys()) - set(expected.keys())), "registry has districts not in side_mapping.py"

    for key, side in expected.items():
        if key in reg_keys:
            assert reg_keys[key] == side, f"{key}: registry says {reg_keys[key]}, side_mapping.py says {side}"


def test_district_registry_contains_no_raw_row_or_host_data():
    instance = _registry_instance()
    raw_text = json.dumps(instance)
    for forbidden in ("host_id", "review_scores", "listing_url", "neighbourhood_cleansed"):
        assert forbidden not in raw_text


@pytest.mark.parametrize("district_id", [d["district_id"] for d in _registry_instance()["districts"]])
def test_every_registry_district_id_matches_stay_option_pattern(district_id: str, registry: Registry):
    validator = _validator("StayOption", registry)
    schema = _load(os.path.join(CONTRACTS_DIR, "StayOption.schema.json"))
    pattern = schema["properties"]["district_id"]["pattern"]
    assert re.match(pattern, district_id), district_id


def test_district_id_lowercase_side_normalization_is_documented_and_deterministic():
    """StayOption.side stays lowercase (already-accepted Phase 1 contract);
    the registry's own canonical side is uppercase. The one documented,
    deterministic normalization rule is side.lower()."""
    instance = _registry_instance()
    for d in instance["districts"]:
        normalized = d["side"].lower()
        assert normalized in ("european", "asian")
