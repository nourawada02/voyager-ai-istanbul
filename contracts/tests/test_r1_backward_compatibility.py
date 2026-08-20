"""RAG-FIRST SYSTEM B R.1 FINAL USER-TEST CLOSURE requirement 5:
backward-compatibility tests for the additive LocalItinerary/RagCandidate
contract extensions made across the R.1 checkpoint and its correction
pass (candidate_provenance, uncovered_interests, accessibility_evaluation,
and candidate_provenance's own additive display_name/matched_interests/
source_id/chunk_id/retrieval_query/retrieval_score fields).

Proves two things a real consumer depends on:
1. A LocalItinerary payload with NONE of the R.1 fields at all (the exact
   shape System B produced before this checkpoint existed) still
   validates -- every new field is genuinely optional, never required.
2. A candidate_provenance entry for a catalog-fallback candidate (only
   poi_id/candidate_origin, no retrieval provenance) still validates on
   its own -- the extended shape never became implicitly required.
"""

from __future__ import annotations

import json
import os

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

CONTRACTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _registry() -> Registry:
    import glob

    resources = []
    for path in glob.glob(os.path.join(CONTRACTS_DIR, "*.schema.json")):
        schema = _load(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _local_itinerary_validator() -> Draft202012Validator:
    schema = _load(os.path.join(CONTRACTS_DIR, "LocalItinerary.schema.json"))
    return Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())


# The exact LocalItinerary shape System B produced before RAG-FIRST
# SYSTEM B CHECKPOINT R.1 existed at all -- no candidate_provenance, no
# uncovered_interests, no accessibility_evaluation.
_PRE_R1_ITINERARY = {
    "schema_version": "1.0.0",
    "session_id": "11111111-1111-4111-8111-111111111111",
    "trace_id": "22222222-2222-4222-8222-222222222222",
    "contract_version": "1.0.0",
    "recommended_base_candidate_id": "stay_001",
    "accessibility_scores": [{"candidate_id": "stay_001", "score": 0.87}],
    "selected_poi_ids": ["poi_ayasofya"],
    "daily_plans": [
        {
            "schema_version": "1.0.0", "date": "2026-09-10", "side": "european",
            "poi_ids": ["poi_ayasofya"], "legs": [], "walking_minutes": 0, "transfer_minutes": 0,
            "activity_minutes": 90, "meal_minutes": 0, "slack_minutes": 0, "warnings": [],
        }
    ],
    "estimated_travel_minutes": 0, "expected_walking_minutes": 0, "side_crossings": 0,
    "citations": [], "assumptions": [], "warnings": [],
    "data_quality": {"schema_version": "1.0.0", "completeness": 0.6, "freshness": "fixture", "assumptions": []},
    "hard_constraint_validation_passed": True,
}


def test_pre_r1_local_itinerary_payload_with_no_r1_fields_still_validates():
    validator = _local_itinerary_validator()
    errors = list(validator.iter_errors(_PRE_R1_ITINERARY))
    assert not errors, [e.message for e in errors]


def test_candidate_provenance_catalog_fallback_entry_valid_without_retrieval_fields():
    """A catalog_fallback entry (just poi_id + candidate_origin, none of
    the retrieval-provenance fields a 'rag'-origin entry carries) is a
    complete, valid candidate_provenance array entry on its own."""
    payload = {**_PRE_R1_ITINERARY, "candidate_provenance": [{"poi_id": "poi_ayasofya", "candidate_origin": "catalog_fallback"}]}
    validator = _local_itinerary_validator()
    errors = list(validator.iter_errors(payload))
    assert not errors, [e.message for e in errors]


def test_candidate_provenance_rag_entry_with_full_extended_fields_validates():
    payload = {
        **_PRE_R1_ITINERARY,
        "candidate_provenance": [
            {
                "poi_id": "poi_ayasofya", "candidate_origin": "rag", "display_name": "Hagia Sophia",
                "matched_interests": ["history"], "source_id": "wiki_en_hagia_sophia", "chunk_id": "chunk_1",
                "retrieval_query": "historic sites and history of Istanbul", "retrieval_score": 0.87,
            }
        ],
    }
    validator = _local_itinerary_validator()
    errors = list(validator.iter_errors(payload))
    assert not errors, [e.message for e in errors]


def test_accessibility_evaluation_accepts_the_precise_status_wording():
    """R.1 FINAL USER-TEST CLOSURE requirement 4: the precise
    'evidence_available_not_candidate_verified' status (never
    'partially_satisfied', which risked implying a verified-accessible
    claim this system cannot support) is a valid enum value."""
    payload = {
        **_PRE_R1_ITINERARY,
        "accessibility_evaluation": {
            "requested": True, "status": "evidence_available_not_candidate_verified",
            "evidence_count": 2, "note": "general evidence only, no per-candidate verification",
        },
    }
    validator = _local_itinerary_validator()
    errors = list(validator.iter_errors(payload))
    assert not errors, [e.message for e in errors]


def test_accessibility_evaluation_rejects_the_old_partially_satisfied_value():
    """The old, imprecise 'partially_satisfied' value is deliberately no
    longer accepted -- proves the wording correction actually took
    effect at the schema level, not just in application code."""
    payload = {
        **_PRE_R1_ITINERARY,
        "accessibility_evaluation": {
            "requested": True, "status": "partially_satisfied", "evidence_count": 2, "note": "x",
        },
    }
    validator = _local_itinerary_validator()
    errors = list(validator.iter_errors(payload))
    assert errors
