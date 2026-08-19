"""Final Evaluation Checkpoint E.1R §3 -- builds and freezes the
INDEPENDENT 50-case live-Qwen holdout (`evaluation/e1r_holdout_cases.json`)
BEFORE the supervisor prompt is touched. New wording, dates, evidence
combinations, and state layouts throughout -- no case text is duplicated
from `evaluation/e1_live_qwen_cases.json` (the original, permanent E.1
baseline, never modified by this script).

Refuses to overwrite an existing frozen holdout file unless invoked with
the narrowly-scoped `--regenerate` flag, so an accidental re-run cannot
silently replace already-frozen, checksummed test evidence.

Run once:
    python -m evaluation.build_e1r_holdout_cases
    python -m evaluation.build_e1r_holdout_cases --regenerate   # explicit, intentional only
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "e1r_holdout_cases.json"

# --- new trip fixtures: different dates, travelers, and budget from the
# original E.1 baseline (2026-09-10/15, 2-3 travelers) -- never reused. ---

TRIP_H_EN = {
    "origin": "BEY", "destination": "IST", "depart_date": "2026-10-05", "return_date": "2026-10-12",
    "traveler_count": 1, "budget": {"amount_minor_units": 800000, "currency": "TRY"},
    "preferences": {"interests": ["food", "architecture"], "pace": "relaxed", "language": "en", "mobility_constraints": []},
}
TRIP_H_TR = dict(TRIP_H_EN, preferences=dict(TRIP_H_EN["preferences"], language="tr"))
TRIP_H_AR = dict(TRIP_H_EN, preferences=dict(TRIP_H_EN["preferences"], language="ar"), traveler_count=2)


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


SUPERVISOR_CASES = [
    # --- category 1: travel-only, missing evidence -> delegation expected
    dict(case_id="H-S01", language="en", category="travel_only_missing_evidence",
         user_message="Book a one-way flight from Beirut to Istanbul departing 2026-10-05.",
         trip_request=None, observations=[], tool_call_count=0, expected_actions=["call_travel_search"]),
    dict(case_id="H-S02", language="tr", category="travel_only_missing_evidence",
         user_message="5-12 Ekim 2026 tarihleri icin Istanbul'da kalacak yer ariyorum.",
         trip_request=None, observations=[], tool_call_count=0, expected_actions=["call_travel_search"]),
    dict(case_id="H-S03", language="ar", category="travel_only_missing_evidence",
         user_message="ما هو الطقس المتوقع في إسطنبول في 5 أكتوبر 2026؟",
         trip_request=None, observations=[], tool_call_count=0, expected_actions=["call_travel_search"]),

    # --- category 2: Istanbul-only, no travel evidence needed at all -> no delegation
    dict(case_id="H-S04", language="en", category="istanbul_only_no_evidence_needed",
         user_message="Which neighborhoods are best for a first-time visitor interested in architecture?",
         trip_request=None, observations=[], tool_call_count=0, expected_actions=["call_istanbul_expert"]),
    dict(case_id="H-S05", language="tr", category="istanbul_only_no_evidence_needed",
         user_message="Sultanahmet cevresinde hangi camileri ve muzeleri gezmeliyim?",
         trip_request=None, observations=[], tool_call_count=0, expected_actions=["call_istanbul_expert"]),
    dict(case_id="H-S06", language="ar", category="istanbul_only_no_evidence_needed",
         user_message="ما هي أفضل المطاعم التقليدية التي يجب أن أزورها في إسطنبول؟",
         trip_request=None, observations=[], tool_call_count=0, expected_actions=["call_istanbul_expert"]),

    # --- category 3: combined travel+Istanbul, NOTHING gathered yet -> delegation expected
    dict(case_id="H-S07", language="en", category="combined_missing_evidence",
         user_message="Plan my whole Istanbul trip, including where I should stay and what to see.",
         trip_request=TRIP_H_EN, observations=[], tool_call_count=0, expected_actions=["call_travel_search"]),
    dict(case_id="H-S08", language="tr", category="combined_missing_evidence",
         user_message="Tum Istanbul gezimi planla, kalacagim yer ve gorulecek yerler dahil.",
         trip_request=TRIP_H_TR, observations=[], tool_call_count=0, expected_actions=["call_travel_search"]),
    dict(case_id="H-S09", language="ar", category="combined_missing_evidence",
         user_message="خطط رحلتي بالكامل إلى إسطنبول، بما في ذلك مكان الإقامة والأماكن التي يجب زيارتها.",
         trip_request=TRIP_H_AR, observations=[], tool_call_count=0, expected_actions=["call_travel_search"]),

    # --- category 4: combined, travel evidence FULLY complete -> call_istanbul_expert, no delegation
    dict(case_id="H-S10", language="en", category="combined_sufficient_evidence",
         user_message="Continue planning my Istanbul trip.", trip_request=TRIP_H_EN,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         tool_call_count=3, expected_actions=["call_istanbul_expert"]),
    dict(case_id="H-S11", language="tr", category="combined_sufficient_evidence",
         user_message="Istanbul gezi planima devam et.", trip_request=TRIP_H_TR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         tool_call_count=3, expected_actions=["call_istanbul_expert"]),
    dict(case_id="H-S12", language="ar", category="combined_sufficient_evidence",
         user_message="تابع تخطيط رحلتي إلى إسطنبول.", trip_request=TRIP_H_AR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         tool_call_count=3, expected_actions=["call_istanbul_expert"]),

    # --- category 5: completed result (Istanbul Expert already done too) -> synthesize, no delegation
    dict(case_id="H-S13", language="en", category="completed_result",
         user_message="Is everything ready?", trip_request=TRIP_H_EN,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"),
                       _obs("search_stays", "success"), _obs("call_istanbul_expert", "success")],
         tool_call_count=4, expected_actions=["synthesize"]),
    dict(case_id="H-S14", language="tr", category="completed_result",
         user_message="Her sey tamam mi?", trip_request=TRIP_H_TR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"),
                       _obs("search_stays", "success"), _obs("call_istanbul_expert", "success")],
         tool_call_count=4, expected_actions=["synthesize"]),
    dict(case_id="H-S15", language="ar", category="completed_result",
         user_message="هل كل شيء جاهز؟", trip_request=TRIP_H_AR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"),
                       _obs("search_stays", "success"), _obs("call_istanbul_expert", "success")],
         tool_call_count=4, expected_actions=["synthesize"]),

    # --- category 6: partial result -- one attempted-and-failed observation,
    # the other two travel facts genuinely settled -> call_istanbul_expert, no delegation
    dict(case_id="H-S16", language="en", category="partial_result_failed_evidence_still_proceed",
         user_message="Continue even though the weather check failed.", trip_request=TRIP_H_EN,
         observations=[_obs("get_weather", "unavailable"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         tool_call_count=3, expected_actions=["call_istanbul_expert"]),
    dict(case_id="H-S17", language="tr", category="partial_result_failed_evidence_still_proceed",
         user_message="Ucus bilgisi alinamasa da devam et.", trip_request=TRIP_H_TR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "unavailable"), _obs("search_stays", "success")],
         tool_call_count=3, expected_actions=["call_istanbul_expert"]),
    dict(case_id="H-S18", language="ar", category="partial_result_failed_evidence_still_proceed",
         user_message="تابع حتى لو تعذر العثور على مكان إقامة.", trip_request=TRIP_H_AR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "unavailable")],
         tool_call_count=3, expected_actions=["call_istanbul_expert"]),

    # --- category 7: time-bounded / shared budget already exhausted -> synthesize, no delegation
    dict(case_id="H-S19", language="en", category="time_bounded_budget_exhausted",
         user_message="Please finish planning my trip.", trip_request=TRIP_H_EN,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success"),
                       _obs("call_istanbul_expert", "success"), _obs("call_istanbul_expert", "success")],
         tool_call_count=8, expected_actions=["synthesize"]),
    dict(case_id="H-S20", language="tr", category="time_bounded_budget_exhausted",
         user_message="Gezi planimi lutfen tamamla.", trip_request=TRIP_H_TR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success"),
                       _obs("call_istanbul_expert", "success"), _obs("call_istanbul_expert", "success")],
         tool_call_count=8, expected_actions=["synthesize"]),
    dict(case_id="H-S21", language="ar", category="time_bounded_budget_exhausted",
         user_message="أكمل خطة رحلتي من فضلك.", trip_request=TRIP_H_AR,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success"),
                       _obs("call_istanbul_expert", "success"), _obs("call_istanbul_expert", "success")],
         tool_call_count=8, expected_actions=["synthesize"]),

    # --- category 8: missing required fields -> ask_clarification, no delegation
    dict(case_id="H-S22", language="en", category="missing_required_fields",
         user_message="I want to travel sometime soon.", trip_request=None, observations=[],
         tool_call_count=0, expected_actions=["ask_clarification"]),
    dict(case_id="H-S23", language="tr", category="missing_required_fields",
         user_message="Bir yere gitmek istiyorum.", trip_request=None, observations=[],
         tool_call_count=0, expected_actions=["ask_clarification"]),
    dict(case_id="H-S24", language="ar", category="missing_required_fields",
         user_message="أريد السفر في وقت ما.", trip_request=None, observations=[],
         tool_call_count=0, expected_actions=["ask_clarification"]),

    # --- category 9: partial evidence, genuinely still needed for the
    # user's own stated need -> delegation expected (unambiguous, unlike
    # the audited S11/S15 pattern: here the user's OWN message explicitly
    # names the still-missing capability)
    dict(case_id="H-S25", language="en", category="partial_evidence_still_delegate",
         user_message="I still need a place to stay for my trip.", trip_request=TRIP_H_EN,
         observations=[_obs("get_weather", "success"), _obs("search_flights", "success")],
         tool_call_count=2, expected_actions=["call_travel_search"]),
    dict(case_id="H-S26", language="tr", category="partial_evidence_still_delegate",
         user_message="Ucus bilgisine hala ihtiyacim var.", trip_request=TRIP_H_TR,
         observations=[_obs("get_weather", "success"), _obs("search_stays", "success")],
         tool_call_count=2, expected_actions=["call_travel_search"]),
    dict(case_id="H-S27", language="ar", category="partial_evidence_still_delegate",
         user_message="ما زلت بحاجة إلى معرفة حالة الطقس لرحلتي.", trip_request=TRIP_H_AR,
         observations=[_obs("search_flights", "success"), _obs("search_stays", "success")],
         tool_call_count=2, expected_actions=["call_travel_search"]),

    # --- category 10: non-Istanbul / irrelevant -> no delegation (structural-immunity style)
    dict(case_id="H-S28", language="en", category="non_istanbul_or_irrelevant",
         user_message="Can you recommend a good ski resort in Switzerland?", trip_request=None, observations=[],
         tool_call_count=0, expected_actions=["degrade", "ask_clarification"]),
    dict(case_id="H-S29", language="tr", category="non_istanbul_or_irrelevant",
         user_message="Bugun hava durumu nasil, genel olarak soruyorum.", trip_request=None, observations=[],
         tool_call_count=0, expected_actions=["ask_clarification", "degrade"],
         notes="Deliberately underspecified (no location at all) -- tests clarification rather than an assumed Istanbul default."),
    dict(case_id="H-S30", language="ar", category="non_istanbul_or_irrelevant",
         user_message="من هو رئيس وزراء اليابان الحالي؟", trip_request=None, observations=[],
         tool_call_count=0, expected_actions=["degrade", "ask_clarification"]),
]

SPECIALIST_CASES = [
    dict(case_id="H-P01", language="en", category="needs_search_flights", trip_request=TRIP_H_EN,
         user_message="Continue planning my Istanbul trip.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["search_flights"]),
    dict(case_id="H-P02", language="tr", category="needs_search_stays", trip_request=TRIP_H_TR,
         user_message="Istanbul gezi planima devam et.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_flights", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["search_stays"]),
    dict(case_id="H-P03", language="ar", category="needs_get_weather", trip_request=TRIP_H_AR,
         user_message="تابع تخطيط رحلتي إلى إسطنبول.",
         inherited_observations=[_obs("search_flights", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["get_weather"]),
    dict(case_id="H-P04", language="en", category="needs_web_search", trip_request=None,
         user_message="What is the current status of the Istanbul tram line near Kabatas?",
         inherited_observations=[], specialist_observations=[], tool_call_count=0, expected_actions=["web_search"]),
    dict(case_id="H-P05", language="tr", category="needs_web_search", trip_request=None,
         user_message="Galata Kulesi'nin guncel ziyaret saatleri nedir?",
         inherited_observations=[], specialist_observations=[], tool_call_count=0, expected_actions=["web_search"]),
    dict(case_id="H-P06", language="ar", category="needs_estimate_fair_price", trip_request=TRIP_H_AR,
         user_message="تابع تخطيط رحلتي إلى إسطنبول.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=3, expected_actions=["estimate_fair_price", "travel_search_complete"],
         notes="Same predeclared asymmetry as the original baseline's P03 -- evidence_collected_so_far never carries a real stay_id."),
    dict(case_id="H-P07", language="en", category="completion_no_additional_call", trip_request=TRIP_H_EN,
         user_message="Continue planning my Istanbul trip.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=3, expected_actions=["travel_search_complete"]),
    dict(case_id="H-P08", language="tr", category="completion_no_additional_call", trip_request=TRIP_H_TR,
         user_message="Istanbul gezi planima devam et.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=3, expected_actions=["travel_search_complete"]),
    dict(case_id="H-P09", language="en", category="provider_failure_terminal", trip_request=TRIP_H_EN,
         user_message="Continue planning my Istanbul trip.",
         inherited_observations=[_obs("search_flights", "unavailable"), _obs("get_weather", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["search_stays"],
         notes="unavailable (not rate_limited) is used deliberately -- a terminal-shaped failure, not a transient one, so 'move on' is the single unambiguous answer (avoids the original baseline's P18 rate_limited ambiguity)."),
    dict(case_id="H-P10", language="tr", category="provider_failure_terminal", trip_request=TRIP_H_TR,
         user_message="Istanbul gezi planima devam et.",
         inherited_observations=[_obs("search_stays", "unavailable"), _obs("get_weather", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["search_flights"]),
    dict(case_id="H-P11", language="ar", category="duplicate_fingerprint", trip_request=TRIP_H_AR,
         user_message="تابع تخطيط رحلتي إلى إسطنبول.",
         inherited_observations=[_obs("get_weather", "success")],
         specialist_observations=[], tool_call_count=1, expected_actions=["search_flights", "search_stays"],
         notes="get_weather already has evidence -- correct behavior is to move on, never repeat it."),
    dict(case_id="H-P12", language="en", category="per_tool_limit", trip_request=None,
         user_message="What's the current status of the Bosphorus ferry crossing near Uskudar?",
         inherited_observations=[],
         specialist_observations=[_obs("web_search", "success"), _obs("web_search", "success")],
         tool_call_count=2, expected_actions=["travel_search_complete"]),
    dict(case_id="H-P13", language="tr", category="global_budget_nearly_exhausted", trip_request=TRIP_H_TR,
         user_message="Istanbul gezi planima devam et.",
         inherited_observations=[_obs("get_weather", "success")],
         specialist_observations=[], tool_call_count=7, expected_actions=["search_flights", "search_stays", "travel_search_complete"]),
    dict(case_id="H-P14", language="ar", category="deadline_or_budget_exhausted", trip_request=TRIP_H_AR,
         user_message="تابع تخطيط رحلتي إلى إسطنبول.",
         inherited_observations=[], specialist_observations=[], tool_call_count=8, expected_actions=["travel_search_complete"]),
    dict(case_id="H-P15", language="en", category="needs_search_flights", trip_request=TRIP_H_EN,
         user_message="I still need to finalize my Istanbul travel plans.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["search_flights"]),
    dict(case_id="H-P16", language="tr", category="needs_search_stays", trip_request=TRIP_H_TR,
         user_message="Istanbul seyahat planlarimi hala tamamlamam gerekiyor.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_flights", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["search_stays"]),
    dict(case_id="H-P17", language="en", category="needs_get_weather", trip_request=TRIP_H_EN,
         user_message="I still need to finalize my Istanbul travel plans.",
         inherited_observations=[_obs("search_flights", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=2, expected_actions=["get_weather"]),
    dict(case_id="H-P18", language="ar", category="multiple_evidence_present", trip_request=TRIP_H_AR,
         user_message="تابع تخطيط رحلتي إلى إسطنبول.",
         inherited_observations=[],
         specialist_observations=[_obs("get_weather", "success"), _obs("search_stays", "success")],
         tool_call_count=2, expected_actions=["search_flights"]),
    dict(case_id="H-P19", language="tr", category="multiple_evidence_present", trip_request=TRIP_H_TR,
         user_message="Istanbul gezi planima devam et.",
         inherited_observations=[_obs("get_weather", "success")],
         specialist_observations=[_obs("search_flights", "success")],
         tool_call_count=2, expected_actions=["search_stays"]),
    dict(case_id="H-P20", language="ar", category="completion_no_additional_call", trip_request=TRIP_H_AR,
         user_message="ما زلت بحاجة إلى إنهاء خطط سفري إلى إسطنبول.",
         inherited_observations=[_obs("get_weather", "success"), _obs("search_flights", "success"), _obs("search_stays", "success")],
         specialist_observations=[], tool_call_count=3, expected_actions=["travel_search_complete"]),
]


def build_manifest() -> dict:
    for case in SUPERVISOR_CASES:
        case.setdefault("notes", "")
        case["agent_role"] = "supervisor"
    for case in SPECIALIST_CASES:
        case.setdefault("notes", "")
        case["agent_role"] = "specialist"

    assert len(SUPERVISOR_CASES) == 30, len(SUPERVISOR_CASES)
    assert len(SPECIALIST_CASES) == 20, len(SPECIALIST_CASES)
    assert len({c["case_id"] for c in SUPERVISOR_CASES + SPECIALIST_CASES}) == 50

    original_case_ids = set()
    try:
        original = json.loads((Path(__file__).parent / "e1_live_qwen_cases.json").read_text(encoding="utf-8"))
        original_case_ids = {c["case_id"] for c in original["supervisor_cases"] + original["specialist_cases"]}
    except FileNotFoundError:
        pass
    holdout_ids = {c["case_id"] for c in SUPERVISOR_CASES + SPECIALIST_CASES}
    assert holdout_ids.isdisjoint(original_case_ids), "holdout case IDs must never collide with the original E.1 baseline"

    supervisor_lang_counts = {lang: sum(1 for c in SUPERVISOR_CASES if c["language"] == lang) for lang in ("en", "tr", "ar")}
    specialist_lang_counts = {lang: sum(1 for c in SPECIALIST_CASES if c["language"] == lang) for lang in ("en", "tr", "ar")}
    assert supervisor_lang_counts == {"en": 10, "tr": 10, "ar": 10}, supervisor_lang_counts
    assert specialist_lang_counts == {"en": 7, "tr": 7, "ar": 6}, specialist_lang_counts

    no_delegation_cases = [c for c in SUPERVISOR_CASES if "call_travel_search" not in c["expected_actions"]]
    assert len(no_delegation_cases) >= 12, f"need >=12 no-delegation-expected supervisor cases, got {len(no_delegation_cases)}"

    category_counts = {}
    for c in SUPERVISOR_CASES + SPECIALIST_CASES:
        category_counts[c["category"]] = category_counts.get(c["category"], 0) + 1

    return {
        "schema_version": "1.0.0",
        "checkpoint": "Final Evaluation Checkpoint E.1R",
        "description": (
            "Independent, frozen live-Qwen holdout -- new wording, dates, evidence "
            "combinations, and state layouts throughout; disjoint case IDs from the "
            "original evaluation/e1_live_qwen_cases.json baseline. Frozen BEFORE the "
            "supervisor prompt correction, so it remains independent test evidence."
        ),
        "supervisor_case_count": len(SUPERVISOR_CASES),
        "specialist_case_count": len(SPECIALIST_CASES),
        "supervisor_language_distribution": supervisor_lang_counts,
        "specialist_language_distribution": specialist_lang_counts,
        "supervisor_no_delegation_expected_count": len(no_delegation_cases),
        "supervisor_no_delegation_expected_case_ids": [c["case_id"] for c in no_delegation_cases],
        "category_distribution": category_counts,
        "scoring_rules": {
            "correct": "actual_action in expected_actions AND final_schema_valid is true",
            "first_attempt_schema_valid": "identical definition to the original E.1 baseline (see evaluation/e1_live_qwen_cases.json).",
            "final_schema_valid": "identical bounded-repair definition to the original E.1 baseline.",
            "unnecessary_action": "identical definition to the original E.1 baseline.",
        },
        "failure_policy": "Identical to the original E.1 baseline -- bounded repair, no infinite retry on transport error, never silently shrinking the denominator.",
        "thresholds": {
            "live_supervisor_routing_accuracy_min": 0.90,
            "live_specialist_tool_selection_correctness_min": 0.90,
            "final_decision_schema_validity_min": 1.00,
            "unnecessary_delegation_or_tool_call_rate_max": 0.10,
            "credential_or_chain_of_thought_leakage_max": 0,
        },
        "supervisor_cases": SUPERVISOR_CASES,
        "specialist_cases": SPECIALIST_CASES,
    }


def main() -> None:
    regenerate = "--regenerate" in sys.argv
    if OUTPUT_PATH.exists() and not regenerate:
        print(f"REFUSING to overwrite already-frozen holdout at {OUTPUT_PATH}.")
        print("Pass --regenerate explicitly if you intend to replace frozen, checksummed test evidence.")
        raise SystemExit(1)

    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {OUTPUT_PATH}")
    print(f"holdout_case_set_sha256={checksum}")
    print(f"supervisor_language_distribution={manifest['supervisor_language_distribution']}")
    print(f"specialist_language_distribution={manifest['specialist_language_distribution']}")
    print(f"supervisor_no_delegation_expected_count={manifest['supervisor_no_delegation_expected_count']}")
    print(f"category_distribution={manifest['category_distribution']}")


if __name__ == "__main__":
    main()
