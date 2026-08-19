"""Final Evaluation Checkpoint E.1V -- builds and freezes the 50-case
independent validation manifest (`evaluation/e1v_cases.json`) BEFORE any
live model call is made. New wording, dates, and evidence combinations
throughout -- no case here is copied or lightly paraphrased from E.1
(`evaluation/e1_live_qwen_cases.json`) or E.1R.

Unlike E.1, every case here has exactly ONE unambiguous expected label
(a single `expected_action`, never a list of acceptable alternatives) --
per this checkpoint's own instruction: "Validate that every expected
label is unambiguous." Cases that are thematically ambiguous/irrelevant
(clarification_required, out_of_scope) are still written so that a
reasonable annotator agrees on the single correct routing outcome, by
keeping the missing-information or off-topic signal unambiguous in the
request itself.

Supervisor cases exercise the REAL, unmodified `phase4.graph.
_make_decide_node` factory end-to-end per case (classification -> decision
-> eligibility gate -> one-shot correction) -- this script never
reimplements or approximates that logic; it only constructs the
`PlannerState` snapshot each case starts from. `travel_search_attempted_
signature`/`istanbul_expert_attempted_signature` are resolved through the
real `phase4.graph._compute_request_signature` function so seeded
"already attempted" markers are byte-exact with what live classification
will independently (re)compute for the same `normalized_request` --
never a hand-computed hash that could silently drift from production
behavior.

Run once:
    python -m evaluation.build_e1v_cases
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "planner-a"))

from phase4.graph import _compute_request_signature  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "e1v_cases.json"

# A fixed "prior turn" placeholder request -- used only to derive a
# deliberately STALE signature (a real, different, earlier turn) for the
# two "follow-up turn" cases, never reused as this case's own request.
_PRIOR_TURN_PLACEHOLDER = {"user_message": "An earlier, now-superseded planning question.", "trip_request": None}


def _sig(user_message: str, trip_request) -> str:
    return _compute_request_signature({"normalized_request": {"user_message": user_message, "trip_request": trip_request}})


_STALE_SIG = _compute_request_signature({"normalized_request": _PRIOR_TURN_PLACEHOLDER})


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


TRIP_EN = {
    "origin": "BEY", "destination": "IST", "depart_date": "2026-10-05", "return_date": "2026-10-12",
    "traveler_count": 2, "budget": {"amount_minor_units": 620000, "currency": "TRY"},
    "preferences": {"interests": ["architecture", "food"], "pace": "relaxed", "language": "en", "mobility_constraints": []},
}
TRIP_TR = dict(TRIP_EN, preferences=dict(TRIP_EN["preferences"], language="tr"), traveler_count=1)
TRIP_AR = dict(TRIP_EN, preferences=dict(TRIP_EN["preferences"], language="ar"), traveler_count=4)


def _sup(
    case_id, language, scope, category, user_message, trip_request,
    observations, tool_call_count, expected_action,
    travel_terminal=None, istanbul_terminal=None, notes="",
):
    """`travel_terminal`/`istanbul_terminal`: None (never attempted this
    turn), "same_turn" (resolved to THIS case's own real request
    signature -- a genuinely completed attempt this turn), or
    "stale_turn" (resolved to `_STALE_SIG` -- a different, earlier turn's
    completed attempt, which must NOT block eligibility for this new
    turn -- Checkpoint E.1S.1's own core fix)."""
    signature = _sig(user_message, trip_request)

    def _resolve(marker):
        if marker is None:
            return None
        if marker == "same_turn":
            return signature
        if marker == "stale_turn":
            return _STALE_SIG
        raise ValueError(marker)

    by_action: dict[str, int] = {}
    for obs in observations:
        by_action[obs["action"]] = by_action.get(obs["action"], 0) + 1

    return dict(
        case_id=case_id, agent_role="supervisor", language=language, expected_capability_scope=scope,
        category=category, user_message=user_message, trip_request=trip_request,
        observations=observations, tool_call_count=tool_call_count, tool_call_count_by_action=by_action,
        expected_action=expected_action,
        travel_search_attempted_signature=_resolve(travel_terminal),
        istanbul_expert_attempted_signature=_resolve(istanbul_terminal),
        request_signature=signature, notes=notes,
    )


def _spec(
    case_id, language, category, user_message, trip_request,
    inherited_observations, specialist_observations, tool_call_count, expected_action, notes="",
):
    all_obs = inherited_observations + specialist_observations
    by_action: dict[str, int] = {}
    for obs in all_obs:
        by_action[obs["action"]] = by_action.get(obs["action"], 0) + 1
    return dict(
        case_id=case_id, agent_role="specialist", language=language, category=category,
        user_message=user_message, trip_request=trip_request,
        inherited_observations=inherited_observations, specialist_observations=specialist_observations,
        tool_call_count=tool_call_count, tool_call_count_by_action=by_action,
        expected_action=expected_action, notes=notes,
    )


# --- 30 supervisor cases: exactly 6 per scope (2 EN / 2 TR / 2 AR each) -------------

SUPERVISOR_CASES = [
    # travel_only (6) -----------------------------------------------------------------
    _sup("V-S01", "en", "travel_only", "initial_missing_evidence",
         "I need a one-way flight from Beirut to Istanbul on 2026-10-05 for one adult -- nothing else.",
         None, [], 0, "call_travel_search"),
    _sup("V-S02", "en", "travel_only", "sufficient_evidence_terminal",
         "The flight and weather look settled -- go ahead and wrap this up.", TRIP_EN,
         [_obs("search_flights", "success"), _obs("get_weather", "success")], 2, "synthesize",
         travel_terminal="same_turn"),
    _sup("V-S03", "tr", "travel_only", "initial_missing_evidence",
         "Istanbul'da 10-15 Eylul arasi 2 kisilik konaklayacak bir yer ariyorum, baska bir sey gerekmiyor.",
         None, [], 0, "call_travel_search"),
    _sup("V-S04", "tr", "travel_only", "degraded_terminal",
         "Ucus bilgisini bulamasan bile devam et, elindekiyle ilerle.", TRIP_TR,
         [_obs("search_flights", "unavailable"), _obs("search_stays", "success")], 2, "synthesize",
         travel_terminal="same_turn", notes="Genuinely degraded (unavailable), not merely claimed -- still terminal for this turn.")
    ,
    _sup("V-S05", "ar", "travel_only", "initial_missing_evidence",
         "أحتاج فقط سعر الفندق وحالة الطقس، لا شيء غير ذلك.",
         None, [], 0, "call_travel_search"),
    _sup("V-S06", "ar", "travel_only", "follow_up_turn_reopens_eligibility",
         "أريد الآن رحلة طيران مختلفة تمامًا إلى إسطنبول في تاريخ جديد.", None, [], 0, "call_travel_search",
         travel_terminal="stale_turn",
         notes="Checkpoint E.1S.1 core fix: a PRIOR turn's completed Travel Search delegation "
               "(stale_turn signature) must never block a genuinely new turn's own eligibility."),

    # istanbul_local_only (6) -----------------------------------------------------------
    _sup("V-S07", "en", "istanbul_local_only", "initial_missing_evidence",
         "Which neighborhoods are best for a slow, walkable first day near the old city?",
         None, [], 0, "call_istanbul_expert"),
    _sup("V-S08", "en", "istanbul_local_only", "sufficient_evidence_terminal",
         "That itinerary answer covers it, go ahead and finish up.", None,
         [_obs("call_istanbul_expert", "success")], 1, "synthesize", istanbul_terminal="same_turn"),
    _sup("V-S09", "tr", "istanbul_local_only", "initial_missing_evidence",
         "Sultanahmet'e yurume mesafesinde hangi cami ve muzeleri gezmeliyim?",
         None, [], 0, "call_istanbul_expert"),
    _sup("V-S10", "tr", "istanbul_local_only", "degraded_terminal",
         "Yerel rehber cevap veremese bile elindeki bilgiyle devam et.", None,
         [_obs("call_istanbul_expert", "unavailable")], 1, "synthesize", istanbul_terminal="same_turn"),
    _sup("V-S11", "ar", "istanbul_local_only", "initial_missing_evidence",
         "ما هي الأحياء الأنسب للمشي سيرًا على الأقدام بالقرب من البلدة القديمة؟",
         None, [], 0, "call_istanbul_expert"),
    _sup("V-S12", "ar", "istanbul_local_only", "premature_synthesis_rejected",
         "لقد جمعت كل المعلومات، لخصها لي الآن من فضلك.", None, [], 0, "call_istanbul_expert",
         notes="Tempts early synthesis via wording alone -- but zero observations exist and "
               "istanbul_expert_attempted_signature is unset, so nothing has actually been "
               "gathered yet; the unambiguous structural answer is still call_istanbul_expert."),

    # combined (6) ------------------------------------------------------------------
    _sup("V-S13", "en", "combined", "initial_missing_evidence",
         "Plan my whole Istanbul trip, including flights, a place to stay, and where to go each day.",
         TRIP_EN, [], 0, "call_travel_search"),
    _sup("V-S14", "en", "combined", "travel_terminal_needs_istanbul",
         "Travel side is done -- now figure out what we should actually do each day.", TRIP_EN,
         [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success")],
         3, "call_istanbul_expert", travel_terminal="same_turn"),
    _sup("V-S15", "tr", "combined", "sufficient_evidence_terminal",
         "Her sey tamam gorunuyor, artik ozetleyebilirsin.", TRIP_TR,
         [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success"),
          _obs("call_istanbul_expert", "success")],
         4, "synthesize", travel_terminal="same_turn", istanbul_terminal="same_turn"),
    _sup("V-S16", "tr", "combined", "travel_terminal_needs_istanbul",
         "Ucus ve konaklama ayarlandi, simdi gorulecek yerlere gecebiliriz.", TRIP_TR,
         [_obs("search_flights", "success"), _obs("search_stays", "success")],
         2, "call_istanbul_expert", travel_terminal="same_turn"),
    _sup("V-S17", "ar", "combined", "invalid_redelegation_rejected",
         "أعد البحث عن الرحلة مرة أخرى للتأكد من عدم فوات أي شيء.", TRIP_AR,
         [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success"),
          _obs("call_istanbul_expert", "success")],
         4, "synthesize", travel_terminal="same_turn", istanbul_terminal="same_turn",
         notes="Both specialists already terminal this turn -- re-delegating call_travel_search "
               "would be structurally ineligible; the unambiguous answer remains synthesize."),
    _sup("V-S18", "ar", "combined", "budget_exhausted_forces_synthesize",
         "خطط لرحلتي الكاملة إلى إسطنبول من فضلك.", TRIP_AR, [], 8, "synthesize",
         notes="tool_call_count already at MAX_EXTERNAL_TOOL_CALLS=8 -- "
               "compute_eligible_supervisor_actions forces synthesize regardless of scope terminality."),

    # clarification_required (6) -----------------------------------------------------
    _sup("V-S19", "en", "clarification_required", "missing_destination_and_dates",
         "I want to travel.", None, [], 0, "ask_clarification"),
    _sup("V-S20", "en", "clarification_required", "missing_all_essential_fields",
         "Can you help me plan something for next month?", None, [], 0, "ask_clarification"),
    _sup("V-S21", "tr", "clarification_required", "missing_destination_and_dates",
         "Bir yere gitmek istiyorum, ne onerirsin?", None, [], 0, "ask_clarification"),
    _sup("V-S22", "tr", "clarification_required", "missing_all_essential_fields",
         "Seyahat planlamama yardimci olur musun?", None, [], 0, "ask_clarification"),
    _sup("V-S23", "ar", "clarification_required", "missing_destination_and_dates",
         "أريد أن أسافر، ماذا تقترح؟", None, [], 0, "ask_clarification"),
    _sup("V-S24", "ar", "clarification_required", "missing_all_essential_fields",
         "هل يمكنك مساعدتي في التخطيط لشيء ما؟", None, [], 0, "ask_clarification"),

    # out_of_scope (6) ----------------------------------------------------------------
    _sup("V-S25", "en", "out_of_scope", "booking_and_payment_request",
         "Can you book my flight right now and charge my card?", None, [], 0, "degrade"),
    _sup("V-S26", "en", "out_of_scope", "irrelevant_general_knowledge",
         "What's the capital of France?", None, [], 0, "degrade"),
    _sup("V-S27", "tr", "out_of_scope", "booking_and_payment_request",
         "Ucusumu hemen rezerve edip kartimdan odeme yapar misin?", None, [], 0, "degrade"),
    _sup("V-S28", "tr", "out_of_scope", "irrelevant_general_knowledge",
         "Fransa'nin baskenti neresidir?", None, [], 0, "degrade"),
    _sup("V-S29", "ar", "out_of_scope", "booking_and_payment_request",
         "هل يمكنك حجز رحلتي الآن والدفع من بطاقتي؟", None, [], 0, "degrade"),
    _sup("V-S30", "ar", "out_of_scope", "irrelevant_general_knowledge",
         "ما هي عاصمة فرنسا؟", None, [], 0, "degrade"),
]

# --- 20 specialist cases: 7 EN / 7 TR / 6 AR, balanced across 5 categories ----------

SPECIALIST_CASES = [
    # weather (4): EN, EN, TR, AR ------------------------------------------------------
    _spec("V-P01", "en", "weather", "Finish planning my Istanbul trip.", TRIP_EN,
          [_obs("search_flights", "success"), _obs("search_stays", "success")], [], 2, "get_weather"),
    _spec("V-P02", "en", "weather", "What should I pack for the trip?", TRIP_EN,
          [], [_obs("search_flights", "success"), _obs("search_stays", "success")], 2, "get_weather"),
    _spec("V-P03", "tr", "weather", "Istanbul gezimi tamamlamama yardim et.", TRIP_TR,
          [_obs("search_flights", "success"), _obs("search_stays", "success")], [], 2, "get_weather"),
    _spec("V-P04", "ar", "weather", "ساعدني في إنهاء خطة رحلتي إلى إسطنبول.", TRIP_AR,
          [_obs("search_flights", "success"), _obs("search_stays", "success")], [], 2, "get_weather"),

    # flight (4): TR, TR, AR, EN --------------------------------------------------------
    _spec("V-P05", "tr", "flight", "Istanbul gezimi tamamlamama yardim et.", TRIP_TR,
          [_obs("search_stays", "success"), _obs("get_weather", "success")], [], 2, "search_flights"),
    _spec("V-P06", "tr", "flight", "Beyrut'tan tek yon ucus lazim, 2026-10-05 tarihinde.", None,
          [], [], 0, "search_flights"),
    _spec("V-P07", "ar", "flight", "أحتاج رحلة طيران من بيروت إلى إسطنبول في 2026-10-05.", None,
          [], [], 0, "search_flights"),
    _spec("V-P08", "en", "flight", "Finish planning my Istanbul trip.", TRIP_EN,
          [_obs("search_stays", "success"), _obs("get_weather", "success")], [], 2, "search_flights"),

    # stay (4): AR, AR, EN, TR -----------------------------------------------------------
    _spec("V-P09", "ar", "stay", "أحتاج مكان إقامة في إسطنبول من 10 إلى 15 أكتوبر لشخصين.", None,
          [], [], 0, "search_stays"),
    _spec("V-P10", "ar", "stay", "أكمل خطة رحلتي إلى إسطنبول من فضلك.", TRIP_AR,
          [_obs("search_flights", "success"), _obs("get_weather", "success")], [], 2, "search_stays"),
    _spec("V-P11", "en", "stay", "Finish planning my Istanbul trip.", TRIP_EN,
          [_obs("search_flights", "success")], [_obs("get_weather", "success")], 2, "search_stays"),
    _spec("V-P12", "tr", "stay", "Istanbul'da kalacak yer ariyorum, 2 kisilik.", None,
          [], [], 0, "search_stays"),

    # fair_price_prereq (3): EN, TR, AR -- estimate_fair_price needs a real stay first --
    _spec("V-P13", "en", "fair_price_prereq", "Is the place we'd stay at actually a fair price?", TRIP_EN,
          [_obs("search_flights", "success"), _obs("get_weather", "success")], [], 2, "search_stays",
          notes="A fair-price estimate needs a real, already-searched stay listing -- since "
                "search_stays has not run yet this turn, the unambiguous prerequisite next step "
                "is search_stays, never estimate_fair_price (evidence_collected_so_far never "
                "carries a real stay_id the model could price)."),
    _spec("V-P14", "tr", "fair_price_prereq", "Bu fiyat gercekten uygun mu, once kontrol et.", TRIP_TR,
          [_obs("search_flights", "success"), _obs("get_weather", "success")], [], 2, "search_stays",
          notes="Same fair-price-prerequisite reasoning as V-P13."),
    _spec("V-P15", "ar", "fair_price_prereq", "هل السعر عادل فعلاً؟ تحقق أولاً.", TRIP_AR,
          [_obs("search_flights", "success"), _obs("get_weather", "success")], [], 2, "search_stays",
          notes="Same fair-price-prerequisite reasoning as V-P13."),

    # completion (5): TR, AR, EN, TR, EN -- all needed evidence already present ---------
    _spec("V-P16", "tr", "completion", "Istanbul gezimi planla.", TRIP_TR,
          [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success")],
          [], 3, "travel_search_complete"),
    _spec("V-P17", "ar", "completion", "خطط لرحلتي إلى إسطنبول.", TRIP_AR,
          [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success")],
          [], 3, "travel_search_complete"),
    _spec("V-P18", "en", "completion", "Plan my Istanbul trip.", TRIP_EN,
          [], [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success")],
          3, "travel_search_complete"),
    _spec("V-P19", "tr", "completion", "Istanbul gezimi planla.", TRIP_TR,
          [], [_obs("web_search", "success"), _obs("web_search", "success")],
          2, "travel_search_complete",
          notes="web_search already at the shared per-tool cap of 2 for this delegation; no other "
                "evidence category is missing/needed for a web-search-only local-status question."),
    _spec("V-P20", "en", "completion", "Plan my Istanbul trip.", TRIP_EN,
          [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success")],
          [], 8, "travel_search_complete",
          notes="tool_call_count already at the shared MAX_EXTERNAL_TOOL_CALLS=8 ceiling."),
]


def build_manifest() -> dict:
    assert len(SUPERVISOR_CASES) == 30, len(SUPERVISOR_CASES)
    assert len(SPECIALIST_CASES) == 20, len(SPECIALIST_CASES)
    assert len({c["case_id"] for c in SUPERVISOR_CASES + SPECIALIST_CASES}) == 50

    sup_lang = {lang: sum(1 for c in SUPERVISOR_CASES if c["language"] == lang) for lang in ("en", "tr", "ar")}
    spec_lang = {lang: sum(1 for c in SPECIALIST_CASES if c["language"] == lang) for lang in ("en", "tr", "ar")}
    assert sup_lang == {"en": 10, "tr": 10, "ar": 10}, sup_lang
    assert sup_lang["en"] + sup_lang["tr"] + sup_lang["ar"] == 30

    scope_counts = {}
    for c in SUPERVISOR_CASES:
        scope_counts[c["expected_capability_scope"]] = scope_counts.get(c["expected_capability_scope"], 0) + 1
    assert scope_counts == {
        "travel_only": 6, "istanbul_local_only": 6, "combined": 6,
        "clarification_required": 6, "out_of_scope": 6,
    }, scope_counts

    spec_category_counts = {}
    for c in SPECIALIST_CASES:
        spec_category_counts[c["category"]] = spec_category_counts.get(c["category"], 0) + 1

    terminal_expected = {"synthesize", "travel_search_complete"}
    unnecessary_denominator = sum(
        1 for c in SUPERVISOR_CASES + SPECIALIST_CASES if c["expected_action"] in terminal_expected
    )
    assert unnecessary_denominator >= 12, unnecessary_denominator

    return {
        "schema_version": "1.0.0",
        "checkpoint": "Final Evaluation Checkpoint E.1V",
        "description": (
            "Predeclared, frozen independent live-Qwen validation case manifest, built against "
            "the current uncommitted E.1S/E.1S.1 candidate (see evaluation/e1v_candidate_manifest.json). "
            "New wording/dates/evidence combinations throughout -- not copied or paraphrased from "
            "E.1 or E.1R. Every expected_action is a single, unambiguous label, never a list of "
            "acceptable alternatives. Case IDs, languages, states, and expected labels are fixed "
            "at generation time and never edited after seeing a live-model result."
        ),
        "supervisor_case_count": len(SUPERVISOR_CASES),
        "specialist_case_count": len(SPECIALIST_CASES),
        "supervisor_language_distribution": sup_lang,
        "specialist_language_distribution": spec_lang,
        "supervisor_scope_distribution": scope_counts,
        "specialist_category_distribution": spec_category_counts,
        "unnecessary_delegation_denominator": unnecessary_denominator,
        "stale_turn_signature_placeholder": _PRIOR_TURN_PLACEHOLDER,
        "stale_turn_signature_sha256": _STALE_SIG,
        "supervisor_cases": SUPERVISOR_CASES,
        "specialist_cases": SPECIALIST_CASES,
    }


def main() -> None:
    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False)
    OUTPUT_PATH.write_text(text, encoding="utf-8")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {OUTPUT_PATH}")
    print(f"case_set_sha256={checksum}")
    print(f"supervisor_language_distribution={manifest['supervisor_language_distribution']}")
    print(f"specialist_language_distribution={manifest['specialist_language_distribution']}")
    print(f"supervisor_scope_distribution={manifest['supervisor_scope_distribution']}")
    print(f"specialist_category_distribution={manifest['specialist_category_distribution']}")
    print(f"unnecessary_delegation_denominator={manifest['unnecessary_delegation_denominator']}")


if __name__ == "__main__":
    main()
