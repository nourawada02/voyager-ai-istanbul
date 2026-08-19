"""Final Evaluation Checkpoint E.1X -- builds and freezes the 30-case
independent supervisor-only validation manifest (`evaluation/e1x_cases.json`)
BEFORE any live model call is made. Supervisor cases only -- the Travel
Search specialist is unchanged by E.1W and already scored 90% in E.1V, so
it is not re-evaluated here (per this checkpoint's own instruction).

New wording, dates, trip data, and evidence combinations throughout -- no
case here is copied or lightly paraphrased from E.1, E.1R, or E.1V.

Every case predeclares exactly one `expected_capability_scope` and one
`expected_action`, plus (for scoring inherited-vs-explicit continuity
behavior specifically) one `expected_scope_source`
("classified" | "inherited" | "explicit") describing what the E.1W
classifier SHOULD structurally produce given the seeded state -- this is
ground truth the evaluator predeclares, never something read back from a
live response before freezing.

Follow-up cases realistically seed `last_successful_capability_scope`,
prior observations, and prior specialist-attempt signatures resolved via
the REAL `phase4.graph._compute_request_signature` -- "same_turn" means
the signature of THIS case's own request (an already-terminal-this-turn
state); "stale_turn" means a different, earlier turn's signature (a
genuinely reopened turn).

Run once:
    python -m evaluation.build_e1x_cases
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "planner-a"))

from phase4.graph import _compute_request_signature  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "e1x_cases.json"

if OUTPUT_PATH.exists():
    raise SystemExit(
        f"REFUSING TO OVERWRITE: {OUTPUT_PATH} already exists. E.1X is one-shot -- "
        "delete it manually first only if you are certain no live run has used it yet."
    )

_PRIOR_TURN_PLACEHOLDER = {"user_message": "An earlier, now-superseded planning question about a different trip.", "trip_request": None}


def _sig(user_message: str, trip_request) -> str:
    return _compute_request_signature({"normalized_request": {"user_message": user_message, "trip_request": trip_request}})


_STALE_SIG = _compute_request_signature({"normalized_request": _PRIOR_TURN_PLACEHOLDER})


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


TRIP_A = {
    "origin": "BEY", "destination": "IST", "depart_date": "2027-03-14", "return_date": "2027-03-20",
    "traveler_count": 2, "budget": {"amount_minor_units": 550000, "currency": "TRY"},
    "preferences": {"interests": ["architecture"], "pace": "relaxed", "language": "en", "mobility_constraints": []},
}
TRIP_C = dict(TRIP_A, depart_date="2027-04-02", return_date="2027-04-08", traveler_count=1)
TRIP_D = dict(TRIP_A, preferences=dict(TRIP_A["preferences"], interests=["history", "food"]), traveler_count=3)
TRIP_E = dict(TRIP_D, depart_date="2027-05-11", return_date="2027-05-17", preferences=dict(TRIP_D["preferences"], language="tr"))
TRIP_F = dict(TRIP_A, depart_date="2027-06-01", return_date="2027-06-07", preferences=dict(TRIP_A["preferences"], language="tr"))
TRIP_G = dict(TRIP_A, depart_date="2027-07-19", return_date="2027-07-25", preferences=dict(TRIP_A["preferences"], language="ar"), traveler_count=4)
TRIP_H = dict(TRIP_A, depart_date="2027-08-09", return_date="2027-08-16", preferences=dict(TRIP_A["preferences"], language="ar"))


def _case(
    case_id, language, scope, category, user_message, trip_request,
    observations, expected_action, expected_scope_source,
    travel_terminal=None, istanbul_terminal=None, is_followup=False, notes="",
):
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

    # `_case` is only ever used for fresh/initial cases (expected_scope_source
    # == "classified"), which by definition have no previous scope at all --
    # follow-up cases (scope_source "inherited"/"explicit") always go
    # through `_followup_case` below, which states `previous_scope` directly.
    assert expected_scope_source == "classified", "_case is for initial cases only -- use _followup_case"

    return dict(
        case_id=case_id, agent_role="supervisor", language=language, expected_capability_scope=scope,
        category=category, user_message=user_message, trip_request=trip_request,
        observations=observations, tool_call_count=len(observations), tool_call_count_by_action=by_action,
        expected_action=expected_action, expected_scope_source=expected_scope_source,
        last_successful_capability_scope=None,
        travel_search_attempted_signature=_resolve(travel_terminal),
        istanbul_expert_attempted_signature=_resolve(istanbul_terminal),
        request_signature=signature, is_followup=is_followup, notes=notes,
    )


def _followup_case(
    case_id, language, scope, category, user_message, trip_request,
    observations, expected_action, expected_scope_source, previous_scope,
    travel_terminal=None, istanbul_terminal=None, notes="",
):
    """Explicit variant for follow-up cases where `previous_scope` must be
    stated directly (needed for `expected_scope_source == "explicit"`,
    where the previous scope differs from the case's own expected scope)."""
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
        observations=observations, tool_call_count=len(observations), tool_call_count_by_action=by_action,
        expected_action=expected_action, expected_scope_source=expected_scope_source,
        last_successful_capability_scope=previous_scope,
        travel_search_attempted_signature=_resolve(travel_terminal),
        istanbul_expert_attempted_signature=_resolve(istanbul_terminal),
        request_signature=signature, is_followup=True, notes=notes,
    )


CASES = [
    # --- travel_only (6): 2 EN, 2 TR, 2 AR ------------------------------------------
    _case("V2-S01", "en", "travel_only", "initial",
          "Find me a one-way flight and somewhere to stay in Istanbul for next spring -- nothing else for now.",
          None, [], "call_travel_search", "classified"),
    _followup_case("V2-S02", "en", "travel_only", "followup_terminal_completed",
          "Great, that's settled -- let's finalize it.", TRIP_A,
          [_obs("search_flights", "success"), _obs("search_stays", "success")],
          "synthesize", "inherited", previous_scope="travel_only", travel_terminal="same_turn",
          notes="short-continuation (EN) + completed evidence + same-turn repeated delegation must remain excluded from eligibility"),
    _case("V2-S03", "tr", "travel_only", "initial",
          "Mart ayinda Istanbul icin sadece ucus ve otel ariyorum, baska bir sey degil.",
          None, [], "call_travel_search", "classified"),
    _followup_case("V2-S04", "tr", "travel_only", "followup_explicit_change_from_local",
          "Aslinda gezilecek yerleri bos ver, sadece ucus bulmama yardim et.", None,
          [], "call_travel_search", "explicit", previous_scope="istanbul_local_only",
          istanbul_terminal="stale_turn"),
    _followup_case("V2-S05", "ar", "travel_only", "followup_new_turn_reopens_eligibility",
          "أريد الآن معلومات عن رحلة مختلفة تمامًا لشهر ديسمبر.", None,
          [], "call_travel_search", "inherited", previous_scope="travel_only", travel_terminal="stale_turn",
          notes="a prior turn's completed Travel Search must never block this genuinely new turn"),
    _followup_case("V2-S06", "ar", "travel_only", "followup_degraded_terminal",
          "تابع حتى لو تعذر إيجاد رحلة طيران، استخدم ما توفر.", TRIP_C,
          [_obs("search_flights", "unavailable"), _obs("search_stays", "success")],
          "synthesize", "inherited", previous_scope="travel_only", travel_terminal="same_turn"),

    # --- istanbul_local_only (6): 2 EN, 2 TR, 2 AR ----------------------------------
    _case("V2-S07", "en", "istanbul_local_only", "initial",
          "What are some good walking routes through the historic peninsula?",
          None, [], "call_istanbul_expert", "classified"),
    _followup_case("V2-S08", "en", "istanbul_local_only", "followup_terminal_completed",
          "Perfect, that covers it -- let's wrap up.", None,
          [_obs("call_istanbul_expert", "success")], "synthesize", "inherited",
          previous_scope="istanbul_local_only", istanbul_terminal="same_turn"),
    _followup_case("V2-S09", "tr", "istanbul_local_only", "followup_not_terminal",
          "Devam et, bana biraz daha gosterecek yer soyle.", None,
          [], "call_istanbul_expert", "inherited", previous_scope="istanbul_local_only",
          notes="short-continuation (TR)"),
    _followup_case("V2-S10", "tr", "istanbul_local_only", "followup_explicit_change_from_travel",
          "Ucus ve otel konusunu unut, sadece gezilecek yerleri anlat.", None,
          [], "call_istanbul_expert", "explicit", previous_scope="travel_only",
          travel_terminal="stale_turn"),
    _followup_case("V2-S11", "ar", "istanbul_local_only", "followup_failed_terminal",
          "أكمل من فضلك حتى لو تعذر الحصول على إجابة الدليل المحلي.", None,
          [_obs("call_istanbul_expert", "provider_error")], "synthesize", "inherited",
          previous_scope="istanbul_local_only", istanbul_terminal="same_turn",
          notes="short-continuation (AR) + failed (not merely degraded) evidence still terminal"),
    _case("V2-S12", "ar", "istanbul_local_only", "initial",
          "ما هي أفضل الأحياء لتجربة الطعام المحلي سيرًا على الأقدام؟",
          None, [], "call_istanbul_expert", "classified"),

    # --- combined (6): 2 EN, 2 TR, 2 AR ----------------------------------------------
    _case("V2-S13", "en", "combined", "initial",
          "Put together my entire Istanbul trip -- travel arrangements and what to do while I'm there.",
          TRIP_D, [], "call_travel_search", "classified"),
    _followup_case("V2-S14", "en", "combined", "followup_travel_terminal_needs_istanbul",
          "Travel side is locked in -- now let's figure out the days.", TRIP_D,
          [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("get_weather", "success")],
          "call_istanbul_expert", "inherited", previous_scope="combined", travel_terminal="same_turn"),
    _followup_case("V2-S15", "tr", "combined", "followup_both_terminal_partial",
          "Her sey tamam gorunuyor, artik ozetleyebilirsin.", TRIP_E,
          [_obs("search_flights", "success"), _obs("get_weather", "unavailable"),
           _obs("search_stays", "success"), _obs("call_istanbul_expert", "success")],
          "synthesize", "inherited", previous_scope="combined",
          travel_terminal="same_turn", istanbul_terminal="same_turn",
          notes="partial evidence (one degraded sub-action) still terminal"),
    _followup_case("V2-S16", "tr", "combined", "followup_explicit_change_into_combined",
          "Aslinda sadece ucus degil, gezilecek yerleri de ogrenmek istiyorum artik.", TRIP_F,
          [_obs("search_flights", "success"), _obs("search_stays", "success")],
          "call_travel_search", "explicit", previous_scope="travel_only", travel_terminal="stale_turn",
          notes="explicit scope expansion into combined -- this NEW turn's own signature has not yet "
                "earned travel-terminal status even though earlier evidence exists"),
    _followup_case("V2-S17", "ar", "combined", "followup_travel_terminal_needs_istanbul",
          "أكمل من فضلك، الرحلة جاهزة الآن انتقل للمعالم.", TRIP_G,
          [_obs("search_flights", "success"), _obs("search_stays", "success")],
          "call_istanbul_expert", "inherited", previous_scope="combined", travel_terminal="same_turn",
          notes="short-continuation (AR)"),
    _case("V2-S18", "ar", "combined", "initial",
          "أرغب في التخطيط الكامل لرحلتي إلى إسطنبول، من الطيران إلى الأماكن التي يجب زيارتها.",
          TRIP_H, [], "call_travel_search", "classified"),

    # --- clarification_required (6): 2 EN, 2 TR, 2 AR --------------------------------
    _case("V2-S19", "en", "clarification_required", "initial",
          "I'm thinking about a trip but haven't figured out the details yet.",
          None, [], "ask_clarification", "classified"),
    _case("V2-S20", "en", "clarification_required", "initial",
          "Can you help me plan something?", None, [], "ask_clarification", "classified"),
    _case("V2-S21", "tr", "clarification_required", "initial",
          "Bir gezi planlamak istiyorum ama daha ne istedigimi bilmiyorum.",
          None, [], "ask_clarification", "classified"),
    _case("V2-S22", "tr", "clarification_required", "initial",
          "Yardimci olur musun, tam olarak ne yapmak istedigimi bilmiyorum.",
          None, [], "ask_clarification", "classified"),
    _case("V2-S23", "ar", "clarification_required", "initial",
          "أفكر في القيام برحلة لكن لم أحدد التفاصيل بعد.",
          None, [], "ask_clarification", "classified"),
    _case("V2-S24", "ar", "clarification_required", "initial",
          "هل يمكنك مساعدتي؟ لا أعرف بالضبط ماذا أريد.",
          None, [], "ask_clarification", "classified"),

    # --- out_of_scope (6): 2 EN, 2 TR, 2 AR ------------------------------------------
    _case("V2-S25", "en", "out_of_scope", "initial",
          "Can you book my hotel room right now and charge my card immediately?",
          None, [], "degrade", "classified"),
    _case("V2-S26", "en", "out_of_scope", "initial",
          "What's the boiling point of water at sea level?", None, [], "degrade", "classified"),
    _case("V2-S27", "tr", "out_of_scope", "initial",
          "Otelimi hemen rezerve edip kartimdan odeme alir misin?",
          None, [], "degrade", "classified"),
    _case("V2-S28", "tr", "out_of_scope", "initial",
          "Suyun deniz seviyesindeki kaynama noktasi kactir?",
          None, [], "degrade", "classified"),
    _case("V2-S29", "ar", "out_of_scope", "initial",
          "هل يمكنك حجز فندقي الآن والدفع من بطاقتي؟",
          None, [], "degrade", "classified"),
    _case("V2-S30", "ar", "out_of_scope", "initial",
          "ما هي نقطة غليان الماء عند مستوى سطح البحر؟",
          None, [], "degrade", "classified"),
]


def build_manifest() -> dict:
    assert len(CASES) == 30, len(CASES)
    assert len({c["case_id"] for c in CASES}) == 30

    lang_dist = {lang: sum(1 for c in CASES if c["language"] == lang) for lang in ("en", "tr", "ar")}
    assert lang_dist == {"en": 10, "tr": 10, "ar": 10}, lang_dist

    scope_dist: dict[str, int] = {}
    for c in CASES:
        scope_dist[c["expected_capability_scope"]] = scope_dist.get(c["expected_capability_scope"], 0) + 1
    assert scope_dist == {
        "travel_only": 6, "istanbul_local_only": 6, "combined": 6,
        "clarification_required": 6, "out_of_scope": 6,
    }, scope_dist

    per_scope_lang: dict[str, dict[str, int]] = {}
    for c in CASES:
        s = c["expected_capability_scope"]
        per_scope_lang.setdefault(s, {"en": 0, "tr": 0, "ar": 0})
        per_scope_lang[s][c["language"]] += 1
    for s, d in per_scope_lang.items():
        assert d == {"en": 2, "tr": 2, "ar": 2}, (s, d)

    followup_count = sum(1 for c in CASES if c["is_followup"])
    assert followup_count >= 12, followup_count

    action_dist: dict[str, int] = {}
    for c in CASES:
        action_dist[c["expected_action"]] = action_dist.get(c["expected_action"], 0) + 1

    scope_source_dist: dict[str, int] = {}
    for c in CASES:
        scope_source_dist[c["expected_scope_source"]] = scope_source_dist.get(c["expected_scope_source"], 0) + 1

    # Unnecessary-delegation denominator: cases whose expected outcome does
    # NOT itself require a specialist call (synthesize / ask_clarification /
    # degrade are all "no delegation needed" outcomes) -- broader than the
    # E.1V definition (which had specialist travel_search_complete cases to
    # draw on); justified here because this checkpoint is supervisor-only,
    # and a model proposing a delegation for an ask_clarification/
    # out_of_scope case is exactly as much an "unnecessary delegation" as
    # one proposed for an already-terminal synthesize case.
    non_delegating_actions = {"synthesize", "ask_clarification", "degrade"}
    unnecessary_denominator = sum(1 for c in CASES if c["expected_action"] in non_delegating_actions)
    assert unnecessary_denominator >= 12, unnecessary_denominator

    return {
        "schema_version": "1.0.0",
        "checkpoint": "Final Evaluation Checkpoint E.1X",
        "description": (
            "Predeclared, frozen independent live-Qwen supervisor-only validation case "
            "manifest for the E.1W capability-classification/continuity repair. New "
            "wording/dates/trip data/evidence throughout -- not copied or paraphrased "
            "from E.1, E.1R, or E.1V. Every expected_action and expected_scope_source is "
            "a single, unambiguous predeclared label. Frozen before any live call."
        ),
        "case_count": len(CASES),
        "language_distribution": lang_dist,
        "scope_distribution": scope_dist,
        "scope_by_language": per_scope_lang,
        "followup_case_count": followup_count,
        "initial_case_count": len(CASES) - followup_count,
        "expected_action_distribution": action_dist,
        "expected_scope_source_distribution": scope_source_dist,
        "unnecessary_delegation_denominator": unnecessary_denominator,
        "unnecessary_delegation_definition": (
            "eligible = expected_action in {synthesize, ask_clarification, degrade}; "
            "unnecessary = a final-schema-valid actual_action in {call_travel_search, call_istanbul_expert} "
            "for such a case."
        ),
        "stale_turn_signature_placeholder": _PRIOR_TURN_PLACEHOLDER,
        "stale_turn_signature_sha256": _STALE_SIG,
        "cases": CASES,
    }


def main() -> None:
    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False)
    OUTPUT_PATH.write_text(text, encoding="utf-8", newline="\n")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {OUTPUT_PATH}")
    print(f"case_set_sha256={checksum}")
    print(f"language_distribution={manifest['language_distribution']}")
    print(f"scope_distribution={manifest['scope_distribution']}")
    print(f"scope_by_language={manifest['scope_by_language']}")
    print(f"followup_case_count={manifest['followup_case_count']}")
    print(f"expected_action_distribution={manifest['expected_action_distribution']}")
    print(f"expected_scope_source_distribution={manifest['expected_scope_source_distribution']}")
    print(f"unnecessary_delegation_denominator={manifest['unnecessary_delegation_denominator']}")


if __name__ == "__main__":
    main()
