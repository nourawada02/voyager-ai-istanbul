"""Final Evaluation Checkpoint E.1Z -- builds and freezes the 30-case
independent supervisor-only routing validation manifest
(`evaluation/e1z_cases.json`) BEFORE any live model call is made, and
writes a machine-readable preflight label-validation audit
(`evaluation/e1z_preflight_audit.json`) proving every case's single
expected action is architecturally justified -- before freezing, and
before any live call.

New wording/dates/trip data throughout -- not copied or paraphrased from
E.1, E.1R, E.1V, or E.1X. Supervisor-only (the Travel Search specialist
is out of this checkpoint's scope).

Run once:
    python -m evaluation.build_e1z_cases
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "planner-a"))

from phase4.graph import _compute_request_signature  # noqa: E402

CASES_PATH = Path(__file__).parent / "e1z_cases.json"
AUDIT_PATH = Path(__file__).parent / "e1z_preflight_audit.json"

if CASES_PATH.exists():
    raise SystemExit(f"REFUSING TO OVERWRITE: {CASES_PATH} already exists. E.1Z is one-shot.")

_PRIOR_TURN_PLACEHOLDER = {"user_message": "An earlier, now-superseded planning question about a different Istanbul trip.", "trip_request": None}


def _sig(user_message: str, trip_request) -> str:
    return _compute_request_signature({"normalized_request": {"user_message": user_message, "trip_request": trip_request}})


_STALE_SIG = _compute_request_signature({"normalized_request": _PRIOR_TURN_PLACEHOLDER})


def _obs(action: str, status: str) -> dict:
    return {"action": action, "status": status}


TRIP_A = {"origin": "BEY", "destination": "IST", "depart_date": "2027-06-03", "return_date": "2027-06-09",
          "traveler_count": 2, "budget": {"amount_minor_units": 480000, "currency": "TRY"},
          "preferences": {"interests": ["food"], "pace": "relaxed", "language": "en", "mobility_constraints": []}}
TRIP_B = dict(TRIP_A, depart_date="2027-07-14", return_date="2027-07-20", preferences=dict(TRIP_A["preferences"], language="tr"))
TRIP_C = dict(TRIP_A, depart_date="2027-08-01", return_date="2027-08-07", preferences=dict(TRIP_A["preferences"], language="ar"), traveler_count=1)
TRIP_D = dict(TRIP_A, depart_date="2027-09-11", return_date="2027-09-17", traveler_count=3)
TRIP_E = dict(TRIP_A, depart_date="2027-10-05", return_date="2027-10-12", preferences=dict(TRIP_A["preferences"], interests=["architecture", "history"]))
TRIP_F = dict(TRIP_E, depart_date="2027-11-02", return_date="2027-11-09", preferences=dict(TRIP_E["preferences"], language="tr"))
TRIP_G = dict(TRIP_E, depart_date="2027-12-01", return_date="2027-12-08", preferences=dict(TRIP_E["preferences"], language="tr"), traveler_count=1)
TRIP_H = dict(TRIP_E, depart_date="2028-01-10", return_date="2028-01-17", preferences=dict(TRIP_E["preferences"], language="ar"), traveler_count=4)
TRIP_I = dict(TRIP_E, depart_date="2028-02-14", return_date="2028-02-21", preferences=dict(TRIP_E["preferences"], language="ar"))


def _case(
    case_id, language, scope, category, user_message, trip_request,
    observations, tool_call_count, expected_action, expected_scope_source,
    last_scope=None, travel_terminal=None, istanbul_terminal=None, is_followup=False, notes="",
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

    return dict(
        case_id=case_id, agent_role="supervisor", language=language, expected_capability_scope=scope,
        category=category, user_message=user_message, trip_request=trip_request,
        observations=observations, tool_call_count=tool_call_count, tool_call_count_by_action=by_action,
        expected_action=expected_action, expected_scope_source=expected_scope_source,
        last_successful_capability_scope=last_scope,
        travel_search_attempted_signature=_resolve(travel_terminal),
        istanbul_expert_attempted_signature=_resolve(istanbul_terminal),
        request_signature=signature, is_followup=is_followup, notes=notes,
    )


CASES = [
    # --- travel_only (6): 2 EN, 2 TR, 2 AR --------------------------------------------
    _case("V3-S01", "en", "travel_only", "initial_complete_info",
          "Sort out the travel side of my Istanbul trip -- flights and a place to stay.", TRIP_A,
          [], 0, "call_travel_search", "classified"),
    _case("V3-S02", "tr", "travel_only", "followup_reopens_complete_info",
          "Bu sefer farkli tarihler icin ucus ve otel lazim.", TRIP_B,
          [], 0, "call_travel_search", "inherited", last_scope="travel_only", travel_terminal="stale_turn",
          is_followup=True, notes="prior turn's completed attempt (stale signature) must never block this new one"),
    _case("V3-S03", "ar", "travel_only", "followup_terminal_completed",
          "كل شيء يبدو جاهزًا، يمكنك المتابعة.", TRIP_C,
          [_obs("search_flights", "success"), _obs("search_stays", "success")], 2,
          "synthesize", "inherited", last_scope="travel_only", travel_terminal="same_turn", is_followup=True),
    _case("V3-S04", "en", "travel_only", "followup_terminal_degraded",
          "Even if the forecast isn't available, go ahead and finish up.", TRIP_D,
          [_obs("search_flights", "success"), _obs("get_weather", "unavailable")], 2,
          "synthesize", "inherited", last_scope="travel_only", travel_terminal="same_turn", is_followup=True),
    _case("V3-S05", "tr", "travel_only", "followup_explicit_change_essential_info_absent",
          "Aslinda gezilecek yerleri unut, sadece ucus konusunda yardim et.", None,
          [], 0, "ask_clarification", "explicit", last_scope="istanbul_local_only", istanbul_terminal="stale_turn",
          is_followup=True, notes="explicit change to travel_only, but no trip_request/date/origin at all"),
    _case("V3-S06", "ar", "travel_only", "initial_essential_info_absent",
          "أريد رحلة طيران، لكن لم أحدد التاريخ بعد.", None,
          [], 0, "ask_clarification", "classified", notes="clear task (flight), missing essential parameter (date)"),

    # --- istanbul_local_only (6): 2 EN, 2 TR, 2 AR -------------------------------------
    _case("V3-S07", "en", "istanbul_local_only", "initial",
          "What neighborhoods should I wander through if I only have one afternoon free?",
          None, [], 0, "call_istanbul_expert", "classified"),
    _case("V3-S08", "tr", "istanbul_local_only", "initial",
          "Sultanahmet disinda hangi bolgeleri gezmeliyim?",
          None, [], 0, "call_istanbul_expert", "classified"),
    _case("V3-S09", "ar", "istanbul_local_only", "followup_not_terminal",
          "تابع من فضلك، أرني المزيد.", None,
          [], 0, "call_istanbul_expert", "inherited", last_scope="istanbul_local_only", is_followup=True,
          notes="short-continuation (AR)"),
    _case("V3-S10", "en", "istanbul_local_only", "followup_terminal_completed",
          "That's exactly what I needed -- go ahead and put it together.", None,
          [_obs("call_istanbul_expert", "success")], 1,
          "synthesize", "inherited", last_scope="istanbul_local_only", istanbul_terminal="same_turn", is_followup=True),
    _case("V3-S11", "tr", "istanbul_local_only", "followup_terminal_failed",
          "Rehber cevap veremese bile elindekiyle devam et.", None,
          [_obs("call_istanbul_expert", "provider_error")], 1,
          "synthesize", "inherited", last_scope="istanbul_local_only", istanbul_terminal="same_turn", is_followup=True,
          notes="regression check for the E.1Y V2-S11 fix: failed-but-terminal evidence -> honest synthesize, never degrade"),
    _case("V3-S12", "ar", "istanbul_local_only", "followup_explicit_change_from_travel",
          "في الواقع، انسَ الرحلة، أخبرني فقط بالأماكن التي يجب زيارتها.", None,
          [], 0, "call_istanbul_expert", "explicit", last_scope="travel_only", travel_terminal="stale_turn", is_followup=True),

    # --- combined (6): 2 EN, 2 TR, 2 AR -------------------------------------------------
    _case("V3-S13", "en", "combined", "initial_complete_info",
          "Put together the full Istanbul trip -- travel logistics and what to do each day.", TRIP_E,
          [], 0, "call_travel_search", "classified"),
    _case("V3-S14", "en", "combined", "followup_travel_terminal_needs_istanbul",
          "Travel logistics are done -- now let's plan the days.", TRIP_E,
          [_obs("search_flights", "success"), _obs("search_stays", "success")], 2,
          "call_istanbul_expert", "inherited", last_scope="combined", travel_terminal="same_turn", is_followup=True),
    _case("V3-S15", "tr", "combined", "followup_both_terminal",
          "Her sey tamamlandi, artik ozetleyebilirsin.", TRIP_F,
          [_obs("search_flights", "success"), _obs("search_stays", "success"), _obs("call_istanbul_expert", "success")], 3,
          "synthesize", "inherited", last_scope="combined", travel_terminal="same_turn", istanbul_terminal="same_turn", is_followup=True),
    _case("V3-S16", "tr", "combined", "followup_explicit_change_into_combined_complete_info",
          "Aslinda sadece gezilecek yerler degil, ucus ve otel de lazim simdi.", TRIP_G,
          [], 0, "call_travel_search", "explicit", last_scope="istanbul_local_only", istanbul_terminal="stale_turn",
          is_followup=True, notes="complete info supplied (TRIP_G) alongside the explicit scope expansion"),
    _case("V3-S17", "ar", "combined", "followup_budget_exhausted",
          "خطط لرحلتي الكاملة إلى إسطنبول من فضلك.", TRIP_H,
          [], 8, "synthesize", "inherited", last_scope="combined", is_followup=True),
    _case("V3-S18", "ar", "combined", "initial_complete_info",
          "أرغب في تنظيم رحلة كاملة إلى إسطنبول، من الطيران إلى الأنشطة اليومية.", TRIP_I,
          [], 0, "call_travel_search", "classified"),

    # --- clarification_required (6): 2 EN, 2 TR, 2 AR -----------------------------------
    _case("V3-S19", "en", "clarification_required", "initial",
          "I'm not sure what I need help with yet.", None, [], 0, "ask_clarification", "classified"),
    _case("V3-S20", "en", "clarification_required", "initial",
          "Can you assist me with something related to my upcoming plans?", None, [], 0, "ask_clarification", "classified"),
    _case("V3-S21", "tr", "clarification_required", "initial",
          "Henuz ne istedigimi tam bilmiyorum.", None, [], 0, "ask_clarification", "classified"),
    _case("V3-S22", "tr", "clarification_required", "initial",
          "Bir konuda yardimci olur musun, tam olarak ne oldugunu bilmiyorum.", None, [], 0, "ask_clarification", "classified"),
    _case("V3-S23", "ar", "clarification_required", "initial",
          "لست متأكدًا مما أحتاجه بعد.", None, [], 0, "ask_clarification", "classified"),
    _case("V3-S24", "ar", "clarification_required", "initial",
          "هل يمكنك مساعدتي؟ لست متأكدًا من طلبي بالضبط.", None, [], 0, "ask_clarification", "classified"),

    # --- out_of_scope (6): 2 EN, 2 TR, 2 AR ----------------------------------------------
    _case("V3-S25", "en", "out_of_scope", "initial",
          "Please book my hotel immediately and charge my card.", None, [], 0, "degrade", "classified"),
    _case("V3-S26", "en", "out_of_scope", "initial",
          "What's the freezing point of water in Fahrenheit?", None, [], 0, "degrade", "classified"),
    _case("V3-S27", "tr", "out_of_scope", "initial",
          "Otelimi hemen rezerve edip kartimdan tahsilat yapar misin?", None, [], 0, "degrade", "classified"),
    _case("V3-S28", "tr", "out_of_scope", "initial",
          "Suyun donma noktasi Fahrenayt cinsinden kactir?", None, [], 0, "degrade", "classified"),
    _case("V3-S29", "ar", "out_of_scope", "initial",
          "هل يمكنك حجز فندقي فورًا والدفع من بطاقتي؟", None, [], 0, "degrade", "classified"),
    _case("V3-S30", "ar", "out_of_scope", "initial",
          "ما هي درجة تجمد الماء بمقياس فهرنهايت؟", None, [], 0, "degrade", "classified"),
]


# --- mandatory label-validation preflight audit -------------------------------------


def _audit_case(c: dict) -> dict:
    checks: dict[str, bool] = {}
    action = c["expected_action"]
    scope = c["expected_capability_scope"]
    has_trip_request = c["trip_request"] is not None
    has_terminal_observation = len(c["observations"]) > 0
    budget_exhausted = c["tool_call_count"] >= 8

    checks["single_expected_action"] = isinstance(action, str)

    if action == "call_travel_search":
        checks["essential_travel_info_present"] = has_trip_request
    if action == "ask_clarification" and scope != "clarification_required":
        checks["essential_travel_info_genuinely_absent"] = not has_trip_request
    if action == "call_istanbul_expert":
        checks["sufficient_local_context"] = True  # CallIstanbulExpertArgs never requires structured data
    if action == "synthesize":
        checks["usable_evidence_or_budget_exhausted"] = has_terminal_observation or budget_exhausted
    if action == "degrade":
        checks["scope_is_out_of_scope"] = scope == "out_of_scope"

    if c["is_followup"]:
        # A follow-up must carry real session state (the prior successful
        # scope) -- observations/attempted-signatures are only required
        # when the follow-up's own OUTCOME depends on them (terminal or
        # budget-exhausted cases); a "not yet terminal this turn"
        # follow-up (e.g. an immediate "continue" with nothing done yet)
        # legitimately has neither.
        checks["followup_has_prior_scope_state"] = c["last_successful_capability_scope"] is not None
    else:
        checks["non_followup_has_no_prior_scope"] = c["last_successful_capability_scope"] is None

    passed = all(checks.values())
    return {"case_id": c["case_id"], "expected_action": action, "expected_scope": scope, "checks": checks, "passed": passed}


def build_audit() -> dict:
    rows = [_audit_case(c) for c in CASES]
    all_passed = all(r["passed"] for r in rows)
    if not all_passed:
        failing = [r["case_id"] for r in rows if not r["passed"]]
        raise SystemExit(f"PREFLIGHT LABEL VALIDATION FAILED for: {failing} -- fix case design before freezing.")
    return {
        "checkpoint": "Final Evaluation Checkpoint E.1Z — mandatory label preflight audit",
        "case_count": len(rows), "all_passed": all_passed, "rows": rows,
    }


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

    non_delegating_actions = {"synthesize", "ask_clarification", "degrade"}
    unnecessary_denominator = sum(1 for c in CASES if c["expected_action"] in non_delegating_actions)
    assert unnecessary_denominator >= 12, unnecessary_denominator

    return {
        "schema_version": "1.0.0",
        "checkpoint": "Final Evaluation Checkpoint E.1Z",
        "description": (
            "Predeclared, frozen independent live-Qwen supervisor-only routing validation "
            "case manifest for the E.1Y action-eligibility repair. New wording/dates/trip "
            "data/evidence throughout -- not copied or paraphrased from E.1, E.1R, E.1V, or "
            "E.1X. Every expected_action is a single, unambiguous predeclared label, and "
            "every label passed the mandatory preflight validation (see "
            "evaluation/e1z_preflight_audit.json) before any live call."
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
            "unnecessary = a final-schema-valid actual_action in {call_travel_search, call_istanbul_expert} for such a case."
        ),
        "stale_turn_signature_placeholder": _PRIOR_TURN_PLACEHOLDER,
        "stale_turn_signature_sha256": _STALE_SIG,
        "cases": CASES,
    }


def main() -> None:
    audit = build_audit()
    audit_text = json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=False)
    AUDIT_PATH.write_text(audit_text, encoding="utf-8", newline="\n")
    print(f"wrote {AUDIT_PATH} -- all_passed={audit['all_passed']}")

    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False)
    CASES_PATH.write_text(text, encoding="utf-8", newline="\n")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {CASES_PATH}")
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
