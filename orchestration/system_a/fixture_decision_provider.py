"""Deterministic, rule-based `DecisionProvider`s for System A's explicit
demo/fixture mode (Checkpoint Phase 4 D.2B, `VOYAGER_SYSTEM_A_MODE=fixture`).

Satisfies the exact same `phase4.qwen_client.DecisionProvider` protocol
`QwenDecisionProvider` does (`generate(system, user) -> str`, returning a
JSON `ActionDecision`), so it plugs into the unmodified bounded LangGraph
loop through the same `decision_provider_factory` seam Checkpoint D.1
already established -- no graph, guard, or bound is touched. Reads the
`user` payload `phase4.graph._build_decision_prompt` /
`phase4.specialist._build_specialist_prompt` already build (`user_message`/
`trip_request`/`evidence_collected_so_far`/`tool_calls_used`) and applies
a small, fixed, explicit action order -- never an LLM call, never network
I/O, never randomness. Pairs naturally with `phase4.tools.FakeToolExecutor`
(already-existing, already-tested, production-owned deterministic fixture
executor -- reused unchanged, not duplicated) to make fixture mode fully
network-free end to end.

Checkpoint Phase 4 D.3 (correction pass): the supervisor and the internal
Travel Search specialist are now served by two SEPARATE classes --
`SupervisorFixtureDecisionProvider` and `SpecialistFixtureDecisionProvider`
-- each instantiated explicitly for its own role by the caller
(`orchestration/system_a/service.py`), exactly mirroring how real mode
constructs two separate `QwenDecisionProvider` instances. Neither class
ever inspects the `system` prompt text to guess which role it is playing
-- the role is fixed at construction/class-identity, not inferred from
prompt content, matching the graph's own explicit
`decision_provider`/`specialist_decision_provider` construction parameters
(`phase4.graph.build_graph`).

This is new orchestration-owned routing code, not a duplication of any
existing business logic: no deterministic "which tool next" policy
existed anywhere in this project before this checkpoint (Qwen decides
that in real mode; test-only `ScriptedDecisionProvider` helpers only ever
replay a fixed script for one specific test scenario, never a general
policy over an arbitrary real trip request).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from phase4.graph import CAPABILITY_SCOPE_PROMPT_MARKER
from phase4.language_detect import is_predominantly_arabic

# The fixed, explicit action order the internal Travel Search specialist
# proposes, one action per call, skipping whatever already has a
# successful observation (visible via the shared evidence registry both
# loops read). Chosen to exercise a real, demonstrable multi-tool plan
# (flights, stays, weather) while deliberately omitting
# `estimate_fair_price` and `web_search`: both are optional actions whose
# arguments in the REAL system also depend on already-executed evidence
# the decision provider itself never directly sees (`estimate_fair_price`'s
# `stay_id` is only ever validated server-side against a prior
# `search_stays` observation, see
# `orchestration/system_a/tool_executor.py::_validated_stay_id`) -- real
# Qwen faces the identical limitation, so skipping them here is not
# fixture mode cutting a corner real mode does not also have.
_SPECIALIST_ACTION_ORDER = ("get_weather", "search_flights", "search_stays")

_REASON_CODE_BY_ACTION = {
    "get_weather": "missing_weather_info",
    "search_flights": "missing_flight_info",
    "search_stays": "missing_stay_info",
    "call_istanbul_expert": "missing_local_expertise",
}


_SYNTHESIZE_EXPLANATION = (
    "This is a deterministic fixture-mode plan (no live provider, MCP, or A2A "
    "call was made). Review the flights, stays, weather, and itinerary tabs below."
)


def _classification(scope: str, reason_code: str) -> str:
    """Checkpoint Final Evaluation E.1S.1: fixture mode's own
    deterministic, rule-based capability-scope classification -- reuses
    this file's own already-established design philosophy ("a small,
    fixed, explicit... policy -- never an LLM call") rather than adding
    a second classification mechanism. A structured trip_request means
    'combined' (this fixture always demonstrates the full flight/stay/
    weather + Istanbul Expert plan); no structured trip_request means
    'out_of_scope' -- preserving the pre-existing fixture-mode behavior
    of an honest, evidence-free degrade for a bare message with no trip
    details, rather than ever calling a specialist for nothing."""
    return json.dumps({"scope": scope, "reason_code": reason_code})


def _decision(action: str, arguments: dict[str, Any], reason_code: str) -> str:
    # `synthesize`'s own `explanation` becomes the user-facing "Summary" line
    # in the frontend (phase4.graph._synthesize_node's `narrative` field,
    # unmodified) -- it gets its own readable sentence rather than the
    # generic per-tool-call phrasing below, which is written for the
    # sanitized SSE/decision trace, not as end-user-facing prose (a real
    # readability defect found and fixed during this checkpoint's own
    # browser smoke test: the generic phrasing read as leaked internal
    # decision-log text, e.g. "deterministically selecting synthesize").
    explanation = (
        _SYNTHESIZE_EXPLANATION if action == "synthesize"
        else f"Fixture mode: deterministically selecting {action}."
    )
    return json.dumps({
        "action": action, "arguments": arguments, "reason_code": reason_code,
        "explanation": explanation,
    })


def _build_arguments(action: str, trip_request: dict[str, Any]) -> Optional[dict[str, Any]]:
    if action == "get_weather":
        depart_date = trip_request.get("depart_date")
        if not depart_date:
            return None
        return {"location": "Istanbul", "date_from": depart_date, "date_to": depart_date}
    if action == "search_flights":
        required = ("origin", "destination", "depart_date", "traveler_count")
        if not all(trip_request.get(field) for field in required):
            return None
        return {
            "origin": trip_request["origin"], "destination": trip_request["destination"],
            "depart_date": trip_request["depart_date"], "passenger_count": trip_request["traveler_count"],
            "cabin_class": "economy",
        }
    if action == "search_stays":
        required = ("depart_date", "return_date", "traveler_count")
        if not all(trip_request.get(field) for field in required):
            return None
        return {
            "check_in": trip_request["depart_date"], "check_out": trip_request["return_date"],
            "guest_count": trip_request["traveler_count"],
        }
    if action == "call_istanbul_expert":
        return {"question": "What should I see, and how should my days be organized around my stay?"}
    return None


def _parse_user_payload(user: str) -> dict[str, Any]:
    # `phase4.graph._build_decision_prompt` / `phase4.specialist.
    # _build_specialist_prompt` both prefix the JSON payload with a fixed
    # human-readable line -- split it off rather than assuming a fixed
    # character offset, so this stays correct even if that prefix's
    # wording changes.
    _, _, json_text = user.partition("\n")
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SupervisorFixtureDecisionProvider:
    """Deterministic `DecisionProvider` for the SUPERVISOR role only.
    Never reads an environment variable, never opens a socket, never
    stores mutable state across calls, and never inspects `system` --
    the role is fixed by this class's own identity, exactly matching
    `phase4.graph.build_graph`'s own explicit `decision_provider`
    construction parameter."""

    def generate(self, system: str, user: str) -> str:
        if CAPABILITY_SCOPE_PROMPT_MARKER in system:
            # Checkpoint Final Evaluation E.1S.1: the once-per-turn
            # capability-scope classification call. This fixture policy
            # always demonstrates the full combined plan for a structured
            # trip request, and honestly degrades (never calling a
            # specialist for nothing) when there is none.
            payload = _parse_user_payload(user)
            trip_request = payload.get("trip_request") or {}
            if trip_request:
                return _classification("combined", "requires_both")
            return _classification("out_of_scope", "outside_project_scope")

        payload = _parse_user_payload(user)
        trip_request = payload.get("trip_request") or {}
        evidence = {
            entry.get("action"): entry.get("status")
            for entry in payload.get("evidence_collected_so_far", [])
            if isinstance(entry, dict)
        }

        if not trip_request:
            # Classified out_of_scope above -- only ask_clarification/
            # degrade are structurally eligible; degrade honestly with no
            # fabricated evidence rather than ever calling a specialist.
            return _decision("degrade", {"reason": "outside_project_scope"}, "irrelevant_to_request")
        if not any(action in evidence for action in _SPECIALIST_ACTION_ORDER):
            # Nothing travel-search-related has been gathered yet for this
            # structured request -- delegate the whole batch to the
            # specialist in one decision, exactly like real Qwen must
            # (the supervisor's own contract never offers the 5 tools
            # directly, see `SUPERVISOR_ACTIONS`).
            return _decision("call_travel_search", {}, _REASON_CODE_BY_ACTION["get_weather"])
        if "call_istanbul_expert" not in evidence:
            # Combined scope requires Istanbul Expert grounding once
            # Travel Search has reached a terminal result, regardless of
            # whether every individual travel sub-action itself
            # succeeded (Checkpoint Final Evaluation E.1S.1: a failed/
            # degraded specialist result is still terminal for this turn).
            arguments = _build_arguments("call_istanbul_expert", trip_request)
            return _decision("call_istanbul_expert", arguments, _REASON_CODE_BY_ACTION["call_istanbul_expert"])
        return _decision("synthesize", {}, "all_required_evidence_present")


class SpecialistFixtureDecisionProvider:
    """Deterministic `DecisionProvider` for the internal Travel Search
    SPECIALIST role only. Never reads an environment variable, never
    opens a socket, never stores mutable state across calls, and never
    inspects `system` -- the role is fixed by this class's own identity,
    exactly matching `phase4.graph.build_graph`'s own explicit
    `specialist_decision_provider` construction parameter."""

    def generate(self, system: str, user: str) -> str:
        payload = _parse_user_payload(user)
        trip_request = payload.get("trip_request") or {}
        evidence = {
            entry.get("action"): entry.get("status")
            for entry in payload.get("evidence_collected_so_far", [])
            if isinstance(entry, dict)
        }

        for action in _SPECIALIST_ACTION_ORDER:
            if action in evidence:
                continue  # already attempted (success or not) -- never repeat
            if not trip_request:
                continue  # no structured trip request at all -- nothing to search for
            arguments = _build_arguments(action, trip_request)
            if arguments is None:
                continue
            return _decision(action, arguments, _REASON_CODE_BY_ACTION[action])
        return _decision("travel_search_complete", {}, "all_required_evidence_present")


# --- Hybrid Chat C.1: deterministic fixture chat-turn provider ---------------------

_CHAT_FIXTURE_CLARIFY_MESSAGE = (
    "This is deterministic fixture mode (no live Qwen call is made), so conversational trip "
    "modification cannot be demonstrated here -- switch to real mode (VOYAGER_SYSTEM_A_MODE=real) "
    "to try changing your trip through chat. I can still tell you: the dashboard above already "
    "reflects your submitted trip request."
)
_CHAT_FIXTURE_CLARIFY_MESSAGE_AR = (
    "هذا وضع العرض التجريبي الحتمي (لا يتم إجراء أي استدعاء حقيقي لنموذج Qwen)، لذا لا يمكن هنا "
    "عرض تعديل الرحلة عبر المحادثة -- بدّل إلى الوضع الحقيقي (VOYAGER_SYSTEM_A_MODE=real) لتجربة "
    "تعديل رحلتك عبر المحادثة. لوحة النتائج أعلاه تعكس بالفعل طلب رحلتك المُرسل."
)


class ChatFixtureDecisionProvider:
    """Deterministic `DecisionProvider` for Hybrid Chat C.1's chat-turn
    call, used only under `VOYAGER_SYSTEM_A_MODE=fixture`. Never opens a
    socket, never uses randomness. Conversational trip modification is
    genuinely a live-Qwen capability (structured-output intent
    classification and patch extraction); this fixture honestly always
    responds `clarify` with an explanatory message rather than faking a
    modification/regeneration, so fixture mode never demonstrates a
    provider call it does not actually make."""

    def generate(self, system: str, user: str) -> str:
        target_language = "en"
        try:
            payload = _parse_user_payload(user)
            preferred = payload.get("preferred_language") or "en"
            target_language = "ar" if preferred == "ar" or is_predominantly_arabic(payload.get("user_message") or "") else preferred
        except Exception:  # noqa: BLE001 -- fixture mode must never raise
            pass
        message = _CHAT_FIXTURE_CLARIFY_MESSAGE_AR if target_language == "ar" else _CHAT_FIXTURE_CLARIFY_MESSAGE
        return json.dumps({
            "intent": "clarify",
            "assistant_message": message,
            "response_language": target_language,
            "patch": None,
            "requires_clarification": True,
            "clarification_reason": "fixture_mode_no_live_classification",
        })
