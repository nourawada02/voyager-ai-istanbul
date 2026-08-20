"""Hermetic tests for Hybrid Chat C.1: `ChatTurnService` (unit-level) and
`POST /v1/chat/turns` (HTTP-level, via `create_app()` with an injected
scripted chat decision provider). No real Qwen/network call anywhere in
this file -- mirrors the existing `orchestration/tests/test_api_endpoints.py`
hermetic-app-building convention exactly.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from orchestration.system_a.api import create_app
from orchestration.system_a.chat_service import ChatProviderUnavailable, ChatSessionAccessRejected, ChatTurnRejected, ChatTurnService
from orchestration.system_a.run_store import RunStatus, RunStore
from orchestration.system_a.service import RunService
from orchestration.tests.conftest import ScriptedDecisionProvider, decision
from phase4.qwen_client import QwenTransportError
from phase4.tools import FakeToolExecutor
from providers.fx import FakeFxProvider

_TEST_WALL_CLOCK = lambda: datetime(2026, 8, 19, tzinfo=timezone.utc)

_TRIP_REQUEST_PARTIAL = {
    "origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "return_date": "2026-09-15",
    "traveler_count": 2, "budget": {"amount_minor_units": 500000, "currency": "TRY"},
    "preferences": {"schema_version": "1.0.0", "interests": ["history"], "pace": "moderate", "language": "en", "mobility_constraints": []},
}


class ScriptedChatDecisionProvider:
    """A `phase4.qwen_client.DecisionProvider` for the chat-turn call
    only -- returns pre-scripted raw JSON strings in order, or raises a
    pre-scripted exception, never a live model call."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append(user)
        response = self.responses[len(self.calls) - 1]
        if isinstance(response, BaseException):
            raise response
        return response


def _chat_decision(intent: str, message: str = "ok", language: str = "en", patch: dict | None = None, requires_clarification: bool = False, clarification_reason: str | None = None) -> str:
    return json.dumps({
        "intent": intent, "assistant_message": message, "response_language": language,
        "patch": patch, "requires_clarification": requires_clarification, "clarification_reason": clarification_reason,
    })


@pytest.fixture()
def store(tmp_db_path) -> RunStore:
    return RunStore(tmp_db_path)


@pytest.fixture()
def run_service(store) -> RunService:
    return RunService(
        run_store=store, db_path=store.db_path,
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: ScriptedDecisionProvider([decision("synthesize", {}, "all_required_evidence_present")]),
        specialist_decision_provider_factory=lambda: ScriptedDecisionProvider([decision("travel_search_complete", {}, "all_required_evidence_present")]),
        max_workers=2, wall_clock=_TEST_WALL_CLOCK, fx_provider_factory=lambda: FakeFxProvider(),
    )


def _create_seed_run(run_service: RunService, trip_request_partial=_TRIP_REQUEST_PARTIAL):
    record, _created = run_service.create_run("Plan a trip to Istanbul.", trip_request_partial, None)
    for _ in range(200):
        current = run_service.get_run(record.run_id)
        if current.status != RunStatus.PENDING and current.status != RunStatus.RUNNING:
            return current
        time.sleep(0.01)
    raise AssertionError("seed run did not reach a terminal status in time")


# --- ChatTurnService: unit-level ------------------------------------------------------


def test_unknown_run_id_is_rejected(store, run_service):
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: ScriptedChatDecisionProvider([]))
    with pytest.raises(ChatTurnRejected):
        service.handle_chat_turn("some-session", "no-such-run", "Why?", "en")


def test_session_id_mismatch_is_rejected(store, run_service):
    seed = _create_seed_run(run_service)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: ScriptedChatDecisionProvider([]))
    with pytest.raises(ChatTurnRejected):
        service.handle_chat_turn("wrong-session-id", seed.run_id, "Why?", "en")


def test_oversized_message_is_rejected_before_any_provider_call(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatTurnRejected):
        service.handle_chat_turn(seed.session_id, seed.run_id, "x" * 3000, "en")
    assert provider.calls == []


def test_explain_plan_never_creates_a_new_run(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "Galata Tower is a well-known landmark.")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Why did you recommend Galata Tower?", "en")
    assert result.intent == "explain_plan"
    assert result.requires_new_run is False
    assert result.new_run_id is None
    assert result.run_id == seed.run_id


def test_modify_trip_valid_patch_creates_exactly_one_new_run_same_session(store, run_service):
    seed = _create_seed_run(run_service)
    patch = {"budget_amount_minor_units": 100000, "budget_currency": "USD", "add_interests": ["shopping"]}
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "Updated your budget.", patch=patch)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Change my budget to 1000 USD and add shopping", "en")
    assert result.requires_new_run is True
    assert result.new_run_id is not None
    assert result.new_run_id != seed.run_id
    new_record = store.get_run(result.new_run_id)
    assert new_record.session_id == seed.session_id
    assert new_record.request["trip_request"]["budget"] == {"amount_minor_units": 100000, "currency": "USD"}
    assert "shopping" in new_record.request["trip_request"]["preferences"]["interests"]


def test_modify_trip_with_past_departure_date_falls_back_to_clarify(store, run_service):
    seed = _create_seed_run(run_service)
    patch = {"depart_date": "2020-01-01"}
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "Moving your trip.", patch=patch)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Move my departure to yesterday", "en")
    assert result.intent == "clarify"
    assert result.requires_new_run is False
    assert result.clarification_required is True


def test_modify_trip_with_unrecognized_interest_warns_but_still_applies_recognized_ones(store, run_service):
    seed = _create_seed_run(run_service)
    patch = {"add_interests": ["shopping", "skydiving"]}
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "Updated.", patch=patch)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Add shopping and skydiving", "en")
    assert result.requires_new_run is True
    assert any("skydiving" in w for w in result.warnings)
    new_record = store.get_run(result.new_run_id)
    assert "shopping" in new_record.request["trip_request"]["preferences"]["interests"]


def test_modify_trip_with_empty_patch_falls_back_to_clarify(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "ok", patch=None)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "change something", "en")
    assert result.intent == "clarify"
    assert result.requires_new_run is False


def test_regenerate_trip_with_no_patch_still_creates_a_new_run_with_unchanged_trip(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("regenerate_trip", "Replanning.", patch=None)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Please replan", "en")
    assert result.requires_new_run is True
    new_record = store.get_run(result.new_run_id)
    assert new_record.request["trip_request"]["budget"] == _TRIP_REQUEST_PARTIAL["budget"]


def test_reset_trip_intent_creates_no_new_run(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("reset_trip", "Starting over.")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "start over", "en")
    assert result.intent == "reset_trip"
    assert result.requires_new_run is False


def test_response_language_is_always_the_deterministic_target_not_whatever_qwen_echoed(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "ok", language="tr")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    assert result.response_language == "en"


def test_arabic_target_with_non_arabic_reply_gets_one_bounded_correction_then_succeeds(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        _chat_decision("explain_plan", "This is in English by mistake.", language="ar"),
        _chat_decision("explain_plan", "تم اقتراح برج غلطة بسبب قربه من الفندق الذي اخترته.", language="ar"),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "لماذا اقترحت برج غلطة؟", "ar")
    assert result.response_language == "ar"
    assert "غلطة" in result.assistant_message
    assert len(provider.calls) == 2


def test_arabic_target_with_repeated_failure_returns_safe_arabic_fallback(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        _chat_decision("explain_plan", "This is in English by mistake.", language="ar"),
        _chat_decision("explain_plan", "Still English.", language="ar"),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "لماذا؟", "ar")
    assert result.response_language == "ar"
    from phase4.language_detect import contains_meaningful_arabic
    assert contains_meaningful_arabic(result.assistant_message)


def test_malformed_json_is_repaired_within_the_bounded_budget(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider(["not valid json", _chat_decision("explain_plan", "Recovered.")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    assert result.assistant_message == "Recovered."
    assert len(provider.calls) == 2


def test_exhausted_repairs_falls_back_to_safe_clarify_never_raises(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider(["not json 1", "not json 2"])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    assert result.intent == "clarify"
    assert result.clarification_required is True


def test_permanent_transport_error_raises_chat_provider_unavailable_not_retriable(store, run_service):
    """A non-transient QwenTransportError (e.g. a permanent auth/quota
    failure) never retries -- retrying would not help -- and is surfaced
    honestly as ChatProviderUnavailable, never silently reclassified as
    user ambiguity (the fixed bug: a real 403 was previously shown to the
    user as "I couldn't confidently understand that request")."""
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([QwenTransportError("boom", transient=False)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatProviderUnavailable) as exc_info:
        service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    assert exc_info.value.retriable is False
    assert len(provider.calls) == 1


def test_permanent_transport_error_persists_a_provider_failed_turn_excluded_from_context(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([QwenTransportError("boom", transient=False)])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatProviderUnavailable):
        service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    turns = store.list_chat_turns(seed.session_id)
    assert len(turns) == 2
    assert turns[0].content == "Why?"
    assert turns[1].status.value == "provider_failed"
    assert turns[1].include_in_context is False


def test_transient_transport_error_is_retried_within_the_bounded_budget_then_succeeds(store, run_service):
    """A transient QwenTransportError (timeout/connect/429/5xx) gets the
    same bounded retry a decision-format repair gets -- mirrors
    `phase4.graph._classify_capability`'s own established
    transient-transport-retry precedent, never an instant give-up on one
    momentary network blip."""
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        QwenTransportError("boom", transient=True), _chat_decision("explain_plan", "Recovered after a transient failure."),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    assert result.assistant_message == "Recovered after a transient failure."
    assert len(provider.calls) == 2


def test_transient_transport_error_exhausted_raises_chat_provider_unavailable_retriable(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        QwenTransportError("boom", transient=True), QwenTransportError("boom again", transient=True),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatProviderUnavailable) as exc_info:
        service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")
    assert exc_info.value.retriable is True
    assert len(provider.calls) == 2


def test_no_prior_trip_request_makes_modify_trip_fall_back_to_clarify(store, run_service):
    seed = _create_seed_run(run_service, trip_request_partial=None)
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "ok", patch={"traveler_count": 3})])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "add a traveler", "en")
    assert result.intent == "clarify"
    assert result.requires_new_run is False


# --- HTTP-level: POST /v1/chat/turns --------------------------------------------------


def _make_app(tmp_db_path: str, chat_decision_provider_factory):
    supervisor = ScriptedDecisionProvider([decision("synthesize", {}, "all_required_evidence_present")])
    specialist = ScriptedDecisionProvider([decision("travel_search_complete", {}, "all_required_evidence_present")])
    return create_app(
        tool_executor_factory=FakeToolExecutor,
        decision_provider_factory=lambda: supervisor,
        specialist_decision_provider_factory=lambda: specialist,
        db_path=tmp_db_path, max_workers=2, wall_clock=_TEST_WALL_CLOCK,
        fx_provider_factory=lambda: FakeFxProvider(),
        chat_decision_provider_factory=chat_decision_provider_factory,
    )


def _poll_until_terminal(client: TestClient, run_id: str, attempts: int = 200, delay: float = 0.01) -> dict:
    data = {}
    for _ in range(attempts):
        response = client.get(f"/v1/runs/{run_id}")
        data = response.json()
        if data["status"] not in ("pending", "running"):
            return data
        time.sleep(delay)
    raise AssertionError(f"run did not reach a terminal status in time: {data}")


def test_http_chat_turn_explain_plan_returns_valid_schema_and_200(tmp_db_path):
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "Galata Tower is a well-known landmark.")])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        response = client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "Why did you recommend Galata Tower?", "preferred_language": "en",
        })
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "explain_plan"
    assert body["requires_new_run"] is False
    assert body["run_id"] == created["run_id"]


def test_http_chat_turn_unknown_run_id_returns_typed_422_never_500(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        response = client.post("/v1/chat/turns", json={
            "session_id": "s1", "run_id": "no-such-run", "user_message": "hi", "preferred_language": "en",
        })
    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "CHAT_TURN_REJECTED"
    assert "trace_id" in body


def test_http_chat_turn_extra_field_is_rejected_by_request_validation(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        response = client.post("/v1/chat/turns", json={
            "session_id": "s1", "run_id": "r1", "user_message": "hi", "preferred_language": "en",
            "trip_result": {"fake": "should never be accepted from the client"},
        })
    assert response.status_code == 422


def test_http_chat_turn_modify_trip_creates_a_new_run_pollable_via_existing_endpoints(tmp_db_path):
    patch = {"traveler_count": 3}
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "Updating your travelers.", patch=patch)])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        chat_response = client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "Add one more traveler", "preferred_language": "en",
        }).json()
        assert chat_response["requires_new_run"] is True
        new_run_id = chat_response["new_run_id"]
        assert new_run_id != created["run_id"]
        final = _poll_until_terminal(client, new_run_id)
        assert final["session_id"] == created["session_id"]


# --- backward compatibility -------------------------------------------------------------


def test_create_app_without_explicit_chat_decision_provider_factory_still_builds(tmp_db_path):
    """Every pre-existing test/caller of `create_app()` omits
    `chat_decision_provider_factory` entirely -- it must default safely
    rather than becoming a required parameter (Hybrid Chat C.1 §8:
    'existing clients/endpoints remain valid')."""
    supervisor = ScriptedDecisionProvider([decision("synthesize", {}, "all_required_evidence_present")])
    app = create_app(
        tool_executor_factory=FakeToolExecutor, decision_provider_factory=lambda: supervisor,
        specialist_decision_provider_factory=lambda: supervisor, db_path=tmp_db_path,
        wall_clock=_TEST_WALL_CLOCK, fx_provider_factory=lambda: FakeFxProvider(),
    )
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_existing_run_endpoints_unaffected_by_chat_turn_field_absence(tmp_db_path):
    """No chat field is ever required on the pre-existing `/v1/runs`
    contract -- a form-only client that never sends a chat turn still
    works exactly as before."""
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        response = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL})
    assert response.status_code == 201
    assert provider.calls == []


# --- persistent-history/grounding correction: persistence -----------------------------


def test_successful_turn_is_persisted_atomically_as_a_user_assistant_pair(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "Galata Tower is a landmark.")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "Why Galata Tower?", "en")
    turns = store.list_chat_turns(seed.session_id)
    assert len(turns) == 2
    assert turns[0].role.value == "user" and turns[0].content == "Why Galata Tower?"
    assert turns[1].role.value == "assistant" and turns[1].content == "Galata Tower is a landmark."
    assert turns[0].status.value == "completed" and turns[1].status.value == "completed"
    assert turns[0].include_in_context is True and turns[1].include_in_context is True


def test_persisted_sequence_numbers_are_deterministic_and_ordered(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        _chat_decision("explain_plan", "first answer"), _chat_decision("explain_plan", "second answer"),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "Q1", "en")
    service.handle_chat_turn(seed.session_id, seed.run_id, "Q2", "en")
    turns = store.list_chat_turns(seed.session_id)
    assert [t.sequence_number for t in turns] == [0, 1, 2, 3]
    assert [t.content for t in turns] == ["Q1", "first answer", "Q2", "second answer"]


def test_rejected_message_is_persisted_with_include_in_context_false(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatTurnRejected):
        service.handle_chat_turn(seed.session_id, seed.run_id, "x" * 3000, "en")
    turns = store.list_chat_turns(seed.session_id)
    assert len(turns) == 2
    assert turns[1].status.value == "rejected"
    assert turns[1].include_in_context is False
    assert provider.calls == []


def test_unknown_run_lookup_persists_nothing(store, run_service):
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatTurnRejected):
        service.handle_chat_turn("s1", "no-such-run", "hi", "en")
    # No session exists at all -- nothing to list, but this also proves
    # no stray row was written under a fabricated session id.
    assert store.list_chat_turns("s1") == []


def test_restart_persistence_a_fresh_run_store_over_the_same_db_file_sees_prior_turns(store, run_service, tmp_db_path):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "persisted answer")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "Why?", "en")

    reopened_store = RunStore(tmp_db_path)
    turns = reopened_store.list_chat_turns(seed.session_id)
    assert len(turns) == 2
    assert turns[1].content == "persisted answer"


def test_new_session_does_not_inherit_a_prior_sessions_turns(store, run_service):
    seed_a = _create_seed_run(run_service)
    seed_b = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "answer for A")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed_a.session_id, seed_a.run_id, "Q for A", "en")
    assert store.list_chat_turns(seed_b.session_id) == []


def test_previous_session_transcript_remains_stored_after_a_new_session_starts(store, run_service):
    seed_a = _create_seed_run(run_service)
    seed_b = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        _chat_decision("explain_plan", "answer for A"), _chat_decision("explain_plan", "answer for B"),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed_a.session_id, seed_a.run_id, "Q for A", "en")
    service.handle_chat_turn(seed_b.session_id, seed_b.run_id, "Q for B", "en")
    turns_a = store.list_chat_turns(seed_a.session_id)
    assert len(turns_a) == 2 and turns_a[1].content == "answer for A"


def test_arabic_content_round_trips_through_sqlite_without_mojibake(store, run_service):
    seed = _create_seed_run(run_service)
    arabic_answer = "تم اقتراح برج غلطة بسبب قربه من الفندق الذي اخترته."
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", arabic_answer, language="ar")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "لماذا برج غلطة؟", "ar")
    turns = store.list_chat_turns(seed.session_id)
    assert turns[0].content == "لماذا برج غلطة؟"
    assert turns[1].content == arabic_answer


def test_sql_injection_shaped_text_is_stored_harmlessly_as_plain_content(store, run_service):
    seed = _create_seed_run(run_service)
    injection_message = "'; DROP TABLE chat_turns; --"
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "noted.")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, injection_message, "en")
    turns = store.list_chat_turns(seed.session_id)
    assert turns[0].content == injection_message
    # The table must still exist and be fully queryable -- proves the
    # parameterized-SQL insert never let the string execute as SQL.
    assert len(store.list_chat_turns(seed.session_id)) == 2


# --- persistent-history/grounding correction: bounded model context -------------------


def test_bounded_history_caps_at_twelve_messages(store, run_service):
    seed = _create_seed_run(run_service)
    responses = [_chat_decision("explain_plan", f"answer {i}") for i in range(10)]
    provider = ScriptedChatDecisionProvider(responses)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    for i in range(10):
        service.handle_chat_turn(seed.session_id, seed.run_id, f"question {i}", "en")
    history = service._load_bounded_history(seed.session_id)
    assert len(history) == 12
    # The most recent exchanges are kept -- chronological order preserved.
    assert history[-1]["content"] == "answer 9"
    assert history[-2]["content"] == "question 9"


def test_bounded_history_caps_total_characters_at_twelve_thousand(store, run_service):
    seed = _create_seed_run(run_service)
    long_answer = "x" * 1999  # under the 2000-char per-message cap, but large enough to blow the 12000 total budget in a few turns
    responses = [_chat_decision("explain_plan", long_answer) for _ in range(6)]
    provider = ScriptedChatDecisionProvider(responses)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    for i in range(6):
        service.handle_chat_turn(seed.session_id, seed.run_id, "q" * 1999, "en")
    history = service._load_bounded_history(seed.session_id)
    total_chars = sum(len(m["content"]) for m in history)
    assert total_chars <= 12000


def test_bounded_history_excludes_provider_failed_and_rejected_turns(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "real answer")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatTurnRejected):
        service.handle_chat_turn(seed.session_id, seed.run_id, "x" * 3000, "en")
    service.handle_chat_turn(seed.session_id, seed.run_id, "a real question", "en")
    history = service._load_bounded_history(seed.session_id)
    assert all("x" * 100 not in m["content"] for m in history)
    assert history == [{"role": "user", "content": "a real question"}, {"role": "assistant", "content": "real answer"}]


def test_deterministic_state_query_never_calls_the_provider(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "What is my budget?", "en")
    assert provider.calls == []
    assert "5,000.00 TRY" in result.assistant_message
    assert result.requires_new_run is False


def test_deterministic_state_query_is_persisted_and_included_in_context(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "What is my budget?", "en")
    turns = store.list_chat_turns(seed.session_id)
    assert turns[1].include_in_context is True
    assert turns[1].status.value == "completed"


def test_deterministic_state_query_arabic(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "ما هي ميزانيتي؟", "ar")
    assert provider.calls == []
    assert "ميزانيتك" in result.assistant_message


def test_contextual_it_reference_resolves_via_bounded_history(store, run_service):
    """The confirmed failure this checkpoint fixes: a prior turn
    discussing budget, followed by 'Change it to 1000', must resolve
    'it' to budget using the supplied conversation_history -- proven here
    by asserting the SECOND call's prompt actually carries the first
    exchange as conversation_history (the scripted fake provider records
    the user payload; the real resolution behavior itself is a live-Qwen
    concern verified in the live test, not re-derivable from a scripted
    fake)."""
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([
        _chat_decision("explain_plan", "Your current budget is 5,000.00 TRY."),
        _chat_decision("modify_trip", "Updated your budget.", patch={"budget_amount_minor_units": 100000}),
    ])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "Tell me about my budget in detail", "en")
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "Change it to 1000", "en")
    assert result.requires_new_run is True
    second_call_user_payload = json.loads(provider.calls[1].split("\n", 1)[1])
    assert second_call_user_payload["conversation_history"][0]["content"] == "Tell me about my budget in detail"


# --- persistent-history/grounding correction: history API -----------------------------


def test_get_session_history_returns_full_ordered_transcript(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "answer")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "question", "en")
    turns = service.get_session_history(seed.session_id, seed.run_id)
    assert [t.content for t in turns] == ["question", "answer"]


def test_get_session_history_rejects_cross_session_access(store, run_service):
    seed_a = _create_seed_run(run_service)
    seed_b = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "answer")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed_a.session_id, seed_a.run_id, "question", "en")
    with pytest.raises(ChatSessionAccessRejected):
        service.get_session_history(seed_a.session_id, seed_b.run_id)


def test_get_session_history_unknown_run_is_rejected(store, run_service):
    provider = ScriptedChatDecisionProvider([])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    with pytest.raises(ChatSessionAccessRejected):
        service.get_session_history("s1", "no-such-run")


def test_http_get_chat_session_turns_returns_full_transcript(tmp_db_path):
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "Galata Tower is a landmark.")])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "Why Galata Tower?", "preferred_language": "en",
        })
        response = client.get(f"/v1/chat/sessions/{created['session_id']}/turns", params={"run_id": created["run_id"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["turns"]) == 2
    assert body["turns"][0]["role"] == "user"
    assert body["turns"][1]["content"] == "Galata Tower is a landmark."


def test_http_get_chat_session_turns_cross_session_returns_typed_422(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        response = client.get("/v1/chat/sessions/wrong-session-id/turns", params={"run_id": created["run_id"]})
    assert response.status_code == 422
    assert response.json()["error_code"] == "CHAT_SESSION_ACCESS_REJECTED"


def test_http_get_chat_session_turns_never_leaks_db_path_or_internal_detail(tmp_db_path):
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "answer")])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "hi", "preferred_language": "en",
        })
        response = client.get(f"/v1/chat/sessions/{created['session_id']}/turns", params={"run_id": created["run_id"]})
    serialized = json.dumps(response.json()).lower()
    assert tmp_db_path.lower() not in serialized
    assert "api_key" not in serialized and "secret" not in serialized


# --- persistent-history/grounding correction: provider-error HTTP mapping -------------


def test_http_chat_turn_permanent_provider_failure_returns_503_not_422(tmp_db_path):
    provider = ScriptedChatDecisionProvider([QwenTransportError("boom", transient=False)])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        response = client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "Why?", "preferred_language": "en",
        })
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "PROVIDER_UNAVAILABLE"
    assert body["retriable"] is False


def test_http_chat_turn_transient_exhausted_provider_failure_returns_503_retriable(tmp_db_path):
    provider = ScriptedChatDecisionProvider([
        QwenTransportError("boom", transient=True), QwenTransportError("boom again", transient=True),
    ])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        response = client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "Why?", "preferred_language": "en",
        })
    assert response.status_code == 503
    assert response.json()["retriable"] is True


# --- persistent-history/grounding correction: money grounding -------------------------


def test_no_flight_claim_when_no_flight_observation_exists(store, run_service):
    """A run with no successful search_flights observation must never let
    the assistant claim flight data exists -- verified here by checking
    the bounded summary handed to the provider never fabricates a
    "flights" key."""
    seed = _create_seed_run(run_service, trip_request_partial=_TRIP_REQUEST_PARTIAL)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "No flight data was returned for this run.")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "What flights did you find?", "en")
    user_payload = json.loads(provider.calls[0].split("\n", 1)[1])
    # The seed run's FakeToolExecutor plan in this file never calls
    # search_flights (only synthesize/travel_search_complete) -- so no
    # flights key should ever reach the prompt.
    assert "flights" not in user_payload["result_summary"]


# --- Hybrid Chat C.2: recent-session sidebar listing -----------------------------------


def test_list_recent_sessions_sorted_most_recent_first(store, run_service):
    seed_a = _create_seed_run(run_service)
    seed_b = _create_seed_run(run_service)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: ScriptedChatDecisionProvider([]))
    summaries, total = service.list_recent_sessions()
    assert [s.session_id for s in summaries[:2]] == [seed_b.session_id, seed_a.session_id]
    assert total >= 2


def test_list_recent_sessions_deterministic_title(store, run_service):
    seed = _create_seed_run(run_service)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: ScriptedChatDecisionProvider([]))
    summaries, _total = service.list_recent_sessions()
    match = next(s for s in summaries if s.session_id == seed.session_id)
    assert match.title == "BEY → Istanbul · 10–15 Sep"
    assert match.origin == "BEY" and match.destination == "IST"
    assert match.depart_date == "2026-09-10" and match.return_date == "2026-09-15"
    assert match.preferred_language == "en"


def test_list_recent_sessions_includes_a_session_with_no_chat_turns(store, run_service):
    seed = _create_seed_run(run_service)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: ScriptedChatDecisionProvider([]))
    summaries, _total = service.list_recent_sessions()
    match = next(s for s in summaries if s.session_id == seed.session_id)
    assert match.chat_turn_count == 0


def test_list_recent_sessions_reflects_chat_turn_count_after_a_turn(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "answer")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "hi", "en")
    summaries, _total = service.list_recent_sessions()
    match = next(s for s in summaries if s.session_id == seed.session_id)
    assert match.chat_turn_count == 2


def test_list_recent_sessions_replan_updates_latest_run_id(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("modify_trip", "Updated.", patch={"traveler_count": 3})])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    result = service.handle_chat_turn(seed.session_id, seed.run_id, "add a traveler", "en")
    summaries, _total = service.list_recent_sessions()
    match = next(s for s in summaries if s.session_id == seed.session_id)
    assert match.latest_run_id == result.new_run_id
    assert match.latest_run_id != seed.run_id


def test_list_recent_sessions_limit_is_clamped_to_maximum(store, run_service):
    for _ in range(3):
        _create_seed_run(run_service)
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: ScriptedChatDecisionProvider([]))
    summaries, _total = service.list_recent_sessions(limit=99999, offset=0)
    assert len(summaries) <= 50


def test_list_recent_sessions_never_exposes_transcript_or_result(store, run_service):
    seed = _create_seed_run(run_service)
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "a secret-shaped answer never meant for the sidebar")])
    service = ChatTurnService(store, run_service, chat_decision_provider_factory=lambda: provider)
    service.handle_chat_turn(seed.session_id, seed.run_id, "hi", "en")
    summaries, _total = service.list_recent_sessions()
    match = next(s for s in summaries if s.session_id == seed.session_id)
    serialized = str(match).lower()
    assert "secret-shaped answer" not in serialized
    assert "observations" not in serialized


def test_http_list_chat_sessions_returns_sorted_summaries(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created_a = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created_a["run_id"])
        created_b = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created_b["run_id"])
        response = client.get("/v1/chat/sessions")
    assert response.status_code == 200
    body = response.json()
    session_ids = [s["session_id"] for s in body["sessions"]]
    assert session_ids.index(created_b["session_id"]) < session_ids.index(created_a["session_id"])
    assert body["limit"] == 20 and body["offset"] == 0


def test_http_list_chat_sessions_rejects_limit_above_maximum(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        response = client.get("/v1/chat/sessions", params={"limit": 51})
    assert response.status_code == 422


def test_http_list_chat_sessions_pagination(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        for _ in range(3):
            created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
            # Each run's background execution updates `updated_at` when it
            # reaches a terminal status -- waiting for that here keeps the
            # two paginated reads below looking at STABLE data (never a
            # session reordering itself between the first and second GET
            # because a background job finished in between).
            _poll_until_terminal(client, created["run_id"])
        first = client.get("/v1/chat/sessions", params={"limit": 2, "offset": 0}).json()
        second = client.get("/v1/chat/sessions", params={"limit": 2, "offset": 2}).json()
    assert len(first["sessions"]) == 2
    assert first["total"] >= 3
    first_ids = {s["session_id"] for s in first["sessions"]}
    second_ids = {s["session_id"] for s in second["sessions"]}
    assert first_ids.isdisjoint(second_ids)


def test_http_list_chat_sessions_never_leaks_internal_detail(tmp_db_path):
    provider = ScriptedChatDecisionProvider([_chat_decision("explain_plan", "answer")])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        created = client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL}).json()
        _poll_until_terminal(client, created["run_id"])
        client.post("/v1/chat/turns", json={
            "session_id": created["session_id"], "run_id": created["run_id"],
            "user_message": "hi", "preferred_language": "en",
        })
        response = client.get("/v1/chat/sessions")
    serialized = json.dumps(response.json()).lower()
    assert tmp_db_path.lower() not in serialized
    assert "api_key" not in serialized and "secret" not in serialized
    assert "answer" not in serialized  # the transcript content itself is never in the listing


def test_http_list_chat_sessions_does_not_call_the_chat_decision_provider(tmp_db_path):
    provider = ScriptedChatDecisionProvider([])
    app = _make_app(tmp_db_path, lambda: provider)
    with TestClient(app) as client:
        client.post("/v1/runs", json={"user_message": "Plan a trip.", "trip_request": _TRIP_REQUEST_PARTIAL})
        client.get("/v1/chat/sessions")
    assert provider.calls == []
