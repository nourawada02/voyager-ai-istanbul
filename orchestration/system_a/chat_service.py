"""Hybrid Chat C.1: the production chat-turn service. Owns the whole
server-side chat-turn boundary the frontend is required to go through --
the frontend never calls Qwen directly and never holds a provider
credential (CLAUDE.md's Hybrid Chat C.1 constraint); this module is the
one place that does, reusing the exact same `phase4.qwen_client.
QwenDecisionProvider`/`DecisionProvider` seam `RunService` already uses
for the planning loop (a SEPARATE instance/call, never the same object,
matching this project's established "each role gets its own explicitly
constructed provider" precedent -- ADR 0017 §3).

A chat turn NEVER constructs an MCP/A2A call itself: the only two things
it can ever produce are (a) a bounded, sanitized assistant message, and
(b) an optional, narrowly allowlisted `TripPatch`
(`phase4.chat_models.TripPatch`) that -- once independently re-validated
here via the SAME `phase4.guards.check_input` every ordinary form
submission already goes through -- is handed to the UNMODIFIED
`RunService.create_run`, exactly like a normal trip request. This module
never reaches into `phase4.graph`, `phase4.tools`, Travel MCP, or System
B directly.

The authoritative previous trip request/result is always loaded HERE,
server-side, from `RunStore` by `(session_id, run_id)` -- a caller-
supplied trip request or result is never trusted (CLAUDE.md's Hybrid Chat
C.1 constraint: "must NOT trust the frontend to send a rewritten full
result").

Persistent-history/grounding correction (this checkpoint): every turn is
persisted atomically to `RunStore`'s `chat_turns` table
(`RunStore.append_chat_turn_pair`), a bounded slice of PRIOR persisted
turns is supplied back to Qwen as untrusted structured context (never
concatenated into the system prompt), a small set of purely-factual
"what is my X" questions are answered deterministically from the
authoritative TripRequest with no provider call at all
(`phase4.chat_state_query`), every monetary value shown to Qwen is an
explicit self-describing structure (`phase4.chat_money`), and a genuine
provider outage (permanent auth/quota failure, or a transient failure
that exhausts its bounded retry) is surfaced honestly as
`ChatProviderUnavailable` -- never silently reclassified as user
ambiguity."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from orchestration.system_a.run_store import ChatTurnRecord, ChatTurnStatus, RunStore
from orchestration.system_a.service import RunService
from orchestration.system_a.session_titles import build_session_title
from phase1.models import Language
from phase4.chat_interests import normalize_interests
from phase4.chat_intent import build_bounded_result_summary, build_chat_turn_prompt
from phase4.chat_models import (
    MAX_CHAT_DECISION_REPAIRS,
    MAX_CHAT_LANGUAGE_CORRECTIONS,
    ChatIntent,
    ChatIntentDecision,
    ChatIntentDecisionValidationError,
    TripPatch,
    apply_patch_to_trip_request,
    parse_chat_intent_decision,
)
from phase4.chat_state_query import answer_state_query, detect_state_query
from phase4.guards import check_input, default_wall_clock, resolve_today
from phase4.language_detect import contains_meaningful_arabic, is_predominantly_arabic
from phase4.qwen_client import DecisionProvider, QwenTransportError

logger = logging.getLogger("voyager.system_a.chat_service")

_SAFE_ARABIC_FALLBACK_MESSAGE = (
    "عذرًا، تعذّر إنشاء رد كامل بالعربية في هذه اللحظة. "
    "يمكنك إعادة صياغة سؤالك أو المحاولة مرة أخرى بعد قليل."
)
_SAFE_CLARIFY_FALLBACK_MESSAGE_EN = (
    "I couldn't confidently understand that request. Could you rephrase it -- "
    "for example, ask why a specific recommendation was made, or state exactly "
    "which trip detail (budget, dates, travelers, pace, interests, or language) "
    "you'd like to change?"
)
_PROVIDER_UNAVAILABLE_MESSAGE_EN = (
    "The AI assistant is temporarily unavailable right now. Please try again in a moment."
)
_PROVIDER_UNAVAILABLE_MESSAGE_AR = (
    "مساعد الذكاء الاصطناعي غير متاح مؤقتًا الآن. يرجى المحاولة مرة أخرى بعد قليل."
)
_REJECTED_MESSAGE_EN = "That message could not be accepted. Please rephrase it."
_REJECTED_MESSAGE_AR = "تعذّر قبول هذه الرسالة. يرجى إعادة صياغتها."

# Persistent-history correction §4: deterministic, fixed bounds on what is
# ever supplied to Qwen as conversation context -- never unbounded, and
# never the same as the full persisted transcript (which the history API,
# §5, returns in full for frontend restoration).
_MAX_HISTORY_ROWS_FETCHED = 60  # a generous recent window read from SQLite before filtering/trimming
_MAX_HISTORY_MESSAGES = 12  # at most 6 completed user/assistant exchanges
_MAX_HISTORY_TOTAL_CHARS = 12000
_MAX_STORED_MESSAGE_CHARS = 2000

# Hybrid Chat C.2: recent-session sidebar listing bounds -- a small,
# fixed, named pair (never unbounded), enforced defensively at BOTH this
# layer and the HTTP layer (api.py's own `Query(ge=1, le=...)`), matching
# this project's established "never trust a single validation layer"
# precedent (e.g. `phase4.guards.check_input` runs in addition to, not
# instead of, the graph's own InputGuard).
DEFAULT_SESSION_LIST_LIMIT = 20
MAX_SESSION_LIST_LIMIT = 50


def default_chat_decision_provider_factory() -> DecisionProvider:
    from phase4.qwen_client import QwenDecisionProvider

    return QwenDecisionProvider()


class ChatTurnRejected(Exception):
    """Raised before any run-mutating side effect when the turn itself is
    unsafe/malformed input (message too long, prompt-injection pattern,
    or the (session_id, run_id) pair does not resolve to a real,
    matching run) -- the HTTP layer maps this to a typed 4xx, never an
    internal error."""

    def __init__(self, reason_code: Optional[str], safe_error: Optional[str]) -> None:
        self.reason_code = reason_code
        self.safe_error = safe_error
        super().__init__(safe_error or "chat_turn_rejected")


class ChatProviderUnavailable(Exception):
    """Raised when the chat decision provider itself is genuinely
    unreachable/unusable -- a permanent failure (auth/quota/other
    non-transient transport error) or a transient failure that exhausted
    its bounded retry budget. NEVER raised for a model output that was
    merely hard to parse (that remains a safe `clarify` response, a
    content-quality outcome, not a provider-health one) -- this is the
    fix for the previous behavior that mislabeled a real 403 as "I
    couldn't confidently understand that request." `retriable` mirrors
    the existing `ErrorEnvelope.retriable` convention: True for a
    transient condition a caller may reasonably retry shortly, False for
    a permanent one (bad credentials/quota) a bare retry will not fix."""

    def __init__(self, message: str, *, retriable: bool) -> None:
        self.retriable = retriable
        super().__init__(message)


@dataclass(frozen=True)
class ChatTurnResult:
    session_id: str
    run_id: str
    intent: str
    assistant_message: str
    response_language: str
    trip_patch: Optional[dict[str, Any]]
    requires_new_run: bool
    new_run_id: Optional[str]
    new_run_status: Optional[str]
    clarification_required: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChatHistoryTurn:
    turn_id: str
    role: str
    content: str
    intent: Optional[str]
    response_language: Optional[str]
    status: str
    created_at: str


@dataclass(frozen=True)
class SessionSummary:
    """Hybrid Chat C.2: one recent-session sidebar row -- a small, closed,
    already-safe set of fields (CLAUDE.md §13.4: no internal detail ever
    reaches a caller). Never carries the transcript, a prompt, a provider
    payload, or the raw `result_json` -- only what
    `orchestration.system_a.run_store.SessionSummaryRecord` itself reads,
    plus the deterministically-computed `title`."""

    session_id: str
    latest_run_id: str
    latest_run_status: str
    title: str
    origin: Optional[str]
    destination: Optional[str]
    depart_date: Optional[str]
    return_date: Optional[str]
    preferred_language: Optional[str]
    chat_turn_count: int
    created_at: str
    updated_at: str


class ChatSessionAccessRejected(Exception):
    """Raised by `get_session_history` when `(session_id, run_id)` do not
    belong together -- the HTTP layer maps this to the same safe 4xx
    convention as `ChatTurnRejected`, never a raw 500, and never
    discloses whether the session exists at all versus the run mismatching
    it."""

    def __init__(self, safe_error: str) -> None:
        self.safe_error = safe_error
        super().__init__(safe_error)


def _truncate_for_storage(text: str, limit: int = _MAX_STORED_MESSAGE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit]


class ChatTurnService:
    def __init__(
        self,
        run_store: RunStore,
        run_service: RunService,
        chat_decision_provider_factory: Callable[[], DecisionProvider] = default_chat_decision_provider_factory,
        wall_clock: Callable[[], datetime] = default_wall_clock,
    ):
        self._store = run_store
        self._run_service = run_service
        self._chat_decision_provider_factory = chat_decision_provider_factory
        self._wall_clock = wall_clock

    def handle_chat_turn(
        self, session_id: str, run_id: str, user_message: str, preferred_language: str
    ) -> ChatTurnResult:
        record = self._store.get_run(run_id)
        if record is None or record.session_id != session_id:
            # No legitimate session context exists to attach anything to
            # -- nothing is persisted for an access-control failure.
            raise ChatTurnRejected("input_rejected", "run_not_found_for_session")

        today = resolve_today(self._wall_clock)
        target_language_guess = "ar" if preferred_language == "ar" or is_predominantly_arabic(user_message) else preferred_language

        # Message-level safety (length/prompt-injection) reuses the exact
        # same deterministic guard every ordinary form submission already
        # goes through -- `trip_request=None` here means only the message
        # itself is checked, not any trip fields. A rejection IS persisted
        # (a real, validated session already exists) so the transcript
        # honestly shows what the user sent and that it was rejected --
        # `include_in_context=False` keeps it out of every future model
        # call.
        message_guard = check_input(user_message, None, today=today)
        if not message_guard.accepted:
            rejection_message = _REJECTED_MESSAGE_AR if target_language_guess == "ar" else _REJECTED_MESSAGE_EN
            self._persist_turn(
                session_id, run_id, user_message, rejection_message, intent=None,
                response_language=target_language_guess, status=ChatTurnStatus.REJECTED, include_in_context=False,
            )
            raise ChatTurnRejected(
                message_guard.reason_code.value if message_guard.reason_code else None, message_guard.safe_error
            )

        trip_request = record.request.get("trip_request") if isinstance(record.request, dict) else None
        result_summary = build_bounded_result_summary(record.result)
        target_language = target_language_guess
        history = self._load_bounded_history(session_id)

        # Persistent-history/grounding correction §7: a small, closed set
        # of purely-factual "what is my X" questions never need a Qwen
        # call at all -- the answer already sits, unambiguous, in the
        # authoritative TripRequest. Only fires on a confident,
        # conservative pattern match; anything else falls through to the
        # bounded Qwen path below, unchanged.
        state_field = detect_state_query(user_message) if trip_request is not None else None
        if state_field is not None:
            answer = answer_state_query(state_field, trip_request, target_language)
            if answer is not None:
                self._persist_turn(
                    session_id, run_id, user_message, answer, intent=ChatIntent.EXPLAIN_PLAN.value,
                    response_language=target_language, status=ChatTurnStatus.COMPLETED, include_in_context=True,
                )
                return ChatTurnResult(
                    session_id=session_id, run_id=run_id, intent=ChatIntent.EXPLAIN_PLAN.value,
                    assistant_message=answer, response_language=target_language, trip_patch=None,
                    requires_new_run=False, new_run_id=None, new_run_status=None,
                    clarification_required=False, warnings=[],
                )

        provider = self._chat_decision_provider_factory()
        try:
            decision = self._classify_with_repairs(
                provider, user_message, trip_request, result_summary, preferred_language, target_language, history
            )
            decision = self._enforce_response_language(
                provider, decision, user_message, trip_request, result_summary, preferred_language, target_language, history
            )
        except ChatProviderUnavailable as exc:
            unavailable_message = _PROVIDER_UNAVAILABLE_MESSAGE_AR if target_language == "ar" else _PROVIDER_UNAVAILABLE_MESSAGE_EN
            self._persist_turn(
                session_id, run_id, user_message, unavailable_message, intent=None,
                response_language=target_language, status=ChatTurnStatus.PROVIDER_FAILED, include_in_context=False,
            )
            raise

        warnings: list[str] = []
        merged_trip_request_partial: Optional[dict[str, Any]] = None
        requires_new_run = False
        intent = decision.intent
        clarification_required = decision.requires_clarification or intent == ChatIntent.CLARIFY

        if intent in (ChatIntent.MODIFY_TRIP, ChatIntent.REGENERATE_TRIP) and not clarification_required:
            if trip_request is None:
                intent = ChatIntent.CLARIFY
                clarification_required = True
            else:
                resolved_patch, patch_warnings = self._resolve_patch(decision.patch)
                warnings.extend(patch_warnings)
                if intent == ChatIntent.MODIFY_TRIP and resolved_patch.is_empty():
                    intent = ChatIntent.CLARIFY
                    clarification_required = True
                else:
                    merged = apply_patch_to_trip_request(trip_request, resolved_patch)
                    trip_guard = check_input(user_message, merged, today=today)
                    if not trip_guard.accepted:
                        intent = ChatIntent.CLARIFY
                        clarification_required = True
                        warnings.append(f"patch_rejected:{trip_guard.safe_error}")
                    else:
                        merged_trip_request_partial = {
                            k: v for k, v in merged.items() if k not in ("schema_version", "session_id", "trace_id")
                        }
                        requires_new_run = True

        new_run_id: Optional[str] = None
        new_run_status: Optional[str] = None
        response_run_id = run_id
        if requires_new_run and merged_trip_request_partial is not None:
            new_record, _created = self._run_service.create_run(
                user_message=user_message,
                trip_request_partial=merged_trip_request_partial,
                idempotency_key=None,
                session_id=session_id,
            )
            new_run_id = new_record.run_id
            new_run_status = new_record.status.value
            response_run_id = new_record.run_id

        self._persist_turn(
            session_id, run_id, user_message, decision.assistant_message, intent=intent.value,
            response_language=decision.response_language.value, status=ChatTurnStatus.COMPLETED, include_in_context=True,
        )

        return ChatTurnResult(
            session_id=session_id,
            run_id=response_run_id,
            intent=intent.value,
            assistant_message=decision.assistant_message,
            response_language=decision.response_language.value,
            trip_patch=merged_trip_request_partial if requires_new_run else None,
            requires_new_run=requires_new_run,
            new_run_id=new_run_id,
            new_run_status=new_run_status,
            clarification_required=clarification_required,
            warnings=warnings,
        )

    def get_session_history(self, session_id: str, run_id: str) -> list[ChatHistoryTurn]:
        """Hybrid Chat C.1 §5: validates `(session_id, run_id)` belong
        together (the exact same ownership check `handle_chat_turn`
        already performs) before returning ANY transcript row -- a
        caller can never enumerate another session's history by guessing
        a session_id. Returns the FULL persisted transcript (every
        status, not the model-context-bounded slice) so the frontend can
        honestly restore exactly what happened, including a past
        rejection or provider failure."""
        record = self._store.get_run(run_id)
        if record is None or record.session_id != session_id:
            raise ChatSessionAccessRejected("run_not_found_for_session")
        rows = self._store.list_chat_turns(session_id)
        return [
            ChatHistoryTurn(
                turn_id=r.turn_id, role=r.role.value, content=r.content, intent=r.intent,
                response_language=r.response_language, status=r.status.value, created_at=r.created_at,
            )
            for r in rows
        ]

    def list_recent_sessions(self, limit: int = DEFAULT_SESSION_LIST_LIMIT, offset: int = 0) -> tuple[list[SessionSummary], int]:
        """Hybrid Chat C.2: the recent-session sidebar's one bounded
        listing call -- never one request per sidebar entry. `limit` is
        defensively re-clamped here (in addition to the HTTP layer's own
        `Query(ge=1, le=MAX_SESSION_LIST_LIMIT)`), `offset` never goes
        negative. Returns `(summaries, total_session_count)` so a caller
        can determine whether another page exists without a second
        query. No Qwen call, no new database, no Qdrant -- pure SQLite
        reads plus deterministic title formatting
        (`orchestration.system_a.session_titles.build_session_title`)."""
        bounded_limit = max(1, min(limit, MAX_SESSION_LIST_LIMIT))
        bounded_offset = max(0, offset)
        rows = self._store.list_sessions(bounded_limit, bounded_offset)
        total = self._store.count_sessions()
        summaries = []
        for row in rows:
            trip_request = row.trip_request or {}
            preferences = trip_request.get("preferences") or {}
            summaries.append(SessionSummary(
                session_id=row.session_id, latest_run_id=row.latest_run_id,
                latest_run_status=row.latest_run_status.value, title=build_session_title(row.trip_request),
                origin=trip_request.get("origin"), destination=trip_request.get("destination"),
                depart_date=trip_request.get("depart_date"), return_date=trip_request.get("return_date"),
                preferred_language=preferences.get("language"), chat_turn_count=row.chat_turn_count,
                created_at=row.created_at, updated_at=row.updated_at,
            ))
        return summaries, total

    # --- internal helpers ------------------------------------------------------------

    def _persist_turn(
        self,
        session_id: str,
        run_id: str,
        user_content: str,
        assistant_content: str,
        *,
        intent: Optional[str],
        response_language: Optional[str],
        status: ChatTurnStatus,
        include_in_context: bool,
    ) -> None:
        self._store.append_chat_turn_pair(
            session_id=session_id, run_id=run_id,
            user_turn_id=str(uuid.uuid4()), user_content=_truncate_for_storage(user_content),
            assistant_turn_id=str(uuid.uuid4()), assistant_content=_truncate_for_storage(assistant_content),
            intent=intent, response_language=response_language, status=status, include_in_context=include_in_context,
        )

    def _load_bounded_history(self, session_id: str) -> list[dict[str, str]]:
        rows: list[ChatTurnRecord] = self._store.list_chat_turns(session_id, limit=_MAX_HISTORY_ROWS_FETCHED)
        included = [r for r in rows if r.include_in_context][-_MAX_HISTORY_MESSAGES:]
        total_chars = sum(len(r.content) for r in included)
        while total_chars > _MAX_HISTORY_TOTAL_CHARS and included:
            # Deterministic oldest-removal (persistent-history correction
            # §4) -- `included` is already in chronological order, so
            # `pop(0)` always removes the OLDEST remaining message first,
            # preserving order for everything that stays.
            removed = included.pop(0)
            total_chars -= len(removed.content)
        return [{"role": r.role.value, "content": r.content} for r in included]

    def _resolve_patch(self, patch: Optional[TripPatch]) -> tuple[TripPatch, list[str]]:
        if patch is None:
            return TripPatch(), []
        warnings: list[str] = []
        add_interests = patch.add_interests
        remove_interests = patch.remove_interests
        if patch.add_interests:
            add_interests, unrecognized = normalize_interests(patch.add_interests)
            if unrecognized:
                warnings.append(f"unrecognized_interests_to_add:{','.join(unrecognized)}")
            add_interests = add_interests or None
        if patch.remove_interests:
            remove_interests, unrecognized = normalize_interests(patch.remove_interests)
            if unrecognized:
                warnings.append(f"unrecognized_interests_to_remove:{','.join(unrecognized)}")
            remove_interests = remove_interests or None
        resolved = patch.model_copy(update={"add_interests": add_interests, "remove_interests": remove_interests})
        return resolved, warnings

    def _classify_with_repairs(
        self,
        provider: DecisionProvider,
        user_message: str,
        trip_request: Optional[dict[str, Any]],
        result_summary: dict[str, Any],
        preferred_language: str,
        target_language: str,
        history: list[dict[str, str]],
    ) -> ChatIntentDecision:
        correction_note: Optional[str] = None
        for attempt in range(MAX_CHAT_DECISION_REPAIRS + 1):
            system, user = build_chat_turn_prompt(
                user_message=user_message, trip_request=trip_request, result_summary=result_summary,
                preferred_language=preferred_language, target_response_language=target_language,
                correction_note=correction_note, history=history,
            )
            try:
                raw_text = provider.generate(system, user)
                decision = parse_chat_intent_decision(json.loads(raw_text))
            except (json.JSONDecodeError, ChatIntentDecisionValidationError) as exc:
                # The validation error text is fed back into the NEXT
                # prompt only (never shown to the end user or written to
                # a log at more than its type name) -- it is Qwen's own
                # output being reflected back to the same model for
                # self-correction, not a security boundary, and it gives
                # a far sharper repair signal than a generic reminder
                # (e.g. exactly which field name it invented). This is a
                # content-quality failure, never a provider-health one --
                # exhausting this budget still falls through to a safe
                # `clarify`, never `ChatProviderUnavailable`.
                detail = str(exc)[:500]
                correction_note = (
                    "Your previous response was not a single valid JSON object matching the required "
                    "contract. Return exactly one valid JSON object with keys: intent, assistant_message, "
                    "response_language, patch, requires_clarification, clarification_reason. The specific "
                    f"validation error was: {detail}"
                )
                logger.info("chat-turn decision repair attempt: %s", type(exc).__name__)
                continue
            except QwenTransportError as exc:
                # A genuine provider-health problem -- fixed boundary
                # (persistent-history/grounding correction §10): a
                # permanent failure (bad credentials/quota, or any other
                # non-transient transport error) is never silently
                # reclassified as "I couldn't confidently understand that
                # request"; it raises `ChatProviderUnavailable` instead. A
                # transient failure (timeout/connect/429/5xx) still gets
                # the same bounded retry a decision-format repair gets --
                # mirrors `phase4.graph._classify_capability`'s own
                # established transient-transport-retry precedent -- but
                # once that budget is exhausted, it is ALSO surfaced as
                # `ChatProviderUnavailable(retriable=True)`, never masked.
                if exc.transient and attempt < MAX_CHAT_DECISION_REPAIRS:
                    logger.info("chat-turn decision provider transient transport failure, retrying: status=%s", exc.status_code)
                    continue
                logger.error(
                    "chat-turn decision provider transport failure (transient=%s): status=%s",
                    exc.transient, exc.status_code,
                )
                raise ChatProviderUnavailable("chat_decision_provider_unavailable", retriable=exc.transient) from exc
            return decision.model_copy(update={"response_language": Language(target_language)})
        return self._safe_clarify_fallback(target_language)

    def _enforce_response_language(
        self,
        provider: DecisionProvider,
        decision: ChatIntentDecision,
        user_message: str,
        trip_request: Optional[dict[str, Any]],
        result_summary: dict[str, Any],
        preferred_language: str,
        target_language: str,
        history: list[dict[str, str]],
    ) -> ChatIntentDecision:
        """Hybrid Chat C.1 §6: when Arabic is the target language, the
        assistant message must actually contain meaningful Arabic text.
        One bounded correction attempt is permitted (MAX_CHAT_LANGUAGE_
        CORRECTIONS); if that also fails, a safe, honest Arabic fallback
        message is returned rather than silently displaying English. A
        transport failure during this purely-cosmetic correction attempt
        is treated the same way (safe fallback, not propagated) -- the
        MAIN decision already succeeded; this step only polishes its
        language, and Qwen having just answered successfully moments ago
        makes a hard provider-unavailable escalation here disproportionate."""
        if target_language != "ar" or contains_meaningful_arabic(decision.assistant_message):
            return decision
        for _attempt in range(MAX_CHAT_LANGUAGE_CORRECTIONS):
            system, user = build_chat_turn_prompt(
                user_message=user_message, trip_request=trip_request, result_summary=result_summary,
                preferred_language=preferred_language, target_response_language="ar",
                correction_note="Your previous assistant_message was not written in Arabic. Rewrite "
                "assistant_message entirely in Arabic this time, preserving proper names where useful.",
                history=history,
            )
            try:
                raw_text = provider.generate(system, user)
                corrected = parse_chat_intent_decision(json.loads(raw_text))
            except (json.JSONDecodeError, ChatIntentDecisionValidationError, QwenTransportError) as exc:
                logger.info("chat-turn language-correction attempt failed: %s", type(exc).__name__)
                continue
            if contains_meaningful_arabic(corrected.assistant_message):
                return corrected.model_copy(update={"response_language": Language.AR})
        return decision.model_copy(update={"assistant_message": _SAFE_ARABIC_FALLBACK_MESSAGE, "response_language": Language.AR})

    def _safe_clarify_fallback(self, target_language: str) -> ChatIntentDecision:
        message = _SAFE_ARABIC_FALLBACK_MESSAGE if target_language == "ar" else _SAFE_CLARIFY_FALLBACK_MESSAGE_EN
        return ChatIntentDecision(
            intent=ChatIntent.CLARIFY, assistant_message=message, response_language=Language(target_language),
            patch=None, requires_clarification=True, clarification_reason="classification_unavailable",
        )
