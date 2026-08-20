"""Hermetic tests for `ProductionToolExecutor` (Checkpoint Phase 4 D.1).
Every dependency is a fake/stub -- no real socket, no real MCP/A2A/
provider process. Proves: correct action dispatch; `estimate_fair_price`
only ever forwards a `stay_id` that actually appeared in a prior
successful `search_stays` observation (never an invented one);
`call_istanbul_expert` builds its `LocalPlanRequest` from context, never
asking the decision provider to reproduce a stay candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from phase4.context import ExecutionContext
from phase4.models import Action

from orchestration.system_a.tool_executor import ProductionToolExecutor, _build_local_plan_request, _validated_stay_id


@dataclass
class _FakeMcpClient:
    calls: list = field(default_factory=list)
    response: dict = field(default_factory=lambda: {"status": "success", "result": {"stays": []}})

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool_name, arguments))
        return self.response


@dataclass
class _FakeA2AClient:
    calls: list = field(default_factory=list)
    response: dict = field(default_factory=lambda: {"status": "success", "result": {"itinerary": "ok"}})

    def call_istanbul_expert(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return self.response


def _executor(mcp=None, a2a=None) -> ProductionToolExecutor:
    return ProductionToolExecutor(
        weather_provider=object(), web_evidence_provider=object(), flight_provider=object(),
        mcp_client=mcp or _FakeMcpClient(), a2a_client=a2a or _FakeA2AClient(),
    )


def _context(observations=(), normalized_request=None) -> ExecutionContext:
    return ExecutionContext(
        session_id="11111111-1111-1111-1111-111111111111",
        trace_id="22222222-2222-2222-2222-222222222222",
        normalized_request=normalized_request or {},
        observations=observations,
        deadline_monotonic=60.0,
        cancellation_check=lambda: False,
    )


_SEARCH_STAYS_OBSERVATION = {
    "action": "search_stays",
    "status": "success",
    # search_stays observations are NOT ProviderResponseEnvelope-wrapped --
    # "envelope" here IS the raw Travel MCP SearchStaysResult directly.
    "envelope": {
        "stays": [
            {"stay": {"stay_id": "stay_real_001", "coordinates": {"lat": 41.0, "lon": 28.9}}, "fair_price": {}, "rank": 1}
        ]
    },
}


def test_search_stays_dispatches_to_mcp_with_enriched_arguments():
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context()
    executor.execute(Action.SEARCH_STAYS, {"guest_count": 2, "check_in": "2026-09-10", "check_out": "2026-09-15"}, context)
    tool_name, args = mcp.calls[0]
    assert tool_name == "search_stays"
    assert args["session_id"] == context.session_id
    assert args["guest_count"] == 2
    assert args["currency"] == "TRY"
    assert args["ranking_mode"] == "preliminary_price_value_desc"


def test_estimate_fair_price_forwards_a_validated_prior_stay_id():
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION,))
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_real_001"}, context)
    assert result["status"] == "success"
    tool_name, args = mcp.calls[0]
    assert tool_name == "estimate_fair_price"
    assert args["stay_id"] == "stay_real_001"


def test_estimate_fair_price_rejects_an_unvalidated_stay_id_never_forwards_it():
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION,))
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_invented_by_qwen"}, context)
    assert result["status"] == "invalid_request"
    assert mcp.calls == []  # never even reached the MCP client


def test_estimate_fair_price_with_no_prior_search_stays_is_invalid_request():
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=())
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "anything"}, context)
    assert result["status"] == "invalid_request"
    assert mcp.calls == []


def test_validated_stay_id_helper_ignores_failed_observations():
    failed_obs = {**_SEARCH_STAYS_OBSERVATION, "status": "timeout"}
    context = _context(observations=(failed_obs,))
    assert _validated_stay_id("stay_real_001", context) is None


TRIP_REQUEST = {
    "depart_date": "2026-09-10", "return_date": "2026-09-15",
    "preferences": {"interests": ["history"], "pace": "relaxed", "language": "en", "mobility_constraints": []},
}


def test_call_istanbul_expert_builds_local_plan_request_from_context():
    a2a = _FakeA2AClient()
    executor = _executor(a2a=a2a)
    context = _context(
        observations=(_SEARCH_STAYS_OBSERVATION,),
        normalized_request={"trip_request": TRIP_REQUEST},
    )
    result = executor.execute(Action.CALL_ISTANBUL_EXPERT, {"question": "what should I see?"}, context)
    assert result["status"] == "success"
    sent = a2a.calls[0]
    assert sent["trip_start_date"] == "2026-09-10"
    assert sent["trip_end_date"] == "2026-09-15"
    assert sent["interests"] == ["history"]
    assert sent["stay_candidates"] == [{"candidate_id": "stay_real_001", "lat": 41.0, "lon": 28.9}]
    assert sent["session_id"] == context.session_id


def test_call_istanbul_expert_without_prior_stays_is_invalid_request():
    a2a = _FakeA2AClient()
    executor = _executor(a2a=a2a)
    context = _context(observations=(), normalized_request={"trip_request": TRIP_REQUEST})
    result = executor.execute(Action.CALL_ISTANBUL_EXPERT, {"question": "x"}, context)
    assert result["status"] == "invalid_request"
    assert a2a.calls == []


def test_call_istanbul_expert_without_trip_request_is_invalid_request():
    a2a = _FakeA2AClient()
    executor = _executor(a2a=a2a)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION,), normalized_request={})
    result = executor.execute(Action.CALL_ISTANBUL_EXPERT, {"question": "x"}, context)
    assert result["status"] == "invalid_request"


def test_build_local_plan_request_includes_weather_context_when_available():
    weather_obs = {"action": "get_weather", "status": "success", "envelope": {"result": {"location": "Istanbul", "kind": "forecast"}}}
    context = _context(
        observations=(_SEARCH_STAYS_OBSERVATION, weather_obs),
        normalized_request={"trip_request": TRIP_REQUEST},
    )
    request = _build_local_plan_request(context)
    assert request is not None
    assert request["weather_context"] == {"location": "Istanbul", "kind": "forecast"}


def test_unknown_action_returns_unavailable_never_raises():
    executor = _executor()
    result = executor.execute(Action.ASK_CLARIFICATION, {}, _context())
    assert result["status"] == "unavailable"


# --- P.1: an unexpected root-provider exception must never reach the caller --------
#
# Production incident P.1: a packaging gap (contracts/ missing from the
# built agent-system-a image) made a real provider call raise a raw
# FileNotFoundError, which propagated uncaught through `execute()` and
# aborted the entire bounded run with `internal_execution_error` -- never
# reaching Observe as a typed status the ReAct loop already knows how to
# handle. `weather_provider`/`web_evidence_provider`/`flight_provider`
# here are bare `object()` instances (this file's own existing `_executor()`
# fixture) -- calling e.g. `.fetch_weather(...)` on one raises a genuine,
# unscripted `AttributeError`, exercising the real exception path rather
# than a mocked one.


def test_unexpected_weather_provider_exception_becomes_provider_error_not_a_raised_exception():
    executor = _executor()
    result = executor.execute(Action.GET_WEATHER, {"date_from": "2026-09-10", "date_to": "2026-09-10"}, _context())
    assert result == {"status": "provider_error", "result": None}


def test_unexpected_web_search_provider_exception_becomes_provider_error_not_a_raised_exception():
    executor = _executor()
    result = executor.execute(Action.WEB_SEARCH, {"query": "Hagia Sophia hours"}, _context())
    assert result == {"status": "provider_error", "result": None}


def test_unexpected_flight_provider_exception_becomes_provider_error_not_a_raised_exception():
    executor = _executor()
    result = executor.execute(
        Action.SEARCH_FLIGHTS,
        {"origin": "BEY", "destination": "IST", "depart_date": "2026-09-10", "passenger_count": 1},
        _context(),
    )
    assert result == {"status": "provider_error", "result": None}


# --- Fair-price correction checkpoint: root cause was NOT a Travel MCP or
# provider defect -- Qwen was never shown the real stay_id values a prior
# search_stays call had already returned, so it had nothing to copy and had
# to guess. The fix grounds phase4/specialist.py's prompt in the real
# values (see services/planner-a/phase4/tests/test_specialist_prompt.py)
# and adds ONE safe, explicit fallback here (a name match) plus a closed
# reason code for the specialist's own bounded retry. Nothing about
# _build_search_stays_arguments/GET_WEATHER/WEB_SEARCH/SEARCH_FLIGHTS/
# CALL_ISTANBUL_EXPERT changed -- every test above this comment still
# passes completely unmodified. ------------------------------------------

_SEARCH_STAYS_OBSERVATION_WITH_NAME = {
    "action": "search_stays",
    "status": "success",
    "envelope": {
        "stays": [
            {
                "stay": {
                    "stay_id": "stay_real_001", "name": "Beyoglu Boutique Hotel",
                    "coordinates": {"lat": 41.0, "lon": 28.9},
                },
                "fair_price": {}, "rank": 1,
            },
            {
                "stay": {
                    "stay_id": "stay_real_002", "name": "Sultanahmet Palace Suites",
                    "coordinates": {"lat": 41.01, "lon": 28.98},
                },
                "fair_price": {}, "rank": 2,
            },
        ]
    },
}


def test_estimate_fair_price_canonical_arguments_pass_unchanged():
    """Test 1/10: the exact valid shape (a real stay_id, copied verbatim)
    still works exactly as before, with no unexpected fields leaking
    into the MCP call and currency always TRY (never relabeled)."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,))
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_real_001"}, context)
    assert result["status"] == "success"
    tool_name, args = mcp.calls[0]
    assert tool_name == "estimate_fair_price"
    assert args == {
        "session_id": context.session_id, "trace_id": context.trace_id,
        "stay_id": "stay_real_001", "currency": "TRY", "schema_version": "1.0.0",
    }


def test_estimate_fair_price_corrects_a_hotel_name_used_in_place_of_its_stay_id():
    """Test 2/10: the exact previously-plausible malformed shape -- Qwen,
    never having been shown the opaque stay_id, echoes the hotel NAME it
    was shown instead -- is corrected once via the safe, explicit
    name-alias resolution, never a fuzzy/partial match."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,))
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "Beyoglu Boutique Hotel"}, context)
    assert result["status"] == "success"
    tool_name, args = mcp.calls[0]
    assert args["stay_id"] == "stay_real_001"  # resolved to the REAL id, name itself never forwarded

    # Case-insensitivity is part of the same safe alias, not a separate rule.
    mcp2 = _FakeMcpClient()
    executor2 = _executor(mcp=mcp2)
    result2 = executor2.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "sultanahmet palace suites"}, context)
    assert result2["status"] == "success"
    assert mcp2.calls[0][1]["stay_id"] == "stay_real_002"


def test_estimate_fair_price_rejects_a_hallucinated_stay_id_matching_neither_id_nor_name():
    """Test 3/10: missing/non-recoverable values remain rejected -- a
    value that matches no real id AND no real name is never guessed at,
    never partially matched, and never forwarded to MCP."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,))
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "The Grand Nonexistent Hotel"}, context)
    assert result["status"] == "invalid_request"
    assert result["reason"] == "stay_id_not_recognized"
    assert mcp.calls == []


def test_estimate_fair_price_reason_distinguishes_no_search_stays_from_unrecognized_id():
    """The two distinct, closed reason codes reported internally (never
    to the end user) so a retry has something concrete to react to."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    no_evidence = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "anything"}, _context())
    assert no_evidence["reason"] == "no_prior_search_stays"

    with_evidence = executor.execute(
        Action.ESTIMATE_FAIR_PRICE, {"stay_id": "invented"}, _context(observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,))
    )
    assert with_evidence["reason"] == "stay_id_not_recognized"


def test_estimate_fair_price_mcp_args_never_include_unexpected_arguments():
    """Test: unknown fields Qwen might emit alongside stay_id are never
    silently forwarded -- mcp_args is built fresh from fixed keys, never
    by spreading the raw decision arguments dict."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,))
    executor.execute(
        Action.ESTIMATE_FAIR_PRICE,
        {"stay_id": "stay_real_001", "hallucinated_extra_field": "x", "currency": "USD"},
        context,
    )
    _, args = mcp.calls[0]
    assert set(args.keys()) == {"session_id", "trace_id", "stay_id", "currency", "schema_version"}
    assert args["currency"] == "TRY"  # the caller-supplied "USD" above is never honored -- see next test


def test_estimate_fair_price_currency_is_always_try_never_relabeled_usd():
    """Do not duplicate currency conversion or mislabel TRY as USD:
    Travel MCP's own contract requires currency == 'TRY' (a const); any
    USD figure the user sees is produced downstream by
    orchestration/system_a/budget_summary.py's FX conversion, never by
    asking Travel MCP itself for a USD-denominated fair price."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(
        observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,),
        normalized_request={"trip_request": {**TRIP_REQUEST, "preferences": {**TRIP_REQUEST["preferences"], "currency": "USD"}}},
    )
    executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_real_001"}, context)
    assert mcp.calls[0][1]["currency"] == "TRY"


def test_estimate_fair_price_arabic_request_reaches_the_same_valid_tool_contract():
    """Arabic input can reach the same valid tool contract -- the fix
    operates only on prior observations, never on request language."""
    mcp = _FakeMcpClient()
    executor = _executor(mcp=mcp)
    context = _context(
        observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,),
        normalized_request={"trip_request": {**TRIP_REQUEST, "preferences": {**TRIP_REQUEST["preferences"], "language": "ar"}}},
    )
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_real_002"}, context)
    assert result["status"] == "success"
    assert mcp.calls[0][1]["stay_id"] == "stay_real_002"


def test_known_stays_helper_uses_most_recent_successful_search_stays_observation_only():
    from orchestration.system_a.tool_executor import _known_stays

    older = {**_SEARCH_STAYS_OBSERVATION_WITH_NAME}
    newer = {
        "action": "search_stays", "status": "success",
        "envelope": {"stays": [{"stay": {"stay_id": "stay_newest", "name": "Newest Hotel"}, "fair_price": {}, "rank": 1}]},
    }
    context = _context(observations=(older, newer))
    known = _known_stays(context)
    assert known == [{"stay_id": "stay_newest", "name": "Newest Hotel"}]


def test_estimate_fair_price_provider_error_from_mcp_remains_visible():
    """Once stay_id validates, a genuine MCP-side provider failure passes
    through completely unmodified -- never silently converted to
    success, never swallowed."""
    mcp = _FakeMcpClient(response={"status": "provider_error", "result": None})
    executor = _executor(mcp=mcp)
    context = _context(observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,))
    result = executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_real_001"}, context)
    assert result == {"status": "provider_error", "result": None}
    assert len(mcp.calls) == 1  # the call really was attempted, not skipped


def test_fair_price_fix_does_not_affect_unrelated_tools_in_the_same_context():
    """search_flights/search_stays/get_weather/call_istanbul_expert all
    still dispatch exactly as before when exercised alongside an
    estimate_fair_price call in the same context."""
    mcp = _FakeMcpClient()
    a2a = _FakeA2AClient()
    executor = _executor(mcp=mcp, a2a=a2a)
    context = _context(
        observations=(_SEARCH_STAYS_OBSERVATION_WITH_NAME,),
        normalized_request={"trip_request": TRIP_REQUEST},
    )
    executor.execute(Action.SEARCH_STAYS, {"guest_count": 2}, context)
    executor.execute(Action.ESTIMATE_FAIR_PRICE, {"stay_id": "stay_real_001"}, context)
    executor.execute(Action.CALL_ISTANBUL_EXPERT, {"question": "x"}, context)
    weather_result = executor.execute(Action.GET_WEATHER, {"date_from": "2026-09-10", "date_to": "2026-09-10"}, context)
    tool_names = [c[0] for c in mcp.calls]
    assert tool_names == ["search_stays", "estimate_fair_price"]
    assert len(a2a.calls) == 1
    assert weather_result == {"status": "provider_error", "result": None}  # bare object() provider, as in the existing tests above
