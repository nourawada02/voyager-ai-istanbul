"""Hermetic unit tests for `orchestration.system_a.session_titles` --
deterministic recent-session sidebar title generation. No network call
anywhere in this file."""

from __future__ import annotations

from orchestration.system_a.session_titles import build_session_title


def test_same_month_date_range_formats_compactly():
    trip_request = {"origin": "BEY", "depart_date": "2026-09-19", "return_date": "2026-09-23"}
    assert build_session_title(trip_request) == "BEY → Istanbul · 19–23 Sep"


def test_cross_month_date_range_includes_both_months():
    trip_request = {"origin": "LHR", "depart_date": "2026-09-28", "return_date": "2026-10-03"}
    assert build_session_title(trip_request) == "LHR → Istanbul · 28 Sep–3 Oct"


def test_cross_year_date_range_includes_both_years():
    trip_request = {"origin": "JFK", "depart_date": "2026-12-29", "return_date": "2027-01-02"}
    assert build_session_title(trip_request) == "JFK → Istanbul · 29 Dec 2026–2 Jan 2027"


def test_none_trip_request_returns_safe_fallback():
    assert build_session_title(None) == "Conversation"


def test_missing_origin_returns_safe_fallback():
    trip_request = {"depart_date": "2026-09-19", "return_date": "2026-09-23"}
    assert build_session_title(trip_request) == "Conversation"


def test_missing_dates_returns_safe_fallback():
    assert build_session_title({"origin": "BEY"}) == "Conversation"


def test_malformed_date_returns_safe_fallback_never_raises():
    trip_request = {"origin": "BEY", "depart_date": "not-a-date", "return_date": "2026-09-23"}
    assert build_session_title(trip_request) == "Conversation"


def test_never_calls_out_to_any_external_provider():
    """Documented as a structural fact: the module has no import of
    `phase4.qwen_client` or any HTTP/network primitive."""
    import orchestration.system_a.session_titles as module
    source = open(module.__file__, encoding="utf-8").read()
    assert "qwen" not in source.lower()
    assert "urllib" not in source and "httpx" not in source and "requests" not in source
