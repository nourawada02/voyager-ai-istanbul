"""Hybrid Chat C.2: deterministic recent-session sidebar titles -- never
an LLM call (matching this project's existing "small, fixed, explicit
formatting -- never a model call for a fact code can already compute"
precedent, e.g. `phase4.chat_state_query`). A title is built ONLY from
the session's own already-validated `TripRequest` fields; when those are
absent (a narrow-scope session that never carried a structured trip),
a safe, honest fallback is used instead of guessing.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any, Optional

# V1 scope is Istanbul-only (phase1.models.TripRequest.destination is
# pattern-locked to "IST") -- a fixed, readable display name is used
# instead of the raw airport code, matching the existing frontend form's
# own "Istanbul (IST)" convention (services/frontend/phase6/app.py).
_DESTINATION_DISPLAY_NAME = "Istanbul"

_FALLBACK_TITLE = "Conversation"


def _format_date_range(depart_date: str, return_date: str) -> Optional[str]:
    try:
        d1 = date_cls.fromisoformat(depart_date)
        d2 = date_cls.fromisoformat(return_date)
    except (TypeError, ValueError):
        return None
    month1, month2 = d1.strftime("%b"), d2.strftime("%b")
    if d1.year != d2.year:
        return f"{d1.day} {month1} {d1.year}–{d2.day} {month2} {d2.year}"
    if d1.month != d2.month:
        return f"{d1.day} {month1}–{d2.day} {month2}"
    return f"{d1.day}–{d2.day} {month1}"


def build_session_title(trip_request: Optional[dict[str, Any]]) -> str:
    """Returns a short, deterministic title, e.g. "BEY → Istanbul ·
    19–23 Sep" -- or a safe, honest fallback ("Conversation") when
    `trip_request` is absent or missing the fields a title needs. Never
    raises on malformed input; never invents an origin/date it cannot
    read directly from the stored request."""
    if not isinstance(trip_request, dict):
        return _FALLBACK_TITLE
    origin = trip_request.get("origin")
    depart_date = trip_request.get("depart_date")
    return_date = trip_request.get("return_date")
    if not origin or not depart_date or not return_date:
        return _FALLBACK_TITLE
    date_range = _format_date_range(depart_date, return_date)
    if date_range is None:
        return _FALLBACK_TITLE
    return f"{origin} → {_DESTINATION_DISPLAY_NAME} · {date_range}"
