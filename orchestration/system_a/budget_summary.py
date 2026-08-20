"""Deterministic, server-side budget summary with genuine currency
conversion (Manual QA remediation Q.1, §B). Built once per run, after the
bounded ReAct loop has already finished, purely from the observations it
already gathered -- never a second round of tool calls, never something
Qwen decides. Every real price this system produces is either the trip's
own budget currency (flights, requested natively) or TRY (accommodation,
which Travel MCP can only ever produce in TRY) -- so at most ONE FX pair
is ever needed for one run, fetched once, reused for every conversion in
this summary (the "one coherent run-level FX quote" requirement).

`raw` on every entry below is the real, provider-native amount/currency,
untouched. `normalized` is that same amount converted into the trip's own
budget currency for direct comparison -- present only when a currency
match needs no conversion, or when a real FX quote was actually obtained.
Never a fabricated `normalized` value: a failed FX fetch leaves it None
and `fx_quote` is also None, so nothing downstream can mistake an
unconverted amount for a converted one.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from providers.fx import FxQuery
from providers.money import ConversionUnavailable, convert_minor_units, quote_from_envelope


def _money(amount_minor_units: int, currency: str) -> dict[str, Any]:
    return {"amount_minor_units": amount_minor_units, "currency": currency}


def _cheapest_flight_price(observations: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for obs in observations:
        if obs.get("action") != "search_flights" or obs.get("status") != "success":
            continue
        envelope = obs.get("envelope") or {}
        options = (envelope.get("result") or {}).get("options") or []
        prices = [
            o["price"] for o in options
            if isinstance(o, dict) and isinstance(o.get("price"), dict) and o["price"].get("amount_minor_units") is not None
        ]
        if prices:
            return min(prices, key=lambda p: p["amount_minor_units"])
    return None


def _cheapest_stay_total(observations: list[dict[str, Any]], nights: Optional[int]) -> Optional[dict[str, Any]]:
    if not nights:
        return None
    for obs in observations:
        if obs.get("action") != "search_stays" or obs.get("status") != "success":
            continue
        envelope = obs.get("envelope") or {}
        stays = envelope.get("stays") or []
        nightly = [
            s["stay"]["nightly_price"] for s in stays
            if isinstance(s, dict) and isinstance(s.get("stay"), dict) and isinstance(s["stay"].get("nightly_price"), dict)
            and s["stay"]["nightly_price"].get("amount_minor_units") is not None
        ]
        if nightly:
            cheapest = min(nightly, key=lambda p: p["amount_minor_units"])
            return _money(cheapest["amount_minor_units"] * nights, cheapest["currency"])
    return None


def _nights_from(trip_request: dict[str, Any]) -> Optional[int]:
    try:
        depart = date.fromisoformat(str(trip_request["depart_date"]))
        ret = date.fromisoformat(str(trip_request["return_date"]))
    except (KeyError, ValueError, TypeError):
        return None
    return max((ret - depart).days, 0)


def build_budget_summary(
    final_result: dict[str, Any], trip_request: Optional[dict[str, Any]], fx_provider_factory: Callable[[], Any],
) -> Optional[dict[str, Any]]:
    """Returns None only when there is no real budget to summarize at all
    (no trip_request, or a malformed budget) -- never a chart/summary
    built from fabricated numbers."""
    if not isinstance(trip_request, dict):
        return None
    budget = trip_request.get("budget")
    if not isinstance(budget, dict) or budget.get("amount_minor_units") is None or not budget.get("currency"):
        return None
    budget_currency = str(budget["currency"]).upper()

    observations = final_result.get("observations") or []
    flight_price = _cheapest_flight_price(observations)
    stay_total = _cheapest_stay_total(observations, _nights_from(trip_request))

    summary: dict[str, Any] = {
        "budget": _money(budget["amount_minor_units"], budget_currency),
        "fx_quote": None,
        "fx_status": None,
        "cheapest_flight": None,
        "cheapest_stay_total": None,
    }

    other_currencies = {
        str(p["currency"]).upper() for p in (flight_price, stay_total) if p is not None
    } - {budget_currency}
    quote = None
    if other_currencies:
        # Every real price this system produces is TRY or the budget
        # currency -- exactly one "other" currency is ever possible.
        other_currency = next(iter(other_currencies))
        provider = fx_provider_factory()
        envelope = provider.fetch_rate(FxQuery(base_currency=budget_currency, quote_currency=other_currency))
        summary["fx_status"] = envelope.get("status")
        quote = quote_from_envelope(envelope)
        if quote is not None:
            summary["fx_quote"] = {
                "base_currency": quote.base_currency, "quote_currency": quote.quote_currency,
                "rate": str(quote.rate), "effective_date": quote.effective_date,
                "provider": quote.provider, "retrieved_at": quote.retrieved_at, "cache_status": quote.cache_status,
            }

    def _entry(raw_money: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if raw_money is None:
            return None
        raw_currency = str(raw_money["currency"]).upper()
        entry: dict[str, Any] = {"raw": _money(raw_money["amount_minor_units"], raw_currency), "normalized": None}
        if raw_currency == budget_currency:
            entry["normalized"] = entry["raw"]
        elif quote is not None:
            try:
                converted = convert_minor_units(raw_money["amount_minor_units"], quote, raw_currency, budget_currency)
                entry["normalized"] = _money(converted, budget_currency)
            except ConversionUnavailable:
                pass  # left None -- never a fabricated conversion
        return entry

    summary["cheapest_flight"] = _entry(flight_price)
    summary["cheapest_stay_total"] = _entry(stay_total)
    return summary
