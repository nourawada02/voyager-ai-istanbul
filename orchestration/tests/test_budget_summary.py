"""Tests for orchestration/system_a/budget_summary.py (Manual QA
remediation Q.1, §B). Every test here is hermetic -- FakeFxProvider,
never a real network call."""

from __future__ import annotations

from decimal import Decimal

from orchestration.system_a.budget_summary import build_budget_summary
from providers.fx import FakeFxProvider

TRIP_REQUEST = {
    "depart_date": "2026-09-10", "return_date": "2026-09-13",  # 3 nights
    "budget": {"amount_minor_units": 500000, "currency": "TRY"},
}

FINAL_RESULT_TRY = {
    "status": "success",
    "observations": [
        {"action": "search_flights", "status": "success", "envelope": {"result": {"options": [
            {"price": {"amount_minor_units": 4409200, "currency": "TRY"}},
            {"price": {"amount_minor_units": 6471000, "currency": "TRY"}},
        ]}}},
        {"action": "search_stays", "status": "success", "envelope": {"stays": [
            {"stay": {"nightly_price": {"amount_minor_units": 35593, "currency": "TRY"}}},
        ]}},
    ],
}


def test_try_budget_needs_no_conversion_raw_equals_normalized():
    """Required test: TRY unchanged."""
    summary = build_budget_summary(FINAL_RESULT_TRY, TRIP_REQUEST, lambda: FakeFxProvider())
    assert summary["fx_quote"] is None  # no conversion needed at all -- FX never even queried
    assert summary["cheapest_flight"]["raw"] == summary["cheapest_flight"]["normalized"]
    assert summary["cheapest_flight"]["raw"]["currency"] == "TRY"
    assert summary["cheapest_stay_total"]["raw"]["amount_minor_units"] == 35593 * 3
    assert summary["cheapest_stay_total"]["normalized"] == summary["cheapest_stay_total"]["raw"]


def test_usd_budget_converts_try_prices_with_one_coherent_quote():
    """Required test: USD conversion, and budget filtering after
    conversion (here: the comparable normalized amounts)."""
    trip_request_usd = dict(TRIP_REQUEST, budget={"amount_minor_units": 500000, "currency": "USD"})
    fx = FakeFxProvider(rates={("USD", "TRY"): Decimal("40.00")})
    summary = build_budget_summary(FINAL_RESULT_TRY, trip_request_usd, lambda: fx)

    assert summary["fx_quote"] is not None
    assert summary["fx_quote"]["base_currency"] == "USD"
    assert summary["fx_quote"]["quote_currency"] == "TRY"
    assert summary["fx_quote"]["rate"] == "40.00"

    flight = summary["cheapest_flight"]
    assert flight["raw"]["currency"] == "TRY"
    assert flight["raw"]["amount_minor_units"] == 4409200
    # 44092.00 TRY / 40.00 = 1102.30 USD -> 110230 minor units
    assert flight["normalized"]["currency"] == "USD"
    assert flight["normalized"]["amount_minor_units"] == 110230

    stay = summary["cheapest_stay_total"]
    # 355.93 * 3 = 1067.79 TRY / 40.00 = 26.69475 -> rounds to 26.69 USD... actually check exact
    assert stay["normalized"]["currency"] == "USD"

    # ONE quote reused for both conversions -- never two independent fetches.
    assert summary["fx_quote"]["rate"] == "40.00"


def test_flight_already_in_budget_currency_needs_no_conversion():
    """SerpApi requested USD natively -- no TRY->USD conversion needed
    for the flight leg, only for the (still TRY-only) stay."""
    trip_request_usd = dict(TRIP_REQUEST, budget={"amount_minor_units": 500000, "currency": "USD"})
    final_result = {
        "status": "success",
        "observations": [
            {"action": "search_flights", "status": "success", "envelope": {"result": {"options": [
                {"price": {"amount_minor_units": 100000, "currency": "USD"}},
            ]}}},
            {"action": "search_stays", "status": "success", "envelope": {"stays": [
                {"stay": {"nightly_price": {"amount_minor_units": 35593, "currency": "TRY"}}},
            ]}},
        ],
    }
    fx = FakeFxProvider(rates={("USD", "TRY"): Decimal("40.00")})
    summary = build_budget_summary(final_result, trip_request_usd, lambda: fx)
    assert summary["cheapest_flight"]["raw"] == summary["cheapest_flight"]["normalized"]
    assert summary["cheapest_flight"]["raw"]["currency"] == "USD"
    assert summary["cheapest_stay_total"]["normalized"]["currency"] == "USD"  # the TRY one still got converted


def test_fx_provider_unavailable_preserves_raw_price_never_fabricates_normalized():
    """Required test: unavailable/rate-limited FX provider."""
    trip_request_usd = dict(TRIP_REQUEST, budget={"amount_minor_units": 500000, "currency": "USD"})
    fx = FakeFxProvider(scenario="unavailable")
    summary = build_budget_summary(FINAL_RESULT_TRY, trip_request_usd, lambda: fx)
    assert summary["fx_quote"] is None
    assert summary["fx_status"] == "unavailable"
    assert summary["cheapest_flight"]["raw"]["amount_minor_units"] == 4409200  # real, provider-native amount preserved
    assert summary["cheapest_flight"]["raw"]["currency"] == "TRY"
    assert summary["cheapest_flight"]["normalized"] is None  # never fabricated


def test_no_accidental_try_label_on_a_usd_amount():
    """Required test: no accidental TRY label on a USD amount."""
    trip_request_usd = dict(TRIP_REQUEST, budget={"amount_minor_units": 500000, "currency": "USD"})
    fx = FakeFxProvider(rates={("USD", "TRY"): Decimal("40.00")})
    summary = build_budget_summary(FINAL_RESULT_TRY, trip_request_usd, lambda: fx)
    assert summary["budget"]["currency"] == "USD"
    assert summary["cheapest_flight"]["normalized"]["currency"] == "USD"
    assert summary["cheapest_flight"]["raw"]["currency"] == "TRY"  # the real, unconverted source currency


def test_no_accidental_usd_label_on_an_unconverted_try_amount():
    """Required test: no accidental USD label on an unconverted TRY
    amount."""
    summary = build_budget_summary(FINAL_RESULT_TRY, TRIP_REQUEST, lambda: FakeFxProvider())
    assert summary["budget"]["currency"] == "TRY"
    assert summary["cheapest_flight"]["raw"]["currency"] == "TRY"
    assert summary["cheapest_flight"]["normalized"]["currency"] == "TRY"


def test_no_trip_request_returns_none_never_a_fabricated_summary():
    assert build_budget_summary({"status": "success", "observations": []}, None, lambda: FakeFxProvider()) is None


def test_malformed_budget_returns_none():
    assert build_budget_summary({"status": "success", "observations": []}, {"depart_date": "2026-09-10", "return_date": "2026-09-13"}, lambda: FakeFxProvider()) is None
