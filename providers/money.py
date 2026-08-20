"""Decimal-exact money conversion (Manual QA remediation Q.1, §B).
Architecture.md's own "Money is Decimal or integer minor units -- never
binary float" rule, applied to currency CONVERSION specifically: every
amount here is `phase1.models.Money`-shaped (amount_minor_units: int,
currency: str), and every rate is a `decimal.Decimal` (never a Python
float) -- the same discipline providers/flights_serpapi.py already uses
for its own TRY price parsing, reused here rather than reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional


class ConversionUnavailable(ValueError):
    """Raised when a conversion is requested but no genuine rate is
    available -- callers must catch this and preserve the original,
    unconverted amount with an explicit degradation, never fabricate a
    rate of 1.0 or silently relabel the currency."""


@dataclass(frozen=True)
class FxQuote:
    """A single, run-coherent FX quote with full provenance -- Manual QA
    remediation Q.1 (§B): "budget totals use one coherent run-level FX
    quote". Every conversion in one run must be built from the SAME
    instance of this, never a second independently-fetched rate."""

    base_currency: str
    quote_currency: str
    rate: Decimal
    effective_date: str
    provider: str
    retrieved_at: str
    cache_status: str


def convert_minor_units(amount_minor_units: int, quote: FxQuote, from_currency: str, to_currency: str) -> int:
    """Converts an integer minor-units amount from `from_currency` to
    `to_currency` using `quote` (which must directly express that exact
    pair, base->quote or quote->base -- never a triangulated/inferred
    cross-rate). Returns the input unchanged if from_currency ==
    to_currency (a currency always converts to itself at rate 1, exactly,
    never routed through the quote at all). Decimal arithmetic throughout,
    rounded to the nearest minor unit (ROUND_HALF_UP, this project's one
    fixed rounding rule -- never banker's rounding, never truncation)."""
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    if from_currency == to_currency:
        return amount_minor_units

    amount = Decimal(amount_minor_units) / Decimal(100)
    if from_currency == quote.base_currency and to_currency == quote.quote_currency:
        rate = quote.rate
    elif from_currency == quote.quote_currency and to_currency == quote.base_currency:
        rate = Decimal(1) / quote.rate
    else:
        raise ConversionUnavailable(
            f"quote for {quote.base_currency}->{quote.quote_currency} cannot convert {from_currency}->{to_currency}"
        )

    converted = (amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int((converted * 100).to_integral_value(rounding=ROUND_HALF_UP))


def quote_from_envelope(envelope: dict) -> Optional[FxQuote]:
    """Builds an FxQuote from a real fx_rate ProviderResponseEnvelope
    (providers/fx_frankfurter.py's own shape) -- returns None if the
    envelope does not represent a successful, usable rate, never raises
    and never invents a placeholder quote."""
    if envelope.get("status") != "success":
        return None
    result = envelope.get("result") or {}
    rate_str = result.get("rate")
    if not rate_str:
        return None
    try:
        rate = Decimal(str(rate_str))
    except Exception:  # noqa: BLE001
        return None
    return FxQuote(
        base_currency=str(result.get("base_currency", "")).upper(),
        quote_currency=str(result.get("quote_currency", "")).upper(),
        rate=rate,
        effective_date=str(result.get("effective_date") or ""),
        provider=str(envelope.get("provider", "")),
        retrieved_at=str(envelope.get("retrieved_at", "")),
        cache_status=str(envelope.get("cache_status", "")),
    )
