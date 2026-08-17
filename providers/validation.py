"""Mandatory capability-result validation (Checkpoint Phase 4 C.0
correction pass). Every envelope any provider adapter (real or fake)
produces must pass through `validate_envelope()` before an orchestrator
trusts it: the outer `ProviderResponseEnvelope` shape is checked first,
then -- when `capability` names a known live-data capability -- the
nested `result` is checked against exactly that capability's own result
schema. A malformed result, an unknown capability, or a result that does
not match its declared capability's shape all raise the same typed
`EnvelopeValidationError`, never pass through silently.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

_PROVIDERS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTRACTS_DIR = os.path.join(os.path.dirname(_PROVIDERS_DIR), "contracts")

# Which result schema each known capability's envelope.result must match.
# Deliberately closed here (unlike ProviderResponseEnvelope.capability's
# own open pattern) -- an envelope naming a capability this module does
# not recognize is rejected as UNKNOWN_CAPABILITY rather than silently
# accepted with no result validation at all.
CAPABILITY_RESULT_SCHEMAS: dict[str, str] = {
    "weather": "WeatherResult",
    "web_search": "WebEvidenceResult",
    "flight_search": "FlightSearchResult",
}


class EnvelopeValidationError(ValueError):
    """Raised by validate_envelope(). `reason` is a short machine-readable
    code (never raw jsonschema internals) -- 'envelope_invalid',
    'unknown_capability', or 'result_invalid_for_capability'."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources = []
    for path in sorted(glob.glob(os.path.join(_CONTRACTS_DIR, "*.schema.json"))):
        schema = _load(path)
        resources.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


@lru_cache(maxsize=None)
def _validator_for(schema_name: str) -> Draft202012Validator:
    schema = _load(os.path.join(_CONTRACTS_DIR, f"{schema_name}.schema.json"))
    return Draft202012Validator(schema, registry=_registry(), format_checker=FormatChecker())


def validate_envelope(envelope: dict, *, require_capability: bool = False) -> None:
    """Validates `envelope` against ProviderResponseEnvelope, then --
    when `capability` is present -- validates `envelope['result']`
    against exactly the schema CAPABILITY_RESULT_SCHEMAS maps it to.

    Raises EnvelopeValidationError on:
      - the envelope itself failing ProviderResponseEnvelope (reason='envelope_invalid')
      - a capability string not in CAPABILITY_RESULT_SCHEMAS (reason='unknown_capability')
      - a result that does not match its declared capability's schema (reason='result_invalid_for_capability')

    A legacy envelope with no 'capability' field (every pre-1.1.0
    instance) is only checked against the outer envelope shape, unless
    require_capability=True -- there is no per-capability result schema
    to check it against, and 1.0.0 instances must remain valid exactly as
    the additive-versioning contract promises.
    """
    envelope_validator = _validator_for("ProviderResponseEnvelope")
    errors = list(envelope_validator.iter_errors(envelope))
    if errors:
        raise EnvelopeValidationError(
            "envelope_invalid", f"envelope failed ProviderResponseEnvelope validation: {[e.message for e in errors]}"
        )

    capability = envelope.get("capability")
    if capability is None:
        if require_capability:
            raise EnvelopeValidationError("unknown_capability", "envelope has no 'capability' field")
        return

    schema_name = CAPABILITY_RESULT_SCHEMAS.get(capability)
    if schema_name is None:
        raise EnvelopeValidationError("unknown_capability", f"capability {capability!r} is not a known live-data capability")

    result_validator = _validator_for(schema_name)
    result_errors = list(result_validator.iter_errors(envelope.get("result")))
    if result_errors:
        raise EnvelopeValidationError(
            "result_invalid_for_capability",
            f"result does not match {schema_name} for capability {capability!r}: {[e.message for e in result_errors]}",
        )
