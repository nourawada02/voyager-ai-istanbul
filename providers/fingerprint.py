"""Deterministic request fingerprinting (Checkpoint Phase 4 C.0).

Mirrors rag/ids.py's canonical-JSON-then-sha256 pattern: a pure function
of stable inputs, never wall-clock time or call order, so the same
logical request always produces the same fingerprint -- used for cache-key
construction and duplicate-call detection in a bounded ReAct loop.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

# Fixed namespace UUID (a random constant, generated once and pinned here
# -- never regenerated) so deterministic_request_id's uuid5 derivation is
# stable across processes and over time, exactly like sha256_hex is.
_REQUEST_ID_NAMESPACE = uuid.UUID("6f1c9b1a-6b2e-4b8b-9b7a-5b6c8d9e0f11")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_request_id(query_fingerprint: str, retrieved_at: str, attempt: int = 0) -> str:
    """A pure function of (query_fingerprint, retrieved_at, attempt) --
    never uuid4/os.urandom. Identical input, clock, and attempt number
    always produce the identical request_id (Checkpoint Phase 4 C.0
    correction pass: "identical input, clock, and configuration must
    produce identical output"). `attempt` is included so a genuine retry
    of the same logical request at the same timestamp still gets a
    distinct id, never colliding with the first attempt's."""
    name = f"{query_fingerprint}:{retrieved_at}:{attempt}"
    return str(uuid.uuid5(_REQUEST_ID_NAMESPACE, name))


def fingerprint_request(capability: str, normalized_request: dict[str, Any]) -> str:
    """A pure function of (capability, normalized_request). Two calls with
    the same capability and the same normalized request fields always
    produce the same fingerprint, regardless of dict key order or when
    the call happens -- capability is included so a weather request and
    a flight request that happen to normalize to the same field values
    never collide on the same fingerprint/cache key."""
    payload = {"capability": capability, "request": normalized_request}
    return sha256_hex(_canonical_json(payload))
