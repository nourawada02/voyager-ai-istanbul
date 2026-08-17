"""Narrowly scoped OPTIONAL live smoke test for the real SerpApi
web-evidence adapter (Checkpoint Phase 4 C.2). Disabled during normal
`pytest` runs -- enabled only by setting the explicit environment flag
below. Makes exactly one bounded search request (max_results <= 3) --
never a loop, never repeated polling, never fetches a returned URL.

Requires SERPAPI_API_KEY to be set in the environment -- unlike Tavily,
SerpApi's `google` engine has no keyless mode. Prints only non-secret,
already-public operational fields -- never the API key, never the
complete request URL (which would contain the key as a query parameter),
never a raw response body, and never writes any file.

Enable with:
    VOYAGER_LIVE_WEB_EVIDENCE_GATE=1 python -m pytest providers/tests/test_web_evidence_serpapi_live.py -v -s
"""

from __future__ import annotations

import os

import pytest

from providers.tests.conftest import make_validator
from providers.web_evidence import WebEvidenceQuery
from providers.web_evidence_serpapi import HARD_MAX_RESULTS, SerpApiWebEvidenceProvider

LIVE_GATE_ENV_VAR = "VOYAGER_LIVE_WEB_EVIDENCE_GATE"

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_GATE_ENV_VAR) != "1",
    reason=f"live SerpApi network gate is disabled by default; set {LIVE_GATE_ENV_VAR}=1 to enable",
)


def test_live_serpapi_istanbul_evidence_query(envelope_validator, registry):
    provider = SerpApiWebEvidenceProvider()  # real transport, real clock; reads SERPAPI_API_KEY from the environment
    if provider.api_key is None:
        pytest.skip("SERPAPI_API_KEY is not set in this environment -- cannot attempt the keyed live gate")

    query = WebEvidenceQuery(query="Hagia Sophia Istanbul visiting information", max_results=3)

    envelope = provider.search(query)

    envelope_errors = list(envelope_validator.iter_errors(envelope))
    assert not envelope_errors, [e.message for e in envelope_errors]
    result_validator = make_validator("WebEvidenceResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]

    assert envelope["provider"] == "serpapi"
    assert envelope["status"] in ("success", "partial"), envelope.get("quality", {}).get("assumptions")
    assert envelope["retrieved_at"]
    # `max_results` only shapes the outgoing request's `num` hint -- it is
    # not enforced as a strict local cap (see
    # providers/tests/test_web_evidence_serpapi.py::test_result_count_is_hard_capped_regardless_of_provider_count).
    # The only ceiling this adapter actually enforces is HARD_MAX_RESULTS.
    assert len(envelope["result"]["items"]) <= HARD_MAX_RESULTS

    print(f"LIVE SERPAPI GATE: provider={envelope['provider']} status={envelope['status']}")
    for item in envelope["result"]["items"]:
        print(
            f"  - rank={item['rank']} source_type={item['source_type']} publisher={item['publisher']} "
            f"url={item['canonical_url']} title={item['title'][:80]!r} snippet={item['snippet'][:80]!r}"
        )
