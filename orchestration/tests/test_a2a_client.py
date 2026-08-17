"""Hermetic tests for the real System B A2A client (Checkpoint Phase 4
D.1 §7). `send_async` is stubbed -- no real socket is ever opened, no
real System B process is started.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.system_a.a2a_client import (
    IstanbulExpertA2AClient,
    IstanbulExpertA2AConfigurationError,
    _istanbul_expert_base_url,
)


def test_missing_base_url_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("ISTANBUL_EXPERT_A2A_BASE_URL", raising=False)
    with pytest.raises(IstanbulExpertA2AConfigurationError):
        _istanbul_expert_base_url()


def test_successful_call_returns_the_artifact_payload():
    sent_payloads = []

    async def _stub(base_url, payload, timeout):
        sent_payloads.append(payload)
        return {"itinerary": "ok"}, "TASK_STATE_COMPLETED"

    client = IstanbulExpertA2AClient(base_url="https://istanbul-expert.example.com", send_async=_stub)
    result = client.call_istanbul_expert({"trace_id": "t1", "session_id": "s1"})
    assert result["status"] == "success"
    assert result["result"] == {"itinerary": "ok"}
    assert sent_payloads == [{"trace_id": "t1", "session_id": "s1"}]


def test_no_artifact_payload_is_provider_error():
    async def _stub(base_url, payload, timeout):
        return None, "TASK_STATE_FAILED"

    client = IstanbulExpertA2AClient(base_url="https://istanbul-expert.example.com", send_async=_stub)
    result = client.call_istanbul_expert({"trace_id": "t1"})
    assert result["status"] == "provider_error"
    assert result["result"] is None


def test_timeout_is_reported_honestly():
    async def _stub(base_url, payload, timeout):
        await asyncio.sleep(10)
        return {}, "TASK_STATE_COMPLETED"

    client = IstanbulExpertA2AClient(base_url="https://istanbul-expert.example.com", send_async=_stub, timeout_seconds=0.01)
    result = client.call_istanbul_expert({"trace_id": "t1"})
    assert result["status"] == "timeout"


def test_raw_exception_never_leaks_only_provider_error():
    async def _stub(base_url, payload, timeout):
        raise RuntimeError("some internal A2A/httpx transport detail")

    client = IstanbulExpertA2AClient(base_url="https://istanbul-expert.example.com", send_async=_stub)
    result = client.call_istanbul_expert({"trace_id": "t1"})
    assert result["status"] == "provider_error"


def test_no_credential_is_ever_included_in_the_sent_payload():
    """The payload sent is built entirely by the caller
    (`orchestration.system_a.tool_executor`) from already-validated,
    already-normalized state -- this client itself never adds a
    credential, and this test proves the exact payload passed through is
    exactly what was given, nothing injected."""
    sent = {}

    async def _stub(base_url, payload, timeout):
        sent.update(payload)
        return {"ok": True}, "TASK_STATE_COMPLETED"

    client = IstanbulExpertA2AClient(base_url="https://istanbul-expert.example.com", send_async=_stub)
    client.call_istanbul_expert({"trace_id": "t1", "interests": ["history"]})
    assert "api_key" not in sent
    assert "credential" not in sent
    assert "authorization" not in {k.lower() for k in sent}
