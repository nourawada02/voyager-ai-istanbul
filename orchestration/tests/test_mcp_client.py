"""Hermetic tests for the real Travel MCP client (Checkpoint Phase 4 D.1
§6). `call_tool_async` is stubbed -- no real socket is ever opened, no
real MCP server process is started.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestration.system_a.mcp_client import TravelMcpClient, TravelMcpConfigurationError, _travel_mcp_base_url


def test_missing_base_url_raises_configuration_error(monkeypatch):
    monkeypatch.delenv("TRAVEL_MCP_BASE_URL", raising=False)
    with pytest.raises(TravelMcpConfigurationError):
        _travel_mcp_base_url()


def test_disallowed_tool_name_is_invalid_request_never_called():
    calls = []

    async def _stub(base_url, tool_name, arguments):
        calls.append(tool_name)
        return {}, False

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub)
    result = client.call_tool("delete_all_listings", {})
    assert result["status"] == "invalid_request"
    assert calls == []


def test_successful_call_returns_structured_result():
    async def _stub(base_url, tool_name, arguments):
        assert tool_name == "search_stays"
        return {"stays": [{"stay": {"stay_id": "s1"}, "fair_price": {}, "rank": 1}]}, False

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub)
    result = client.call_tool("search_stays", {"guest_count": 2})
    assert result["status"] == "success"
    assert result["result"]["stays"][0]["stay"]["stay_id"] == "s1"


def test_is_error_flag_maps_to_provider_error():
    async def _stub(base_url, tool_name, arguments):
        return {}, True

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub)
    result = client.call_tool("search_stays", {})
    assert result["status"] == "provider_error"


def test_error_envelope_maps_to_invalid_request():
    async def _stub(base_url, tool_name, arguments):
        return {"error_code": "UNSUPPORTED_ROOM_TYPE", "message": "x", "trace_id": "t", "retriable": False}, False

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub)
    result = client.call_tool("search_stays", {})
    assert result["status"] == "invalid_request"
    assert result["result"] is None


def test_error_envelope_serving_resource_unavailable_maps_to_unavailable():
    async def _stub(base_url, tool_name, arguments):
        return {"error_code": "SERVING_RESOURCE_UNAVAILABLE", "message": "x", "trace_id": "t", "retriable": True}, False

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub)
    result = client.call_tool("estimate_fair_price", {})
    assert result["status"] == "unavailable"


def test_timeout_is_reported_honestly():
    async def _stub(base_url, tool_name, arguments):
        await asyncio.sleep(10)
        return {}, False

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub, timeout_seconds=0.01)
    result = client.call_tool("search_stays", {})
    assert result["status"] == "timeout"


def test_raw_exception_never_leaks_only_provider_error():
    async def _stub(base_url, tool_name, arguments):
        raise RuntimeError("some internal MCP transport detail with a stack trace")

    client = TravelMcpClient(base_url="https://travel-mcp.example.com", call_tool_async=_stub)
    result = client.call_tool("search_stays", {})
    assert result["status"] == "provider_error"
    assert result["result"] is None
