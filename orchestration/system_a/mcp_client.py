"""Real Travel MCP client (Checkpoint Phase 4 D.1 §6). Uses the official
MCP SDK client and the exact Streamable HTTP pattern already proven live
in `services/planner-a/phase0/graph_client.py::_call_mcp` -- a real
`initialize()` handshake, a real `call_tool()`, never a direct in-process
call to Travel MCP's own Python service. Calls only the two allowlisted
domain tools; a well-formed `ErrorEnvelope` returned by the server's own
`phase2/mcp/adapters.py::run_tool` boundary is mapped to the shared
result-status vocabulary, never treated as success, and its raw
`error_code`/message details are never re-exposed as an exception.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

ALLOWED_TOOLS = frozenset({"search_stays", "estimate_fair_price"})

DEFAULT_TIMEOUT_SECONDS = 25.0

# services/travel-mcp/phase2/mcp/adapters.py's own fixed error_code
# vocabulary, mapped to the shared providers-style result-status
# vocabulary this project already uses everywhere else.
_RETRIABLE_UNAVAILABLE_CODES = frozenset({
    "SERVING_RESOURCE_UNAVAILABLE", "REGISTRY_INCOMPATIBLE", "ARTIFACT_INCOMPATIBLE",
    "DATASET_MISMATCH", "NO_MATCHING_STAYS", "STAY_NOT_FOUND",
})
_INVALID_REQUEST_CODES = frozenset({
    "INVALID_REQUEST", "UNSUPPORTED_ROOM_TYPE", "UNKNOWN_DISTRICT", "CONTRADICTORY_FILTERS", "REQUEST_TOO_BROAD",
})


def _status_for_error_code(error_code: Optional[str]) -> str:
    if error_code in _RETRIABLE_UNAVAILABLE_CODES:
        return "unavailable"
    if error_code in _INVALID_REQUEST_CODES:
        return "invalid_request"
    return "provider_error"


class TravelMcpConfigurationError(RuntimeError):
    """Raised when `TRAVEL_MCP_BASE_URL` is not configured. Never guesses
    an internal endpoint -- fails before any network call."""


def _travel_mcp_base_url() -> str:
    base_url = os.environ.get("TRAVEL_MCP_BASE_URL")
    if not base_url:
        raise TravelMcpConfigurationError("TRAVEL_MCP_BASE_URL is required and must be set explicitly")
    return base_url


async def _call_tool_async(base_url: str, tool_name: str, arguments: dict[str, Any]) -> tuple[dict, bool]:
    endpoint = base_url.rstrip("/") + "/mcp"
    async with streamable_http_client(endpoint) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            init_result = await session.initialize()
            if not isinstance(init_result, mcp_types.InitializeResult):
                raise RuntimeError("MCP initialize() returned an unexpected type")
            call_result = await session.call_tool(tool_name, arguments)
            if not isinstance(call_result, mcp_types.CallToolResult):
                raise RuntimeError("MCP call_tool() returned an unexpected type")
            structured: dict[str, Any] = call_result.structured_content or {}
            if not structured and call_result.content:
                first = call_result.content[0]
                text = getattr(first, "text", None)
                if text:
                    structured = json.loads(text)
            return structured, bool(call_result.is_error)


@dataclass
class TravelMcpClient:
    """Production MCP binding. `base_url` defaults to `TRAVEL_MCP_BASE_URL`
    (read once at construction time, never guessed). Injectable
    `call_tool_async` seam lets hermetic tests stub the SDK/transport
    boundary without opening a real socket."""

    base_url: str = field(default_factory=_travel_mcp_base_url)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    call_tool_async: Any = _call_tool_async  # (base_url, tool_name, arguments) -> awaitable[(dict, bool)]

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in ALLOWED_TOOLS:
            # Structurally unreachable via the real graph (Action's own
            # closed enum has no field naming an arbitrary tool), kept as
            # an independent second check at this boundary too.
            return {"status": "invalid_request", "result": None}
        try:
            structured, is_error = asyncio.run(
                asyncio.wait_for(self.call_tool_async(self.base_url, tool_name, arguments), timeout=self.timeout_seconds)
            )
        except (TimeoutError, asyncio.TimeoutError):
            return {"status": "timeout", "result": None}
        except Exception:  # noqa: BLE001 -- a raw MCP/transport exception must never reach the caller
            return {"status": "provider_error", "result": None}

        if is_error or not isinstance(structured, dict):
            return {"status": "provider_error", "result": None}
        if "error_code" in structured:
            return {"status": _status_for_error_code(structured.get("error_code")), "result": None}
        return {"status": "success", "result": structured}
