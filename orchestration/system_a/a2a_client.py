"""Real System B A2A client (Checkpoint Phase 4 D.1 §7). Reuses the exact
authoritative Phase 0 client sequence
(`services/planner-a/phase0/graph_client.py::_call_a2a`) unchanged in
spirit: Agent Card discovery via `A2ACardResolver`'s standard
well-known-path behavior, a client built through the official
`ClientFactory`, a real task/message send, and Artifact-only evidence
extraction (a status/standalone Message is never accepted as evidence).
Never hardcodes a guessed RPC path, never imports System B's own Python
implementation, never calls it in-process, and never sends a provider
credential -- the only payload sent is an already-normalized
`LocalPlanRequest`-shaped JSON object built entirely from validated
System A state.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from a2a.client.card_resolver import A2ACardResolver
from a2a.client.client_factory import ClientConfig, ClientFactory
from a2a.types import a2a_pb2 as a2a_types

DEFAULT_TIMEOUT_SECONDS = 25.0


class IstanbulExpertA2AConfigurationError(RuntimeError):
    """Raised when `ISTANBUL_EXPERT_A2A_BASE_URL` is not configured.
    Never guesses an internal endpoint -- fails before any network call."""


def _istanbul_expert_base_url() -> str:
    base_url = os.environ.get("ISTANBUL_EXPERT_A2A_BASE_URL")
    if not base_url:
        raise IstanbulExpertA2AConfigurationError(
            "ISTANBUL_EXPERT_A2A_BASE_URL is required and must be set explicitly"
        )
    return base_url


async def _send_local_plan_request_async(
    base_url: str, payload: dict[str, Any], timeout_seconds: float
) -> tuple[Optional[dict], str]:
    """Returns (payload_dict_or_None, task_state_name). Never raises a
    raw A2A/httpx exception outward -- the caller (`IstanbulExpertA2AClient`)
    maps every exception to a safe status."""
    async with httpx.AsyncClient(timeout=timeout_seconds) as httpx_client:
        resolver = A2ACardResolver(httpx_client, base_url)
        card = await resolver.get_agent_card()
        if not isinstance(card, a2a_types.AgentCard) or not card.supported_interfaces:
            raise RuntimeError("Agent Card discovery failed or advertised no interfaces")

        card_bindings = [i.protocol_binding for i in card.supported_interfaces]
        config = ClientConfig(httpx_client=httpx_client, streaming=True, supported_protocol_bindings=card_bindings)
        client = ClientFactory(config).create(card)

        message = a2a_types.Message(
            message_id=f"system-a-{payload.get('trace_id', 'unknown')}",
            role=a2a_types.Role.ROLE_USER,
            parts=[a2a_types.Part(text=json.dumps(payload))],
        )
        request = a2a_types.SendMessageRequest(message=message)

        collected_artifacts: list = []
        final_state = a2a_types.TaskState.TASK_STATE_UNSPECIFIED

        async for response in client.send_message(request):
            which = response.WhichOneof("payload")
            if which == "task":
                task = response.task
                collected_artifacts.extend(task.artifacts)
                final_state = task.status.state
            elif which == "status_update":
                final_state = response.status_update.status.state
            elif which == "artifact_update":
                collected_artifacts.append(response.artifact_update.artifact)
            # "message" (a standalone Message) is intentionally never
            # collected as evidence here, matching Phase 0's own rule.

        if final_state != a2a_types.TaskState.TASK_STATE_COMPLETED:
            return None, a2a_types.TaskState.Name(final_state)

        for artifact in collected_artifacts:
            for part in artifact.parts:
                if not part.text:
                    continue
                try:
                    found = json.loads(part.text)
                except json.JSONDecodeError:
                    continue
                if isinstance(found, dict):
                    return found, a2a_types.TaskState.Name(final_state)

        return None, a2a_types.TaskState.Name(final_state)


@dataclass
class IstanbulExpertA2AClient:
    """Production A2A binding. `base_url` defaults to
    `ISTANBUL_EXPERT_A2A_BASE_URL` (read once at construction, never
    guessed). Injectable `send_async` seam lets hermetic tests stub the
    SDK/client boundary without opening a real socket."""

    base_url: str = field(default_factory=_istanbul_expert_base_url)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    send_async: Any = _send_local_plan_request_async  # (base_url, payload, timeout) -> awaitable[(dict|None, str)]

    def call_istanbul_expert(self, local_plan_request: dict[str, Any]) -> dict[str, Any]:
        try:
            payload, _task_state_name = asyncio.run(
                asyncio.wait_for(
                    self.send_async(self.base_url, local_plan_request, self.timeout_seconds),
                    timeout=self.timeout_seconds,
                )
            )
        except (TimeoutError, asyncio.TimeoutError):
            return {"status": "timeout", "result": None}
        except Exception:  # noqa: BLE001 -- a raw A2A/network exception must never reach the caller
            return {"status": "provider_error", "result": None}

        if payload is None:
            return {"status": "provider_error", "result": None}
        return {"status": "success", "result": payload}
