#!/usr/bin/env python3
"""Phase 0 executable cross-process compatibility verification harness.

Starts the real ADK/A2A server (services/istanbul-expert-b) and the real MCP
Streamable HTTP server (services/travel-mcp) as separate subprocesses, then
runs the real LangGraph planner spike (services/planner-a) as a third
process. Asserts a live, real round trip across genuine OS-process and HTTP
boundaries -- no mocks, no fallbacks. A compatibility failure is reported as
a failure, never hidden.

This script itself depends only on the Python standard library. The real
A2A/MCP client calls used for readiness checks are performed by invoking the
already-installed, already-verified planner-a environment (which has httpx,
a2a-sdk, and mcp) as a subprocess, so this script does not need those
packages in its own interpreter.

Usage:
    python phase0/verify.py \\
        --planner-python <path to planner-a venv python> \\
        --adk-python <path to istanbul-expert-b venv python> \\
        --mcp-python <path to travel-mcp venv python>

Interpreter paths may also be given via the PHASE0_PLANNER_PYTHON,
PHASE0_ADK_PYTHON, and PHASE0_MCP_PYTHON environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANNER_DIR = REPO_ROOT / "services" / "planner-a"
ADK_DIR = REPO_ROOT / "services" / "istanbul-expert-b"
MCP_DIR = REPO_ROOT / "services" / "travel-mcp"

HOST = "127.0.0.1"  # never anything but localhost
READY_TIMEOUT_SECONDS = 25.0
READY_POLL_INTERVAL_SECONDS = 0.5
PLANNER_TIMEOUT_SECONDS = 30.0


class Phase0Failure(RuntimeError):
    """Raised for any Phase 0 compatibility failure. Never suppressed."""


def _free_port() -> int:
    """Allocate a free localhost port without hard-coding one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


A2A_READY_PROBE = """
import asyncio, sys
import httpx
from a2a.client.card_resolver import A2ACardResolver

async def main():
    base_url = sys.argv[1]
    async with httpx.AsyncClient(timeout=2.0) as client:
        resolver = A2ACardResolver(client, base_url)
        card = await resolver.get_agent_card()
        print(card.name)

asyncio.run(main())
"""

MCP_READY_PROBE = """
import asyncio, sys
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

async def main():
    endpoint = sys.argv[1]
    async with streamable_http_client(endpoint) as streams:
        read_stream, write_stream = streams[0], streams[1]
        async with ClientSession(read_stream, write_stream) as session:
            result = await session.initialize()
            print(result.protocol_version)

asyncio.run(main())
"""


def _wait_ready(probe_code: str, target_url: str, probe_python: str, label: str) -> None:
    """Poll a real readiness probe (real Agent Card resolution / real MCP
    initialize handshake) with a bounded timeout. Never assumes readiness
    from a guessed endpoint or a raw TCP connect alone."""
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    last_error = None
    while time.monotonic() < deadline:
        proc = subprocess.run(
            [probe_python, "-c", probe_code, target_url],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return
        last_error = proc.stderr.strip().splitlines()[-1] if proc.stderr else "(no stderr)"
        time.sleep(READY_POLL_INTERVAL_SECONDS)
    raise Phase0Failure(
        f"{label} did not become ready within {READY_TIMEOUT_SECONDS}s. "
        f"Last probe error: {last_error}"
    )


def _terminate(procs: list[subprocess.Popen], log_dir: Path) -> None:
    """Guaranteed cleanup: terminate every launched child, on success,
    failure, exception, or interruption."""
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    deadline = time.monotonic() + 5.0
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except OSError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--planner-python",
        default=os.environ.get("PHASE0_PLANNER_PYTHON"),
        required=os.environ.get("PHASE0_PLANNER_PYTHON") is None,
    )
    parser.add_argument(
        "--adk-python",
        default=os.environ.get("PHASE0_ADK_PYTHON"),
        required=os.environ.get("PHASE0_ADK_PYTHON") is None,
    )
    parser.add_argument(
        "--mcp-python",
        default=os.environ.get("PHASE0_MCP_PYTHON"),
        required=os.environ.get("PHASE0_MCP_PYTHON") is None,
    )
    args = parser.parse_args()

    nonce = uuid.uuid4().hex  # unique, unpredictable, generated by the harness
    a2a_port = _free_port()
    mcp_port = _free_port()
    a2a_base_url = f"http://{HOST}:{a2a_port}"
    mcp_base_url = f"http://{HOST}:{mcp_port}"

    log_dir = Path(tempfile.mkdtemp(prefix="phase0-verify-"))  # outside every repo
    procs: list[subprocess.Popen] = []

    def _sigterm_handler(signum, frame):
        raise Phase0Failure(f"Interrupted by signal {signum}")

    old_sigterm = signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        adk_log = open(log_dir / "a2a_server.log", "w", encoding="utf-8")
        mcp_log = open(log_dir / "mcp_server.log", "w", encoding="utf-8")
        planner_log = open(log_dir / "planner.log", "w", encoding="utf-8")

        adk_pid_file = log_dir / "a2a_server.pid"
        mcp_pid_file = log_dir / "mcp_server.pid"

        adk_proc = subprocess.Popen(
            [
                args.adk_python, "-m", "phase0.adk_a2a_server",
                "--host", HOST, "--port", str(a2a_port), "--protocol", "http",
                "--pid-file", str(adk_pid_file),
            ],
            cwd=str(ADK_DIR), stdout=adk_log, stderr=subprocess.STDOUT,
        )
        procs.append(adk_proc)

        mcp_proc = subprocess.Popen(
            [
                args.mcp_python, "-m", "phase0.mcp_server",
                "--host", HOST, "--port", str(mcp_port),
                "--pid-file", str(mcp_pid_file),
            ],
            cwd=str(MCP_DIR), stdout=mcp_log, stderr=subprocess.STDOUT,
        )
        procs.append(mcp_proc)

        # Bounded readiness: real Agent Card discovery, real MCP initialize.
        _wait_ready(A2A_READY_PROBE, a2a_base_url, args.planner_python, "ADK/A2A server")
        if adk_proc.poll() is not None:
            raise Phase0Failure(f"ADK/A2A server exited early with code {adk_proc.returncode}")

        _wait_ready(MCP_READY_PROBE, mcp_base_url.rstrip("/") + "/mcp", args.planner_python, "MCP server")
        if mcp_proc.poll() is not None:
            raise Phase0Failure(f"MCP server exited early with code {mcp_proc.returncode}")

        # On this platform, subprocess.Popen(...).pid for a venv python can be
        # a launcher-stub PID that differs from the real interpreter's
        # os.getpid() (discovered live during this harness's development --
        # confirmed via an isolated repro unrelated to ADK/A2A/MCP). Each
        # server writes its own true PID to --pid-file at startup; that is
        # the authoritative value for the "self-reported PID matches the
        # launched process" assertion below. Popen.pid is still recorded,
        # for transparency, as the launcher PID.
        if not adk_pid_file.exists() or not mcp_pid_file.exists():
            raise Phase0Failure("A server reported ready but never wrote its --pid-file")
        adk_true_pid = int(adk_pid_file.read_text(encoding="utf-8").strip())
        mcp_true_pid = int(mcp_pid_file.read_text(encoding="utf-8").strip())

        planner_proc = subprocess.Popen(
            [
                args.planner_python, "-m", "phase0.graph_client",
                "--nonce", nonce,
                "--mcp-base-url", mcp_base_url,
                "--a2a-base-url", a2a_base_url,
            ],
            cwd=str(PLANNER_DIR), stdout=planner_log, stderr=subprocess.STDOUT,
        )
        procs.append(planner_proc)

        try:
            planner_proc.wait(timeout=PLANNER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            raise Phase0Failure(
                f"Planner (LangGraph) process did not finish within {PLANNER_TIMEOUT_SECONDS}s"
            )

        planner_log.flush()
        planner_output = (log_dir / "planner.log").read_text(encoding="utf-8")
        if planner_proc.returncode != 0:
            raise Phase0Failure(
                f"Planner process exited with code {planner_proc.returncode}. "
                f"Output:\n{planner_output}"
            )

        result_line = next(
            (line for line in planner_output.splitlines() if line.startswith("PHASE0_RESULT ")),
            None,
        )
        if result_line is None:
            raise Phase0Failure(
                f"Planner process produced no PHASE0_RESULT line. Output:\n{planner_output}"
            )

        result = json.loads(result_line[len("PHASE0_RESULT "):])

        # ---- Assertions (10-15) ----
        planner_true_pid = result.get("pid")
        if not isinstance(planner_true_pid, int):
            raise Phase0Failure(
                f"Planner process did not report its own PID: {planner_true_pid!r}"
            )

        # The three ACTUAL application PIDs (self-reported by each running
        # process) are what must be mutually distinct -- launcher PIDs
        # (subprocess.Popen(...).pid) are recorded separately, below, purely
        # as diagnostic information, since on this platform a launcher PID
        # can differ from the real interpreter's own os.getpid().
        true_pids = {
            "planner": planner_true_pid,
            "a2a_server": adk_true_pid,
            "mcp_server": mcp_true_pid,
        }
        if len(set(true_pids.values())) != 3:
            raise Phase0Failure(f"Expected 3 distinct true application PIDs, got: {true_pids}")

        launcher_pids_diagnostic = {
            "planner_launcher": planner_proc.pid,
            "a2a_server_launcher": adk_proc.pid,
            "mcp_server_launcher": mcp_proc.pid,
        }

        pids = {**true_pids, **launcher_pids_diagnostic}

        mcp_result = result.get("mcp_result") or {}
        a2a_result = result.get("a2a_result") or {}

        if mcp_result.get("pid") != mcp_true_pid:
            raise Phase0Failure(
                f"MCP self-reported PID {mcp_result.get('pid')} != server's own --pid-file PID {mcp_true_pid}"
            )
        if a2a_result.get("pid") != adk_true_pid:
            raise Phase0Failure(
                f"A2A self-reported PID {a2a_result.get('pid')} != server's own --pid-file PID {adk_true_pid}"
            )

        if mcp_result.get("nonce") != nonce:
            raise Phase0Failure(
                f"MCP response nonce mismatch: expected {nonce}, got {mcp_result.get('nonce')}"
            )
        if a2a_result.get("nonce") != nonce:
            raise Phase0Failure(
                f"A2A response nonce mismatch: expected {nonce}, got {a2a_result.get('nonce')}"
            )

        if mcp_result.get("is_error"):
            raise Phase0Failure(f"MCP tool reported is_error=True: {mcp_result}")

        if a2a_result.get("task_state") != "TASK_STATE_COMPLETED":
            raise Phase0Failure(
                f"A2A task did not complete: state={a2a_result.get('task_state')}, result={a2a_result}"
            )
        if not a2a_result.get("artifact_count"):
            raise Phase0Failure(
                f"A2A task completed but reported no validated artifact: {a2a_result}"
            )
        if not a2a_result.get("selected_transport_binding"):
            raise Phase0Failure(
                f"A2A selected transport binding could not be established: {a2a_result}"
            )
        if not a2a_result.get("selected_rpc_url"):
            raise Phase0Failure(
                f"A2A selected RPC URL (card-derived) could not be established: {a2a_result}"
            )

        observed_nodes = set(result.get("observed_nodes") or [])
        expected_nodes = {"call_mcp", "call_a2a"}
        if not expected_nodes.issubset(observed_nodes):
            raise Phase0Failure(
                f"LangGraph did not emit updates for both expected nodes. "
                f"Expected {expected_nodes}, observed {observed_nodes}"
            )

        summary = {
            "status": "PASS",
            "nonce": nonce,
            "pids": pids,
            "observed_nodes": sorted(observed_nodes),
            "mcp_result": mcp_result,
            "a2a_result": a2a_result,
            "a2a_base_url": a2a_base_url,
            "mcp_base_url": mcp_base_url,
            "log_dir": str(log_dir),
        }
        print(json.dumps(summary, indent=2))
        return 0

    except Phase0Failure as exc:
        failure = {
            "status": "FAIL",
            "error": str(exc),
            "nonce": nonce,
            "log_dir": str(log_dir),
        }
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        _terminate(procs, log_dir)
        for f in ("adk_log", "mcp_log", "planner_log"):
            obj = locals().get(f)
            if obj is not None and not obj.closed:
                obj.close()


if __name__ == "__main__":
    sys.exit(main())
