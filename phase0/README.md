# Phase 0 — Compatibility Spike

Deterministic, credential-free compatibility spike proving: a LangGraph
client can (1) call a Google ADK agent over a real A2A network boundary,
discovering the Agent Card and RPC endpoint at runtime rather than guessing
them, and (2) call a real MCP server over Streamable HTTP. No travel-domain
logic, no LLM, no databases, no Docker. See `docs/phase0-evidence.md` for
recorded results.

## Fixed technical baseline

Python `3.12.10`. Package versions are fixed and recorded in
`phase0/constraints-py312.txt` (the single joint resolution every service's
lock is derived from):

- `google-adk[a2a]==2.6.3`
- `a2a-sdk==1.1.2`
- `mcp==2.0.0`
- `langgraph==1.2.10`
- `uvicorn==0.52.1`
- `pytest==9.1.1`
- `httpx==0.28.1`

## Environment setup (reproducible, per service)

All virtual environments must be created **outside every repository**, at a
short path (Windows path-length limits were hit during discovery at a long
nested temp path). None of the `.venv` directories are committed.

For each of `services/planner-a`, `services/istanbul-expert-b`,
`services/travel-mcp`, from a clean checkout:

```
py -3.12 -m venv <short-outside-repo-path>/<service>-venv
<venv>/Scripts/python.exe -m pip install -r phase0/requirements.in -r phase0/requirements-test.in -c ../../phase0/constraints-py312.txt
<venv>/Scripts/python.exe -m pip check
```

To regenerate a service's `phase0/requirements.lock` from scratch:

```
<venv>/Scripts/python.exe -m pip freeze > phase0/requirements.lock
```

## Running the focused unit tests

From each service's root directory, using that service's own venv:

```
<venv>/Scripts/python.exe -m pytest phase0/tests -v
```

(`python -m pytest`, not the bare `pytest` executable, so the current
directory is on `sys.path` and `phase0` resolves as a package.)

## Running the live cross-process harness

From the superproject root, using the three services' venvs:

```
python phase0/verify.py \
    --planner-python <planner-a venv>/Scripts/python.exe \
    --adk-python <istanbul-expert-b venv>/Scripts/python.exe \
    --mcp-python <travel-mcp venv>/Scripts/python.exe
```

(Interpreter paths may also be supplied via `PHASE0_PLANNER_PYTHON`,
`PHASE0_ADK_PYTHON`, `PHASE0_MCP_PYTHON`.) `phase0/verify.py` itself needs no
third-party dependencies — it drives the three real service processes and
uses the already-installed `planner-a` environment (which has `httpx`,
`a2a-sdk`, and `mcp`) as the interpreter for its readiness probes.

The harness starts the ADK/A2A server and MCP server as separate
subprocesses, waits for real readiness (Agent Card resolution, MCP
`initialize` handshake), runs the LangGraph planner as a third subprocess
with a fresh unpredictable nonce, and asserts: three distinct PIDs, the
self-reported server PIDs match the launched subprocesses, both the MCP and
A2A responses contain the exact nonce, the A2A task reached
`TASK_STATE_COMPLETED`, and LangGraph emitted node-update events for both
`call_mcp` and `call_a2a`. It exits nonzero with a specific error on any
failure, and always terminates every child process it started.

## Reproducibility check

1. Run the focused tests and the live harness (at least twice) using the
   environments built above.
2. Discard those environments entirely.
3. Rebuild three fresh, short-path environments **solely** from each
   service's `phase0/requirements.lock` (`pip install -r phase0/requirements.lock`,
   no `.in` files, no constraints file).
4. Run `pip check` in each.
5. Rerun the focused tests and the live harness again, using only the
   freshly-locked environments.

A successful `pip install` from the lock files alone is not sufficient by
itself — the tests and the live harness must both pass again from those
environments.
