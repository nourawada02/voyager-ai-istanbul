"""System A's production composition root (Checkpoint Phase 4 D.1).

Owns exactly what ADR 0009 §6.1 and this checkpoint's own instructions
require: the boundary that imports planner-a's graph/ToolExecutor
protocol (`services/planner-a/phase4`), root provider adapters
(`providers/`), and concrete MCP/A2A clients, and wires them into one
real `ProductionToolExecutor` satisfying planner-a's own `ToolExecutor`
protocol -- injected into the exact same bounded LangGraph loop
Checkpoint D.0 already proved, in place of `FakeToolExecutor`.

Composition mechanism (temporary, until Checkpoint D.2 packages this
into the real System A service/image): `services/planner-a` is put on
`PYTHONPATH` at process/test-invocation time (an external, standard
multi-package composition technique -- not `sys.path` mutation inside
any module's own source, which this package never does). No provider
implementation is copied into planner-a; no root `providers`/`rag` code
is copied into System B; no provider credential is ever passed to
System B.
"""
