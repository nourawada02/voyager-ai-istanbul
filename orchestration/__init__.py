"""Root-owned production composition packages (Checkpoint Phase 4 D.1;
docs/adr/0014-phase4-checkpoint-d1-system-a-real-wiring.md). Never
imported by any `services/*` submodule -- submodules stay independently
buildable and this package is the only place that reaches across the
System A / root-provider / Travel MCP / System B boundary.
"""
