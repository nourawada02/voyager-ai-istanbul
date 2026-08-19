"""Final Evaluation Checkpoint E.1X -- runs the frozen 30-case supervisor-
only validation manifest (`evaluation/e1x_cases.json`) against the REAL,
unmodified E.1W production decision path: `phase4.graph._make_decide_node`,
called exactly as production calls it (capability classification with
cross-turn continuity, decision-prompt construction, the bounded repair
loop, the eligibility gate, and the one-shot ineligible-action correction).
This script never reproduces or approximates a production prompt itself.

No tool, MCP, or A2A call is ever made -- `_decide_node` only ever
PROPOSES the next action; Execute is a separate node, never invoked here.

One-shot: refuses to run if `e1x_live_qwen_results.jsonl` already exists
non-empty.

Run with:
    python -m evaluation.run_e1x_live_qwen_eval

Writes:
    evaluation/e1x_live_qwen_results.jsonl
    evaluation/e1x_live_qwen_summary.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "services" / "planner-a"))

from phase4.graph import _make_decide_node, compute_eligible_supervisor_actions  # noqa: E402
from phase4.qwen_client import QwenDecisionProvider, qwen_api_key_present  # noqa: E402

CASES_PATH = Path(__file__).parent / "e1x_cases.json"
RESULTS_PATH = Path(__file__).parent / "e1x_live_qwen_results.jsonl"
SUMMARY_PATH = Path(__file__).parent / "e1x_live_qwen_summary.json"

_FORBIDDEN_SUBSTRINGS = (
    "chain_of_thought", "chain-of-thought", "authorization", "bearer",
    "qwen_api_key", "dashscope_api_key", "sk-ws-",
)

_NON_DELEGATING_ACTIONS = {"synthesize", "ask_clarification", "degrade"}
_DELEGATING_ACTIONS = {"call_travel_search", "call_istanbul_expert"}


def _load_cases() -> dict:
    text = CASES_PATH.read_text(encoding="utf-8")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = json.loads(text)
    manifest["_loaded_checksum"] = checksum
    return manifest


def _leak_scan(*blobs: Any) -> list[str]:
    haystack = json.dumps(blobs, default=str, ensure_ascii=False).lower()
    return [term for term in _FORBIDDEN_SUBSTRINGS if term in haystack]


def _run_case(case: dict, provider: QwenDecisionProvider) -> dict:
    state: dict[str, Any] = {
        "started_at_monotonic": 0.0,
        "tool_call_count": case["tool_call_count"],
        "tool_call_count_by_action": dict(case["tool_call_count_by_action"]),
        "executed_fingerprints": [],
        "observations": list(case["observations"]),
        "trace": [],
        "graph_transition_count": 0,
        "repair_count": 0,
        "consecutive_duplicate_count": 0,
        "normalized_request": {"user_message": case["user_message"], "trip_request": case["trip_request"]},
        "capability_plan": None,
        "travel_search_attempted_signature": case["travel_search_attempted_signature"],
        "istanbul_expert_attempted_signature": case["istanbul_expert_attempted_signature"],
        "last_successful_capability_scope": case["last_successful_capability_scope"],
    }
    decide = _make_decide_node(decision_provider=provider, cancellation_check=lambda: False, monotonic_clock=lambda: 0.0)

    start = time.monotonic()
    updates = decide(state)
    latency_seconds = time.monotonic() - start

    trace = updates.get("trace", [])
    statuses = {t.get("status") for t in trace}
    capability_plan = updates.get("capability_plan") or {}
    classification_succeeded = bool(capability_plan.get("classification_succeeded"))
    actual_scope = capability_plan.get("scope")
    actual_scope_source = capability_plan.get("scope_source")

    final_decide_entries = [t for t in trace if t.get("node") == "Decide" and "action" in t and "status" not in t]
    repair_attempts_first_call = final_decide_entries[-1]["repair_attempts"] if final_decide_entries else None

    has_transport_failed = "transport_failed" in statuses
    has_capability_classification_failed = "capability_classification_failed" in statuses
    has_ineligible_rejected = "action_ineligible_rejected" in statuses
    has_ineligible_correction_failed = "action_ineligible_correction_failed" in statuses
    transport_retry_count = sum(1 for t in trace if t.get("status") == "transport_retry")

    final_schema_valid = (
        classification_succeeded and not has_transport_failed and not has_ineligible_correction_failed
    )
    first_attempt_schema_valid = (
        final_schema_valid and repair_attempts_first_call == 0 and not has_ineligible_rejected
    )

    if final_schema_valid:
        safe_error_category = "none"
    elif has_transport_failed:
        safe_error_category = "transport_error"
    elif has_capability_classification_failed:
        safe_error_category = "classification_format_invalid"
    elif has_ineligible_correction_failed:
        safe_error_category = "ineligible_action_uncorrected"
    else:
        safe_error_category = "decision_format_invalid"

    pending = updates.get("pending_action") or {}
    actual_action = pending.get("action")
    expected_action = case["expected_action"]
    expected_scope = case["expected_capability_scope"]
    expected_scope_source = case["expected_scope_source"]

    correct_scope = final_schema_valid and actual_scope == expected_scope
    correct_action = final_schema_valid and actual_action == expected_action
    correct_scope_source = final_schema_valid and actual_scope_source == expected_scope_source

    eligible_actions = sorted(a.value for a in (
        compute_eligible_supervisor_actions({**state, "capability_plan": capability_plan}) if capability_plan else set()
    ))

    unnecessary_action = (
        final_schema_valid and expected_action in _NON_DELEGATING_ACTIONS and actual_action in _DELEGATING_ACTIONS
    )

    leak_hits = _leak_scan(trace, pending)

    return {
        "case_id": case["case_id"], "language": case["language"],
        "is_followup": case["is_followup"],
        "expected_capability_scope": expected_scope, "actual_capability_scope": actual_scope,
        "correct_scope": correct_scope,
        "expected_scope_source": expected_scope_source, "actual_scope_source": actual_scope_source,
        "correct_scope_source": correct_scope_source,
        "expected_action": expected_action, "actual_action": actual_action, "correct_action": correct_action,
        "eligible_action_set": eligible_actions,
        "first_attempt_schema_valid": bool(first_attempt_schema_valid), "final_schema_valid": bool(final_schema_valid),
        "repair_count": updates.get("repair_count", 0), "transient_retry_count": transport_retry_count,
        "safe_error_category": safe_error_category, "unnecessary_action": bool(unnecessary_action),
        "latency_seconds": round(latency_seconds, 3), "model": provider.model, "provider": "qwen",
        "timestamp": datetime.now(timezone.utc).isoformat(), "leak_scan_hits": leak_hits,
    }


def main() -> None:
    if RESULTS_PATH.exists() and RESULTS_PATH.read_text(encoding="utf-8").strip():
        print("BLOCKED: results file already exists and is non-empty -- one-shot rule forbids a second live run.")
        raise SystemExit(3)

    key_present = qwen_api_key_present()
    print(f"QWEN_API_KEY/DASHSCOPE_API_KEY present: {str(key_present).lower()}")
    if not key_present:
        print("BLOCKED: no Qwen credential present -- live evaluation cannot start.")
        raise SystemExit(2)

    manifest = _load_cases()
    print(f"loaded case_set checksum: {manifest['_loaded_checksum']}")
    all_cases = manifest["cases"]
    print(f"total predeclared cases: {len(all_cases)}")

    provider = QwenDecisionProvider()
    results: list[dict] = []
    RESULTS_PATH.write_text("", encoding="utf-8")

    consecutive_transport_errors = 0
    blocked = False
    blocked_reason = ""

    for case in all_cases:
        try:
            record = _run_case(case, provider)
        except Exception as exc:  # noqa: BLE001
            record = {
                "case_id": case["case_id"], "language": case["language"], "is_followup": case["is_followup"],
                "expected_capability_scope": case["expected_capability_scope"], "actual_capability_scope": None,
                "correct_scope": False, "expected_scope_source": case["expected_scope_source"],
                "actual_scope_source": None, "correct_scope_source": False,
                "expected_action": case["expected_action"], "actual_action": None,
                "correct_action": False, "eligible_action_set": [],
                "first_attempt_schema_valid": False, "final_schema_valid": False,
                "repair_count": 0, "transient_retry_count": 0, "safe_error_category": "transport_error",
                "unnecessary_action": False, "latency_seconds": 0.0, "model": provider.model, "provider": "qwen",
                "timestamp": datetime.now(timezone.utc).isoformat(), "leak_scan_hits": [],
                "_unexpected_exception_type": type(exc).__name__,
            }
        results.append(record)
        with RESULTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{record['case_id']} [{record['language']}{'/followup' if record['is_followup'] else '/initial'}] "
              f"expected_scope={record['expected_capability_scope']} actual_scope={record.get('actual_capability_scope')} "
              f"expected_action={record['expected_action']} actual_action={record.get('actual_action')} "
              f"correct={record['correct_action']} error={record['safe_error_category']} latency={record['latency_seconds']}s")

        if record["safe_error_category"] == "transport_error":
            consecutive_transport_errors += 1
        else:
            consecutive_transport_errors = 0
        if consecutive_transport_errors >= 5:
            blocked = True
            blocked_reason = "5 consecutive transport errors -- stopping rather than fabricating completion"
            print(f"STOPPING EARLY: {blocked_reason}")
            break

    completed = len(results)
    total = len(all_cases)

    def _rate(rows, key):
        n = len(rows)
        if n == 0:
            return {"correct": 0, "n": 0, "rate": None}
        c = sum(1 for r in rows if r[key])
        return {"correct": c, "n": n, "rate": c / n}

    def _validity(rows, key):
        n = len(rows)
        if n == 0:
            return {"valid": 0, "n": 0, "rate": None}
        v = sum(1 for r in rows if r[key])
        return {"valid": v, "n": n, "rate": v / n}

    def _by(rows, field):
        out = {}
        for value in sorted({r[field] for r in rows}):
            subset = [r for r in rows if r[field] == value]
            out[value] = {"n": len(subset), "scope_accuracy": _rate(subset, "correct_scope"), "action_accuracy": _rate(subset, "correct_action")}
        return out

    initial_rows = [r for r in results if not r["is_followup"]]
    followup_rows = [r for r in results if r["is_followup"]]
    inherited_rows = [r for r in results if r["expected_scope_source"] == "inherited"]
    explicit_rows = [r for r in results if r["expected_scope_source"] == "explicit"]

    unnecessary_eligible = [r for r in results if r["expected_action"] in _NON_DELEGATING_ACTIONS]
    unnecessary_count = sum(1 for r in unnecessary_eligible if r["unnecessary_action"])
    leak_hits_total = sum(len(r["leak_scan_hits"]) for r in results)

    summary = {
        "checkpoint": "Final Evaluation Checkpoint E.1X",
        "case_set_sha256": manifest["_loaded_checksum"],
        "model": provider.model, "provider": "qwen",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_case_count": total, "completed_case_count": completed,
        "blocked": blocked, "blocked_reason": blocked_reason,
        "capability_scope_accuracy": _rate(results, "correct_scope"),
        "supervisor_next_action_accuracy": _rate(results, "correct_action"),
        "initial_request_scope_accuracy": _rate(initial_rows, "correct_scope"),
        "followup_scope_accuracy": _rate(followup_rows, "correct_scope"),
        "initial_request_action_accuracy": _rate(initial_rows, "correct_action"),
        "followup_action_accuracy": _rate(followup_rows, "correct_action"),
        "inherited_scope_accuracy": _rate(inherited_rows, "correct_scope_source"),
        "explicit_scope_change_accuracy": _rate(explicit_rows, "correct_scope_source"),
        "first_attempt_schema_validity": _validity(results, "first_attempt_schema_valid"),
        "final_schema_validity": _validity(results, "final_schema_valid"),
        "provider_transport_completion_rate": {
            "completed_without_transport_error": sum(1 for r in results if r["safe_error_category"] != "transport_error"),
            "n": completed,
            "rate": (sum(1 for r in results if r["safe_error_category"] != "transport_error") / completed) if completed else None,
        },
        "unnecessary_delegation": {
            "count": unnecessary_count, "eligible_n": len(unnecessary_eligible),
            "rate": (unnecessary_count / len(unnecessary_eligible)) if unnecessary_eligible else None,
        },
        "correction_rate": {
            "corrected_or_repaired": sum(1 for r in results if r["repair_count"] > 0),
            "n": completed,
            "rate": (sum(1 for r in results if r["repair_count"] > 0) / completed) if completed else None,
        },
        "retry_rate": {
            "retried": sum(1 for r in results if r.get("transient_retry_count", 0) > 0),
            "n": completed,
            "rate": (sum(1 for r in results if r.get("transient_retry_count", 0) > 0) / completed) if completed else None,
        },
        "credential_or_chain_of_thought_leak_hits": leak_hits_total,
        "total_repair_attempts": sum(r["repair_count"] for r in results),
        "mean_latency_seconds": (sum(r["latency_seconds"] for r in results) / completed) if completed else None,
        "results_by_language": _by(results, "language"),
        "results_by_capability_scope": _by(results, "expected_capability_scope"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print("")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
