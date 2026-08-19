"""Final Evaluation Checkpoint E.1 §4 -- runs the frozen 50-case manifest
(`evaluation/e1_live_qwen_cases.json`) against the REAL Qwen decision
provider. No tool, MCP, or A2A call is ever made -- this measures
decision ROUTING/SELECTION only, exactly the boundary the checkpoint
requires.

Reuses, never reimplements: `phase4.graph._build_decision_prompt` /
`phase4.specialist._build_specialist_prompt` for prompt construction
(the exact real production prompts), and mirrors -- not reinvents --
the real bounded repair loop already implemented in
`phase4.graph._decide_node` / `phase4.specialist._specialist_decide`
(same repair-budget constants, same repair-prompt-append pattern).

Run with:
    python -m evaluation.run_e1_live_qwen_eval

Writes:
    evaluation/e1_live_qwen_results.jsonl   (one sanitized record per case)
    evaluation/e1_live_qwen_summary.json    (aggregate metrics)

Never saves: complete system prompts, raw model output, chain-of-thought,
credential-bearing URLs, or credential values.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from phase4.graph import _build_decision_prompt
from phase4.models import (
    MAX_DECISION_REPAIRS,
    MAX_SPECIALIST_DECISION_REPAIRS,
    SPECIALIST_ACTIONS,
    SUPERVISOR_ACTIONS,
    Action,
    ActionDecisionValidationError,
    parse_action_decision,
)
from phase4.qwen_client import QwenConfigurationError, QwenDecisionProvider, QwenTransportError, qwen_api_key_present
from phase4.specialist import _build_specialist_prompt

CASES_PATH = Path(__file__).parent / "e1_live_qwen_cases.json"
RESULTS_PATH = Path(__file__).parent / "e1_live_qwen_results.jsonl"
SUMMARY_PATH = Path(__file__).parent / "e1_live_qwen_summary.json"

_FORBIDDEN_SUBSTRINGS = (
    "chain_of_thought", "chain-of-thought", "authorization", "bearer",
)


def _load_cases() -> dict:
    text = CASES_PATH.read_text(encoding="utf-8")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest = json.loads(text)
    manifest["_loaded_checksum"] = checksum
    return manifest


def _supervisor_state(case: dict) -> dict:
    return {
        "normalized_request": {"user_message": case["user_message"], "trip_request": case["trip_request"]},
        "user_message": case["user_message"],
        "observations": case["observations"],
        "tool_call_count": case["tool_call_count"],
    }


def _specialist_state(case: dict) -> dict:
    return {
        "normalized_request": {"user_message": case["user_message"], "trip_request": case["trip_request"]},
        "inherited_observations": case["inherited_observations"],
        "specialist_observations": case["specialist_observations"],
        "tool_call_count": case["tool_call_count"],
    }


def _run_one_case(case: dict, provider: QwenDecisionProvider) -> dict:
    role = case["agent_role"]
    if role == "supervisor":
        allowed_actions = SUPERVISOR_ACTIONS
        system, user = _build_decision_prompt(_supervisor_state(case))
        max_attempts = MAX_DECISION_REPAIRS + 1
    else:
        allowed_actions = SPECIALIST_ACTIONS
        system, user = _build_specialist_prompt(_specialist_state(case))
        max_attempts = MAX_SPECIALIST_DECISION_REPAIRS + 1

    start = time.monotonic()
    decision = None
    first_attempt_valid = False
    attempts = 0
    safe_error_code = "none"
    raw_text_for_leak_scan = ""

    while attempts < max_attempts and decision is None:
        attempts += 1
        try:
            raw_text = provider.generate(system, user)
            raw_text_for_leak_scan += raw_text
            raw_obj = json.loads(raw_text)
            candidate = parse_action_decision(raw_obj)
            if candidate.action not in allowed_actions:
                raise ActionDecisionValidationError(f"{role} decision named an out-of-role action {candidate.action.value!r}")
            decision = candidate
            if attempts == 1:
                first_attempt_valid = True
        except (json.JSONDecodeError, ActionDecisionValidationError) as exc:
            last_error = str(exc)[:200]
            safe_error_code = "decision_format_invalid"
            if attempts < max_attempts:
                user = user + f"\n\nYour previous response was invalid ({last_error}). Respond again with a single valid JSON object matching the required schema exactly."
        except QwenConfigurationError as exc:
            safe_error_code = "transport_error"
            break
        except QwenTransportError as exc:
            safe_error_code = "transport_error"
            break

    latency_seconds = time.monotonic() - start
    final_valid = decision is not None
    actual_action = decision.action.value if decision is not None else None
    expected_actions = case["expected_actions"]
    correct = final_valid and actual_action in expected_actions
    unnecessary_action = (
        final_valid and not correct
        and set(expected_actions).issubset({"synthesize", "travel_search_complete"})
    )

    leak_hits = [term for term in _FORBIDDEN_SUBSTRINGS if term in raw_text_for_leak_scan.lower()]

    return {
        "case_id": case["case_id"],
        "agent_role": role,
        "language": case["language"],
        "category": case["category"],
        "expected_actions": expected_actions,
        "actual_action": actual_action,
        "correct": correct,
        "first_attempt_schema_valid": first_attempt_valid,
        "final_schema_valid": final_valid,
        "repair_count": max(0, attempts - 1),
        "safe_error_code": safe_error_code if not final_valid else "none",
        "latency_seconds": round(latency_seconds, 3),
        "unnecessary_action": unnecessary_action,
        "model": provider.model,
        "provider": "qwen",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "leak_scan_hits": leak_hits,
    }


def main() -> None:
    key_present = qwen_api_key_present()
    print(f"QWEN_API_KEY/DASHSCOPE_API_KEY present: {str(key_present).lower()}")
    if not key_present:
        print("BLOCKED: no Qwen credential present -- live evaluation cannot start.")
        raise SystemExit(2)

    manifest = _load_cases()
    print(f"loaded case_set checksum: {manifest['_loaded_checksum']}")
    all_cases = manifest["supervisor_cases"] + manifest["specialist_cases"]
    print(f"total predeclared cases: {len(all_cases)}")

    provider = QwenDecisionProvider()
    results: list[dict] = []
    RESULTS_PATH.write_text("", encoding="utf-8")  # fresh file this run

    consecutive_transport_errors = 0
    blocked = False
    blocked_reason = ""

    for case in all_cases:
        try:
            record = _run_one_case(case, provider)
        except Exception as exc:  # noqa: BLE001 -- never let one case's unexpected failure abort the run silently
            record = {
                "case_id": case["case_id"], "agent_role": case["agent_role"], "language": case["language"],
                "category": case["category"], "expected_actions": case["expected_actions"],
                "actual_action": None, "correct": False, "first_attempt_schema_valid": False,
                "final_schema_valid": False, "repair_count": 0, "safe_error_code": "transport_error",
                "latency_seconds": 0.0, "unnecessary_action": False, "model": provider.model,
                "provider": "qwen", "timestamp": datetime.now(timezone.utc).isoformat(), "leak_scan_hits": [],
            }
        results.append(record)
        with RESULTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{record['case_id']} [{record['agent_role']}/{record['language']}] "
              f"expected={record['expected_actions']} actual={record['actual_action']} "
              f"correct={record['correct']} repair={record['repair_count']} "
              f"error={record['safe_error_code']} latency={record['latency_seconds']}s")

        if record["safe_error_code"] == "transport_error":
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

    def _rate(rows, pred):
        subset = [r for r in rows if pred(r)]
        return (len(subset), len(subset))

    supervisor_rows = [r for r in results if r["agent_role"] == "supervisor"]
    specialist_rows = [r for r in results if r["agent_role"] == "specialist"]

    def _accuracy(rows):
        n = len(rows)
        if n == 0:
            return {"correct": 0, "n": 0, "rate": None}
        c = sum(1 for r in rows if r["correct"])
        return {"correct": c, "n": n, "rate": c / n}

    def _validity(rows, key):
        n = len(rows)
        if n == 0:
            return {"valid": 0, "n": 0, "rate": None}
        v = sum(1 for r in rows if r[key])
        return {"valid": v, "n": n, "rate": v / n}

    unnecessary_eligible = [r for r in results if set(r["expected_actions"]).issubset({"synthesize", "travel_search_complete"})]
    unnecessary_count = sum(1 for r in unnecessary_eligible if r["unnecessary_action"])

    leak_hits_total = sum(len(r["leak_scan_hits"]) for r in results)

    summary = {
        "checkpoint": "Final Evaluation Checkpoint E.1",
        "case_set_sha256": manifest["_loaded_checksum"],
        "model": provider.model,
        "provider": "qwen",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_case_count": total,
        "completed_case_count": completed,
        "blocked": blocked,
        "blocked_reason": blocked_reason,
        "supervisor_routing_accuracy": _accuracy(supervisor_rows),
        "specialist_tool_selection_correctness": _accuracy(specialist_rows),
        "first_attempt_schema_validity": _validity(results, "first_attempt_schema_valid"),
        "final_schema_validity_after_bounded_repair": _validity(results, "final_schema_valid"),
        "unnecessary_delegation_or_tool_call": {
            "count": unnecessary_count, "eligible_n": len(unnecessary_eligible),
            "rate": (unnecessary_count / len(unnecessary_eligible)) if unnecessary_eligible else None,
        },
        "credential_or_chain_of_thought_leak_hits": leak_hits_total,
        "total_repair_attempts": sum(r["repair_count"] for r in results),
        "mean_latency_seconds": (sum(r["latency_seconds"] for r in results) / completed) if completed else None,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
