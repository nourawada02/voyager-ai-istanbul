"""Small, targeted live smoke evaluation for known Phase 3 behavioral
categories (Checkpoint Phase 3 remediation §7), meant to run before
spending Groq quota on the full 45-question `rag.run_generation_eval`.

Exercises the exact same GroundedAnswerService / `evaluate_one` pipeline
as the full run, over a fixed 9-question subset selected for behaviors
previously observed to fail on a live Groq run (false refusals on
Beyoglu/hospitality cross-lingual claims, unsafe answers on dynamic
current-price questions), plus one already-correct weather refusal kept
as a regression control. Question selection is by fixed question_id list
below -- never by matching or hardcoding expected answers/facts, and no
ground-truth answer content is ever fed into generation, exactly like
the full run.

Writes evaluation/datasets/rag_smoke_eval_report.json -- a distinct file
from rag_generation_eval_report(_remediated).json, which this script
never touches. Does not use or write to the 45-case checkpoint
(rag/_gen_eval_checkpoint.jsonl); nine questions do not need resumability.

Reproducibility command:
    python -m rag.run_smoke_eval
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rag.answer_service import REFUSAL_TEXT, GroundedAnswerService
from rag.generation_eval import GenerationEvalRow, evaluate_one
from rag.ground_truth import QUESTIONS
from rag.llm_providers import detect_available_providers, enforce_required_provider
from rag.run_generation_eval import OUT_DIR, _select_evaluation_providers, build_retrieval_context

# (question_id, why it's in the smoke set)
SMOKE_QUESTIONS: tuple[tuple[str, str], ...] = (
    ("gt_04_en", "Beyoglu, English -- previously a false refusal"),
    ("gt_04_tr", "Beyoglu, Turkish -- previously a false refusal"),
    ("gt_07_tr", "hospitality, Turkish -- previously a false refusal"),
    ("gt_07_ar", "hospitality, Arabic -- previously a false refusal"),
    ("gt_09_en", "current ticket price, English -- previously an unsafe answer"),
    ("gt_09_tr", "current ticket price, Turkish -- previously an unsafe answer"),
    ("gt_09_ar", "current ticket price, Arabic -- required example"),
    ("gt_13_tr", "cross-lingual Istanbulkart, Turkish question over an English-only source"),
    ("gt_14_en", "weather refusal, English -- regression control (previously correct)"),
)

SMOKE_REPORT_PATH = OUT_DIR / "rag_smoke_eval_report.json"

# The behaviors this smoke run exists to catch: false refusals must be
# answered, unsafe live-data answers must refuse. Not every smoke
# question participates in this gate (gt_13_tr and gt_14_en are
# informational/regression controls, not root-cause regression targets).
_MUST_ANSWER_QUESTION_IDS = frozenset({"gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar"})
_MUST_REFUSE_QUESTION_IDS = frozenset({"gt_09_en", "gt_09_tr", "gt_09_ar", "gt_14_en"})


def _questions_by_id(question_ids: tuple[str, ...]) -> list:
    by_id = {q.question_id: q for q in QUESTIONS}
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise RuntimeError(f"unknown smoke question_id(s): {missing}")
    return [by_id[qid] for qid in question_ids]


def _row_refused(row: GenerationEvalRow) -> bool:
    """Whether this row's answer is the canonical refusal -- derived from
    the answer text rather than re-deriving AnswerResult.refused, since
    GenerationEvalRow does not carry that field directly, and every
    refusal path returns REFUSAL_TEXT verbatim (never a paraphrase)."""
    return row.answer_text == REFUSAL_TEXT


def _row_summary(row: GenerationEvalRow, reason: str) -> dict[str, Any]:
    return {
        "question_id": row.question_id,
        "reason_in_smoke_set": reason,
        "language": row.language,
        "category": row.category,
        "refusal_required": row.refusal_required,
        "refused": _row_refused(row),
        "correctly_refused": row.correctly_refused,
        "used_fallback": row.used_fallback,
        "degradation_reason": row.degradation_reason,
        "citation_count": row.citation_count,
        "citations": row.citations,
        "emitted_claim_count": row.emitted_claim_count,
        "dropped_claim_count": row.dropped_claim_count,
        "judge_error": row.judge_error,
        "judge_score": (
            {
                "faithfulness": row.judge_score.faithfulness,
                "correctness": row.judge_score.correctness,
                "relevance": row.judge_score.relevance,
                "reason": row.judge_score.reason,
            }
            if row.judge_score
            else None
        ),
        "failure_categories": row.failure_categories,
        "answer_text": row.answer_text,
    }


def _provider_transport_failed_question_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Question IDs where either role's provider failed at the transport
    layer (connection reset, timeout, TLS handshake failure) after its
    own bounded retry budget -- see llm_providers.ProviderTransportError.
    Detected via failure_categories, never by parsing judge_error text.
    """
    return [
        r["question_id"]
        for r in rows
        if "generator_transport_failed" in r["failure_categories"]
        or "judge_transport_failed" in r["failure_categories"]
    ]


def _smoke_gate_passed(rows: list[dict[str, Any]]) -> bool:
    by_id = {r["question_id"]: r for r in rows}
    for question_id in _MUST_ANSWER_QUESTION_IDS:
        row = by_id[question_id]
        if row["refused"] or row["citation_count"] == 0:
            return False
    for question_id in _MUST_REFUSE_QUESTION_IDS:
        row = by_id[question_id]
        if not row["refused"]:
            return False
    if any(r["judge_error"] is not None for r in rows):
        return False
    # A provider transport failure must fail the gate even when
    # answer_service's safe degradation happened to still produce a
    # cited, non-refused answer for a must-answer question -- never a
    # superficial green result from a provider outage.
    if _provider_transport_failed_question_ids(rows):
        return False
    return True


def run() -> dict[str, Any]:
    detection = detect_available_providers()
    enforce_required_provider(detection)

    context = build_retrieval_context()
    generator, judge = _select_evaluation_providers(detection)

    print(f"provider_detection: {detection.description}", flush=True)
    print(f"generator_model: {generator.model if generator is not None else None}", flush=True)
    print(f"judge_model: {judge.model if judge is not None else None}", flush=True)

    questions = _questions_by_id(tuple(qid for qid, _ in SMOKE_QUESTIONS))
    reasons = dict(SMOKE_QUESTIONS)

    rows: list[dict[str, Any]] = []
    for question in questions:
        service = GroundedAnswerService(
            retrieval=context.retrieval_service,
            provider=generator,
            chunk_config=context.winner_config,
            top_k=context.top_k,
            retrieval_mode=context.winner_mode,
        )
        row = evaluate_one(question, service, context.chunk_texts_by_id, context.source_metadata_by_id, judge)
        row_summary = _row_summary(row, reasons[question.question_id])
        rows.append(row_summary)
        print(
            f"[{question.question_id}] refused={row_summary['refused']} "
            f"used_fallback={row_summary['used_fallback']} "
            f"citations={row_summary['citation_count']} "
            f"judge_error={row_summary['judge_error']!r} "
            f"failure_categories={row_summary['failure_categories']}",
            flush=True,
        )

    gate_passed = _smoke_gate_passed(rows)
    provider_transport_failed_question_ids = _provider_transport_failed_question_ids(rows)
    report = {
        "schema_version": "1.0.0",
        "reproducibility_command": "python -m rag.run_smoke_eval",
        "provider_detection": detection.description,
        "generator_model": generator.model if generator is not None else None,
        "judge_model": judge.model if judge is not None else None,
        "smoke_gate_passed": gate_passed,
        "judge_failed_question_ids": [r["question_id"] for r in rows if r["judge_error"] is not None],
        "provider_transport_failed_question_ids": provider_transport_failed_question_ids,
        "rows": rows,
    }

    _write_json_atomic(SMOKE_REPORT_PATH, report)
    return report


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write the smoke report as one complete, valid JSON file or not at
    all. A crash or a killed process mid-write must never leave a
    partially written/corrupt report behind -- same atomic
    write-to-temp-then-replace pattern as
    run_generation_eval._write_checkpoint_atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _gate_exit_code(report: dict[str, Any]) -> int:
    return 0 if report.get("smoke_gate_passed") is True else 1


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2, ensure_ascii=False))
    raise SystemExit(_gate_exit_code(r))
