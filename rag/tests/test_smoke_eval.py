"""Tests for the targeted live smoke evaluation runner (Checkpoint Phase
3 remediation §7, and transport resilience): question selection, the
derived-refusal helper, and the smoke-specific pass/fail gate -- all
hermetic, no provider/network calls."""

from __future__ import annotations

import json

import pytest

from rag import llm_providers, run_generation_eval, run_smoke_eval
from rag.answer_service import REFUSAL_TEXT
from rag.generation_eval import GenerationEvalRow
from rag.ground_truth import QUESTIONS


def _row(
    question_id: str,
    *,
    refused: bool,
    citation_count: int = 1,
    judge_error: str | None = None,
    failure_categories: list[str] | None = None,
) -> GenerationEvalRow:
    return GenerationEvalRow(
        question_id=question_id,
        language="en",
        category="factual",
        refusal_required=False,
        answer_text=REFUSAL_TEXT if refused else "Some grounded answer.",
        correctly_refused=None,
        retrieved_chunk_ids=("c1",),
        citations=[] if refused else [{"source_id": "s1", "chunk_ids": ["c1"]}],
        citation_count=0 if refused else citation_count,
        emitted_claim_count=0 if refused else 1,
        dropped_claim_count=0,
        used_fallback=False,
        degradation_reason="live_data_required" if refused else None,
        judge_score=None,
        judge_error=judge_error,
        failure_categories=failure_categories if failure_categories is not None else [],
    )


def test_smoke_question_ids_all_exist_in_ground_truth():
    known_ids = {q.question_id for q in QUESTIONS}
    for question_id, _reason in run_smoke_eval.SMOKE_QUESTIONS:
        assert question_id in known_ids


def test_smoke_question_set_covers_every_required_category():
    ids = {qid for qid, _ in run_smoke_eval.SMOKE_QUESTIONS}
    assert {"gt_04_en", "gt_04_tr"} <= ids  # Beyoglu EN/TR
    assert {"gt_07_tr", "gt_07_ar"} <= ids  # hospitality TR/AR
    assert {"gt_09_en", "gt_09_tr", "gt_09_ar"} <= ids  # current ticket price EN/TR/AR
    assert "gt_13_tr" in ids  # cross-lingual Istanbulkart
    assert "gt_14_en" in ids  # weather/opening-hours refusal


def test_smoke_report_path_is_distinct_from_the_45_case_report_paths():
    assert run_smoke_eval.SMOKE_REPORT_PATH != run_generation_eval.OUT_DIR / "rag_generation_eval_report.json"
    assert run_smoke_eval.SMOKE_REPORT_PATH != run_generation_eval.OUT_DIR / "rag_generation_eval_report_remediated.json"
    assert run_smoke_eval.SMOKE_REPORT_PATH.name == "rag_smoke_eval_report.json"


def test_smoke_never_imports_or_uses_the_45_case_checkpoint_path():
    assert not hasattr(run_smoke_eval, "DEFAULT_CHECKPOINT_PATH")
    assert not hasattr(run_smoke_eval, "GEN_EVAL_CHECKPOINT_PATH")


def test_questions_by_id_raises_on_unknown_id():
    with pytest.raises(RuntimeError, match="unknown smoke question_id"):
        run_smoke_eval._questions_by_id(("not_a_real_question_id",))


def test_row_refused_matches_exact_refusal_text_only():
    refused_row = _row("gt_09_en", refused=True)
    answered_row = _row("gt_04_en", refused=False)
    assert run_smoke_eval._row_refused(refused_row) is True
    assert run_smoke_eval._row_refused(answered_row) is False


def test_gate_fails_when_a_must_answer_question_is_refused():
    rows = [
        run_smoke_eval._row_summary(_row(qid, refused=(qid == "gt_04_en")), "why")
        for qid in ("gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar", "gt_09_en", "gt_09_tr", "gt_09_ar", "gt_13_tr", "gt_14_en")
    ]
    assert run_smoke_eval._smoke_gate_passed(rows) is False


def test_gate_fails_when_a_must_refuse_question_is_answered():
    rows = [
        run_smoke_eval._row_summary(_row(qid, refused=(qid != "gt_09_en")), "why")
        for qid in ("gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar", "gt_09_en", "gt_09_tr", "gt_09_ar", "gt_13_tr", "gt_14_en")
    ]
    assert run_smoke_eval._smoke_gate_passed(rows) is False


def test_gate_fails_on_any_judge_error():
    rows = [
        run_smoke_eval._row_summary(
            _row(qid, refused=qid in run_smoke_eval._MUST_REFUSE_QUESTION_IDS, judge_error=("boom" if qid == "gt_04_en" else None)),
            "why",
        )
        for qid in ("gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar", "gt_09_en", "gt_09_tr", "gt_09_ar", "gt_13_tr", "gt_14_en")
    ]
    assert run_smoke_eval._smoke_gate_passed(rows) is False


def test_gate_passes_when_all_conditions_are_met():
    rows = [
        run_smoke_eval._row_summary(_row(qid, refused=qid in run_smoke_eval._MUST_REFUSE_QUESTION_IDS), "why")
        for qid in ("gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar", "gt_09_en", "gt_09_tr", "gt_09_ar", "gt_13_tr", "gt_14_en")
    ]
    assert run_smoke_eval._smoke_gate_passed(rows) is True


def test_gate_exit_code_reflects_smoke_gate():
    assert run_smoke_eval._gate_exit_code({"smoke_gate_passed": True}) == 0
    assert run_smoke_eval._gate_exit_code({"smoke_gate_passed": False}) == 1
    assert run_smoke_eval._gate_exit_code({}) == 1


def test_gate_fails_on_a_provider_transport_failure_even_on_a_non_gated_question():
    """A provider transport failure on gt_13_tr (not a must-answer or
    must-refuse question) must still fail the gate -- a transport
    failure must never produce a superficial green result just because
    it happened to land on a question the answer/refuse checks don't
    directly cover."""
    rows = [
        run_smoke_eval._row_summary(
            _row(
                qid,
                refused=qid in run_smoke_eval._MUST_REFUSE_QUESTION_IDS,
                failure_categories=["generator_transport_failed"] if qid == "gt_13_tr" else [],
            ),
            "why",
        )
        for qid in ("gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar", "gt_09_en", "gt_09_tr", "gt_09_ar", "gt_13_tr", "gt_14_en")
    ]
    # Without the transport check, this row set would otherwise pass every other condition.
    assert run_smoke_eval._smoke_gate_passed(rows) is False
    assert run_smoke_eval._provider_transport_failed_question_ids(rows) == ["gt_13_tr"]


def test_gate_fails_on_a_judge_transport_failure():
    rows = [
        run_smoke_eval._row_summary(
            _row(
                qid,
                refused=qid in run_smoke_eval._MUST_REFUSE_QUESTION_IDS,
                judge_error="boom" if qid == "gt_04_en" else None,
                failure_categories=["judge_call_failed", "judge_transport_failed"] if qid == "gt_04_en" else [],
            ),
            "why",
        )
        for qid in ("gt_04_en", "gt_04_tr", "gt_07_tr", "gt_07_ar", "gt_09_en", "gt_09_tr", "gt_09_ar", "gt_13_tr", "gt_14_en")
    ]
    assert run_smoke_eval._smoke_gate_passed(rows) is False
    assert run_smoke_eval._provider_transport_failed_question_ids(rows) == ["gt_04_en"]


def test_smoke_run_completes_all_rows_and_writes_a_failing_report_on_a_transport_failure(monkeypatch, tmp_path):
    """End-to-end (but fully hermetic) proof that a single provider
    transport failure partway through the smoke set does not crash
    run_smoke_eval.run(): every question still gets a row, the report is
    written, and its gate correctly fails."""
    report_path = tmp_path / "rag_smoke_eval_report.json"
    monkeypatch.setattr(run_smoke_eval, "SMOKE_REPORT_PATH", report_path)
    monkeypatch.setattr(
        run_smoke_eval, "detect_available_providers",
        lambda: llm_providers.ProviderDetectionResult(hosted_provider=None, local_ollama_model=None),
    )
    monkeypatch.setattr(run_smoke_eval, "enforce_required_provider", lambda detection: None)

    class _FakeRetrievalService:
        def search(self, **kwargs):
            return {
                "schema_version": "1.0.0", "query": kwargs["query"], "query_language": kwargs["query_language"],
                "retrieval_mode": kwargs["mode"], "chunk_config": kwargs["chunk_config"], "top_k": kwargs["top_k"],
                "collection_fingerprint": "fp",
                "items": [{"rank": 1, "chunk_id": "c1", "source_id": "s1", "score": 0.9}],
                "latency_ms": 1.0, "zero_result": False,
            }

    fake_context = run_generation_eval.RetrievalContext(
        retrieval_service=_FakeRetrievalService(),
        chunk_texts_by_id={"c1": "Beyoglu was historically known as Pera."},
        source_metadata_by_id={"s1": {"title": "Beyoglu", "url": "https://x", "language": "en"}},
        top_k=3,
        winner_mode="dense",
        winner_config="A",
        collection_fingerprint="fp",
    )
    monkeypatch.setattr(run_smoke_eval, "build_retrieval_context", lambda: fake_context)

    # Non-live-data smoke questions, in SMOKE_QUESTIONS order: gt_04_en,
    # gt_04_tr, gt_07_tr, gt_07_ar, gt_13_tr (the other four -- gt_09_*,
    # gt_14_en -- are live-data and never reach the generator at all).
    # The 5th such call (gt_13_tr, not a must-answer/must-refuse
    # question) fails at the transport layer.
    call_state = {"n": 0}

    class _FlakyGenerator:
        name = "qwen"
        model = "test-model"

        def generate_json(self, system, user, validate, max_retries=2):
            call_state["n"] += 1
            if call_state["n"] == 5:
                raise llm_providers.ProviderTransportError("qwen", 3)
            data = {
                "answer_status": "grounded",
                "claims": [{
                    "text": "Beyoglu was historically known as Pera.",
                    "evidence_quote": "Beyoglu was historically known as Pera.",
                    "chunk_ids": ["c1"],
                }],
            }
            validate(data)
            return data

    class _StubJudge:
        name = "groq"
        model = "test-model"

        def generate_json(self, system, user, validate, max_retries=2):
            data = {"faithfulness": 2, "correctness": 2, "relevance": 2, "reason": "ok"}
            validate(data)
            return data

    monkeypatch.setattr(
        run_smoke_eval, "_select_evaluation_providers",
        lambda detection: (_FlakyGenerator(), _StubJudge()),
    )

    report = run_smoke_eval.run()

    assert len(report["rows"]) == len(run_smoke_eval.SMOKE_QUESTIONS)  # completed every question, never crashed
    assert report["smoke_gate_passed"] is False
    assert "gt_13_tr" in report["provider_transport_failed_question_ids"]
    assert report_path.exists()
    written = json.loads(report_path.read_text(encoding="utf-8"))
    assert written == report
    # No temp file left behind by the atomic write.
    assert not report_path.with_name(f".{report_path.name}.tmp").exists()
