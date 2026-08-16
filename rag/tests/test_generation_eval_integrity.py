"""Regression tests for Phase 3 generation-evaluation integrity."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rag.answer_service import AnswerResult
from rag.generation_eval import evaluate_one
from rag.ground_truth import GroundTruthQuestion
from rag.llm_providers import (
    GROQ_GENERATOR_MODEL,
    GROQ_JUDGE_MODEL,
    ProviderDetectionResult,
    ProviderRateLimitExceeded,
    ProviderTransportError,
)
from rag import run_generation_eval


@dataclass
class _AnswerService:
    result: AnswerResult

    def answer(self, *args, **kwargs) -> AnswerResult:
        return self.result


@dataclass
class _Judge:
    calls: list[str]
    name: str = "stub"
    model: str = "stub-model"

    def generate_json(self, system, user, validate, max_retries=2):
        self.calls.append(user)
        data = {"faithfulness": 2, "correctness": 2, "relevance": 2, "reason": "grounded"}
        validate(data)
        return data


def _question() -> GroundTruthQuestion:
    return GroundTruthQuestion(
        question_id="q1",
        slot=1,
        language="en",
        question="What was Beyoglu historically called?",
        expected_source_ids=("s1",),
        expected_section="History",
        poi_id=None,
        district_id="district_beyoglu",
        expected_facts=("SECRET_GROUND_TRUTH_SENTINEL",),
        refusal_required=False,
        category="neighborhood",
    )


def test_judge_receives_exact_retrieved_context_without_ground_truth_leakage():
    result = AnswerResult(
        answer_text="Beyoglu was historically called Pera.",
        refused=False,
        citations=[{"source_id": "s1", "chunk_ids": ["wanted"]}],
        degraded=False,
        degradation_reason=None,
        retrieval_zero_result=False,
        total_claim_count=1,
        retrieved_chunk_ids=("wanted",),
    )
    judge = _Judge(calls=[])

    row = evaluate_one(
        _question(),
        _AnswerService(result),
        {
            "wrong": "This is the first global chunk but was not retrieved.",
            "wanted": "Beyoglu was historically known as Pera.",
        },
        {"s1": {"title": "Beyoglu"}},
        judge,
    )

    assert row.judge_score is not None
    assert "Beyoglu was historically known as Pera" in judge.calls[0]
    assert "first global chunk" not in judge.calls[0]
    assert "SECRET_GROUND_TRUTH_SENTINEL" not in judge.calls[0]


def test_judge_receives_only_cited_chunks_not_all_retrieved_chunks_for_a_grounded_answer():
    """Repair 5: a non-refused, cited answer must expose only the chunks
    its emitted claims actually cite to the judge -- not every chunk the
    retriever returned. Reduces judge token usage and avoids exposing
    uncited retrieved content the judge has no reason to see."""
    result = AnswerResult(
        answer_text="Beyoglu was historically known as Pera.",
        refused=False,
        citations=[{"source_id": "s1", "chunk_ids": ["cited1"]}],
        degraded=False,
        degradation_reason=None,
        retrieval_zero_result=False,
        total_claim_count=1,
        retrieved_chunk_ids=("cited1", "uncited2", "uncited3"),
    )
    judge = _Judge(calls=[])

    row = evaluate_one(
        _question(),
        _AnswerService(result),
        {
            "cited1": "Beyoglu was historically known as Pera.",
            "uncited2": "UNCITED_CHUNK_MARKER_TWO must never reach the judge.",
            "uncited3": "UNCITED_CHUNK_MARKER_THREE must never reach the judge either.",
        },
        {"s1": {"title": "Beyoglu"}},
        judge,
    )

    assert row.judge_score is not None
    assert "cited1" in judge.calls[0]
    assert "Beyoglu was historically known as Pera" in judge.calls[0]
    assert "uncited2" not in judge.calls[0]
    assert "uncited3" not in judge.calls[0]
    assert "UNCITED_CHUNK_MARKER" not in judge.calls[0]


def test_judge_receives_full_retrieved_context_for_a_false_refusal():
    """A refusal (on a question that was actually answerable) has no
    citations to trim to -- the judge needs the full retrieved context
    to be able to tell that the refusal was false."""
    result = AnswerResult(
        answer_text="I don't have grounded information on this in my current sources.",
        refused=True,
        citations=[],
        degraded=False,
        degradation_reason=None,
        retrieval_zero_result=False,
        total_claim_count=0,
        retrieved_chunk_ids=("r1", "r2"),
    )
    judge = _Judge(calls=[])

    row = evaluate_one(
        _question(),
        _AnswerService(result),
        {
            "r1": "Beyoglu was historically known as Pera.",
            "r2": "Today Beyoglu is a center for cultural activities.",
        },
        {"s1": {"title": "Beyoglu"}},
        judge,
    )

    assert row.judge_score is not None
    assert "r1" in judge.calls[0] and "Beyoglu was historically known as Pera" in judge.calls[0]
    assert "r2" in judge.calls[0] and "center for cultural activities" in judge.calls[0]


@dataclass
class _TransportFailingJudge:
    """Simulates a judge whose provider fails at the transport layer
    (connection reset, timeout, TLS handshake failure) after its own
    bounded retry budget -- exactly the live failure this checkpoint
    fixes: a Groq judge call died with
    URLError(WinError 10054 ...) and crashed the whole run."""

    name: str = "groq"
    model: str = "stub-model"

    def generate_json(self, system, user, validate, max_retries=2):
        raise ProviderTransportError(self.name, 3)


def test_judge_transport_failure_is_recorded_as_judge_error_without_raising():
    result = AnswerResult(
        answer_text="Beyoglu was historically known as Pera.",
        refused=False,
        citations=[{"source_id": "s1", "chunk_ids": ["c1"]}],
        degraded=False,
        degradation_reason=None,
        retrieval_zero_result=False,
        total_claim_count=1,
        retrieved_chunk_ids=("c1",),
    )

    row = evaluate_one(
        _question(),
        _AnswerService(result),
        {"c1": "Beyoglu was historically known as Pera."},
        {"s1": {"title": "Beyoglu"}},
        _TransportFailingJudge(),
    )

    assert row.judge_score is None
    assert row.judge_error is not None
    assert "groq" in row.judge_error
    assert "judge_call_failed" in row.failure_categories
    assert "judge_transport_failed" in row.failure_categories
    # The safe ProviderTransportError message only -- never a raw traceback.
    assert "Traceback" not in row.judge_error


def test_judge_transport_failure_causes_the_gate_to_fail():
    row_dict = {
        "question_id": "q1",
        "language": "en",
        "category": "factual",
        "refusal_required": False,
        "correctly_refused": None,
        "emitted_claim_count": 1,
        "citation_count": 1,
        "used_fallback": False,
        "judge_error": "provider 'groq' transport failed after 3 attempt(s)...",
        "judge_score": None,
        "failure_categories": ["judge_call_failed", "judge_transport_failed"],
    }

    summary = run_generation_eval._summarize_rows([row_dict])

    assert summary["judge_transport_failure_count"] == 1
    assert summary["gate_passed"] is False


def test_generator_transport_failure_causes_the_gate_to_fail_even_with_perfect_citations():
    """A generator transport failure must fail the gate even when
    answer_service's safe degraded/fallback result happened to still
    produce a cited, non-refused answer -- never a superficial green
    result from a provider outage."""
    row_dict = {
        "question_id": "q1",
        "language": "en",
        "category": "factual",
        "refusal_required": False,
        "correctly_refused": None,
        "emitted_claim_count": 1,
        "citation_count": 1,
        "used_fallback": True,
        "judge_error": None,
        "judge_score": {"faithfulness": 2, "correctness": 2, "relevance": 2},
        "failure_categories": ["generator_transport_failed"],
    }

    summary = run_generation_eval._summarize_rows([row_dict])

    assert summary["grounded_claim_citation_coverage"] == 1.0  # would otherwise look perfect
    assert summary["generator_transport_failure_count"] == 1
    assert summary["gate_passed"] is False


def test_groq_generation_and_judging_use_independent_model_budgets():
    generator, judge = run_generation_eval._select_evaluation_providers(
        ProviderDetectionResult(hosted_provider="groq", local_ollama_model=None)
    )

    assert generator is not None and generator.model == GROQ_GENERATOR_MODEL
    assert judge is not None and judge.model == GROQ_JUDGE_MODEL
    assert generator.model != judge.model


def test_false_refusal_cannot_pass_even_when_emitted_claim_citations_are_perfect():
    cited_answer = {
        "question_id": "answered",
        "refusal_required": False,
        "correctly_refused": None,
        "emitted_claim_count": 1,
        "citation_count": 1,
        "used_fallback": False,
        "judge_error": None,
        "judge_score": {"faithfulness": 2, "correctness": 2, "relevance": 2},
        "failure_categories": [],
    }
    false_refusal = {
        "question_id": "refused",
        "refusal_required": False,
        "correctly_refused": None,
        "emitted_claim_count": 0,
        "citation_count": 0,
        "used_fallback": False,
        "judge_error": None,
        "judge_score": {"faithfulness": 2, "correctness": 1, "relevance": 0},
        "failure_categories": ["refusal_failure"],
    }

    summary = run_generation_eval._summarize_rows([cited_answer, false_refusal])

    assert summary["grounded_claim_citation_coverage"] == 1.0
    assert summary["citation_coverage_over_emitted_claims"] == 1.0
    assert summary["answerable_response_rate"] == 0.5
    assert summary["false_refusal_count"] == 1
    assert summary["gate_passed"] is False


def test_judge_detected_hallucination_is_a_hard_gate_failure():
    row = {
        "question_id": "hallucinated",
        "refusal_required": False,
        "correctly_refused": None,
        "emitted_claim_count": 1,
        "citation_count": 1,
        "used_fallback": False,
        "judge_error": None,
        "judge_score": {"faithfulness": 1, "correctness": 1, "relevance": 2},
        "failure_categories": ["generation_hallucination"],
    }

    summary = run_generation_eval._summarize_rows([row])

    assert summary["generation_hallucination_count"] == 1
    assert summary["gate_passed"] is False


def test_legacy_checkpoint_is_rejected_without_modification(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    original = json.dumps({"question_id": "q1"}) + "\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(run_generation_eval.IncompatibleCheckpointError, match="older evaluation schema"):
        run_generation_eval._load_checkpoint(path, "fingerprint", {"q1"})

    assert path.read_text(encoding="utf-8") == original


def test_checkpoint_with_different_evaluation_fingerprint_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    row = {
        "checkpoint_schema_version": run_generation_eval.CHECKPOINT_SCHEMA_VERSION,
        "evaluation_fingerprint": "old-fingerprint",
        "question_id": "q1",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(run_generation_eval.IncompatibleCheckpointError, match="fingerprint differs"):
        run_generation_eval._load_checkpoint(path, "new-fingerprint", {"q1"})


def test_rate_limit_failure_leaves_checkpoint_byte_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.jsonl"
    original = b'{"existing": true}\n'
    path.write_bytes(original)

    def fail_before_row(*args, **kwargs):
        raise ProviderRateLimitExceeded(36000.0, 1)

    monkeypatch.setattr(run_generation_eval, "evaluate_one", fail_before_row)

    with pytest.raises(ProviderRateLimitExceeded):
        run_generation_eval._evaluate_and_checkpoint(
            question=SimpleNamespace(question_id="q1"),
            service=None,
            chunk_texts_by_id={},
            source_metadata_by_id={},
            judge=None,
            evaluation_fingerprint="fingerprint",
            checkpoint_path=path,
            completed={},
        )

    assert path.read_bytes() == original


def test_atomic_checkpoint_writer_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    rows = [{"question_id": "q1"}, {"question_id": "q2"}]

    run_generation_eval._write_checkpoint_atomic(path, rows)

    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == rows
    assert not path.with_name(f".{path.name}.tmp").exists()


def _row(
    question_id: str,
    language: str,
    category: str,
    *,
    refusal_required: bool = False,
    correctly_refused: bool | None = None,
    emitted_claim_count: int = 1,
    citation_count: int = 1,
    used_fallback: bool = False,
    judge_error: str | None = None,
    judge_score: dict | None = None,
    failure_categories: list[str] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "language": language,
        "category": category,
        "refusal_required": refusal_required,
        "correctly_refused": correctly_refused,
        "emitted_claim_count": emitted_claim_count,
        "citation_count": citation_count,
        "used_fallback": used_fallback,
        "judge_error": judge_error,
        "judge_score": judge_score if judge_score is not None else {"faithfulness": 2, "correctness": 2, "relevance": 2},
        "failure_categories": failure_categories if failure_categories is not None else [],
    }


def test_per_language_and_per_category_breakdowns_isolate_their_own_rows():
    rows = [
        _row("q_en_1", "en", "factual"),
        _row("q_en_2", "en", "dynamic", refusal_required=True, correctly_refused=True, emitted_claim_count=0, citation_count=0, judge_score=None),
        _row("q_tr_1", "tr", "factual", correctly_refused=None, failure_categories=["refusal_failure"], emitted_claim_count=0, citation_count=0, judge_score={"faithfulness": 2, "correctness": 1, "relevance": 0}),
        _row("q_ar_1", "ar", "etiquette", used_fallback=True),
    ]

    summary = run_generation_eval._summarize_rows(rows)

    assert set(summary["by_language"]) == {"ar", "en", "tr"}
    assert set(summary["by_category"]) == {"dynamic", "etiquette", "factual"}

    en_metrics = summary["by_language"]["en"]
    assert en_metrics["question_count"] == 2
    assert en_metrics["false_refusal_count"] == 0

    tr_metrics = summary["by_language"]["tr"]
    assert tr_metrics["question_count"] == 1
    assert tr_metrics["false_refusal_count"] == 1
    assert tr_metrics["answerable_response_rate"] == 0.0

    ar_metrics = summary["by_language"]["ar"]
    assert ar_metrics["fallback_count"] == 1
    assert ar_metrics["fallback_rate"] == 1.0

    # The global gate must still reflect all rows, not a cherry-picked subset.
    assert summary["false_refusal_count"] == 1
    assert summary["gate_passed"] is False


def test_per_language_breakdown_does_not_change_global_gate_or_top_level_metrics():
    """The breakdowns are purely additive diagnostics: the pre-existing
    top-level keys/values must be identical to what a global-only
    computation would produce."""
    rows = [_row("q1", "en", "factual"), _row("q2", "tr", "cross_lingual")]

    summary = run_generation_eval._summarize_rows(rows)
    global_only = run_generation_eval._compute_metrics(rows)

    for key, value in global_only.items():
        assert summary[key] == value
    assert summary["gate_passed"] is True


def test_generation_eval_process_exit_code_reflects_quality_gate():
    assert run_generation_eval._gate_exit_code({"gate_passed": True}) == 0
    assert run_generation_eval._gate_exit_code({"gate_passed": False}) == 1
    assert run_generation_eval._gate_exit_code({}) == 1
