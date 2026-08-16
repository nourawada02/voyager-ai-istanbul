"""Runs generation evaluation over all 45 ground-truth questions
(Checkpoint Phase 3 §8, remediated §5), using the frozen winning
retriever (dense config B, per evaluation/datasets/rag_experiment_report.json).

Writes evaluation/datasets/rag_generation_eval_report_remediated.json,
preserving the original (pre-remediation) report at
rag_generation_eval_report.json as baseline evidence -- never
overwritten, so the improvement is auditable.

Each question's result is checkpointed to a JSONL scratch file as soon as
it is computed (path from GEN_EVAL_CHECKPOINT_PATH, default is a local
scratch file next to this script). The complete small file is atomically
replaced after each row, and every row carries an evaluation fingerprint;
an interrupted compatible run resumes without redoing completed questions,
while rows produced by different code/configuration are rejected instead
of being mixed into one report. The checkpoint file is not evaluation
output; only the final aggregated report is.

Reproducibility command:
    python -m rag.run_generation_eval
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag import embeddings, ingest, qdrant_store
from rag.answer_service import GroundedAnswerService, SYSTEM_INSTRUCTION
from rag.generation_eval import JUDGE_SYSTEM_PROMPT, GenerationEvalRow, evaluate_one
from rag.ground_truth import QUESTIONS
from rag.llm_providers import (
    GROQ_GENERATOR_MODEL,
    GROQ_JUDGE_MODEL,
    GroqProvider,
    LLMProvider,
    ProviderDetectionResult,
    RequiredProviderUnavailable,
    construct_provider,
    detect_available_providers,
    enforce_required_provider,
    provider_credential_present,
)
from rag.retrieval_service import RetrievalService

OUT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "datasets"
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent / "_gen_eval_checkpoint.jsonl"
CHECKPOINT_SCHEMA_VERSION = "2.0.0"


class IncompatibleCheckpointError(RuntimeError):
    """Raised before provider calls when checkpoint rows came from a
    different evaluation implementation/configuration."""


def _row_to_dict(r: GenerationEvalRow, evaluation_fingerprint: str) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evaluation_fingerprint": evaluation_fingerprint,
        "question_id": r.question_id,
        "language": r.language,
        "category": r.category,
        "refusal_required": r.refusal_required,
        "answer_text": r.answer_text,
        "correctly_refused": r.correctly_refused,
        "retrieved_chunk_ids": list(r.retrieved_chunk_ids),
        "citations": r.citations,
        "citation_count": r.citation_count,
        "emitted_claim_count": r.emitted_claim_count,
        "dropped_claim_count": r.dropped_claim_count,
        "used_fallback": r.used_fallback,
        "degradation_reason": r.degradation_reason,
        "judge_score": (
            {
                "faithfulness": r.judge_score.faithfulness,
                "correctness": r.judge_score.correctness,
                "relevance": r.judge_score.relevance,
                "reason": r.judge_score.reason,
            }
            if r.judge_score
            else None
        ),
        "judge_error": r.judge_error,
        "failure_categories": r.failure_categories,
    }


def _evaluation_fingerprint(
    *,
    generator_name: str | None,
    generator_model: str | None,
    judge_name: str | None,
    judge_model: str | None,
    winner_config: str,
    winner_mode: str,
    collection_fingerprint: str,
) -> str:
    """Fingerprint every input that makes a checkpoint row comparable.

    Code bytes are included deliberately: resuming rows after changing
    prompts, validation, judging, or provider behavior would mix two
    experiments and invalidate the final aggregate.
    """
    code_dir = Path(__file__).resolve().parent
    code_hashes = {}
    for filename in (
        "answer_service.py",
        "generation_eval.py",
        "ground_truth.py",
        "llm_providers.py",
        "retrieval_service.py",
    ):
        code_hashes[filename] = hashlib.sha256((code_dir / filename).read_bytes()).hexdigest()
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "generator_name": generator_name,
        "generator_model": generator_model,
        "judge_name": judge_name,
        "judge_model": judge_model,
        "winner_config": winner_config,
        "winner_mode": winner_mode,
        "collection_fingerprint": collection_fingerprint,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_INSTRUCTION.encode("utf-8")).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Role-specific provider selection (Checkpoint Phase 3: direct Qwen
# generator integration). Each role is resolved completely
# independently -- presence of one provider's credential must never
# influence which provider serves the *other* role. Without an explicit
# override, each role falls back to the original single-detected-
# provider behavior for full backward compatibility.
_ROLE_PROVIDER_ENV_VARS = {"generator": "GEN_EVAL_GENERATOR_PROVIDER", "judge": "GEN_EVAL_JUDGE_PROVIDER"}
_ROLE_REQUIRE_ENV_VARS = {
    "generator": "GEN_EVAL_REQUIRE_GENERATOR_PROVIDER",
    "judge": "GEN_EVAL_REQUIRE_JUDGE_PROVIDER",
}


def _model_for_role(provider_name: str, role: str) -> str | None:
    """Role-specific default model, where a provider distinguishes one.
    Groq splits generator/judge models to avoid exhausting one model's
    daily token allowance; Qwen (generator-only in this checkpoint) and
    Ollama have a single default model regardless of role, so `None`
    lets `construct_provider` use each provider's own default."""
    if provider_name == "groq":
        return GROQ_GENERATOR_MODEL if role == "generator" else GROQ_JUDGE_MODEL
    return None


def _legacy_role_fallback(role: str, detection: ProviderDetectionResult) -> LLMProvider | None:
    """The provider a role falls back to when it has no explicit
    GEN_EVAL_GENERATOR_PROVIDER/GEN_EVAL_JUDGE_PROVIDER override --
    exactly the original (pre-Qwen) single-detected-provider behavior."""
    if detection.hosted_provider == "groq":
        return GroqProvider(model=_model_for_role("groq", role))
    return detection.selected


def _resolved_role_provider_name(role: str, detection: ProviderDetectionResult) -> str | None:
    """The provider name that will actually serve this role: an explicit
    override if set, else whatever the legacy single-provider detection
    resolved to. Used both to construct the role's provider and to
    evaluate `GEN_EVAL_REQUIRE_GENERATOR_PROVIDER`/
    `GEN_EVAL_REQUIRE_JUDGE_PROVIDER`."""
    explicit = os.environ.get(_ROLE_PROVIDER_ENV_VARS[role])
    return explicit or detection.provider_name


def _resolve_explicit_role_provider(role: str) -> LLMProvider:
    """Construct the provider explicitly requested for this role via
    GEN_EVAL_GENERATOR_PROVIDER/GEN_EVAL_JUDGE_PROVIDER. Raises
    RequiredProviderUnavailable -- before any network call -- if that
    provider's credential/reachability requirement is not met, rather
    than silently falling back to a different provider for an
    acceptance run."""
    env_var = _ROLE_PROVIDER_ENV_VARS[role]
    provider_name = os.environ.get(env_var)
    if not provider_credential_present(provider_name):
        raise RequiredProviderUnavailable(
            f"{env_var}={provider_name!r} but its credential/reachability requirement is not satisfied"
        )
    return construct_provider(provider_name, model=_model_for_role(provider_name, role))


def enforce_required_role_provider(role: str, detection: ProviderDetectionResult) -> None:
    """Fail fast, before any provider call, if
    GEN_EVAL_REQUIRE_GENERATOR_PROVIDER/GEN_EVAL_REQUIRE_JUDGE_PROVIDER
    is set and does not match the provider that would actually be
    resolved for that role. A role can only satisfy this by an explicit
    GEN_EVAL_GENERATOR_PROVIDER/GEN_EVAL_JUDGE_PROVIDER override or by
    the legacy detection genuinely resolving to it -- e.g. Qwen is never
    reachable through legacy detection (see `detect_available_providers`),
    so `GEN_EVAL_REQUIRE_GENERATOR_PROVIDER=qwen` can never be satisfied
    by accident, only by deliberately setting `GEN_EVAL_GENERATOR_PROVIDER=qwen`.
    """
    required = os.environ.get(_ROLE_REQUIRE_ENV_VARS[role])
    if not required:
        return
    actual = _resolved_role_provider_name(role, detection)
    if actual != required:
        raise RequiredProviderUnavailable(
            f"{_ROLE_REQUIRE_ENV_VARS[role]}={required!r} but the resolved {role} provider is {actual!r}; "
            "refusing to silently substitute a different provider for an acceptance run"
        )


def _select_evaluation_providers(
    detection: ProviderDetectionResult,
) -> tuple[LLMProvider | None, LLMProvider | None]:
    """Select independent generation and judge roles.

    If either GEN_EVAL_GENERATOR_PROVIDER or GEN_EVAL_JUDGE_PROVIDER is
    set, each role is resolved completely independently: an explicit
    override always wins for its own role (e.g. GEN_EVAL_GENERATOR_PROVIDER=qwen
    selects Qwen for generation regardless of whether GROQ_API_KEY is
    also present), and a role without an override falls back to legacy
    single-provider detection. This is what a Qwen-generator/Groq-judge
    acceptance run uses.

    With neither override set, behavior is byte-for-byte the original
    (pre-Qwen) logic: Groq supplies both roles with independent models
    when detected, otherwise one locally detected/Ollama provider
    supplies both roles from the same instance.
    """
    generator_override = os.environ.get(_ROLE_PROVIDER_ENV_VARS["generator"])
    judge_override = os.environ.get(_ROLE_PROVIDER_ENV_VARS["judge"])

    if generator_override or judge_override:
        generator = (
            _resolve_explicit_role_provider("generator")
            if generator_override
            else _legacy_role_fallback("generator", detection)
        )
        judge = (
            _resolve_explicit_role_provider("judge") if judge_override else _legacy_role_fallback("judge", detection)
        )
        return generator, judge

    if detection.hosted_provider == "groq":
        return GroqProvider(model=GROQ_GENERATOR_MODEL), GroqProvider(model=GROQ_JUDGE_MODEL)
    selected = detection.selected
    return selected, selected


def _load_checkpoint(
    path: Path,
    expected_fingerprint: str,
    valid_question_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IncompatibleCheckpointError(
                f"checkpoint line {line_number} is not valid JSON; preserve the file and start a new checkpoint"
            ) from exc
        if row.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise IncompatibleCheckpointError(
                "checkpoint was produced by an older evaluation schema; preserve it as diagnostic evidence "
                "and start a new checkpoint"
            )
        if row.get("evaluation_fingerprint") != expected_fingerprint:
            raise IncompatibleCheckpointError(
                "checkpoint evaluation fingerprint differs from the current code/configuration; "
                "mixing its rows would invalidate the final report"
            )
        question_id = row.get("question_id")
        if question_id not in valid_question_ids:
            raise IncompatibleCheckpointError(
                f"checkpoint line {line_number} has unknown question_id {question_id!r}"
            )
        if question_id in completed:
            raise IncompatibleCheckpointError(f"checkpoint contains duplicate question_id {question_id!r}")
        completed[question_id] = row
    return completed


def _write_checkpoint_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the complete small checkpoint atomically after one row.

    A crash before ``os.replace`` leaves the previous valid checkpoint
    untouched; a partially appended JSONL line can never become the next
    run's input.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as checkpoint_file:
            for row in rows:
                checkpoint_file.write(json.dumps(row, sort_keys=True) + "\n")
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _evaluate_and_checkpoint(
    *,
    question,
    service: GroundedAnswerService,
    chunk_texts_by_id: dict[str, str],
    source_metadata_by_id: dict[str, Any],
    judge,
    evaluation_fingerprint: str,
    checkpoint_path: Path,
    completed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate first and checkpoint second.

    Any provider/transport exception propagates before the atomic write,
    leaving all previously completed rows byte-valid and resumable.
    """
    row = evaluate_one(question, service, chunk_texts_by_id, source_metadata_by_id, judge)
    row_dict = _row_to_dict(row, evaluation_fingerprint)
    _write_checkpoint_atomic(checkpoint_path, [*completed.values(), row_dict])
    completed[question.question_id] = row_dict
    return row_dict


def _compute_metrics(row_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the independent citation, answerability, refusal, fallback,
    and judge metrics for a set of rows -- the full 45-row result set, or
    any language/category subset of it (see `_summarize_rows`).

    Citation coverage is correctly scoped to claims that were emitted;
    empty answers are separately and visibly penalized by the false-
    refusal/answerable-response metric so a system cannot pass by
    deleting every claim. This function never computes `gate_passed` --
    the gate is a global acceptance decision, deliberately never
    evaluated per-language/per-category (see `_summarize_rows`), so a
    system cannot pass by cherry-picking a favorable subset.
    """
    total = len(row_dicts)
    answerable_rows = [row for row in row_dicts if not row["refusal_required"]]
    refusal_rows = [row for row in row_dicts if row["refusal_required"]]

    rows_with_emitted_claims = [row for row in answerable_rows if row["emitted_claim_count"] > 0]
    total_emitted_claims = sum(row["emitted_claim_count"] for row in rows_with_emitted_claims)
    # AnswerResult's structural invariant is all-or-none per emitted row:
    # every accepted claim contributes at least one valid source-backed
    # citation. Count claims, not rows or grouped SourceReference objects.
    cited_emitted_claims = sum(
        row["emitted_claim_count"] for row in rows_with_emitted_claims if row["citation_count"] > 0
    )
    citation_coverage = (
        cited_emitted_claims / total_emitted_claims if total_emitted_claims else 1.0
    )
    total_citations = sum(row["citation_count"] for row in rows_with_emitted_claims)

    refusal_accuracy = (
        sum(1 for row in refusal_rows if row["correctly_refused"]) / len(refusal_rows)
        if refusal_rows
        else 1.0
    )
    false_refusal_count = sum(
        1
        for row in answerable_rows
        if row["correctly_refused"] is None and "refusal_failure" in row["failure_categories"]
    )
    answerable_response_count = len(answerable_rows) - false_refusal_count
    answerable_response_rate = (
        answerable_response_count / len(answerable_rows) if answerable_rows else 1.0
    )
    false_refusal_rate = false_refusal_count / len(answerable_rows) if answerable_rows else 0.0
    fallback_rows = [row for row in row_dicts if row["used_fallback"]]
    fallback_rate = len(fallback_rows) / total if total else 0.0

    judge_failed_rows = [row for row in answerable_rows if row["judge_error"] is not None]
    judged_rows = [row for row in answerable_rows if row["judge_score"] is not None]
    judge_summary = None
    if judged_rows:
        judge_summary = {
            "mean_faithfulness": sum(row["judge_score"]["faithfulness"] for row in judged_rows)
            / len(judged_rows),
            "mean_correctness": sum(row["judge_score"]["correctness"] for row in judged_rows)
            / len(judged_rows),
            "mean_relevance": sum(row["judge_score"]["relevance"] for row in judged_rows)
            / len(judged_rows),
            "n_judged": len(judged_rows),
        }

    all_failure_categories: dict[str, int] = {}
    for row in row_dicts:
        for failure_category in row["failure_categories"]:
            all_failure_categories[failure_category] = all_failure_categories.get(failure_category, 0) + 1
    generation_hallucination_count = all_failure_categories.get("generation_hallucination", 0)
    # A provider transport failure (connection reset, timeout, TLS
    # handshake failure exhausting its bounded retry budget -- see
    # llm_providers.ProviderTransportError) on either the generator or
    # judge role must be a hard, visible gate failure, never a silently
    # accepted degraded answer or missing judge score (Checkpoint Phase
    # 3 remediation: transport resilience).
    generator_transport_failure_rows = [
        row for row in row_dicts if "generator_transport_failed" in row["failure_categories"]
    ]
    judge_transport_failure_rows = [
        row for row in row_dicts if "judge_transport_failed" in row["failure_categories"]
    ]

    return {
        "question_count": total,
        "answerable_question_count": len(answerable_rows),
        "answerable_response_count": answerable_response_count,
        "answerable_response_rate": answerable_response_rate,
        "refusal_required_question_count": len(refusal_rows),
        "grounded_claim_citation_coverage": citation_coverage,
        "citation_coverage_over_emitted_claims": citation_coverage,
        "cited_emitted_claim_count": cited_emitted_claims,
        "total_emitted_claims": total_emitted_claims,
        "total_citations": total_citations,
        "dynamic_refusal_accuracy": refusal_accuracy,
        "false_refusal_count": false_refusal_count,
        "false_refusal_rate": false_refusal_rate,
        "fallback_count": len(fallback_rows),
        "fallback_rate": fallback_rate,
        "fallback_question_ids": [row["question_id"] for row in fallback_rows],
        "judge_failed_question_ids": [row["question_id"] for row in judge_failed_rows],
        "judge_summary": judge_summary,
        "failure_categories": all_failure_categories,
        "generation_hallucination_count": generation_hallucination_count,
        "generator_transport_failure_count": len(generator_transport_failure_rows),
        "generator_transport_failure_question_ids": [row["question_id"] for row in generator_transport_failure_rows],
        "judge_transport_failure_count": len(judge_transport_failure_rows),
        "judge_transport_failure_question_ids": [row["question_id"] for row in judge_transport_failure_rows],
    }


def _summarize_rows(row_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Global metrics (unchanged acceptance semantics) plus diagnostic
    per-language and per-category breakdowns (Checkpoint Phase 3
    remediation §6).

    The breakdowns reuse the exact same `_compute_metrics` formulas as
    the global result, scoped to each language/category's own rows; they
    are purely diagnostic and never participate in `gate_passed`, which
    is computed only once, over the full row set.
    """
    metrics = _compute_metrics(row_dicts)

    judged_count = metrics["judge_summary"]["n_judged"] if metrics["judge_summary"] else 0
    gate_passed = (
        len(metrics["judge_failed_question_ids"]) == 0
        and judged_count == metrics["answerable_question_count"]
        and metrics["grounded_claim_citation_coverage"] == 1.0
        and metrics["dynamic_refusal_accuracy"] == 1.0
        and metrics["false_refusal_count"] == 0
        and metrics["generation_hallucination_count"] == 0
        # A provider transport failure (connection reset, timeout, TLS
        # handshake failure) on either role must fail the gate even if
        # answer_service's safe degradation happened to still produce a
        # cited, non-refused answer -- never a superficial green result
        # from a provider outage. judge_transport_failure_count is
        # already implied by judge_failed_question_ids above; checked
        # explicitly here too rather than assumed.
        and metrics["generator_transport_failure_count"] == 0
        and metrics["judge_transport_failure_count"] == 0
    )

    languages = sorted({row.get("language", "unknown") for row in row_dicts})
    categories = sorted({row.get("category", "unknown") for row in row_dicts})
    by_language = {
        language: _compute_metrics([row for row in row_dicts if row.get("language", "unknown") == language])
        for language in languages
    }
    by_category = {
        category: _compute_metrics([row for row in row_dicts if row.get("category", "unknown") == category])
        for category in categories
    }

    return {
        **metrics,
        "gate_passed": gate_passed,
        "by_language": by_language,
        "by_category": by_category,
    }


@dataclass(frozen=True)
class RetrievalContext:
    """Everything a generation-eval run (the full 45-case run or the
    smaller targeted smoke run) needs to build a GroundedAnswerService,
    built once against the frozen winning retriever config."""

    retrieval_service: RetrievalService
    chunk_texts_by_id: dict[str, str]
    source_metadata_by_id: dict[str, Any]
    top_k: int
    winner_mode: str
    winner_config: str
    collection_fingerprint: str


def build_retrieval_context() -> RetrievalContext:
    """Ingest the frozen winning retriever config and assemble the
    chunk-text/source-metadata lookups shared by every generation-eval
    entry point. Never modifies the frozen retrieval winner itself --
    only reads `rag_experiment_report.json` to learn which config it is.
    """
    experiment_report_path = OUT_DIR / "rag_experiment_report.json"
    if not experiment_report_path.exists():
        raise RuntimeError("run `python -m rag.run_experiment` first -- no frozen winner report found")
    experiment_report = json.loads(experiment_report_path.read_text(encoding="utf-8"))
    final_selection = experiment_report["final_selection"]
    winner_config = experiment_report["dense_winner"]
    winner_mode = "hybrid_rrf" if final_selection.endswith("hybrid_rrf") else "dense"

    client = qdrant_store.local_client(path=None)
    tokenizer = embeddings.load_tokenizer()
    embed_fp = embeddings.fingerprint().as_dict()
    documents = ingest.load_documents()
    ingestion = ingest.ingest_config(client, winner_config, tokenizer, embeddings.embed_passages, embed_fp, documents)

    chunk_texts_by_id = dict(zip(ingestion.chunk_ids, ingestion.chunk_texts))
    manifest_docs = _manifest_docs()
    source_metadata_by_id = {
        d.source_id: {
            "title": d.title,
            "url": next((p["url"] for p in manifest_docs if p["source_id"] == d.source_id), None),
            "retrieved_at": next((p["retrieved_at"] for p in manifest_docs if p["source_id"] == d.source_id), None),
            "language": d.language,
        }
        for d in documents
    }

    retrieval_service = RetrievalService(
        client=client,
        collection_name=ingestion.collection_name,
        bm25_index=ingestion.bm25_index,
        embed_query_fn=embeddings.embed_query,
        collection_fingerprint=ingestion.fingerprint,
    )
    top_k = ingest.CHUNK_CONFIGS[winner_config]["top_k"]

    return RetrievalContext(
        retrieval_service=retrieval_service,
        chunk_texts_by_id=chunk_texts_by_id,
        source_metadata_by_id=source_metadata_by_id,
        top_k=top_k,
        winner_mode=winner_mode,
        winner_config=winner_config,
        collection_fingerprint=ingestion.fingerprint,
    )


def run() -> dict[str, Any]:
    # Fail before any ingestion/retrieval setup, checkpoint I/O, or
    # provider call if this is an acceptance run pinned to a specific
    # provider -- globally (GEN_EVAL_REQUIRE_PROVIDER, backward
    # compatible) or per role (GEN_EVAL_REQUIRE_GENERATOR_PROVIDER /
    # GEN_EVAL_REQUIRE_JUDGE_PROVIDER) -- and that provider was not
    # actually available. Never silently substitute a different provider
    # for a run meant as acceptance evidence. Provider resolution itself
    # (_select_evaluation_providers) can also raise here, before any
    # ingestion/retrieval setup, if an explicitly requested role
    # provider's credential is missing.
    detection = detect_available_providers()
    enforce_required_provider(detection)
    enforce_required_role_provider("generator", detection)
    enforce_required_role_provider("judge", detection)
    generator, judge = _select_evaluation_providers(detection)

    context = build_retrieval_context()
    retrieval_service = context.retrieval_service
    chunk_texts_by_id = context.chunk_texts_by_id
    source_metadata_by_id = context.source_metadata_by_id
    top_k = context.top_k
    winner_mode = context.winner_mode
    winner_config = context.winner_config

    evaluation_fingerprint = _evaluation_fingerprint(
        generator_name=generator.name if generator is not None else None,
        generator_model=generator.model if generator is not None else None,
        judge_name=judge.name if judge is not None else None,
        judge_model=judge.model if judge is not None else None,
        winner_config=winner_config,
        winner_mode=winner_mode,
        collection_fingerprint=context.collection_fingerprint,
    )

    checkpoint_path = Path(os.environ.get("GEN_EVAL_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH)))
    completed = _load_checkpoint(
        checkpoint_path,
        expected_fingerprint=evaluation_fingerprint,
        valid_question_ids={question.question_id for question in QUESTIONS},
    )
    if completed:
        print(f"resuming from checkpoint: {len(completed)}/{len(QUESTIONS)} already done", flush=True)

    row_dicts: list[dict[str, Any]] = []
    for i, q in enumerate(QUESTIONS, start=1):
        if q.question_id in completed:
            row_dict = completed[q.question_id]
        else:
            svc = GroundedAnswerService(
                retrieval=retrieval_service, provider=generator, chunk_config=winner_config,
                top_k=top_k, retrieval_mode=winner_mode,
            )
            row_dict = _evaluate_and_checkpoint(
                question=q,
                service=svc,
                chunk_texts_by_id=chunk_texts_by_id,
                source_metadata_by_id=source_metadata_by_id,
                judge=judge,
                evaluation_fingerprint=evaluation_fingerprint,
                checkpoint_path=checkpoint_path,
                completed=completed,
            )
        row_dicts.append(row_dict)
        print(
            f"[{i}/{len(QUESTIONS)}] {row_dict['question_id']} refusal_required={row_dict['refusal_required']} "
            f"refused={row_dict['correctly_refused'] if row_dict['refusal_required'] else (row_dict['emitted_claim_count'] == 0)} "
            f"emitted_claims={row_dict['emitted_claim_count']} dropped={row_dict['dropped_claim_count']} "
            f"used_fallback={row_dict['used_fallback']} judge_error={row_dict['judge_error']!r}",
            flush=True,
        )

    summary = _summarize_rows(row_dicts)

    report = {
        "schema_version": "1.0.0",
        "reproducibility_command": "python -m rag.run_generation_eval",
        "provider_detection": detection.description,
        "generator_available": generator is not None,
        "generator_provider": generator.name if generator is not None else None,
        "generator_model": generator.model if generator is not None else None,
        "judge_available": judge is not None,
        "judge_provider": judge.name if judge is not None else None,
        "judge_model": judge.model if judge is not None else None,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evaluation_fingerprint": evaluation_fingerprint,
        **summary,
        "rows": row_dicts,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rag_generation_eval_report_remediated.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


_MANIFEST_CACHE: list[dict[str, Any]] | None = None


def _manifest_docs() -> list[dict[str, Any]]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is None:
        path = Path(__file__).resolve().parent / "manifests" / "corpus_manifest.json"
        _MANIFEST_CACHE = json.loads(path.read_text(encoding="utf-8"))["documents"]
    return _MANIFEST_CACHE


def _gate_exit_code(report: dict[str, Any]) -> int:
    """A completed evaluation that misses its quality gate is a process failure."""
    return 0 if report.get("gate_passed") is True else 1


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}, indent=2))
    raise SystemExit(_gate_exit_code(r))
