"""Retrieval experiment runner (Checkpoint Phase 3 §6, architecture.md
§9.5): configs A-D dense retrieval, predeclared winner-selection rule,
then winning-dense-vs-hybrid-RRF comparison.

Predeclared decision rule (fixed BEFORE looking at results, per the
"never adjust ground truth/rules to improve metrics" instruction):
  1. Rank configs by mean Recall@K (primary -- did we find the right
     document at all).
  2. Break ties by mean Precision@K (secondary).
  3. Break remaining ties by lower mean latency_ms (tertiary).
  4. Break any remaining tie by config name ascending (A < B < C < D),
     a fully deterministic total order.

Hybrid is only reported as winning over the dense winner if its mean
Recall@K is strictly greater, OR (equal Recall@K AND strictly greater
Precision@K) -- otherwise the dense winner stands, and the report says so
explicitly rather than defaulting to "hybrid wins" by assumption.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from rag.ground_truth import QUESTIONS, GroundTruthQuestion


@dataclass(frozen=True)
class PerQueryResult:
    question_id: str
    language: str
    chunk_config: str
    mode: str
    retrieved_source_ids: list[str]
    relevant_source_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    latency_ms: float
    zero_result: bool


def _precision_recall_mrr(retrieved_source_ids: list[str], relevant_source_ids: set[str]) -> tuple[float, float, float]:
    if not retrieved_source_ids:
        return 0.0, (0.0 if relevant_source_ids else 1.0), 0.0
    hits = [1 if sid in relevant_source_ids else 0 for sid in retrieved_source_ids]
    precision = sum(hits) / len(retrieved_source_ids)
    if not relevant_source_ids:
        # A deliberately-unanswerable question: recall is vacuously 1.0
        # iff nothing irrelevant was returned as if it were relevant --
        # here we define recall as 1.0 only when zero items were
        # retrieved (a true, correct empty/refusal result); otherwise 0.0,
        # since anything retrieved for a source-less question is a
        # (spurious) miss against the "nothing relevant exists" truth.
        recall = 1.0 if len(retrieved_source_ids) == 0 else 0.0
    else:
        found = len(relevant_source_ids & set(retrieved_source_ids))
        recall = found / len(relevant_source_ids)
    rr = 0.0
    for i, sid in enumerate(retrieved_source_ids, start=1):
        if sid in relevant_source_ids:
            rr = 1.0 / i
            break
    return precision, recall, rr


def evaluate_query(question: GroundTruthQuestion, retrieval_result: dict[str, Any]) -> PerQueryResult:
    retrieved = [item["source_id"] for item in retrieval_result["items"]]
    relevant = set(question.expected_source_ids)
    precision, recall, rr = _precision_recall_mrr(retrieved, relevant)
    return PerQueryResult(
        question_id=question.question_id,
        language=question.language,
        chunk_config=retrieval_result["chunk_config"],
        mode=retrieval_result["retrieval_mode"],
        retrieved_source_ids=retrieved,
        relevant_source_ids=sorted(relevant),
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=rr,
        latency_ms=retrieval_result["latency_ms"],
        zero_result=retrieval_result["zero_result"],
    )


@dataclass(frozen=True)
class AggregateMetrics:
    label: str
    mean_precision_at_k: float
    mean_recall_at_k: float
    mrr: float
    mean_latency_ms: float
    zero_result_rate: float
    n: int
    per_language: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mrr": self.mrr,
            "mean_latency_ms": self.mean_latency_ms,
            "zero_result_rate": self.zero_result_rate,
            "n": self.n,
            "per_language": self.per_language,
        }


def aggregate(results: list[PerQueryResult], label: str) -> AggregateMetrics:
    n = len(results)
    per_lang: dict[str, dict[str, float]] = {}
    for lang in ("en", "tr", "ar"):
        lang_results = [r for r in results if r.language == lang]
        if not lang_results:
            continue
        per_lang[lang] = {
            "mean_precision_at_k": statistics.fmean(r.precision_at_k for r in lang_results),
            "mean_recall_at_k": statistics.fmean(r.recall_at_k for r in lang_results),
            "mrr": statistics.fmean(r.reciprocal_rank for r in lang_results),
            "zero_result_rate": sum(1 for r in lang_results if r.zero_result) / len(lang_results),
            "n": len(lang_results),
        }
    return AggregateMetrics(
        label=label,
        mean_precision_at_k=statistics.fmean(r.precision_at_k for r in results) if n else 0.0,
        mean_recall_at_k=statistics.fmean(r.recall_at_k for r in results) if n else 0.0,
        mrr=statistics.fmean(r.reciprocal_rank for r in results) if n else 0.0,
        mean_latency_ms=statistics.fmean(r.latency_ms for r in results) if n else 0.0,
        zero_result_rate=(sum(1 for r in results if r.zero_result) / n) if n else 0.0,
        n=n,
        per_language=per_lang,
    )


def select_dense_winner(config_metrics: dict[str, AggregateMetrics]) -> str:
    """The predeclared decision rule (see module docstring), applied
    mechanically -- never adjusted after seeing results."""

    def sort_key(name: str) -> tuple[float, float, float, str]:
        m = config_metrics[name]
        return (-m.mean_recall_at_k, -m.mean_precision_at_k, m.mean_latency_ms, name)

    return min(config_metrics.keys(), key=sort_key)


def hybrid_beats_dense(dense: AggregateMetrics, hybrid: AggregateMetrics) -> bool:
    if hybrid.mean_recall_at_k > dense.mean_recall_at_k:
        return True
    if hybrid.mean_recall_at_k == dense.mean_recall_at_k and hybrid.mean_precision_at_k > dense.mean_precision_at_k:
        return True
    return False


def categorize_failure(question: GroundTruthQuestion, result: PerQueryResult) -> str | None:
    """Best-effort real failure categorization for the winner-selection
    report -- returns None for a fully correct result."""
    if question.refusal_required:
        return None  # retrieval-level correctness for refusal cases is about zero_result, handled separately
    if result.recall_at_k >= 1.0:
        return None
    if result.zero_result:
        return "corpus_gap_or_retrieval_miss"
    if question.category == "cross_lingual" and result.recall_at_k < 1.0:
        return "cross_lingual_retrieval_miss"
    if question.category == "alias" and result.recall_at_k < 1.0:
        return "alias_resolution_failure"
    return "retrieval_ranking_failure"
