"""Tests for retrieval-metric correctness and the predeclared
winner-selection rule -- pure arithmetic, no model/Qdrant needed."""

from __future__ import annotations

from rag.experiment import (
    AggregateMetrics,
    PerQueryResult,
    _precision_recall_mrr,
    aggregate,
    hybrid_beats_dense,
    select_dense_winner,
)


def test_precision_recall_exact_values():
    precision, recall, rr = _precision_recall_mrr(["a", "b", "c"], {"b"})
    assert precision == 1 / 3
    assert recall == 1.0
    assert rr == 1 / 2  # b is at rank 2


def test_recall_with_multiple_relevant_documents():
    precision, recall, rr = _precision_recall_mrr(["a", "b"], {"a", "b", "c"})
    assert recall == 2 / 3


def test_zero_relevant_and_zero_retrieved_is_a_correct_refusal_case():
    precision, recall, rr = _precision_recall_mrr([], set())
    assert recall == 1.0


def test_zero_relevant_but_something_retrieved_is_a_miss():
    precision, recall, rr = _precision_recall_mrr(["spurious"], set())
    assert recall == 0.0


def test_mrr_is_zero_when_relevant_document_never_retrieved():
    _, _, rr = _precision_recall_mrr(["x", "y"], {"z"})
    assert rr == 0.0


def _mk(precision, recall, rr, latency, zero_result=False) -> PerQueryResult:
    return PerQueryResult("q", "en", "A", "dense", [], [], precision, recall, rr, latency, zero_result)


def test_aggregate_computes_correct_means():
    results = [_mk(1.0, 1.0, 1.0, 10.0), _mk(0.0, 0.0, 0.0, 20.0)]
    agg = aggregate(results, "test")
    assert agg.mean_precision_at_k == 0.5
    assert agg.mean_recall_at_k == 0.5
    assert agg.mrr == 0.5
    assert agg.mean_latency_ms == 15.0
    assert agg.zero_result_rate == 0.0


def test_select_dense_winner_prefers_higher_recall():
    metrics = {
        "A": AggregateMetrics("A", 0.5, 0.5, 0.5, 10, 0.0, 1, {}),
        "B": AggregateMetrics("B", 0.5, 0.9, 0.5, 10, 0.0, 1, {}),
    }
    assert select_dense_winner(metrics) == "B"


def test_select_dense_winner_ties_break_on_precision_then_latency_then_name():
    metrics = {
        "A": AggregateMetrics("A", 0.5, 0.8, 0.5, 20, 0.0, 1, {}),
        "B": AggregateMetrics("B", 0.9, 0.8, 0.5, 10, 0.0, 1, {}),
    }
    assert select_dense_winner(metrics) == "B"  # same recall, higher precision wins

    metrics2 = {
        "C": AggregateMetrics("C", 0.5, 0.8, 0.5, 20, 0.0, 1, {}),
        "D": AggregateMetrics("D", 0.5, 0.8, 0.5, 10, 0.0, 1, {}),
    }
    assert select_dense_winner(metrics2) == "D"  # same recall+precision, lower latency wins


def test_hybrid_beats_dense_requires_strict_improvement():
    dense = AggregateMetrics("dense", 0.5, 0.7, 0.5, 10, 0.0, 1, {})
    hybrid_equal = AggregateMetrics("hybrid", 0.5, 0.7, 0.5, 10, 0.0, 1, {})
    assert hybrid_beats_dense(dense, hybrid_equal) is False

    hybrid_better_recall = AggregateMetrics("hybrid", 0.4, 0.8, 0.5, 10, 0.0, 1, {})
    assert hybrid_beats_dense(dense, hybrid_better_recall) is True

    hybrid_same_recall_better_precision = AggregateMetrics("hybrid", 0.9, 0.7, 0.5, 10, 0.0, 1, {})
    assert hybrid_beats_dense(dense, hybrid_same_recall_better_precision) is True

    hybrid_worse = AggregateMetrics("hybrid", 0.3, 0.6, 0.4, 10, 0.0, 1, {})
    assert hybrid_beats_dense(dense, hybrid_worse) is False
