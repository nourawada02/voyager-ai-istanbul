"""Tests that the experiment's winner-selection outcome is reproducible:
running select_dense_winner/hybrid_beats_dense twice on the same input
metrics always yields the same decision (pure functions, no randomness)."""

from __future__ import annotations

from rag.experiment import AggregateMetrics, hybrid_beats_dense, select_dense_winner


def _metrics(**overrides) -> dict[str, AggregateMetrics]:
    base = {
        "A": AggregateMetrics("A", 0.6, 0.7, 0.6, 15, 0.0, 15, {}),
        "B": AggregateMetrics("B", 0.65, 0.75, 0.62, 18, 0.0, 15, {}),
        "C": AggregateMetrics("C", 0.55, 0.7, 0.58, 25, 0.0, 15, {}),
        "D": AggregateMetrics("D", 0.6, 0.72, 0.6, 30, 0.0, 15, {}),
    }
    base.update(overrides)
    return base


def test_winner_selection_is_reproducible_across_repeated_calls():
    metrics = _metrics()
    w1 = select_dense_winner(metrics)
    w2 = select_dense_winner(metrics)
    assert w1 == w2


def test_winner_selection_is_order_independent():
    metrics = _metrics()
    w_forward = select_dense_winner(metrics)
    reordered = dict(reversed(list(metrics.items())))
    w_reversed = select_dense_winner(reordered)
    assert w_forward == w_reversed


def test_hybrid_decision_is_reproducible():
    dense = AggregateMetrics("dense", 0.6, 0.7, 0.6, 15, 0.0, 15, {})
    hybrid = AggregateMetrics("hybrid", 0.62, 0.72, 0.61, 20, 0.0, 15, {})
    r1 = hybrid_beats_dense(dense, hybrid)
    r2 = hybrid_beats_dense(dense, hybrid)
    assert r1 == r2 is True
