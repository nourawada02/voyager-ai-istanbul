"""Runs the full Checkpoint Phase 3 §6 retrieval experiment: configs A-D
dense, predeclared winner selection, then winner-vs-hybrid-RRF. Writes
per-query raw results, aggregate metrics, fingerprints, and the
winner-selection report to evaluation/datasets/.

Reproducibility command (recorded verbatim in the output report too):
    python -m rag.run_experiment
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rag import embeddings, ingest, qdrant_store
from rag.experiment import (
    aggregate,
    categorize_failure,
    evaluate_query,
    hybrid_beats_dense,
    select_dense_winner,
)
from rag.ground_truth import QUESTIONS

OUT_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "datasets"


def run() -> dict[str, Any]:
    t_start = time.perf_counter()
    client = qdrant_store.local_client(path=None)  # in-memory embedded Qdrant, real qdrant-client engine
    tokenizer = embeddings.load_tokenizer()
    embed_fp = embeddings.fingerprint().as_dict()

    documents = ingest.load_documents()

    per_config_results: dict[str, list] = {}
    per_config_agg = {}
    ingestion_results = {}
    raw_rows: list[dict[str, Any]] = []

    for config_name, cfg in ingest.CHUNK_CONFIGS.items():
        result = ingest.ingest_config(
            client, config_name, tokenizer, embeddings.embed_passages, embed_fp, documents
        )
        ingestion_results[config_name] = result
        chunk_texts_by_id = dict(zip(result.chunk_ids, result.chunk_texts))

        from rag.retrieval_service import RetrievalService

        service = RetrievalService(
            client=client,
            collection_name=result.collection_name,
            bm25_index=result.bm25_index,
            embed_query_fn=embeddings.embed_query,
            collection_fingerprint=result.fingerprint,
        )

        query_results = []
        for q in QUESTIONS:
            top_k = cfg["top_k"]
            retrieval = service.search(q.question, q.language, config_name, top_k, mode="dense")
            pr = evaluate_query(q, retrieval)
            query_results.append(pr)
            raw_rows.append(
                {
                    "question_id": q.question_id,
                    "language": q.language,
                    "chunk_config": config_name,
                    "mode": "dense",
                    "retrieved_source_ids": pr.retrieved_source_ids,
                    "relevant_source_ids": pr.relevant_source_ids,
                    "precision_at_k": pr.precision_at_k,
                    "recall_at_k": pr.recall_at_k,
                    "reciprocal_rank": pr.reciprocal_rank,
                    "latency_ms": pr.latency_ms,
                    "zero_result": pr.zero_result,
                    "refusal_required": q.refusal_required,
                    "category": q.category,
                    "failure_category": categorize_failure(q, pr),
                }
            )
        per_config_results[config_name] = query_results
        per_config_agg[config_name] = aggregate(query_results, config_name)

    winner = select_dense_winner(per_config_agg)
    winner_cfg = ingest.CHUNK_CONFIGS[winner]
    winner_result = ingestion_results[winner]

    # --- Hybrid comparison on the winning config -------------------------
    from rag.retrieval_service import RetrievalService

    winner_service = RetrievalService(
        client=client,
        collection_name=winner_result.collection_name,
        bm25_index=winner_result.bm25_index,
        embed_query_fn=embeddings.embed_query,
        collection_fingerprint=winner_result.fingerprint,
    )
    hybrid_results = []
    for q in QUESTIONS:
        retrieval = winner_service.search(
            q.question, q.language, winner, winner_cfg["top_k"], mode="hybrid_rrf"
        )
        pr = evaluate_query(q, retrieval)
        hybrid_results.append(pr)
        raw_rows.append(
            {
                "question_id": q.question_id,
                "language": q.language,
                "chunk_config": winner,
                "mode": "hybrid_rrf",
                "retrieved_source_ids": pr.retrieved_source_ids,
                "relevant_source_ids": pr.relevant_source_ids,
                "precision_at_k": pr.precision_at_k,
                "recall_at_k": pr.recall_at_k,
                "reciprocal_rank": pr.reciprocal_rank,
                "latency_ms": pr.latency_ms,
                "zero_result": pr.zero_result,
                "refusal_required": q.refusal_required,
                "category": q.category,
                "failure_category": categorize_failure(q, pr),
            }
        )
    hybrid_agg = aggregate(hybrid_results, f"{winner}+hybrid_rrf")
    dense_winner_agg = per_config_agg[winner]
    hybrid_wins = hybrid_beats_dense(dense_winner_agg, hybrid_agg)

    report = {
        "schema_version": "1.0.0",
        "reproducibility_command": "python -m rag.run_experiment",
        "corpus_fingerprint": winner_result.corpus_fp,
        "embedding_fingerprint": embed_fp,
        "config_fingerprints": {name: r.fingerprint for name, r in ingestion_results.items()},
        "question_count": len(QUESTIONS),
        "decision_rule": (
            "1) max mean Recall@K; 2) tie-break max mean Precision@K; "
            "3) tie-break min mean latency_ms; 4) tie-break config name ascending"
        ),
        "config_metrics": {name: agg.to_dict() for name, agg in per_config_agg.items()},
        "dense_winner": winner,
        "hybrid_metrics": hybrid_agg.to_dict(),
        "hybrid_wins_over_dense_winner": hybrid_wins,
        "final_selection": f"{winner}+hybrid_rrf" if hybrid_wins else f"{winner}+dense",
        "total_wall_clock_seconds": round(time.perf_counter() - t_start, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "rag_experiment_raw_results.json").write_text(
        json.dumps(raw_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "rag_experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2))
