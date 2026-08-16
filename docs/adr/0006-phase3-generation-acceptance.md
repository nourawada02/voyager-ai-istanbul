# ADR 0006: Phase 3 — generation acceptance and known evaluator limitation

Status: accepted. Date: 2026-08-16.

Short status note, not a rewrite of `docs/architecture.md`. Records the
decision to close Phase 3 (multilingual RAG: corpus, ingestion,
retrieval, grounded generation, evaluation) despite the strict
generation-acceptance report reading `gate_passed: false`, and why that
is the right call.

## Decision

Phase 3 retrieval and RAG implementation are **accepted and frozen**.
Phase 4 (System B / A2A vertical slice) is the next gate, ahead of the
August 21 deadline.

## Frozen retrieval configuration and measured results

Per `evaluation/datasets/rag_experiment_report.json` (unchanged since
2026-08-15, reproducibility command `python -m rag.run_experiment`):

- Winning dense config: **B** (350 chunk tokens, 50 overlap, top-K 5).
- Final selection: **B+dense** — hybrid dense+sparse RRF did not beat it
  (`hybrid_wins_over_dense_winner: false`).
- Measured over all 45 ground-truth questions, all three languages
  (EN/TR/AR), zero-result rate 0.0 in every language:

  | Metric | Value |
  |---|---|
  | Mean Recall@K | 0.7111 |
  | Mean Precision@K | 0.3822 |
  | MRR | 0.6759 |
  | Mean latency | 31.6 ms |

- Embedding: `intfloat/multilingual-e5-small`, revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`, dim 384.

This configuration, the corpus, chunking, and embeddings are unmodified
by this acceptance decision and must remain frozen going forward.

## 45-case generation results

Final run — `evaluation/datasets/rag_generation_eval_report_remediated.json`,
fingerprint `d99959b8851b2f3be92b4229d202e12dc2dc07d3e26c72e764c17d89fe37fac4`,
reproducibility command `python -m rag.run_generation_eval`:

- `generator_provider`: `qwen`, `generator_model`: `qwen3.7-flash`
- `judge_provider`: `groq`, `judge_model`: `openai/gpt-oss-20b`
- `answerable_response_rate`: 1.0 (33/33 answerable questions answered)
- `dynamic_refusal_accuracy`: 1.0 (12/12 dynamic/unanswerable questions correctly refused)
- `false_refusal_count`: 0
- `grounded_claim_citation_coverage` / `citation_coverage_over_emitted_claims`: 1.0
- `generator_transport_failure_count`: 0, `judge_transport_failure_count`: 0
- `judge_failed_question_ids`: `[]`
- `fallback_count`: 2 (`gt_03_ar`, `gt_13_ar`)
- `generation_hallucination_count`: 1
- `gate_passed`: **false**

The strict generation acceptance report is **red**, and this ADR does
not claim otherwise. It is red because of exactly one diagnosed
evaluation false negative, not because of a product-correctness defect.

## The failed row: `gt_03_en`

Question: "When did Ayasofya (Hagia Sophia) become a mosque again?"

Generated answer: "Hagia Sophia became a mosque again in 2020. Ayasofya
was converted back to mosque status in 2020 by a presidential decree."

Cited sources: `wiki_en_hagia_sophia` (`chunk_8894707de46fdce1a2776fde`)
and `wiki_tr_hagia_sophia` (`chunk_1de1013dfe5b2758afc3b993`).

The Turkish source chunk (`rag/documents/wiki_tr_hagia_sophia.json`)
verified to literally contain:

> "2020 yılında Ayasofya, bir cumhurbaşkanlığı kararnamesiyle yeniden
> cami statüsüne dönüştürüldü."

— i.e. "in 2020, Ayasofya was converted back to mosque status by a
presidential decree." The generated claim was genuinely supported by a
chunk it correctly cited.

The Groq judge (`openai/gpt-oss-20b`) scored `faithfulness: 1,
correctness: 1` with the stated reason: "Answer correctly states the
2020 reclassification but incorrectly claims a presidential decree,
which is not in the context." This is factually wrong about what the
cited context contained, and is what drove `gate_passed` to `false`.

## Root cause diagnosis

**Failure class: evaluation design failure**, not a generation defect.
The probable mechanism is that judge-context assembly (`rag/generation_eval.py`
`_judge_context_chunk_ids`) did not reliably surface the correct
lower-ranked but cited Turkish chunk to the judge in the same shape/order
the generator itself had it in, letting the judge miss evidence that was
genuinely present and cited.

- **Product impact**: none demonstrated. The generated claim was cited,
  the citation is real, and the cited source supports the claim exactly
  as written.
- **Evaluation impact**: one false-negative hallucination classification
  out of 33 judged answers (`judge_summary.mean_faithfulness: 1.97/2`,
  `mean_correctness: 1.94/2` — both driven down by this single row).

## Deferred repair

**Citation-first judge-context assembly** is the identified fix (make
the judge context reflect exactly and only the claim's own cited
chunks, in a form that can't be lost or reordered relative to what the
generator saw) — **explicitly deferred**, not implemented in this
checkpoint.

**Reason for deferral**: mandatory Phase 4–6 architecture work is now
the critical path ahead of the August 21 deadline; this is a narrowly
scoped evaluation-harness fix with no demonstrated product-correctness
impact, and does not block Phase 4.

This limitation will be documented in `EVALUATION.md` and in the final
presentation. No claim is made anywhere that the strict generation gate
passed — it is recorded honestly as red, with the single cause above.

## What this does not decide

- Any change to `rag/generation_eval.py`'s judge-context logic — deferred
  to a future checkpoint.
- Phase 4 System B / A2A design — out of scope here.
