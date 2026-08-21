# VoyagerAI Istanbul — Final Evaluation

This is the final evaluation of the committed system, covering retrieval, generation, agent routing, tool selection, schema/transport reliability, the accommodation fair-price model, the itinerary optimizer, and end-to-end resilience. It reports the system as it exists in the current committed repository, using only real, committed evaluation artifacts — no metric here is invented or estimated.

## 1. Executive summary

| Area | Result | Threshold | Status |
|---|---|---|---|
| Supervisor routing accuracy (final) | 27/30 = **90.0%** | ≥ 90% | **PASS** |
| Specialist tool-selection correctness | 18/20 = **90.0%** | ≥ 90% | **PASS** |
| Final decision schema validity | 30/30 = **100%** | 100% | **PASS** |
| Unnecessary delegation/tool-call rate | 2/20 = **10.0%** | ≤ 10% | **PASS** |
| Dense retrieval (config B) selection | Recall@5 = 0.711 | winner by measured comparison | selected |
| RAG generation faithfulness/citation gate | 32/33 answerable cases pass; **1 open failure** | 100% | **BLOCKED** (disclosed, unresolved — see §14) |
| RAG-first candidate origin (System B) | 24/27 = **88.9%** | ≥ 80% | **PASS** |
| Accommodation fair-price model promotion | MAE **−40.7%** vs. strong baseline | ≥ 10% | **PASS** |
| Itinerary optimizer (side-aware vs. naive) | Side crossings **−84.6%**, 0 hard violations | — | **PASS** |
| End-to-end / resilience scenarios | 21/21 | — | **PASS** |
| Credential / chain-of-thought leakage | 0 hits, every run | 0 | **PASS** |

Routing reached its required threshold only after two targeted, evidenced prompt corrections, applied one at a time and each validated against an independently frozen holdout case set never seen during the fix (§13). The RAG generation gate has one disclosed, unresolved failure (§14) — it is reported as failed, not reinterpreted as a pass.

**Post-evaluation code note:** one additional stability fix (`services/planner-a` commit `39bffc4`, "stabilize live planning and deadline finalization" — corrects a deadline-vs-complete-evidence race so a workflow whose required evidence had already been gathered is no longer incorrectly reported `degraded` merely because the 60s deadline was crossed while selecting the trivial, local `synthesize` action) was committed to `phase4/graph.py` **after** the routing measurements in §7 were taken. It was not itself re-validated by a live-Qwen routing checkpoint. It changes deadline-boundary finalization logic only — the decision-prompt content, the eligibility gate, and the action allowlist §7 actually measures are unchanged by it. Disclosed here rather than silently assumed unaffected.

## 2. Scope and baseline

| Repository | Evaluated code baseline SHA |
|---|---|
| Superproject (`phase-4/istanbul-expert`) | `094929f0a398997ac986f83dbb1b110ce09c494d` |
| `services/planner-a` | `ef4a59ff805307a972253c05818965c337be91d7` |
| `services/istanbul-expert-b` | `0b458b8d3900486f4ee5e373bd93ee7a4209a305` |
| `services/travel-mcp` | `2111002f0fa8c6366c38ef329326da3ad672f5c2` |
| `services/frontend` | `cde3d07c70ff9577f4c82bcb742c2ee1c53936b8` |

The routing/RAG/ML/itinerary/resilience measurements in this document were taken against the planner-a/istanbul-expert-b/travel-mcp commits that introduced the measured behavior (`phase4/graph.py`, `phase4/specialist.py`, `rag/`, `phase2/` ML pipeline, System B's itinerary engine) — every one of those files is unchanged between the measured commit and the baseline SHA above, except for the single post-measurement stability fix disclosed in §1. Hybrid Chat (post-trip conversation) and the recent-session sidebar are outside this document's metric framework by design — they are decision-support UI/persistence features, not a retrieval, generation, or routing capability, and are validated instead by their own automated test suites (`orchestration/tests/test_chat_service.py`, `services/frontend/phase6/tests/test_app.py`) and the manual English/Arabic smoke sequence in [`README.md`](README.md#manual-smoke-tests-english-and-arabic).

The documentation-only commit that follows this evaluation report (`README.md`, `EVALUATION.md`, `docs/DEMO_SCRIPT.md`) changes none of the evaluated production code, tests, runtime configuration, or evaluation artifacts referenced above or anywhere in this document — every SHA, metric, and file path in this report remains accurate against that commit.

**Evidence classes used below:** *live* (a real network call — Qwen, Open-Meteo, Travel MCP, System B A2A, or Qdrant — made specifically for that measurement); *fixture* (deterministic, network-free executors/providers); *scripted* (a predeclared ground-truth decision sequence run through the real compiled graph, hermetic); *historical* (an existing, previously-generated, committed artifact, read not regenerated).

## 3. Test sets and ground truth

| Evaluation | n | Composition | Ground truth |
|---|---|---|---|
| RAG retrieval/generation | 45 questions | 15 EN / 15 TR / 15 AR, 33 answerable + 12 refusal-required | `evaluation/datasets/rag_ground_truth.json` — each question carries `expected_source_ids` (real Wikipedia document IDs, e.g. `wiki_en_hagia_sophia`), `expected_facts`, `expected_section`, `poi_id`, `refusal_required` |
| RAG-first candidate origin (System B) | 8 demonstration cases | interest-facet coverage across history/culture, food, nightlife/shopping, nature/islands, family, accessibility, plus one Arabic-language and one Turkish-language request | `evaluation/datasets/istanbul_expert_r1_summary_report.json` — per-case scheduled-POI provenance |
| Supervisor/specialist routing (final) | 30 cases | 10 EN / 10 TR / 10 AR; 5 capability scopes × 6 each; 18 initial + 12 follow-up turns | `evaluation/e1z_cases.json` (SHA-256 `5d09fea13629f133735ff0b00539e522ab4f53b20a2d02cfa223b278cdb6ef98`), preflight-audited before any live call (`evaluation/e1z_preflight_audit.json`) |
| Specialist tool-selection | 20 cases | weather / flight / stay / fair-price-prerequisite / completion, EN/TR/AR | `evaluation/e1_live_qwen_cases.json` |
| Accommodation fair-price ML | 23,943 rows (post-filter) | real Inside Airbnb Istanbul snapshot, `snapshot_date=2026-06-30`, host-disjoint 70/10/10/10 split | `ml/reports/phase2_final_2026-06-30.json` |
| Itinerary optimizer | 22 cases | all feasible (≥ declared 20-case minimum) | `evaluation/datasets/istanbul_expert_baseline_comparison_report.json` |
| End-to-end / resilience | 21 scenarios | 13 core + 8 explicit failure injections, through the real public API | `evaluation/e1_end_to_end_results.json` |

## 4. Retrieval: Precision@K, Recall@K, MRR

Source: `evaluation/datasets/rag_experiment_report.json` (historical, Phase 3, read not regenerated). Corpus: real, stable Wikipedia content (EN/TR/AR), CC BY-SA 4.0. Embedding: `intfloat/multilingual-e5-small` (384-dim, dense). K = 5 (config B) unless noted.

**Chunk/top-K sweep, dense retrieval, all 45 questions:**

| Config | chunk tokens | overlap | top_k | Precision@K | Recall@K | MRR | Mean latency |
|---|---|---|---|---|---|---|---|
| A | 350 | 50 | 3 | 0.467 | 0.667 | 0.670 | 34.9ms |
| **B (selected)** | 350 | 50 | 5 | 0.382 | **0.711** | 0.676 | 31.6ms |
| C | 700 | 100 | 3 | 0.467 | 0.667 | 0.670 | 31.5ms |
| D | 700 | 100 | 5 | 0.382 | 0.711 | 0.676 | 32.0ms |

Selection rule: max Recall@K → tie-break max Precision@K → tie-break min latency → tie-break name-ascending. B and D tie on every measured metric; B wins on name-ascending tie-break.

**Per-language, config B:** EN P=0.427 R=0.733 MRR=0.733 · TR P=0.360 R=0.700 MRR=0.606 · AR P=0.360 R=0.700 MRR=0.689. Zero-result rate: 0.0% in every language.

NDCG@5 was not measured in this original Phase 3 sweep. It **was** measured separately, per-facet, in the RAG-first evaluation (§6) — the only place in this project NDCG is a real, committed number.

## 5. Generation: faithfulness, correctness, relevance

Source: `evaluation/datasets/rag_generation_eval_report_remediated.json` (historical, Phase 3). 45 questions, 33 answerable / 12 refusal-required. Generator `qwen3.7-flash`; judge `openai/gpt-oss-20b` via Groq.

| Metric | Result |
|---|---|
| Answerable response rate | 33/33 = 100% |
| Dynamic refusal accuracy (12 refusal-required questions) | 100% |
| Grounded-claim citation coverage | 100% |
| Citation coverage over emitted claims | 100% |
| Judge mean correctness / faithfulness / relevance (0–2 scale, n=33) | 1.94 / 1.97 / 1.94 |
| **Gate passed** | **False** — see §14 |

Per-language judge means: EN 1.91/1.91/1.91 · TR 2.00/2.00/2.00 · AR 1.91/2.00/1.91 (n=11 each).

## 6. RAG-first candidate selection vs. catalog fallback (System B)

Source: `evaluation/datasets/istanbul_expert_r1_summary_report.json`. For every supported interest, Qdrant retrieval is the first-priority candidate source; the catalog is used only when RAG evidence is insufficient.

Measured against the R.1 development shadow collection's 69-document corpus content (39 EN / 14 TR / 16 AR). The official-promotion checkpoint (`istanbul_rag_B_v2`, see [Qdrant multilingual RAG](README.md#qdrant-multilingual-rag)) reproducibly serves that exact same corpus content under a new collection name and a corpus-aware fingerprint — a deployment-reproducibility and naming change, not a retrieval-algorithm change. The numbers below were **not** re-measured against `istanbul_rag_B_v2` and are not claimed to be.

| Metric | Result | Target | Status |
|---|---|---|---|
| RAG-origin scheduled points of interest | 24/27 = **88.9%** | ≥ 80% | **PASS** |
| Catalog-fallback scheduled points of interest | 3/27 = 11.1% | — | disclosed |
| Mean Recall@5 across 17 interest facets | 0.263 | — | measured |
| Mean Precision@5 across 17 interest facets | 0.579 | — | measured |
| Mean NDCG@5 across 17 interest facets | 0.518 | — | measured |
| Mean Hit Rate@5 across 17 interest facets | 0.941 | — | measured |
| Turkish native-query mean Recall@5 / NDCG@5 | 0.440 / 0.492 | — | measured |
| Arabic native-query mean Recall@5 / NDCG@5 | 0.535 / 0.531 | — | measured |

**One honest failed case**, disclosed not hidden: the `accessibility` demo case had zero valid RAG candidate evidence for that interest and is reported as `uncovered_interest` — accessibility is treated as cross-cutting evidence (a note attached to the itinerary), never a fabricated schedulable POI category, and the itinerary's `accessibility_evaluation` field reports `partially_satisfied` honestly rather than claiming full coverage.

**Qdrant-unavailable fallback**, exercised directly: all 4 candidates in a Qdrant-down simulation fell back to the catalog cleanly, with explicit warnings (`candidate_pool_no_rag_candidates_catalog_fallback_used`, `rag_evidence_insufficient_for_interests:history,food`, `rag_unavailable_structured_catalog_only`) and zero fabricated RAG evidence (`unnecessary_fallback: false`).

## 7. Supervisor routing accuracy and specialist tool-selection correctness (final)

The live-Qwen supervisor routing measurement required two rounds of targeted, evidenced prompt remediation before it cleared its required 90% threshold. Each round is fully detailed as a failure case in §13. This section reports only the **final, passing** measurement and the **original baseline** for direct before/after comparison — the two intermediate rounds are in §13, not repeated here.

| Metric | Original baseline | **Final (after remediation)** | Threshold | Status |
|---|---|---|---|---|
| Supervisor routing / next-action accuracy | 23/30 = 76.7% | **27/30 = 90.0%** | ≥ 90% | **PASS** |
| Capability-scope classification accuracy | not separately measured at baseline | 27/30 = 90.0% | ≥ 90% | **PASS** |
| Specialist tool-selection correctness | 18/20 = 90.0% (unchanged by either fix — specialist prompt untouched) | 18/20 = 90.0% | ≥ 90% | **PASS** |
| Unnecessary delegation/tool-call rate | 2/6 = 33.3% | **2/20 = 10.0%** | ≤ 10% | **PASS** |

Final measurement source: `evaluation/e1z_live_qwen_results.jsonl` (30 rows) / `evaluation/e1z_live_qwen_summary.json`. Model `qwen3.7-flash`, run once, 30/30 completed, zero transport errors. Original baseline source: `evaluation/e1_live_qwen_results.jsonl` (50 rows, 30 supervisor + 20 specialist) / `evaluation/e1_live_qwen_summary.json`.

**By language (final measurement):**

| Language | n | Scope accuracy | Action accuracy |
|---|---|---|---|
| EN | 10 | 100% | 100% |
| TR | 10 | 100% | 90.0% |
| AR | 10 | 70.0% | 80.0% |

**Two of the four PASS thresholds above landed exactly at the boundary** (90.0% action accuracy on 27/30, 10.0% unnecessary-delegation on 2/20) rather than comfortably inside it — reported precisely, not rounded favorably. A one-case shift in either direction would flip the result; §14 discloses this margin as a real, unresolved risk, not a closed one.

## 8. Schema validity, transport, retry, and correction metrics

Source: `evaluation/e1z_live_qwen_summary.json` (final measurement) and `evaluation/e1_end_to_end_resilience` suite (hermetic, real API).

| Metric | Result |
|---|---|
| First-attempt decision schema validity (final measurement) | 30/30 = 100% |
| Final decision schema validity after bounded repair | 30/30 = **100%** |
| Correction (repair) rate | 0/30 = 0.0% |
| Transient-transport retry rate | 0/30 = 0.0% |
| Provider/transport completion rate | 30/30 = 100% |
| Mean latency | 5.73s/case |
| Shared 8-call external-tool budget: exhaustion honestly reported (not silently truncated) | verified (`test_e1_end_to_end_resilience.py`, injection scenario) |
| Credential / chain-of-thought leakage across every live artifact cited in this document | 0 hits |

## 9. Configuration comparisons

At least two independently measured configuration comparisons, each with a real, measured winner — never asserted in advance:

| Comparison | Winner | Margin |
|---|---|---|
| RAG chunk/top-K sweep, dense retrieval, configs A–D | **B** (350 tokens / 50 overlap / top-5) | Recall@K +4.4pp over A/C (0.711 vs 0.667); tied with D, won on tie-break |
| Dense config B vs. dense+sparse RRF hybrid retrieval | **Dense B** | Recall@K +13.3pp over hybrid (0.711 vs 0.578) |
| Accommodation price model: weak/strong baseline vs. HistGradientBoosting | **HistGradientBoosting** | MAE −40.7% vs. strong baseline (test set) |
| Itinerary: naive popularity-based vs. side-aware/ferry-aware scheduling | **Side-aware** | Side crossings −84.6% (0.591 → 0.091 mean); walking minutes +27.0% (disclosed tradeoff) |

## 10. Accommodation fair-price ML and itinerary optimizer

Source: `ml/reports/phase2_final_2026-06-30.json`, `ml/reports/phase2_selection_2026-06-30.json` (historical, Phase 2).

| Model | MAE | RMSE | Median AE | R² | n |
|---|---|---|---|---|---|
| Weak baseline (validation) | 3179.71 | 12288.00 | 1952.09 | −0.018 | 2366 |
| Strong baseline (**test**) | 2773.60 | 6092.35 | 1574.73 | 0.031 | 2331 |
| **Selected: HistGradientBoosting (test)** | **1643.68** | **4639.41** | **632.40** | **0.438** | 2331 |

Improvement over strong baseline on the held-out test set: **40.7%**, against a predeclared 10% promotion threshold → **promoted**. Host-disjoint split (zero hosts shared across train/validation/calibration/test). Disclosed subgroup weakness: Hotel-room MAE=18750 (n=23, small-n, negative R²) — not hidden.

Itinerary optimizer (`evaluation/datasets/istanbul_expert_baseline_comparison_report.json`, 22 cases, all feasible):

| Metric | Naive baseline | Side-aware/ferry-aware |
|---|---|---|
| Mean side crossings | 0.591 | **0.091** |
| Hard violations | 0 | 0 |
| Mean daily slack (min) | 280.94 | 273.03 |
| Mean walking minutes | 25.32 | 32.16 (disclosed +27.0% cost) |

## 11. End-to-end and resilience

`orchestration/tests/test_e1_end_to_end_resilience.py` — 21 scenarios (13 core + 8 explicit failure injections) through the real public System A API, real SSE stream, real SQLite `RunStore`. Raw rows: `evaluation/e1_end_to_end_results.json`. **All 21 pass.** Failure injections cover: Qwen decision-format-invalid with exhausted repair, weather-provider failure, flight-provider rate-limiting, Travel MCP timeout, malformed MCP result, System B/A2A timeout, invalid A2A artifact, and shared external-tool-budget exhaustion across two delegations — every one degrades honestly (`degraded`/`cancelled`, never a fabricated success) with the specific cause reported in `warnings`.

A real five-container Docker Compose smoke (fixture mode, zero paid calls) confirmed all five services reach `healthy`, a real trip streams the honest SSE sequence to `run_completed`, and the result survives `docker compose restart agent-system-a` byte-for-byte.

## 12. Thresholds and pass/block outcomes — summary

| Gate | Threshold | Result | Status |
|---|---|---|---|
| Supervisor routing accuracy | ≥ 90% | 90.0% | **PASS** |
| Specialist tool-selection correctness | ≥ 90% | 90.0% | **PASS** |
| Final decision schema validity | 100% | 100% | **PASS** |
| Unnecessary delegation rate | ≤ 10% | 10.0% | **PASS** |
| Credential/chain-of-thought leakage | 0 | 0 | **PASS** |
| RAG-first candidate origin | ≥ 80% | 88.9% | **PASS** |
| ML MAE improvement over strong baseline | ≥ 10% | 40.7% | **PASS** |
| Itinerary hard-constraint violations | 0 | 0 | **PASS** |
| RAG generation faithfulness/citation gate | 100% pass | 32/33 (1 disclosed failure) | **BLOCKED** |

## 13. Three failure cases

Full machine-readable detail: Case 1 — `evaluation/e1_failure_cases.json`; Case 2 — `evaluation/datasets/rag_generation_eval_report_remediated.json`; Case 3 — `evaluation/e1r_baseline_failure_audit.json`, `evaluation/e1y_e1x_failure_adjudication.json`, and the routing result files cited inline.

### Case 1 — A2A subprocess stdout-pipe deadlock (Design failure)

**Symptom:** the opt-in real-integration test (`orchestration/tests/test_cross_process_integration.py`, real Open-Meteo + Travel MCP + System B A2A) hung indefinitely with no error, indistinguishable from a genuine A2A protocol hang.

**Root cause:** the test harness started child processes with an undrained `stdout=subprocess.PIPE`; once System B's own ADK logging exceeded the OS pipe buffer, the child process blocked writing to a pipe nobody was reading, and the parent blocked waiting on the child — a real OS-level deadlock, not an A2A defect.

**Fix:** redirect subprocess output to a real log file instead of an unread pipe, matching System B's own already-proven process-management pattern. Test harness change only — no production code touched.

**Measured post-fix result:** the real cross-process gate completed successfully — `runtime_seconds=20.92`, observed actions `['get_weather','search_stays','call_istanbul_expert']`, all `status=success`, `final_status=success`.

### Case 2 — RAG judge-context construction false negative (Design failure)

**Symptom:** the RAG generation faithfulness/citation gate reported `gate_passed: false`. One case (`gt_03_en`, of 33 answerable questions) was judged `generation_hallucination` even though the generated claim (a 2020 reclassification fact) was correct on the merits.

**Root cause:** the judge's own context window was constructed without including the specific retrieved chunk that named the fact the generated answer stated — an evaluation-harness design issue in how judge context is assembled, not a defect in the generator's output or its citations.

**Fix:** not implemented in this evaluation cycle. The identified fix (citation-first judge-context construction) is explicitly deferred and unstarted.

**Measured post-fix result:** none — this fix has not been applied. The gate remains reported as failed, exactly as measured, with no reinterpretation as a pass. See §14 for the current disclosure of this open item; it is not re-analyzed a second time there.

### Case 3 — Supervisor scope/eligibility prompt gap (Prompt failure)

**Symptom:** the original live-Qwen routing baseline measured supervisor next-action accuracy at 23/30 = 76.7%, below the required 90% threshold. An independent follow-up validation against a freshly frozen 50-case set isolated the largest single contributor: capability-scope classification accuracy of only 20/30 = **66.7%** — the classifier frequently misjudged whether a request needed travel evidence, Istanbul-local grounding, both, clarification, or was out of scope, especially on short, context-light follow-up phrasings with no explicit destination/evidence-type keyword. After a first repair resolved scope classification, a second validation round then isolated a *separate* remaining gap: supervisor next-action accuracy still measured only 24/30 = 80.0%, because 5 of 6 remaining misses had the *correct* scope but chose the *wrong* action within it (e.g. `ask_clarification` when a single eligible specialist action should have been chosen).

**Root cause:** two distinct, evidenced prompt gaps in `phase4/graph.py::_build_decision_prompt` — (1) the capability-scope classification prompt carried no continuity context across turns, so each classification was made from scratch with no signal of what the previous turn had already established; (2) the decision prompt had no explicit eligibility policy stating which actions are legitimately co-eligible under a given scope (e.g., when essential trip details are genuinely absent) versus which single action is required, so the model chose between structurally-valid but ambiguous options with no stated tiebreak.

**Fix:** applied as two targeted, evidenced, additive prompt corrections, one at a time, each validated against an independently frozen holdout never seen during that fix — (1) a scope-continuity repair added explicit prior-turn scope/evidence context to the classification prompt, letting the model recognize an elliptical continuation ("continue", "show me") of an already-classified task rather than reclassifying from nothing; (2) an eligibility-policy repair added the co-eligible action set per scope and the conditions that select between them to the decision prompt, plus a one-shot correction step when the model's first choice falls outside its own eligible set.

**Measured post-fix result:** after repair (1), an independent 30-case validation measured capability-scope accuracy at 27/30 = 90.0% (+23.3pp) and unnecessary delegation fell to 0.0%, but next-action accuracy still stood at only 80.0% (`evaluation/e1v_live_qwen_summary.json` → `evaluation/e1x_live_qwen_summary.json`). After repair (2), a further independent, freshly frozen 30-case validation (disjoint from every prior case set, preflight-audited before any live call) measured supervisor next-action accuracy at **27/30 = 90.0%** and capability-scope accuracy holding at **27/30 = 90.0%**, with final schema validity 100% and unnecessary delegation at 10.0% — all four required thresholds met for the first time (`evaluation/e1x_live_qwen_summary.json` → `evaluation/e1z_live_qwen_summary.json`, §7).

## 14. Honest unresolved limitations

- **The RAG generation faithfulness/citation gate remains failed (`gate_passed: false`), not resolved, not reinterpreted as a pass** — see Case 2 (§13) for the full root-cause analysis; not repeated here.
- **The post-E.1Z deadline-finalization stability fix (`39bffc4`, §1) has not been independently re-validated by a live-Qwen routing checkpoint.** It changes only deadline-boundary finalization (never the decision prompt, eligibility gate, or action allowlist §7 measures), so the routing numbers in §7 are believed still representative, but this is a disclosed assumption, not a re-measured fact.
- **Two of the four final routing thresholds (§7) landed exactly at their boundary** (90.0% action accuracy on a 30-case set; 10.0% unnecessary-delegation on a 20-case denominator) — a single additional miss in either metric would flip the result. This is reported precisely, not smoothed over.
- **Specialist tool-selection correctness (90.0%) was last independently measured at the original baseline** and was not re-measured after Case 3's supervisor-only prompt fixes, because the specialist prompt itself was never touched by either repair.
- Deal-ranking evaluation (`architecture.md` §14.4, ≥25 pairwise/listwise cases) was added and passes fully: 25/25 (`evaluation/e1r_deal_ranking_results.json`) — no model retraining, deterministic scoring function only.
- Hybrid Chat (post-trip conversation) and the recent-session sidebar are validated by automated tests and a manual English/Arabic smoke sequence (`README.md`), not by this document's retrieval/generation/routing metric framework — this is a scope boundary, not a gap in either evaluation.

## 15. Raw artifact index

| Area | Artifact |
|---|---|
| RAG retrieval/generation ground truth | `evaluation/datasets/rag_ground_truth.json` |
| RAG retrieval sweep (Phase 3) | `evaluation/datasets/rag_experiment_report.json` |
| RAG generation (Phase 3, remediated) — Case 2 | `evaluation/datasets/rag_generation_eval_report_remediated.json` |
| RAG-first candidate origin (System B) | `evaluation/datasets/istanbul_expert_r1_summary_report.json`, `istanbul_expert_r1_raw_results.json` |
| Routing baseline (Case 3, original weakness) | `evaluation/e1_live_qwen_cases.json`, `e1_live_qwen_results.jsonl`, `e1_live_qwen_summary.json` |
| Routing Case 3 (continuity-context repair) | `evaluation/e1v_cases.json`, `e1v_live_qwen_results.jsonl`, `e1v_live_qwen_summary.json`, `e1x_cases.json`, `e1x_live_qwen_results.jsonl`, `e1x_live_qwen_summary.json` |
| Routing Case 3 (eligibility-policy repair, final) | `evaluation/e1z_cases.json`, `e1z_preflight_audit.json`, `e1z_live_qwen_results.jsonl`, `e1z_live_qwen_summary.json`, `e1z_post_run_integrity_check.json` |
| A2A deadlock failure case — Case 1 | `evaluation/e1_failure_cases.json` |
| Accommodation ML | `ml/reports/phase2_final_2026-06-30.json`, `ml/reports/phase2_selection_2026-06-30.json` |
| Itinerary optimizer | `evaluation/datasets/istanbul_expert_baseline_comparison_report.json` |
| End-to-end / resilience | `evaluation/e1_end_to_end_results.json` |
| Deal ranking | `evaluation/e1r_deal_ranking_results.json` |

## 16. Reproduction commands

```bash
# RAG retrieval/generation and RAG-first artifacts are read-only historical
# records -- do not re-run to "get new numbers"; they are frozen evidence.

# Routing (final, §7/§13 Case 3) -- requires real QWEN_API_KEY/QWEN_BASE_URL,
# already executed once; the runner refuses to overwrite existing results.
python -m evaluation.run_e1z_live_qwen_eval

# End-to-end / resilience -- hermetic, no network
python -m pytest orchestration/tests/test_e1_end_to_end_resilience.py -q

# Deal ranking -- deterministic, hermetic, no network
cd services/travel-mcp && python -m pytest phase2/tests/test_e1r_deal_ranking_evaluation.py -v

# Five-container Docker smoke
docker compose up -d
docker compose ps
curl -s http://localhost:8002/health
```
