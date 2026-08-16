"""Exports the 45 ground-truth questions to evaluation/datasets/ as a
tracked JSON artifact (Checkpoint Phase 3 §3)."""

from __future__ import annotations

import json
from pathlib import Path

from rag.ground_truth import QUESTIONS

OUT_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "datasets" / "rag_ground_truth.json"


def build() -> dict:
    payload = {
        "schema_version": "1.0.0",
        "question_count": len(QUESTIONS),
        "language_counts": {
            lang: sum(1 for q in QUESTIONS if q.language == lang) for lang in ("en", "tr", "ar")
        },
        "questions": [
            {
                "question_id": q.question_id,
                "slot": q.slot,
                "language": q.language,
                "question": q.question,
                "expected_source_ids": list(q.expected_source_ids),
                "expected_section": q.expected_section,
                "poi_id": q.poi_id,
                "district_id": q.district_id,
                "expected_facts": list(q.expected_facts),
                "refusal_required": q.refusal_required,
                "category": q.category,
            }
            for q in QUESTIONS
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    p = build()
    print(f"wrote {p['question_count']} questions to {OUT_PATH}")
