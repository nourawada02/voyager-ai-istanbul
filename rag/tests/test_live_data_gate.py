"""Tests for the deterministic, multilingual (EN/TR/AR) live-data intent
gate in `rag.answer_service._requires_live_data`.

Regression coverage for the remediated root cause: a prior version only
matched exact contiguous phrases ("current price", "güncel fiyat") and
missed real question phrasing such as "current ticket price" / "güncel
giriş ücreti" -- these are drawn directly from the live Groq run's
observed false answers (gt_09_en, gt_09_tr) and the task's required
examples (gt_09_ar). Static corpus content must never be able to
override this gate; these tests exercise only the question-text gate
itself, never a retrieved chunk.
"""

from __future__ import annotations

import pytest

from rag.answer_service import _requires_live_data
from rag.ground_truth import QUESTIONS


# Every dynamic/accessibility ground-truth question (refusal_required=True)
# must trigger the gate; every answerable one must not.
@pytest.mark.parametrize("question", [q for q in QUESTIONS if q.refusal_required], ids=lambda q: q.question_id)
def test_every_refusal_required_ground_truth_question_triggers_the_gate(question):
    assert _requires_live_data(question.question) is True


@pytest.mark.parametrize("question", [q for q in QUESTIONS if not q.refusal_required], ids=lambda q: q.question_id)
def test_every_answerable_ground_truth_question_does_not_trigger_the_gate(question):
    assert _requires_live_data(question.question) is False


# Root cause A: the previously missed "current ticket price" phrasing,
# in all three required languages.
@pytest.mark.parametrize(
    "question",
    [
        "What is the current ticket price for Hagia Sophia?",
        "Ayasofya'nın güncel giriş ücreti nedir?",
        "ما هو سعر تذكرة الدخول الحالي لآيا صوفيا؟",
        "How much does admission to Topkapı Palace cost right now?",
        "Kapalıçarşı'ya giriş ücreti ne kadar?",
    ],
)
def test_current_ticket_price_phrasing_is_gated_in_all_languages(question):
    assert _requires_live_data(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Is there availability at the Grand Bazaar tomorrow?",
        "Topkapı Sarayı'nda müsaitlik durumu nedir?",
        "ما هو التوفر في السوق الكبير؟",
    ],
)
def test_availability_phrasing_is_gated_in_all_languages(question):
    assert _requires_live_data(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is the weather like in Istanbul?",
        "İstanbul'da hava durumu nasıl?",
        "كيف هو الطقس في اسطنبول؟",
    ],
)
def test_weather_phrasing_is_gated_without_needing_a_recency_word(question):
    assert _requires_live_data(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What are the opening hours of the Grand Bazaar?",
        "Kapalıçarşı'nın açılış saatleri nedir?",
        "ما هي ساعات عمل البازار الكبير؟",
    ],
)
def test_opening_hours_phrasing_is_gated_without_needing_a_recency_word(question):
    assert _requires_live_data(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "What wheelchair accessibility features does Topkapı Palace currently have?",
        "Topkapı Sarayı'nda şu anda hangi tekerlekli sandalye erişilebilirlik özellikleri var?",
        "ما ميزات إمكانية الوصول للكراسي المتحركة المتوفرة حالياً في قصر توب قابي؟",
    ],
)
def test_accessibility_phrasing_requires_a_recency_cue_and_is_gated(question):
    assert _requires_live_data(question) is True


def test_accessibility_phrasing_without_a_recency_cue_is_not_gated():
    """The corpus boundary explicitly permits stable accessibility notes
    (architecture.md §9.1); a plain accessibility question without a
    "currently"/"şu anda"/"حالياً" cue must be allowed to reach retrieval
    rather than being blocked outright."""
    assert _requires_live_data("Does Topkapı Palace have wheelchair accessibility features?") is False


def test_bare_short_price_cue_does_not_false_positive_inside_an_unrelated_word():
    """Regression: a naive substring check for the cue "fee" also matches
    inside "coffee". Word-bounded matching must not repeat that mistake."""
    assert _requires_live_data("What role does tea or coffee play in Turkish hospitality customs?") is False


def test_turkish_dotted_i_normalization_still_matches_price_cue():
    assert _requires_live_data("AYASOFYA'NIN GÜNCEL GİRİŞ ÜCRETİ NEDİR?") is True
