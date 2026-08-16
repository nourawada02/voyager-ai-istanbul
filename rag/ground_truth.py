"""45 ground-truth retrieval/generation questions (15 EN, 15 TR, 15 AR),
Checkpoint Phase 3 §3.

Authored independently of any retrieval run, directly from the real
corpus documents in `corpus_sources.py` -- never adjusted afterward to
improve measured metrics. Relevance is document-level and reproducibly
resolvable: a retrieved chunk counts as relevant to a question iff its
`source_id` is in that question's `expected_source_ids` (never a manually
guessed chunk_id, which would depend on the chunking config).

15 "slots" (topics/intents), each translated into English, Turkish, and
Arabic -- deliberately exploiting the corpus's real, honestly asymmetric
per-language coverage (see corpus_sources.py) so several slots are
genuine cross-language-retrieval tests: the query language has no
same-language source document, so a correct retrieval can only happen
through multilingual-e5's cross-lingual semantic matching, not a lexical
shortcut.

4 of the 15 slots (ids 6, 9, 14, 15) are deliberately unanswerable from
this corpus by design -- accessibility infrastructure detail, current
ticket prices, current weather, and current opening hours are all
explicitly excluded from the corpus boundary (architecture.md §9.1).
`refusal_required=True` for these; `expected_source_ids=()` confirms
there is truly nothing in the corpus that should be cited.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroundTruthQuestion:
    question_id: str
    slot: int
    language: str  # "en" | "tr" | "ar"
    question: str
    expected_source_ids: tuple[str, ...]
    expected_section: str | None
    poi_id: str | None
    district_id: str | None
    expected_facts: tuple[str, ...]
    refusal_required: bool
    category: str  # free-text label: factual | alias | neighborhood | cross_lingual | etiquette | accessibility | dynamic


QUESTIONS: tuple[GroundTruthQuestion, ...] = (
    # --- Slot 1: Hagia Sophia construction (direct factual) --------------
    GroundTruthQuestion(
        "gt_01_en", 1, "en",
        "When was Hagia Sophia originally constructed and who commissioned it?",
        ("wiki_en_hagia_sophia",), "Construction History", "poi_hagia_sophia", "district_fatih",
        ("built between 532 and 537 AD", "commissioned by Emperor Justinian I"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_01_tr", 1, "tr",
        "Ayasofya ilk olarak ne zaman inşa edildi ve kim tarafından yaptırıldı?",
        ("wiki_tr_hagia_sophia",), "İnşa Tarihi", "poi_hagia_sophia", "district_fatih",
        ("532-537 yılları arasında inşa edildi", "İmparator I. Justinianus tarafından yaptırıldı"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_01_ar", 1, "ar",
        "متى بُنيت آيا صوفيا أصلاً ومن الذي أمر ببنائها؟",
        ("wiki_ar_hagia_sophia",), "تاريخ البناء", "poi_hagia_sophia", "district_fatih",
        ("بُنيت بين 532 و537 ميلادية", "أمر ببنائها الإمبراطور جستنيان الأول"),
        False, "factual",
    ),
    # --- Slot 2: Topkapi Palace commissioning (cross-lingual for TR) -----
    GroundTruthQuestion(
        "gt_02_en", 2, "en",
        "Who ordered the construction of Topkapı Palace and when did construction begin?",
        ("wiki_en_topkapi",), "Construction and Development", "poi_topkapi_palace", "district_fatih",
        ("Sultan Mehmed II ordered construction", "began in 1459"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_02_tr", 2, "tr",
        "Topkapı Sarayı'nın inşasını kim emretti ve inşaat ne zaman başladı?",
        ("wiki_en_topkapi", "wiki_ar_topkapi"), "Construction and Development / البناء والتاريخ",
        "poi_topkapi_palace", "district_fatih",
        ("Sultan II. Mehmed inşaatı emretti", "inşaat 1459'da başladı"),
        False, "cross_lingual",
    ),
    GroundTruthQuestion(
        "gt_02_ar", 2, "ar",
        "من أمر ببناء قصر توب قابي ومتى بدأ البناء؟",
        ("wiki_ar_topkapi",), "البناء والتاريخ", "poi_topkapi_palace", "district_fatih",
        ("أمر به السلطان محمد الثاني الفاتح", "بدأ البناء سنة 1459"),
        False, "factual",
    ),
    # --- Slot 3: Alias/transliteration resolution -------------------------
    GroundTruthQuestion(
        "gt_03_en", 3, "en",
        "When did Ayasofya (Hagia Sophia) become a mosque again?",
        ("wiki_en_hagia_sophia",), "Modern Period: Museum and Mosque", "poi_hagia_sophia", "district_fatih",
        ("reclassified as a mosque in 2020",),
        False, "alias",
    ),
    GroundTruthQuestion(
        "gt_03_tr", 3, "tr",
        "Aya Sofya günümüzde ne zaman yeniden camiye dönüştürüldü?",
        ("wiki_tr_hagia_sophia",), "Dini ve Siyasi Dönüşümler", "poi_hagia_sophia", "district_fatih",
        ("2020 yılında yeniden cami statüsüne dönüştürüldü",),
        False, "alias",
    ),
    GroundTruthQuestion(
        "gt_03_ar", 3, "ar",
        "متى تحولت آيا صوفيا مرة أخرى إلى مسجد؟",
        ("wiki_ar_hagia_sophia",), "التحولات الدينية", "poi_hagia_sophia", "district_fatih",
        ("أُعيد تصنيفها كمسجد سنة 2020",),
        False, "alias",
    ),
    # --- Slot 4: Beyoglu neighborhood/POI filter --------------------------
    GroundTruthQuestion(
        "gt_04_en", 4, "en",
        "What was the Beyoğlu district of Istanbul historically known as, and what is it culturally known for today?",
        ("wiki_en_beyoglu",), "Etymology and Historical Names", None, "district_beyoglu",
        ("historically known as Pera", "today a center for cultural activities, leisure and entertainment"),
        False, "neighborhood",
    ),
    GroundTruthQuestion(
        "gt_04_tr", 4, "tr",
        "Beyoğlu ilçesi tarihte hangi isimle bilinirdi ve bugün kültürel olarak neyle tanınır?",
        ("wiki_tr_beyoglu",), "Etimoloji", None, "district_beyoglu",
        ("tarihte Pera olarak bilinirdi", "günümüzde kültürel etkinliklerin merkezi"),
        False, "neighborhood",
    ),
    GroundTruthQuestion(
        "gt_04_ar", 4, "ar",
        "بأي اسم كانت تُعرف منطقة بي أوغلو (Beyoğlu) تاريخياً؟",
        ("wiki_en_beyoglu", "wiki_tr_beyoglu"), "Etymology and Historical Names / Etimoloji",
        None, "district_beyoglu",
        ("كانت تُعرف تاريخياً باسم پيرا (Pera)",),
        False, "cross_lingual",
    ),
    # --- Slot 5: Galata Tower (EN-only source -> cross-lingual for TR/AR) -
    GroundTruthQuestion(
        "gt_05_en", 5, "en",
        "When was Galata Tower built and by whom?",
        ("wiki_en_galata_tower",), "Construction and Early History", "poi_galata_tower", "district_beyoglu",
        ("built in 1348", "built by the Genoese"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_05_tr", 5, "tr",
        "Galata Kulesi ne zaman ve kimin tarafından inşa edildi?",
        ("wiki_en_galata_tower",), "Construction and Early History", "poi_galata_tower", "district_beyoglu",
        ("1348 yılında inşa edildi", "Cenevizliler tarafından inşa edildi"),
        False, "cross_lingual",
    ),
    GroundTruthQuestion(
        "gt_05_ar", 5, "ar",
        "متى بُني برج غالطة ومن الذي بناه؟",
        ("wiki_en_galata_tower",), "Construction and Early History", "poi_galata_tower", "district_beyoglu",
        ("بُني سنة 1348", "بناه الجنويون"),
        False, "cross_lingual",
    ),
    # --- Slot 6: Accessibility (genuinely not in corpus -> refusal) ------
    GroundTruthQuestion(
        "gt_06_en", 6, "en",
        "What wheelchair accessibility features does Topkapı Palace currently have?",
        (), None, "poi_topkapi_palace", "district_fatih", (),
        True, "accessibility",
    ),
    GroundTruthQuestion(
        "gt_06_tr", 6, "tr",
        "Topkapı Sarayı'nda şu anda hangi tekerlekli sandalye erişilebilirlik özellikleri var?",
        (), None, "poi_topkapi_palace", "district_fatih", (),
        True, "accessibility",
    ),
    GroundTruthQuestion(
        "gt_06_ar", 6, "ar",
        "ما ميزات إمكانية الوصول للكراسي المتحركة المتوفرة حالياً في قصر توب قابي؟",
        (), None, "poi_topkapi_palace", "district_fatih", (),
        True, "accessibility",
    ),
    # --- Slot 7: Etiquette (tea/coffee hospitality) -----------------------
    GroundTruthQuestion(
        "gt_07_en", 7, "en",
        "What role does tea or coffee play in Turkish hospitality customs?",
        ("wiki_en_culture",), "Hospitality and Food Culture", None, None,
        ("tea/coffee sharing is a key aspect of hospitality", "guests offered refreshment as a sign of respect"),
        False, "etiquette",
    ),
    GroundTruthQuestion(
        "gt_07_tr", 7, "tr",
        "Çay veya kahvenin Türk misafirperverlik geleneklerindeki rolü nedir?",
        ("wiki_tr_coffee", "wiki_en_culture"), "Misafirperverlik ve Görgü Kuralları", None, None,
        ("kahve misafirperverliğin temel taşlarından biridir", "önemli ziyaretlere eşlik eder"),
        False, "etiquette",
    ),
    GroundTruthQuestion(
        "gt_07_ar", 7, "ar",
        "ما هو دور الشاي أو القهوة في تقاليد الضيافة التركية؟",
        ("wiki_en_culture", "wiki_tr_coffee"), "Hospitality and Food Culture / Misafirperverlik ve Görgü Kuralları",
        None, None,
        ("مشاركة الشاي والقهوة جانب أساسي من الضيافة",),
        False, "cross_lingual",
    ),
    # --- Slot 8: Grand Bazaar founding --------------------------------
    GroundTruthQuestion(
        "gt_08_en", 8, "en",
        "When did construction of the Grand Bazaar begin and who commissioned it?",
        ("wiki_en_grand_bazaar",), "Founding and Early Development", "poi_grand_bazaar", "district_fatih",
        ("construction began in winter 1455/56", "commissioned by Sultan Mehmed II"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_08_tr", 8, "tr",
        "Kapalıçarşı'nın inşasına ne zaman başlandı ve kim tarafından yaptırıldı?",
        ("wiki_tr_grand_bazaar",), "Tarihsel Gelişim", "poi_grand_bazaar", "district_fatih",
        ("1455/56 kışında başlamıştır", "Fatih Sultan Mehmed tarafından yaptırılmıştır"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_08_ar", 8, "ar",
        "متى بدأ بناء البازار الكبير ومن أمر به؟",
        ("wiki_en_grand_bazaar", "wiki_tr_grand_bazaar"),
        "Founding and Early Development / Tarihsel Gelişim", "poi_grand_bazaar", "district_fatih",
        ("بدأ البناء في شتاء 1455/56", "أمر به السلطان محمد الثاني"),
        False, "cross_lingual",
    ),
    # --- Slot 9: Dynamic/unanswerable -- current price --------------------
    GroundTruthQuestion(
        "gt_09_en", 9, "en",
        "What is the current ticket price for Hagia Sophia?",
        (), None, "poi_hagia_sophia", "district_fatih", (),
        True, "dynamic",
    ),
    GroundTruthQuestion(
        "gt_09_tr", 9, "tr",
        "Ayasofya'nın güncel giriş ücreti nedir?",
        (), None, "poi_hagia_sophia", "district_fatih", (),
        True, "dynamic",
    ),
    GroundTruthQuestion(
        "gt_09_ar", 9, "ar",
        "ما هو سعر تذكرة الدخول الحالي لآيا صوفيا؟",
        (), None, "poi_hagia_sophia", "district_fatih", (),
        True, "dynamic",
    ),
    # --- Slot 10: Bosphorus geography (cross-lingual for TR) -------------
    GroundTruthQuestion(
        "gt_10_en", 10, "en",
        "How long is the Bosphorus strait and what two bodies of water does it connect?",
        ("wiki_en_bosphorus",), "Geographic Overview", None, None,
        ("approximately 31 kilometers long", "connects the Black Sea and the Sea of Marmara"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_10_tr", 10, "tr",
        "İstanbul Boğazı ne kadar uzundur ve hangi iki denizi birbirine bağlar?",
        ("wiki_en_bosphorus", "wiki_ar_bosphorus"), "Geographic Overview / الموقع والجغرافيا", None, None,
        ("yaklaşık 31 kilometre uzunluğundadır", "Karadeniz ile Marmara Denizi'ni birbirine bağlar"),
        False, "cross_lingual",
    ),
    GroundTruthQuestion(
        "gt_10_ar", 10, "ar",
        "كم يبلغ طول مضيق البوسفور وماذا يربط؟",
        ("wiki_ar_bosphorus",), "الموقع والجغرافيا", None, None,
        ("يبلغ طوله نحو 30 كيلومتراً", "يربط البحر الأسود وبحر مرمرة"),
        False, "factual",
    ),
    # --- Slot 11: Kadikoy (EN-only -> cross-lingual for TR/AR) ------------
    GroundTruthQuestion(
        "gt_11_en", 11, "en",
        "What was Kadıköy historically called and what is it known for culturally today?",
        ("wiki_en_kadikoy",), "Etymology and Historical Names", None, "district_kadikoy",
        ("historically known as Chalcedon", "the liberal cultural centre of the Anatolian side"),
        False, "neighborhood",
    ),
    GroundTruthQuestion(
        "gt_11_tr", 11, "tr",
        "Kadıköy tarihte hangi isimle biliniyordu ve bugün kültürel olarak neyle tanınıyor?",
        ("wiki_en_kadikoy",), "Etymology and Historical Names", None, "district_kadikoy",
        ("tarihte Chalcedon olarak biliniyordu", "Anadolu yakasının liberal kültür merkezi"),
        False, "cross_lingual",
    ),
    GroundTruthQuestion(
        "gt_11_ar", 11, "ar",
        "بأي اسم كانت تُعرف كاديكوي (Kadıköy) تاريخياً؟",
        ("wiki_en_kadikoy",), "Etymology and Historical Names", None, "district_kadikoy",
        ("كانت تُعرف تاريخياً باسم خلقيدونية (Chalcedon)",),
        False, "cross_lingual",
    ),
    # --- Slot 12: Uskudar (EN-only -> cross-lingual) ----------------------
    GroundTruthQuestion(
        "gt_12_en", 12, "en",
        "What was Üsküdar historically called and what is its cultural character?",
        ("wiki_en_uskudar",), "Etymology and Ancient Foundations", None, "district_uskudar",
        ("historically called Chrysopolis", "a conservative cultural center of the Anatolian side"),
        False, "neighborhood",
    ),
    GroundTruthQuestion(
        "gt_12_tr", 12, "tr",
        "Üsküdar tarihte hangi isimle biliniyordu ve kültürel karakteri nedir?",
        ("wiki_en_uskudar",), "Etymology and Ancient Foundations", None, "district_uskudar",
        ("tarihte Chrysopolis olarak biliniyordu", "Anadolu yakasının muhafazakar kültür merkezi"),
        False, "cross_lingual",
    ),
    GroundTruthQuestion(
        "gt_12_ar", 12, "ar",
        "بأي اسم كانت تُعرف أسكودار (Üsküdar) تاريخياً؟",
        ("wiki_en_uskudar",), "Etymology and Ancient Foundations", None, "district_uskudar",
        ("كانت تُعرف تاريخياً باسم خريسوبوليس (Chrysopolis)",),
        False, "cross_lingual",
    ),
    # --- Slot 13: Transport / Istanbulkart ---------------------------------
    GroundTruthQuestion(
        "gt_13_en", 13, "en",
        "What is the Istanbulkart and what transit modes does it work across?",
        ("wiki_en_transport",), "Integrated Payment System", None, None,
        ("a contactless smart card", "works across metro, tram, bus, ferry, and funicular"),
        False, "factual",
    ),
    GroundTruthQuestion(
        "gt_13_tr", 13, "tr",
        "İstanbulkart nedir ve hangi ulaşım türlerinde kullanılabilir?",
        ("wiki_en_transport",), "Integrated Payment System", None, None,
        ("temassız akıllı bir karttır", "metro, tramvay, otobüs, vapur ve füniküler için geçerlidir"),
        False, "cross_lingual",
    ),
    GroundTruthQuestion(
        "gt_13_ar", 13, "ar",
        "ما هي بطاقة إسطنبول كارت (Istanbulkart) وما وسائل النقل التي تعمل بها؟",
        ("wiki_en_transport",), "Integrated Payment System", None, None,
        ("بطاقة ذكية بدون تلامس", "تعمل عبر المترو والترام والحافلات والعبارات"),
        False, "cross_lingual",
    ),
    # --- Slot 14: Dynamic/unanswerable -- weather --------------------------
    GroundTruthQuestion(
        "gt_14_en", 14, "en",
        "What is the weather in Istanbul today?",
        (), None, None, None, (),
        True, "dynamic",
    ),
    GroundTruthQuestion(
        "gt_14_tr", 14, "tr",
        "İstanbul'da bugün hava durumu nasıl?",
        (), None, None, None, (),
        True, "dynamic",
    ),
    GroundTruthQuestion(
        "gt_14_ar", 14, "ar",
        "كيف هو الطقس في إسطنبول اليوم؟",
        (), None, None, None, (),
        True, "dynamic",
    ),
    # --- Slot 15: Dynamic/unanswerable -- opening hours --------------------
    GroundTruthQuestion(
        "gt_15_en", 15, "en",
        "What are today's opening hours for the Grand Bazaar?",
        (), None, "poi_grand_bazaar", "district_fatih", (),
        True, "dynamic",
    ),
    GroundTruthQuestion(
        "gt_15_tr", 15, "tr",
        "Kapalıçarşı'nın bugünkü açılış saatleri nedir?",
        (), None, "poi_grand_bazaar", "district_fatih", (),
        True, "dynamic",
    ),
    GroundTruthQuestion(
        "gt_15_ar", 15, "ar",
        "ما هي ساعات فتح البازار الكبير اليوم؟",
        (), None, "poi_grand_bazaar", "district_fatih", (),
        True, "dynamic",
    ),
)

assert len(QUESTIONS) == 45
assert sum(1 for q in QUESTIONS if q.language == "en") == 15
assert sum(1 for q in QUESTIONS if q.language == "tr") == 15
assert sum(1 for q in QUESTIONS if q.language == "ar") == 15
