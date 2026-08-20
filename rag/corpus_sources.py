"""Raw corpus source records (Checkpoint Phase 3).

Every entry here is real, stable content extracted (via a fetch-and-
extract pass over the cited canonical URL) from Wikipedia articles in
English, Turkish, and Arabic, all licensed CC BY-SA 4.0 -- a clearly
reusable, attributable source per the corpus-boundary requirement. The
`text` field is an extraction/paraphrase of the cited source produced
during ingestion, not a verbatim scrape; it is what is actually indexed
and checksummed, and its provenance (source publisher, canonical URL,
license) is recorded honestly alongside it, never a fabricated fact.

Corpus boundary (architecture.md §9.1): every entry below is deliberately
restricted to stable historical/cultural/geographic/architectural content.
Current opening hours, current prices, current availability, weather, and
anything requiring live verification were explicitly excluded during
extraction and must never be added here.

Language coverage is honestly asymmetric: English Wikipedia has the
deepest per-topic coverage of Istanbul-specific subjects, Turkish
Wikipedia covers the core topics well, and Arabic Wikipedia's coverage of
individual Istanbul landmarks (as opposed to the city and strait
themselves) is narrower -- reflected directly in which languages exist
for which source_id below, not padded to force artificial symmetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LICENSE = "CC BY-SA 4.0"
PUBLISHER = "Wikipedia contributors"


@dataclass(frozen=True)
class SourceDocument:
    source_id: str
    title: str
    language: str  # "en" | "tr" | "ar"
    url: str
    content_type: str
    text: str
    poi_id: str | None = None
    district_id: str | None = None
    topic_group: str = ""  # links same-topic documents across languages
    # RAG-FIRST SYSTEM B CHECKPOINT R.1 (additive, backward-compatible --
    # every field below defaults such that all 20 Phase 3 entries above
    # are unaffected unless explicitly retrofitted with a value):
    # interest_tags drives facet-specific retrieval query routing
    # (rag/interests.py's closed vocabulary); lat/lon/side/display_name
    # let a document that describes one concrete, visitable entity be
    # promoted to a schema-valid RAG itinerary candidate (see
    # services/istanbul-expert-b/phase4/rag_candidates.py) without
    # requiring a second lookup into phase4.poi_catalog. A document with
    # no lat/lon (general knowledge, transport, etiquette, geography) can
    # still contribute citations/evidence but never becomes a schedulable
    # candidate itself -- entity_kind records that distinction explicitly.
    interest_tags: tuple[str, ...] = ()
    lat: float | None = None
    lon: float | None = None
    side: str | None = None  # "european" | "asian", only set when lat/lon are
    display_name: str | None = None  # defaults to `title` when None
    preferred_period: str = "any"  # "any" | "morning" | "afternoon" | "evening" | "night"
    entity_kind: str = "general_knowledge"  # "poi_candidate" | "general_knowledge"
    retrieved_at: str | None = None  # None = the original Phase 3 fetch date (build_documents.RETRIEVED_AT)
    # RAG-FIRST SYSTEM B R.1 CORRECTION requirements 6/7 (additive):
    # publisher_type distinguishes official/primary sources (a
    # government/UNESCO/transit-authority publication) from Wikipedia
    # (a reputable tertiary/secondary source) -- reported per Phase 5's
    # source-counts-by-publisher-type metric. transformation_method
    # records honestly that `text` is an LLM extraction/paraphrase of the
    # cited page, never verbatim official text, even when the cited
    # source itself is official/primary; derived_from_source is the
    # exact URL that paraphrase was produced from (identical to `url`
    # today, but a distinct, explicit field per requirement 7 rather than
    # inferred from `url`'s dual use as both citation link and
    # transformation source).
    publisher_type: str = "wikipedia"  # "official_primary" | "reputable_secondary" | "wikipedia"
    transformation_method: str = "llm_paraphrase_of_cited_source"
    derived_from_source: str | None = None  # None -> defaults to `url` in build_documents.py


DOCUMENTS: tuple[SourceDocument, ...] = (
    # --- Istanbul (city history/geography) -----------------------------
    SourceDocument(
        source_id="wiki_en_istanbul",
        title="Istanbul",
        language="en",
        url="https://en.wikipedia.org/wiki/Istanbul",
        content_type="history",
        topic_group="istanbul",
        interest_tags=("history", "culture"),
        text=(
            "# Founding and Early History\n\n"
            "Istanbul's documented history begins around 660 BC when Greek settlers "
            "from Megara established Byzantium on the European side of the Bosphorus. "
            "Archaeological evidence suggests even earlier settlement, with Neolithic "
            "artifacts indicating habitation as far back as the sixth millennium BC. "
            "The city was named Byzantium by its Megarian founders.\n\n"
            "# Historical Names\n\n"
            "The city has carried three primary names reflecting its successive "
            "periods of dominance. Byzantium served as the original Greek "
            "designation. Constantinople emerged following Constantine the Great's "
            "refoundation in 330 AD, deriving from the Latin name Constantinus, "
            "after Constantine the Great, the Roman emperor who refounded the city "
            "in 324 AD. The modern name Istanbul derives from the Medieval Greek "
            "phrase eis ten Polin, literally 'to the city', and was officially "
            "adopted in 1930.\n\n"
            "# Byzantine Era\n\n"
            "Constantinople was proclaimed capital of the Roman Empire on May 11, "
            "330. Following the empire's division in 395, the city became the "
            "capital of the Eastern Roman (Byzantine) Empire, serving in this role "
            "until 1453. During this period numerous churches were built across the "
            "city, including Hagia Sophia, built during the reign of Justinian I, "
            "which remained the world's largest cathedral for a thousand years.\n\n"
            "# Ottoman Period\n\n"
            "The Ottomans conquered Constantinople on May 29, 1453. Sultan Mehmed II "
            "declared it the new Ottoman capital and undertook massive urban "
            "revitalization. The city remained the capital of the Ottoman Empire and "
            "seat of its caliphate for nearly five centuries, until the empire's "
            "dissolution following World War I.\n\n"
            "# Turkish Republic\n\n"
            "Following Turkish independence (1919-1922), the new Turkish Republic "
            "established its capital in Ankara in 1923. The city's official name "
            "changed to Istanbul on March 28, 1930. From the 1930s onward, with "
            "large-scale urban planning efforts initially led by Henri Prost "
            "between 1936 and 1950, Istanbul underwent significant structural "
            "modernization.\n\n"
            "# Geographic Position\n\n"
            "Istanbul occupies a strategic location straddling the Bosphorus "
            "strait. The city is uniquely positioned on two continents; about "
            "two-thirds of its population live in Europe and the rest in Asia. It "
            "sits between the Sea of Marmara and the Black Sea in northwestern "
            "Turkey. This geographic positioning along major historical trade "
            "routes, particularly the historic Silk Road and maritime passages, "
            "fundamentally shaped its development and significance."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_istanbul",
        title="İstanbul",
        language="tr",
        url="https://tr.wikipedia.org/wiki/%C4%B0stanbul",
        content_type="history",
        topic_group="istanbul",
        interest_tags=("history", "culture"),
        text=(
            "# Kuruluş ve Antik Dönem\n\n"
            "İstanbul'un yerleşim tarihi oldukça uzun bir geçmişe sahiptir. MÖ "
            "667'de Megara şehir devletinden gelen Yunan yerleşimciler bugünkü "
            "İstanbul üzerinde bir koloni kurdu ve bu koloni Kral Byzas'ın adını "
            "taşıyarak Bizantion adını aldı. Şehir daha sonra Roma'nın kontrolüne "
            "geçti ve İmparator I. Konstantin tarafından Roma İmparatorluğu'nun "
            "başkenti ilan edildi.\n\n"
            "# Bizans İmparatorluğu Dönemi\n\n"
            "Bizans döneminde İstanbul (Konstantinopolis), Avrupa ve Asya arasında "
            "stratejik bir kapı olarak hizmet etti. Bu uzun dönem, saldırılar "
            "arasında en yıkıcı olanı 1204 yılında Haçlılar tarafından "
            "gerçekleştirilen yağmalamayla kesintiye uğradı. Şehir daha sonra Bizans "
            "kontrolüne geri döndü ve 1453'e kadar Bizans hakimiyeti altında "
            "kaldı.\n\n"
            "# Osmanlı İmparatorluğu Dönemi\n\n"
            "1453 yılında Osmanlı Padişahı II. Mehmed şehri fethetmiş ve Osmanlı "
            "İmparatorluğu'nun başkenti haline getirmiştir. Bu dönemde şehir yeniden "
            "inşa edildi; Topkapı Sarayı ve Kapalıçarşı kuruldu. Çeşitli dinlere "
            "mensup insanların bir arada yaşadığı kozmopolit bir toplum oluştu. 1923 "
            "yılına kadar İstanbul, Osmanlı İmparatorluğu'nun başkenti olarak "
            "kaldı.\n\n"
            "# Cumhuriyet Dönemi\n\n"
            "Cumhuriyet'in ilanıyla başkent Ankara'ya taşındı. Ancak İstanbul "
            "ülkenin ticaret, sanayi, ulaşım, turizm, eğitim, kültür ve sanat "
            "merkezi olmaya devam etmiştir.\n\n"
            "# Coğrafi Konum\n\n"
            "İstanbul iki kıtada yer almaktadır. Şehir, Karadeniz ile Marmara "
            "Denizi'ni bağlayan ve Asya ile Avrupa'yı ayıran İstanbul Boğazı'na ev "
            "sahipliği yapmaktadır. Boğaz boyunca inşa edilen köprüler (15 Temmuz "
            "Şehitler, Fatih Sultan Mehmet ve Yavuz Sultan Selim) iki yakasını "
            "birleştirmektedir. Nüfusunun yaklaşık üçte ikisi Avrupa yakasında, "
            "kalanı Anadolu yakasında yaşamaktadır."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_istanbul",
        title="إسطنبول",
        language="ar",
        url="https://ar.wikipedia.org/wiki/%D8%A5%D8%B3%D8%B7%D9%86%D8%A8%D9%88%D9%84",
        content_type="history",
        topic_group="istanbul",
        interest_tags=("history", "culture"),
        text=(
            "# التأسيس والعصور القديمة\n\n"
            "أسس المستوطنون اليونانيون من ميغارا مدينة بيزنطة حوالي سنة 660 ق.م. "
            "كانت المدينة موقعاً استراتيجياً مهماً على مضيق البسفور، فاجتذبت انتباه "
            "الإمبراطور الروماني قسطنطين الأول الذي جعلها عاصمة الإمبراطورية "
            "الرومانية سنة 330 ميلادية وأعاد تسميتها للقسطنطينية.\n\n"
            "# العصر البيزنطي\n\n"
            "ظلت المدينة عاصمة الإمبراطورية البيزنطية لقرون، حيث ازدهرت كمركز "
            "تجاري وثقافي وديني كبير. اشتهرت ببناء الكنائس الضخمة كآيا صوفيا، "
            "وأصبحت منارة للحضارة المسيحية الأرثوذكسية في المنطقة.\n\n"
            "# الفتح العثماني والعصر العثماني\n\n"
            "في 29 مايو 1453، فتح السلطان محمد الفاتح المدينة بعد حصار استمر 53 "
            "يوماً، مما أنهى الحقبة البيزنطية. غيّر اسمها إلى إسلامبول وجعلها عاصمة "
            "الدولة العثمانية. شهدت المدينة إعماراً مكثفاً وتطوراً معمارياً كبيراً "
            "خلال هذه الفترة، خاصة في عهد سليمان القانوني.\n\n"
            "# الحقبة الجمهورية\n\n"
            "بعد تأسيس الجمهورية التركية سنة 1923، نُقلت العاصمة إلى أنقرة. شهدت "
            "المدينة تحولاً عمرانياً جذرياً في الأربعينات والخمسينات.\n\n"
            "# الموقع الجغرافي\n\n"
            "تقع إسطنبول على مضيق البسفور، الذي يفصل بين قارتي أوروبا وآسيا. "
            "تمتد المدينة على كلا الضفتين، مما يجعلها المدينة الوحيدة الموجودة على "
            "قارتين. يفصل بينها وبين البحر الأسود شمالاً بحر مرمرة جنوباً."
        ),
    ),
    # --- Hagia Sophia -----------------------------------------------------
    SourceDocument(
        source_id="wiki_en_hagia_sophia",
        title="Hagia Sophia",
        language="en",
        url="https://en.wikipedia.org/wiki/Hagia_Sophia",
        content_type="attraction",
        poi_id="poi_hagia_sophia",
        district_id="district_fatih",
        topic_group="hagia_sophia",
        interest_tags=("history", "culture", "attractions", "religious_heritage"),
        lat=41.0086, lon=28.9802, side="european", entity_kind="poi_candidate",
        text=(
            "# Construction History\n\n"
            "The current Hagia Sophia represents the third church built on its "
            "Constantinople site. Emperor Justinian I commissioned architects "
            "Anthemius of Tralles and Isidore of Miletus to design a replacement "
            "after the second church was destroyed during the Nika Riots on "
            "January 13-14, 532. Construction began on February 23, 532, and the "
            "basilica was inaugurated on December 27, 537. More than ten thousand "
            "workers participated in the construction process. A significant "
            "earthquake in 558 caused the eastern semi-dome to collapse, prompting "
            "Justinian to commission Isidore the Younger for repairs; the dome was "
            "rebuilt and reinforced, giving the structure its present 55.6-meter "
            "interior height, completed in 562.\n\n"
            "# Architectural Significance\n\n"
            "The basilica measures 82 meters long and 73 meters wide, with a "
            "maximum height of 55 meters. Its architecture is considered the "
            "epitome of Byzantine architecture. The structure influenced countless "
            "subsequent religious buildings, including the Suleymaniye Mosque and "
            "Sehzade Mosque. The interior combined polychrome marbles with gold "
            "mosaics and intricate decorative elements.\n\n"
            "# Religious and Imperial Significance (537-1453)\n\n"
            "For over nine hundred years, Hagia Sophia served as the cathedral of "
            "Constantinople and the seat of the Ecumenical Patriarch. It functioned "
            "as the principal setting for Byzantine imperial ceremonies, including "
            "coronations. In 1054, the excommunication of Patriarch Michael I "
            "Cerularius occurred here, marking the beginning of the East-West "
            "Schism. During the Fourth Crusade in 1204, Latin Crusaders stripped "
            "the church of gold ornaments. From 1204 to 1261 the church functioned "
            "as a Latin Catholic cathedral under the Latin Empire; upon Byzantine "
            "reconquest in 1261 it returned to Orthodox control.\n\n"
            "# Transition to Islamic Use (1453 onward)\n\n"
            "Following the Ottoman conquest of Constantinople in 1453, Sultan "
            "Mehmed II converted Hagia Sophia into a mosque. The building received "
            "minarets and underwent modifications to accommodate Islamic worship "
            "practices. It functioned as the principal mosque of Istanbul until the "
            "completion of the Sultan Ahmed Mosque in 1616.\n\n"
            "# Modern Period: Museum and Mosque\n\n"
            "The complex remained a mosque until 1931, when it was closed to the "
            "public. In 1935 the secular Turkish Republic reopened the structure as "
            "a museum. In 2020, Turkey's Council of State reclassified Hagia Sophia "
            "as a mosque once again, a decision that proved highly controversial "
            "internationally."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_hagia_sophia",
        title="Ayasofya",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Ayasofya",
        content_type="attraction",
        poi_id="poi_hagia_sophia",
        district_id="district_fatih",
        topic_group="hagia_sophia",
        interest_tags=("history", "culture", "attractions", "religious_heritage"),
        lat=41.0086, lon=28.9802, side="european", entity_kind="poi_candidate",
        text=(
            "# İnşa Tarihi\n\n"
            "Ayasofya, 532-537 yılları arasında Bizans İmparatoru I. Justinianus "
            "döneminde inşa edilmiştir. Mimarlar Milet'li İsidoros ve Tralles'li "
            "Anthemios inşaatı yönetmiş, yaklaşık 10.000 işçi çalışmıştır. Yapı "
            "malzemelerinin çoğu, Efes'teki Artemis Tapınağı'ndan getirilen "
            "sütunlar dahil olmak üzere imparatorluk genelindeki eski tapınaklardan "
            "temin edilmiştir. Yapı, aynı yerdeki iki önceki kiliseyi "
            "değiştirmiştir; her ikisi de Nika İsyanı sırasında yıkılmıştır.\n\n"
            "# Mimari Önem\n\n"
            "Dört büyük ayağa oturan kubbe, mimarlık tarihinde devrim niteliğinde "
            "bir başarıyı temsil eder. Bu tasarım, merkezi ve bazilika planlarını "
            "eşi görülmemiş şekillerde birleştirmiştir. Ana kubbe 55,60 metre "
            "yüksekliğe ve 30,80-32,6 metre iç çapa sahiptir.\n\n"
            "# Dini ve Siyasi Dönüşümler\n\n"
            "1453'te Konstantinopolis'in fethinin ardından, Osmanlı Padişahı II. "
            "Mehmed kiliseyi camiye çevirdi. Yapı, minareler, minber ve İslami "
            "hat sanatı panelleri dahil önemli değişiklikler geçirdi. 1935 yılında, "
            "Atatürk'ün direktifiyle Ayasofya müze haline getirildi; bu dönüşüm "
            "kapsamlı restorasyon çalışmalarını ve Bizans mozaiklerinin ortaya "
            "çıkarılmasını içeriyordu. 2020 yılında Ayasofya, bir cumhurbaşkanlığı "
            "kararnamesiyle yeniden cami statüsüne dönüştürüldü."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_hagia_sophia",
        title="آيا صوفيا",
        language="ar",
        url="https://ar.wikipedia.org/wiki/%D8%A2%D9%8A%D8%A7_%D8%B5%D9%88%D9%81%D9%8A%D8%A7",
        content_type="attraction",
        poi_id="poi_hagia_sophia",
        district_id="district_fatih",
        topic_group="hagia_sophia",
        interest_tags=("history", "culture", "attractions", "religious_heritage"),
        lat=41.0086, lon=28.9802, side="european", entity_kind="poi_candidate",
        text=(
            "# تاريخ البناء\n\n"
            "تمثل آيا صوفيا الحالية ثالث هيكل رئيسي بُني في موقعها بالقسطنطينية. "
            "الكنيسة الأولى شُيدت حوالي سنة 360 ميلادية في عهد قسطنطين الثاني، "
            "واحترقت خلال أعمال الشغب في سنة 404. كنيسة ثانية بناها ثيودوسيوس "
            "الثاني (415-532) دُمرت في انتفاضة نيكا سنة 532. الهيكل الحالي "
            "كلّف به الإمبراطور جستنيان الأول وشُيد بين سنتي 532 و537 ميلادية. "
            "صمم المبنى المهندسان إيسيدوروس الميليتي وأنثيميوس الترالسي، واستُخدم "
            "أكثر من 10,000 عامل على مدى خمس سنوات متواصلة.\n\n"
            "# الأهمية المعمارية\n\n"
            "يبلغ طول المبنى 100 متر مع قبة يصل ارتفاعها إلى 55 متراً وقطرها 30 "
            "متراً. تُعتبر القبة إنجازاً هندسياً ثورياً لعصرها، إذ تبدو معلقة في "
            "الهواء دون دعامات ظاهرة.\n\n"
            "# التحولات الدينية\n\n"
            "خلال العصر البيزنطي (537-1453) خدمت الكاتدرائية كمقر رئيسي "
            "للبطريركية الأرثوذكسية المسكونية واستضافت مراسم تتويج الأباطرة. بعد "
            "فتح 1453، حوّل السلطان محمد الفاتح المبنى إلى مسجد. جرى تجصيص "
            "الجداريات والفسيفساء امتثالاً للممارسات الإسلامية. حولت الجمهورية "
            "التركية المبنى إلى متحف سنة 1935. في سنة 2020 ألغت المحكمة العليا "
            "التركية مرسوم العلمنة لسنة 1934، وأعادت تصنيف الموقع كوقف إسلامي "
            "ليعمل مجدداً كمسجد."
        ),
    ),
    # --- Topkapi Palace -----------------------------------------------------
    SourceDocument(
        source_id="wiki_en_topkapi",
        title="Topkapı Palace",
        language="en",
        url="https://en.wikipedia.org/wiki/Topkap%C4%B1_Palace",
        content_type="attraction",
        poi_id="poi_topkapi_palace",
        district_id="district_fatih",
        topic_group="topkapi",
        interest_tags=("history", "culture", "attractions"),
        lat=41.0115, lon=28.9833, side="european", entity_kind="poi_candidate",
        text=(
            "# Overview\n\n"
            "Topkapi Palace served as the administrative center and primary "
            "residence of Ottoman sultans from the 1460s until 1853, when "
            "Dolmabahce Palace became the official seat. Today it functions as a "
            "museum administered by Turkey's Ministry of Culture and Tourism. The "
            "complex covers approximately 59,260 to 70,000 square meters on the "
            "Seraglio Point, overlooking the Golden Horn and Bosphorus Strait.\n\n"
            "# Construction and Development\n\n"
            "Sultan Mehmed II ordered construction to begin in 1459, six years "
            "following Constantinople's conquest. The palace was originally called "
            "the New Palace. Construction of the inner core proceeded between 1459 "
            "and the late 1460s. The complex expanded significantly during Suleyman "
            "the Magnificent's reign (1520-1560) and again when the Imperial Harem "
            "permanently relocated there after 1541, following a fire that "
            "destroyed the Old Palace.\n\n"
            "# Architectural Design\n\n"
            "The palace represents an extensive complex rather than a single "
            "monolithic structure, comprising low buildings organized around "
            "courtyards, galleries, and passages. Few structures exceed two "
            "stories. The layout follows four main courtyards arranged "
            "hierarchically, from the most accessible first courtyard to the "
            "highly restricted harem and fourth courtyard, reflecting the Ottoman "
            "principle of imperial seclusion codified by Mehmed II in 1477 and "
            "1481.\n\n"
            "# Museum Status\n\n"
            "Following the Ottoman Empire's collapse in 1923, the Turkish "
            "government converted the palace into a museum on April 3, 1924. "
            "Today it houses extensive collections of ceramics, textiles, weapons, "
            "Islamic manuscripts, Ottoman miniatures, and jewelry, including sacred "
            "Islamic artifacts. The site was designated a UNESCO World Heritage "
            "Site in 1985 as part of the Historic Areas of Istanbul."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_topkapi",
        title="طوب قابي سراي",
        language="ar",
        url="https://ar.wikipedia.org/wiki/%D8%B7%D9%88%D8%A8_%D9%82%D8%A7%D8%A8%D9%8A_%D8%B3%D8%B1%D8%A7%D9%8A",
        content_type="attraction",
        poi_id="poi_topkapi_palace",
        district_id="district_fatih",
        topic_group="topkapi",
        interest_tags=("history", "culture", "attractions"),
        lat=41.0115, lon=28.9833, side="european", entity_kind="poi_candidate",
        text=(
            "# البناء والتاريخ\n\n"
            "بدأ بناء القصر سنة 1459 على يد السلطان محمد الثاني الفاتح بعد فتحه "
            "القسطنطينية سنة 1453. استمر البناء حتى 1478، وكان يُعرف مبدئياً باسم "
            "'السراي الجديد' لتمييزه عن مقار الإقامة الملكية السابقة.\n\n"
            "# الأهمية التاريخية\n\n"
            "كان القصر مقر إقامة السلاطين العثمانيين لأربعة قرون، من 1465 إلى "
            "1856. عمل كمركز إداري ومقر ملكي في آن واحد، وفقد أهميته عندما كلّف "
            "السلطان عبد المجيد الأول ببناء قصر دولمة بهجة سنة 1856 كأول مقر ملكي "
            "على الطراز الأوروبي.\n\n"
            "# السمات المعمارية\n\n"
            "يتألف المجمع من أربع ساحات رئيسية مترابطة بممرات ومساحات خضراء. "
            "البوابة الرئيسية، باب الهمايون، يبلغ ارتفاعها نحو 15 متراً. بُني "
            "القصر على موقع حصن بيزنطي سابق يطل على القرن الذهبي وبحر مرمرة.\n\n"
            "# التحول إلى متحف\n\n"
            "بعد انهيار الدولة العثمانية سنة 1923، حوّلت الحكومة التركية القصر "
            "إلى متحف في 3 أبريل 1924. يضم اليوم مجموعات واسعة من الخزف والمنسوجات "
            "والأسلحة والمخطوطات الإسلامية والمنمنمات العثمانية والمجوهرات، بالإضافة "
            "إلى آثار إسلامية مقدسة. أُدرج الموقع ضمن قائمة التراث العالمي لليونسكو "
            "سنة 1985."
        ),
    ),
    # --- Grand Bazaar -----------------------------------------------------
    SourceDocument(
        source_id="wiki_en_grand_bazaar",
        title="Grand Bazaar, Istanbul",
        language="en",
        url="https://en.wikipedia.org/wiki/Grand_Bazaar,_Istanbul",
        content_type="attraction",
        poi_id="poi_grand_bazaar",
        district_id="district_fatih",
        topic_group="grand_bazaar",
        interest_tags=("history", "shopping", "attractions", "culture"),
        lat=41.0106, lon=28.9681, side="european", entity_kind="poi_candidate",
        text=(
            "# Founding and Early Development\n\n"
            "The Grand Bazaar's core construction began in winter 1455/56, shortly "
            "after the Ottoman conquest of Constantinople. Sultan Mehmed II "
            "commissioned an edifice devoted to trading textiles and jewels, named "
            "the Cevahir Bedestan (Bedesten of Gems). This initial structure was "
            "completed by winter 1460/61. A second covered market, the Sandal "
            "Bedesten, was erected later. Traders of the same goods concentrated "
            "along specific roads, gradually creating an entire quarter dedicated "
            "exclusively to commerce.\n\n"
            "# Architectural Composition\n\n"
            "The Ic Bedesten features a rectangular plan measuring 43.30 by 29.50 "
            "meters, with stone piers supporting three rows of brick domes. The "
            "structure includes 44 cellars serving as storage and safes. The "
            "Sandal Bedesten similarly employs a rectangular design with 12 stone "
            "piers bearing 20 domed bays. Originally constructed with wood, the "
            "buildings were rebuilt in stone and brick after the 1700 fire.\n\n"
            "# Historical Scale and Significance\n\n"
            "By 1638, traveler Evliya Celebi documented approximately 3,000 shops "
            "within the bazaar, plus 300 in surrounding caravanserais. An 1890 "
            "survey recorded 4,399 active shops across the complex.\n\n"
            "# Social Organization and Cultural Role\n\n"
            "The complex operated under strict guild systems, with merchants "
            "organized by trade type. Gates were always closed at night, and the "
            "bazaar was patrolled by guards paid by the merchants' guilds. The "
            "bazaar functioned as Istanbul's primary social gathering place, "
            "particularly significant as one of few venues where women could "
            "venture relatively freely during the Ottoman period."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_grand_bazaar",
        title="Kapalıçarşı",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Kapal%C4%B1%C3%A7ar%C5%9F%C4%B1",
        content_type="attraction",
        poi_id="poi_grand_bazaar",
        district_id="district_fatih",
        topic_group="grand_bazaar",
        interest_tags=("history", "shopping", "attractions", "culture"),
        lat=41.0106, lon=28.9681, side="european", entity_kind="poi_candidate",
        text=(
            "# Giriş\n\n"
            "Kapalıçarşı, İstanbul'un merkezi ilçelerinden birinde bulunan "
            "dünyanın en büyük ve en eski kapalı çarşılarından biridir. Çarşı "
            "yaklaşık 4.000 dükkanı barındırmaktadır.\n\n"
            "# Tarihsel Gelişim\n\n"
            "Çarşının çekirdek yapıları iki bedestenden köken almıştır. İç "
            "bedesten muhtemelen Bizans dönemine tarihlenmektedir ve 48m x 36m "
            "ölçülerindedir. Çarşının temel inşası, Konstantinopolis'in Osmanlı "
            "fethinin hemen ardından, 1455/56 kışında başlamıştır. İkinci büyük "
            "yapı olan Sandal Bedesteni, Fatih Sultan Mehmed tarafından yaklaşık "
            "1460 yılında yaptırılmıştır. 1460 yılı, çarşının resmi kuruluş tarihi "
            "olarak kabul edilir. Kanuni Sultan Süleyman döneminde çarşı ahşap "
            "yapılarla önemli ölçüde genişletilmiştir.\n\n"
            "# Mimari Ölçek ve Düzen\n\n"
            "Kompleks, yaklaşık 30.700 metrekarelik bir alanda, 66 sokak ve 4.000 "
            "dükkan içermektedir. Tarihsel olarak yapı beş cami, bir okul, yedi "
            "çeşme, on kuyu ve çok sayıda han barındırmıştır.\n\n"
            "# Modern İşlevler\n\n"
            "Kapalıçarşı 2014 yılında 91,25 milyon ziyaretçi almış, dünyanın en "
            "çok ziyaret edilen turistik yerleri arasına girmiştir."
        ),
    ),
    # --- Galata Tower -----------------------------------------------------
    SourceDocument(
        source_id="wiki_en_galata_tower",
        title="Galata Tower",
        language="en",
        url="https://en.wikipedia.org/wiki/Galata_Tower",
        content_type="attraction",
        poi_id="poi_galata_tower",
        district_id="district_beyoglu",
        topic_group="galata_tower",
        interest_tags=("history", "attractions", "photography"),
        lat=41.0256, lon=28.9741, side="european", entity_kind="poi_candidate",
        text=(
            "# Construction and Early History\n\n"
            "The Galata Tower was built in 1348 during the expansion of a Genoese "
            "colony established in Constantinople in 1267. Originally known as the "
            "Christea Turris (Tower of Christ), it was constructed in Romanesque "
            "style at the highest point of the Walls of Galata. At the time the "
            "Galata Tower, at 66.9 meters, was the tallest building in the city. A "
            "Byzantine tower had previously occupied this location, erected during "
            "Emperor Justinian's reign, but it was destroyed by Crusaders during "
            "the 1204 Sack of Constantinople.\n\n"
            "# Architectural Features\n\n"
            "The tower is a nine-story cylindrical structure built from stone "
            "masonry, 62.59 meters in height, with an external diameter of 16.45 "
            "meters at the base and a wall thickness of 3.75 meters. The structure "
            "features a conical roof, reconstructed during 1965-1967 restoration "
            "work.\n\n"
            "# Functional Evolution\n\n"
            "Following the 1453 conquest of Constantinople, the Genoese colony was "
            "abolished and the tower was converted into a prison, then repurposed "
            "starting in 1717 as a fire lookout tower. The tower underwent major "
            "restoration between 1965-1967, during which the wooden interior was "
            "replaced with a concrete structure."
        ),
    ),
    # --- Districts: Beyoglu -----------------------------------------------
    SourceDocument(
        source_id="wiki_en_beyoglu",
        title="Beyoğlu",
        language="en",
        url="https://en.wikipedia.org/wiki/Beyo%C4%9Flu",
        content_type="neighborhood",
        poi_id="poi_beyoglu_district",
        district_id="district_beyoglu",
        topic_group="beyoglu",
        interest_tags=("neighborhoods", "culture", "nightlife", "shopping"),
        lat=41.0330, lon=28.9773, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        text=(
            "# Etymology and Historical Names\n\n"
            "Beyoglu was historically known as Pera, derived from the Greek word "
            "meaning 'across' or 'beyond', referencing its location on the "
            "opposite side of the Golden Horn from Constantinople's historic "
            "peninsula.\n\n"
            "# Byzantine and Medieval Periods\n\n"
            "The district has been inhabited since the 7th century BC. During the "
            "Byzantine era, the northern shore of the Golden Horn developed as a "
            "suburban area by the 5th century and became known as Galata.\n\n"
            "# Genoese and Venetian Dominance (1273-1453)\n\n"
            "In 1273, Byzantine Emperor Michael VIII Palaiologos granted Pera to "
            "the Republic of Genoa. The Genoese built the Galata Tower in 1348 and "
            "established the area as a flourishing trade colony.\n\n"
            "# Ottoman Period and Modernization\n\n"
            "After Ottoman conquest, European merchant communities continued "
            "flourishing in Pera. The 19th century witnessed rapid modernization, "
            "including the Tunel, the second-oldest underground urban railway in "
            "the world after the London Underground, opened in 1875.\n\n"
            "# 19th Century European Character\n\n"
            "The district became home to prominent European embassies and a "
            "cosmopolitan Levantine population, with schools, theaters, cafes, and "
            "patisseries reflecting its sophisticated international character.\n\n"
            "# Contemporary Cultural Revival\n\n"
            "Beginning in the 1980s-1990s, urban renewal projects revitalized the "
            "district, including pedestrianization of Istiklal Avenue and "
            "reinstatement of nostalgic trams in 1990. Present-day Beyoglu "
            "functions as one of the main centers for cultural activities, "
            "leisure and entertainment in Istanbul.\n\n"
            "# Religious and Ethnic Diversity\n\n"
            "The district contains the largest Catholic church in Turkey (S. "
            "Antonio di Padova), the largest synagogue in Turkey (Neve Shalom), "
            "and numerous Orthodox churches, reflecting centuries of multicultural "
            "coexistence."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_beyoglu",
        title="Beyoğlu",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Beyo%C4%9Flu",
        content_type="neighborhood",
        poi_id="poi_beyoglu_district",
        district_id="district_beyoglu",
        topic_group="beyoglu",
        interest_tags=("neighborhoods", "culture", "nightlife", "shopping"),
        lat=41.0330, lon=28.9773, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        text=(
            "# Etimoloji\n\n"
            "İlçe, adını 'karşı kıyı' veya 'öbür taraf' anlamına gelen Yunanca "
            "'Pera' kelimesinden alır ve tarihi yarımadanın karşısındaki konumunu "
            "ifade eder. 1925 yılında Pera kullanımı resmi belgelerden "
            "kaldırılarak Beyoğlu adı benimsenmiştir.\n\n"
            "# Tarihsel Gelişim\n\n"
            "Beyoğlu, 16. yüzyılda Hristiyanların ve yabancıların diplomatik "
            "misyonlar yakınında yerleştiği Avrupa tarzı bir yerleşim olarak "
            "ortaya çıktı. 19. yüzyılda Osmanlı-Avrupa ticaretinin yoğunlaşmasıyla "
            "uluslararası bir ticaret merkezi haline geldi; tiyatrolar, modern "
            "altyapı ve seçkin Art Nouveau mimarisiyle öne çıktı.\n\n"
            "# Kültürel Önem\n\n"
            "Günümüz Beyoğlu'su, kozmopolit kimliğiyle 'İstanbul'un en karakteristik "
            "İstanbul ilçesi' olarak tanımlanır. İstiklal Caddesi ve çevresindeki "
            "sokaklar, müzeler, tiyatrolar, galeriler ve gösteri mekanlarıyla "
            "ilçenin ticari ve kültürel kalbini oluşturur."
        ),
    ),
    # --- Kadikoy ------------------------------------------------------------
    SourceDocument(
        source_id="wiki_en_kadikoy",
        title="Kadıköy",
        language="en",
        url="https://en.wikipedia.org/wiki/Kad%C4%B1k%C3%B6y",
        content_type="neighborhood",
        poi_id="poi_kadikoy_district",
        district_id="district_kadikoy",
        topic_group="kadikoy",
        interest_tags=("neighborhoods", "culture", "food", "nightlife"),
        lat=40.9903, lon=29.0275, side="asian", entity_kind="poi_candidate",
        preferred_period="evening",
        text=(
            "# Etymology and Historical Names\n\n"
            "Kadikoy was formerly known as Chalcedon during ancient and Byzantine "
            "periods. The current name, adopted after the 1453 Ottoman conquest, "
            "translates to 'Village of the Judge', referencing the jurisdiction "
            "granted to Istanbul's first judge, Hizir Bey.\n\n"
            "# Ancient Settlement\n\n"
            "Archaeological evidence indicates continuous habitation since "
            "prehistoric times, with relics dating to 5500-3500 BC found at the "
            "Fikirtepe Mound. Greeks from Megara established Chalcedon as their "
            "first Bosphorus settlement in 685 BC, several years before founding "
            "Byzantium across the strait.\n\n"
            "# Medieval and Ottoman Periods\n\n"
            "The Council of Chalcedon convened there in 451 AD as a significant "
            "ecclesiastical gathering. Ottoman control began in 1353, over a "
            "century before Constantinople's fall. Following the 1453 conquest, "
            "Chalcedon transitioned from rural settlement to market center.\n\n"
            "# Modern Development\n\n"
            "Kadikoy separated administratively from Uskudar in 1928, establishing "
            "its current district status.\n\n"
            "# Cultural Character\n\n"
            "The district emerged as the liberal cultural centre of the Anatolian "
            "side of Istanbul, distinguished by its concentration of bookshops, "
            "cinemas, bars, and cafes. It ranks first nationally on the Human "
            "Development Index among Turkish districts.\n\n"
            "# Religious Diversity\n\n"
            "Kadikoy historically housed populations representing Judaism, "
            "Christianity, and Islam. The Metropolis of Chalcedon remains "
            "headquartered there as one of four surviving metropolises within the "
            "Ecumenical Patriarchate of Constantinople."
        ),
    ),
    # --- Uskudar --------------------------------------------------------
    SourceDocument(
        source_id="wiki_en_uskudar",
        title="Üsküdar",
        language="en",
        url="https://en.wikipedia.org/wiki/%C3%9Csk%C3%BCdar",
        content_type="neighborhood",
        poi_id="poi_uskudar_district",
        district_id="district_uskudar",
        topic_group="uskudar",
        interest_tags=("neighborhoods", "religious_heritage", "culture"),
        lat=41.0214, lon=29.0161, side="asian", entity_kind="poi_candidate",
        text=(
            "# Etymology and Ancient Foundations\n\n"
            "Uskudar was established in the 7th century BC by Greek colonists "
            "from Megara, predating Byzantium's founding across the Bosphorus. "
            "Originally called Chrysopolis, meaning 'Golden City', the settlement "
            "served as a critical harbor and shipyard.\n\n"
            "# Byzantine Period Significance\n\n"
            "During Byzantine times the city became known as Skoutarion. The "
            "district maintained strategic importance as all trade routes to Asia "
            "started there, and all Byzantine army units headed to Asia mustered "
            "there.\n\n"
            "# Ottoman Era Development\n\n"
            "In 1338, Ottoman leader Orhan Gazi captured Skoutarion, establishing "
            "the Ottomans' first foothold within sight of Constantinople. The "
            "district evolved into a major burial ground, with extensive "
            "cemeteries becoming defining features that persist today.\n\n"
            "# Conservative Cultural Character\n\n"
            "Uskudar developed as a conservative cultural center of the Anatolian "
            "side of Istanbul, distinguished by its landmark as well as numerous "
            "tiny mosques and dergahs.\n\n"
            "# Architectural and Religious Heritage\n\n"
            "The district contains over 180 mosques, numerous designed by "
            "celebrated architect Mimar Sinan, including structures built for "
            "imperial harem women. Historic churches, synagogues, and Sufi lodges "
            "(tekkes) reflect the area's multicultural heritage before 20th-century "
            "demographic shifts."
        ),
    ),
    # --- Bosphorus ----------------------------------------------------------
    SourceDocument(
        source_id="wiki_en_bosphorus",
        title="Bosphorus",
        language="en",
        url="https://en.wikipedia.org/wiki/Bosphorus",
        content_type="geography",
        topic_group="bosphorus",
        interest_tags=("nature", "photography", "transportation"),
        text=(
            "# Geographic Overview\n\n"
            "The Bosphorus is a natural strait located in northwestern Turkey, "
            "straddling the city of Istanbul. It connects the Black Sea to the Sea "
            "of Marmara and forms one of the continental boundaries between Asia "
            "and Europe. The strait measures approximately 31 kilometers in "
            "length, with widths varying from 700 meters at its narrowest point to "
            "3,420 meters at its widest.\n\n"
            "# Division of Europe and Asia\n\n"
            "The Bosphorus serves as a geographic divider between two continents. "
            "Its western banks mark the starting point of Europe, while its "
            "eastern banks represent the beginning of Asia, making Istanbul one of "
            "very few intercontinental cities globally.\n\n"
            "# Historical Significance\n\n"
            "Athens maintained critical alliances with cities which controlled the "
            "straits in the 5th century BC due to its dependence on grain from the "
            "Black Sea. Roman Emperor Constantine the Great founded Constantinople "
            "in AD 330, recognizing the strait's strategic value. Following the "
            "Ottoman conquest in 1453, the empire controlled access to the Black "
            "Sea region.\n\n"
            "# Modern Treaties and Control\n\n"
            "The Treaty of Lausanne (1923) restored Turkish control while allowing "
            "foreign navigation. The Montreux Convention (1936), still in force, "
            "designates the straits as an international shipping lane while "
            "permitting Turkey to restrict non-Black Sea naval traffic.\n\n"
            "# Bridge Crossings and Infrastructure\n\n"
            "Three major bridges span the Bosphorus: the 15 July Martyrs Bridge "
            "(1973), the Fatih Sultan Mehmet Bridge (1988), and the Yavuz Sultan "
            "Selim Bridge (2016). Additional infrastructure includes the Marmaray "
            "railway tunnel and the Eurasia Tunnel for vehicular traffic."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_bosphorus",
        title="مضيق البوسفور",
        language="ar",
        url="https://ar.wikipedia.org/wiki/%D9%85%D8%B6%D9%8A%D9%82_%D8%A7%D9%84%D8%A8%D9%88%D8%B3%D9%81%D9%88%D8%B1",
        content_type="geography",
        topic_group="bosphorus",
        interest_tags=("nature", "photography", "transportation"),
        text=(
            "# الموقع والجغرافيا\n\n"
            "يقع مضيق البوسفور في تركيا، ويربط البحر الأسود وبحر مرمرة بطول 30 "
            "كيلومتراً تقريباً. يتراوح عرضه بين 550 و3000 متر، ويعتبر ممراً حاسماً "
            "يصل بين البحار والمحيطات المختلفة.\n\n"
            "# الأهمية الجيوسياسية\n\n"
            "المضيق يعمل كحدود طبيعية تفصل بين قارتي آسيا وأوروبا، بالاشتراك مع "
            "مضيق الدردنيل. يوفر طريقاً بحرياً حيوياً يربط البحر الأسود بالبحر "
            "الأبيض المتوسط والمحيط الأطلسي.\n\n"
            "# الخلفية التاريخية\n\n"
            "سُمي المضيق تاريخياً 'ممر البقرة' حسب الأساطير اليونانية القديمة. "
            "تُرجع بعض الدراسات الجيولوجية تكونه إلى حوالي 5600 قبل الميلاد.\n\n"
            "# البنية الحديثة\n\n"
            "يعبره حالياً ثلاثة جسور رئيسية، بالإضافة إلى نفق مرمراي تحت مياهه، "
            "مما يسهل الربط بين الضفاف الآسيوية والأوروبية."
        ),
    ),
    # --- Transport ------------------------------------------------------
    SourceDocument(
        source_id="wiki_en_transport",
        title="Istanbul Metro (transit system overview)",
        language="en",
        url="https://en.wikipedia.org/wiki/Istanbul_Metro",
        content_type="transport",
        topic_group="transport",
        interest_tags=("transportation",),
        text=(
            "# System Architecture\n\n"
            "Istanbul's public transportation network comprises multiple "
            "integrated modes serving the city's geography across the European "
            "and Asian sides of the Bosphorus: rapid transit metro lines, "
            "commuter rail, trams, ferries, funiculars, and bus services, "
            "coordinated through unified payment infrastructure.\n\n"
            "# Metro Network Structure\n\n"
            "The metro system operates eleven lines across two geographical "
            "zones. Lines designated M1A, M1B, M2, M3, M6, M7, M9 and M11 are on "
            "the European side of the Bosphorus, while lines M4, M5 and M8 are on "
            "the Asian side. Due to Istanbul's unique geography and the depth of "
            "the Bosphorus strait which divides the city, the European and Asian "
            "metro networks do not connect directly by rail across the strait.\n\n"
            "# Cross-Strait Connectivity\n\n"
            "The Marmaray commuter rail provides the primary rail bridge between "
            "sides, connecting to the metro at major stations like Yenikapi and "
            "Uskudar. Traditional ferries also serve as vital connectors between "
            "seabus ports such as Bostanci, Kadikoy, Bakirkoy and Kabatas.\n\n"
            "# Integrated Payment System\n\n"
            "The Istanbulkart represents the network's unified payment approach: a "
            "contactless smart card enabling passengers to pay across all transit "
            "modes -- metro, tram, bus, ferry, and funicular services -- using "
            "standardized fare structures.\n\n"
            "# Complementary Transit Modes\n\n"
            "Supporting the metro are dedicated systems including the Istanbul "
            "Tram network, funicular lines, cable cars, and the Metrobus rapid bus "
            "service, all coordinated within the broader metropolitan transit "
            "ecosystem."
        ),
    ),
    # --- Culture / etiquette -------------------------------------------
    SourceDocument(
        source_id="wiki_en_culture",
        title="Culture of Turkey (etiquette and hospitality)",
        language="en",
        url="https://en.wikipedia.org/wiki/Culture_of_Turkey",
        content_type="etiquette",
        topic_group="culture",
        interest_tags=("etiquette", "culture", "food"),
        text=(
            "# Hospitality and Food Culture\n\n"
            "Turkish society places significant emphasis on hospitality through "
            "food and beverage sharing. Food plays a central role in Turkish "
            "social life, with customs such as sharing tea or coffee forming key "
            "aspects of hospitality. Meals are often family-centered, and guests "
            "are traditionally offered food or drink as a sign of respect and "
            "generosity.\n\n"
            "# Tea, Coffee, and Raki Traditions\n\n"
            "Turkish tea (cay), coffee (kahve), and raki (a traditional alcoholic "
            "drink) transcend their culinary purposes and are often associated "
            "with ritual, conversation, and social bonding, serving as focal "
            "points for social interaction and relationship-building.\n\n"
            "# Regional Dining Customs\n\n"
            "Turkish cuisine varies considerably by region. Mediterranean coastal "
            "areas emphasize vegetables, herbs, and fish prepared with olive oil. "
            "Central Anatolia favors pastry-based dishes, while Southeastern "
            "regions are known for kebabs and layered desserts.\n\n"
            "# Life-Cycle Ceremonies\n\n"
            "Traditional rituals mark major life events. Weddings constitute "
            "elaborate events that may span several days, involving music, "
            "dancing, and symbolic customs such as henna nights (kina gecesi).\n\n"
            "# Family and Social Structure\n\n"
            "Traditional Turkish family dynamics emphasize extended kinship "
            "networks, though urbanization has shifted toward nuclear families."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_coffee",
        title="Türk kahvesi",
        language="tr",
        url="https://tr.wikipedia.org/wiki/T%C3%BCrk_kahvesi",
        content_type="etiquette",
        topic_group="culture",
        interest_tags=("etiquette", "food", "culture"),
        text=(
            "# Tarihçe\n\n"
            "Türk kahvesi, 14. yüzyılda Etiyopya'dan Osmanlı İmparatorluğu "
            "aracılığıyla dünyaya yayılmıştır. Yemen valisi Özdemir Paşa, kahveyi "
            "İstanbul'a getirmiş ve cezve kullanılarak yeni bir hazırlama yöntemi "
            "ortaya çıkmıştır.\n\n"
            "# Kültürel Önem\n\n"
            "Türk kahvesinin Türk kültüründe derin bir sosyal önemi vardır. 'Bir "
            "fincan kahvenin kırk yıl hatırı vardır' sözü, törensel değerini "
            "yansıtır. Kahve, nişan törenlerinde, dini bayramlarda ve fal "
            "geleneklerinde yer alır. UNESCO, bu kültürel mirası 2013 yılında "
            "Somut Olmayan Kültürel Miras listesine dahil etmiştir.\n\n"
            "# Misafirperverlik ve Görgü Kuralları\n\n"
            "Kahve, Türk geleneğinde misafirperverliğin temel taşlarından "
            "biridir. Önemli sosyal etkinliklere ve resmi ziyaretlere eşlik eder. "
            "Lokum ve suyla birlikte sunulması, misafire saygı gösterir ve Türk "
            "sosyal geleneklerinin merkezindeki cömertlik ilkelerini yansıtır.\n\n"
            "# Hazırlama ve Özellikler\n\n"
            "Türk kahvesi, kendine özgü demleme yöntemi ve sunumuyla öne çıkar; "
            "kahve telvesi çok ince öğütülür, cezve adı verilen küçük bir tencerede "
            "demlenir ve küçük fincanlarda telveler dibe çökene kadar servis "
            "edilir."
        ),
    ),
    # === RAG-FIRST SYSTEM B CHECKPOINT R.1 corpus expansion =============
    # Real, stable Wikipedia content (CC BY-SA 4.0), fetched and
    # paraphrased the same way as the Phase 3 corpus above -- extending
    # coverage from history/attraction/neighborhood/geography/transport/
    # etiquette to every interest in the closed vocabulary
    # (services/istanbul-expert-b/phase4/interests.py): shopping,
    # nightlife, nature, islands, family, accessibility,
    # religious_heritage, photography, plus deeper neighborhoods/
    # transportation/food/seasonal_planning coverage.
    # --- Shopping / nightlife: Istiklal Avenue --------------------------
    SourceDocument(
        source_id="wiki_en_istiklal_avenue",
        title="Istiklal Avenue",
        language="en",
        url="https://en.wikipedia.org/wiki/%C4%B0stiklal_Avenue",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_istiklal_avenue",
        district_id="district_beyoglu",
        topic_group="istiklal_avenue",
        interest_tags=("shopping", "nightlife", "neighborhoods", "culture"),
        lat=41.0328, lon=28.9784, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        text=(
            "# Overview and Location\n\n"
            "Istiklal Avenue is a 1.4-kilometre pedestrian street in the "
            "historic Beyoglu (Pera) district, running from Tunel Square in "
            "the south to Taksim Square in the north.\n\n"
            "# Naming and Historical Significance\n\n"
            "The street acquired its modern name after the declaration of "
            "the Republic on 29 October 1923 -- Istiklal means "
            "'Independence', commemorating Turkey's War of Independence. It "
            "was previously known as the Grand Avenue of Pera.\n\n"
            "# Architectural Character\n\n"
            "Buildings reflect Neo-Classical, Neo-Gothic, Renaissance "
            "Revival, Beaux-Arts, Art Nouveau and First Turkish National "
            "Architecture styles from the 19th and early 20th centuries, "
            "supplemented by Art Deco examples from the early Republic.\n\n"
            "# Commercial and Nightlife Character\n\n"
            "The avenue hosts boutiques, music stores, art galleries, "
            "cinemas, theatres, cafes, pubs, and nightclubs with live "
            "music, alongside historic pastry shops and chocolateries. A "
            "historic nostalgic tram, reinstated in 1990, runs the length "
            "of the avenue.\n\n"
            "# Notable Institutions\n\n"
            "Galatasaray High School, the oldest secondary school in "
            "Turkey, marks the avenue's midpoint. Several European nations "
            "established consulates along the avenue in the Ottoman period, "
            "including France, the Netherlands, Russia, Spain, and Sweden."
        ),
    ),
    # --- Nightlife / photography / religious heritage: Ortakoy ----------
    SourceDocument(
        source_id="wiki_en_ortakoy",
        title="Ortakoy",
        language="en",
        url="https://en.wikipedia.org/wiki/Ortak%C3%B6y",
        content_type="neighborhood",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_ortakoy",
        district_id="district_besiktas",
        topic_group="ortakoy",
        interest_tags=("nightlife", "nature", "religious_heritage", "photography", "neighborhoods"),
        lat=41.0478, lon=29.0271, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        text=(
            "# Location and Basic Information\n\n"
            "Ortakoy is a neighbourhood in the Besiktas district of "
            "Istanbul, on the European shore of the Bosphorus. The name "
            "translates to 'Middle Village' in Turkish.\n\n"
            "# Ortakoy Mosque\n\n"
            "This Neo-Baroque mosque sits directly beside the Bosphorus, "
            "visible from passing boats and framed by the first Bosphorus "
            "Bridge behind it. It was built between 1854 and 1856 under "
            "Sultan Abdulmecid I, designed by architects Garabet Amira "
            "Balyan and Nigogayos Balyan.\n\n"
            "# Cirağan Palace\n\n"
            "Constructed in 1871 by Sultan Abdulaziz, the palace later "
            "housed the Ottoman Parliament before a 1910 fire caused "
            "significant damage. Restored in the 1980s, it now operates as "
            "a hotel.\n\n"
            "# Waterfront and Cultural Character\n\n"
            "The Esma Sultan Mansion, an Ottoman-era waterfront residence "
            "from 1875, now operates as an event venue after renovation. "
            "Ortakoy's waterfront square, cafes, and weekend art-and-craft "
            "market give the district a lively evening character, and it "
            "hosts Galatasaray University and Kabatas Erkek Lisesi as "
            "centres of higher education."
        ),
    ),
    # --- Shopping / food: Spice Bazaar -----------------------------------
    SourceDocument(
        source_id="wiki_en_spice_bazaar",
        title="Spice Bazaar (Egyptian Bazaar)",
        language="en",
        url="https://en.wikipedia.org/wiki/Spice_Bazaar,_Istanbul",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_spice_bazaar",
        district_id="district_fatih",
        topic_group="spice_bazaar",
        interest_tags=("shopping", "food", "history", "attractions"),
        lat=41.0165, lon=28.9702, side="european", entity_kind="poi_candidate",
        text=(
            "# Founding and Historical Origins\n\n"
            "The Spice Bazaar was established in 1660 in Istanbul's "
            "Eminonu quarter, built with revenues from the Ottoman eyalet "
            "of Egypt -- hence its Turkish name, Misir Carsisi (Egyptian "
            "Bazaar). Construction followed the Great Fire of 1660 and was "
            "designed by court architect Koca Kasim Aga.\n\n"
            "# Architectural Significance\n\n"
            "The bazaar forms part of the kulliye (religious complex) of "
            "the New Mosque; rent from its shops historically funded the "
            "mosque's upkeep. Construction was patronized by Turhan "
            "Hatice, the Valide Sultan (Queen Mother) of Sultan Mehmed "
            "IV.\n\n"
            "# Merchandise and Scale\n\n"
            "The bazaar contains around 85 shops selling spices, Turkish "
            "delight and other sweets, jewellery, souvenirs, and dried "
            "fruits and nuts.\n\n"
            "# Position in Istanbul's Shopping Culture\n\n"
            "It ranks as the most famous covered shopping complex after "
            "the Grand Bazaar, and remains a working spice market as well "
            "as a tourist destination."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_spice_bazaar",
        title="Mısır Çarşısı",
        language="tr",
        url="https://tr.wikipedia.org/wiki/M%C4%B1s%C4%B1r_%C3%87ar%C5%9F%C4%B1s%C4%B1",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_spice_bazaar",
        district_id="district_fatih",
        topic_group="spice_bazaar",
        interest_tags=("shopping", "food", "history", "attractions"),
        lat=41.0165, lon=28.9702, side="european", entity_kind="poi_candidate",
        text=(
            "# Kuruluş ve Tarihçe\n\n"
            "Mısır Çarşısı, 1660 yılında Turhan Sultan'ın himayesinde, baş "
            "mimar Kazım Ağa yönetiminde inşa edilmiştir. Bizans "
            "döneminde aynı yerde bir çarşı bulunduğu aktarılır. İlk "
            "olarak 'Yeni Çarşı' veya 'Valide Çarşısı' olarak anılmış, "
            "günümüzdeki adını 18. yüzyıldan sonra almıştır.\n\n"
            "# Konum ve Ürünler\n\n"
            "Eminönü'nde Yeni Cami'nin arkasında konumlanan çarşı, "
            "İstanbul'un en eski kapalı çarşılarından biridir. Geleneksel "
            "şifalı bitkiler, baharatlar, çiçek tohumları ve kurutulmuş "
            "bitkisel ürünlerin yanı sıra kuruyemiş ve şarküteri ürünleri "
            "de satılmaktadır.\n\n"
            "# Mimari\n\n"
            "L biçimli yapı altı kapıya sahiptir; bunlardan biri Hasekiler "
            "Kapısı'dır. Üst katlar bir zamanlar esnaf arasındaki "
            "uyuşmazlıkların çözüldüğü bir mahkeme işlevi görmüştür.\n\n"
            "# Kültürel Önem\n\n"
            "Çarşı, İstanbul'un ticari mirasını yansıtan önemli bir turistik "
            "merkez olmaya devam etmektedir."
        ),
    ),
    # --- Nature: Belgrad Forest -------------------------------------------
    SourceDocument(
        source_id="wiki_en_belgrad_forest",
        title="Belgrad Forest",
        language="en",
        url="https://en.wikipedia.org/wiki/Belgrad_Forest",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_belgrad_forest",
        district_id="district_sariyer",
        topic_group="belgrad_forest",
        interest_tags=("nature",),
        lat=41.1804, lon=28.9982, side="european", entity_kind="poi_candidate",
        text=(
            "# Location and Geography\n\n"
            "Belgrad Forest sits adjacent to Istanbul, spanning "
            "approximately 5,524 hectares across the Sariyer and Eyup "
            "districts, at the easternmost point of the Thracian "
            "Peninsula.\n\n"
            "# Historical Origins\n\n"
            "The forest takes its name from a nearby village established "
            "by Serbian deportees after Suleiman the Magnificent's 1521 "
            "conquest of Belgrade relocated thousands of Serbs to "
            "Constantinople.\n\n"
            "# Forest Composition\n\n"
            "The ecosystem is mixed deciduous woodland, with sessile oak "
            "as the dominant species, supporting diverse plant, bird, and "
            "animal life.\n\n"
            "# Ottoman Water Infrastructure\n\n"
            "The forest contains historical reservoirs and aqueducts built "
            "over 150 years of Ottoman administration, including the "
            "Kirkcesme system upgraded by architect Sinan in the 16th "
            "century, and the Maglova Aqueduct.\n\n"
            "# Recreation\n\n"
            "The forest is used for jogging, hiking, and picnicking, with "
            "a 6.5-kilometre jogging track and the Ataturk Arboretum, "
            "which maintains over 2,000 plant species and is open daily "
            "except Mondays."
        ),
    ),
    # --- Nature / family / seasonal: Emirgan Park ------------------------
    SourceDocument(
        source_id="wiki_en_emirgan_park",
        title="Emirgan Park",
        language="en",
        url="https://en.wikipedia.org/wiki/Emirgan_Park",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_emirgan_park",
        district_id="district_sariyer",
        topic_group="emirgan_park",
        interest_tags=("nature", "family", "photography", "seasonal_planning"),
        lat=41.1073, lon=29.0522, side="european", entity_kind="poi_candidate",
        text=(
            "# Overview\n\n"
            "Emirgan Park is a major public green space on the European "
            "shore of the Bosphorus in Sariyer, encompassing 117 acres of "
            "hillside terrain.\n\n"
            "# Historical Development\n\n"
            "Originally a cypress grove in the Byzantine era, the land was "
            "granted to Ottoman Lord Chancellor Nisanci Feridun Bey in the "
            "16th century, then to the Safavid commander Emirgune Han "
            "under Sultan Murad IV -- the source of the modern name. In "
            "the 1860s, Khedive Isma'il Pasha of Egypt acquired the "
            "property and built the wooden pavilions that remain today.\n\n"
            "# The Three Pavilions\n\n"
            "The Yellow, Pink, and White pavilions -- Ottoman-era wooden "
            "structures restored between 1979 and 1983 -- now operate as "
            "public cafeterias and event venues.\n\n"
            "# Flora and Recreation\n\n"
            "The park contains over 120 plant species, including stone "
            "pine, cedar varieties, and maidenhair trees, with jogging "
            "tracks and picnic areas popular with families.\n\n"
            "# Annual Tulip Festival\n\n"
            "A dedicated tulip garden established in the 1960s hosts an "
            "annual international tulip festival every April since 2005, "
            "commemorating the Ottoman Tulip Period (1718-1730) -- one of "
            "the most reliable seasonal reasons to visit the park."
        ),
    ),
    # --- Islands / nature / family: Buyukada -----------------------------
    SourceDocument(
        source_id="wiki_en_buyukada",
        title="Buyukada (Princes' Islands)",
        language="en",
        url="https://en.wikipedia.org/wiki/Princes%27_Islands",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_buyukada",
        district_id="district_adalar",
        topic_group="buyukada",
        interest_tags=("islands", "nature", "family"),
        lat=40.8767, lon=29.1206, side="asian", entity_kind="poi_candidate",
        text=(
            "# Overview\n\n"
            "The Princes' Islands (Adalar) are an archipelago of nine "
            "islands in the Sea of Marmara near Istanbul, with a total "
            "area of 11 square kilometres. Buyukada, at 5.46 square "
            "kilometres, is the largest.\n\n"
            "# Historical Background\n\n"
            "During the Byzantine Empire, out-of-favor princes and royalty "
            "were exiled to these islands; after the 1453 Ottoman "
            "conquest, members of sultans' families were banished here "
            "too, giving the islands their name. The 19th century turned "
            "them into fashionable retreats for Istanbul's wealthy "
            "classes.\n\n"
            "# Motor Vehicle Ban\n\n"
            "Motorized vehicles, except service vehicles, are forbidden on "
            "most islands. Visitors get around by foot, bicycle, "
            "horse-drawn carriage, or electric taxi.\n\n"
            "# Ferry Access\n\n"
            "Ferries run from multiple Istanbul terminals, including "
            "Bostanci, Kadikoy, and Kabatas, calling at the four largest "
            "islands.\n\n"
            "# Notable Landmarks and History\n\n"
            "Buyukada's landmarks include the Prinkipo Greek Orthodox "
            "Orphanage, said to be the largest wooden building in Europe, "
            "and the 1908 Splendid Palace Hotel. Leon Trotsky lived on the "
            "island from 1929 to 1933 after his Soviet exile."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_buyukada",
        title="Büyükada",
        language="tr",
        url="https://tr.wikipedia.org/wiki/B%C3%BCy%C3%BCkada",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_buyukada",
        district_id="district_adalar",
        topic_group="buyukada",
        interest_tags=("islands", "nature", "family"),
        lat=40.8767, lon=29.1206, side="asian", entity_kind="poi_candidate",
        text=(
            "# Tarihsel Genel Bakış\n\n"
            "Yunanca Prinkipos (prens) olarak da bilinen Büyükada, Prens "
            "Adaları grubunun en büyüğüdür. 1930 yılında adada bulunan ve "
            "Makedonya Kralı II. Filip'e ait altın sikkeler İstanbul "
            "Arkeoloji Müzesi'nde sergilenmektedir.\n\n"
            "# Coğrafya ve Ulaşım\n\n"
            "Ada 5,4 km² alana sahiptir. Yücetepe (203 m) ve Manastır "
            "Tepesi (164 m) adanın iki tepesidir. Bostancı'dan deniz "
            "otobüsleri yaklaşık 35 dakikada, Kabataş'tan vapurlar "
            "yaklaşık 80 dakikada adaya ulaşır.\n\n"
            "# Ada İçi Ulaşım\n\n"
            "Ada içinde ulaşım bisiklet ve elektrikli araçlarla "
            "sağlanmaktadır; tarihi faytonlar son yıllarda "
            "kaldırılmıştır.\n\n"
            "# Kültürel ve Doğal Özellikler\n\n"
            "Lev Trotski 1929-1933 yılları arasında adada yaşamıştır. En "
            "yüksek noktada bulunan Aya Yorgi Manastırı özel günlerde "
            "hacılar çekmektedir. Adanın bitki örtüsüne kızılçam "
            "ormanları, servi, defne ve laden türleri hakimdir."
        ),
    ),
    # --- Islands / nature: Heybeliada ------------------------------------
    SourceDocument(
        source_id="wiki_en_heybeliada",
        title="Heybeliada",
        language="en",
        url="https://en.wikipedia.org/wiki/Heybeliada",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_heybeliada",
        district_id="district_adalar",
        topic_group="heybeliada",
        interest_tags=("islands", "nature", "family"),
        lat=40.8791, lon=29.0919, side="asian", entity_kind="poi_candidate",
        text=(
            "# Location and Size\n\n"
            "Heybeliada is the second largest of the Princes' Islands "
            "(Adalar) in the Sea of Marmara, covering approximately 2.35 "
            "square kilometres across four hills, the highest reaching "
            "136 metres.\n\n"
            "# Historical Background\n\n"
            "The island was historically known by Greek names referencing "
            "copper -- Halki, Halkitis, Demonesos -- reflecting its "
            "ancient reputation for copper ore.\n\n"
            "# Naval Academy\n\n"
            "A Naval High School, originally founded in 1773, overlooks "
            "the jetty and houses Byzantine church ruins, serving as a "
            "major educational facility for Turkey's maritime forces.\n\n"
            "# Transportation\n\n"
            "The island is served by Sehir Hatlari ferries from multiple "
            "Istanbul terminals. Horse-drawn phaetons provided land "
            "transport until 2020, when they were replaced with electric "
            "vehicles.\n\n"
            "# Religious Heritage\n\n"
            "The Halki Theological Seminary operated as the main Greek "
            "Orthodox seminary in Turkey until its 1971 closure; the "
            "island also holds several Orthodox monasteries and churches."
        ),
    ),
    # --- Family: Miniaturk ------------------------------------------------
    SourceDocument(
        source_id="wiki_en_miniaturk",
        title="Miniaturk",
        language="en",
        url="https://en.wikipedia.org/wiki/Miniat%C3%BCrk",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_miniaturk",
        district_id="district_beyoglu",
        topic_group="miniaturk",
        interest_tags=("family", "attractions"),
        lat=41.0576, lon=28.9377, side="european", entity_kind="poi_candidate",
        text=(
            "# Overview and Location\n\n"
            "Miniaturk is a miniature park on the northeastern shore of "
            "the Golden Horn, occupying 60,000 square metres. It opened on "
            "2 May 2003 and had attracted approximately 5 million visitors "
            "by 2015.\n\n"
            "# Scale and Model Collection\n\n"
            "The park contains 135 architectural reproductions built at "
            "1:25 scale, depicting structures from in and around Turkey: "
            "60 models depict Istanbul landmarks, 63 represent Anatolian "
            "sites, and 13 show former Ottoman territories now outside "
            "Turkey's borders, including the Temple of Artemis at "
            "Ephesus.\n\n"
            "# Space Allocation\n\n"
            "Of the total area, 40,000 square metres is open space, 3,500 "
            "is covered, and 2,000 contains pools and waterways, making it "
            "one of the world's largest miniature parks and a popular "
            "family destination."
        ),
    ),
    # --- Family / nature: Gulhane Park ------------------------------------
    SourceDocument(
        source_id="wiki_en_gulhane_park",
        title="Gulhane Park",
        language="en",
        url="https://en.wikipedia.org/wiki/G%C3%BClhane_Park",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_gulhane_park",
        district_id="district_fatih",
        topic_group="gulhane_park",
        interest_tags=("family", "nature", "attractions"),
        lat=41.0128, lon=28.9814, side="european", entity_kind="poi_candidate",
        text=(
            "# Overview\n\n"
            "Gulhane Park is a historic urban park in Istanbul's Fatih "
            "district, adjacent to Topkapi Palace, encompassing 9.7 "
            "hectares.\n\n"
            "# Historical Origins\n\n"
            "The park's name derives from the Gulhane (Rosehouse), where "
            "the 1839 Edict of Gulhane was proclaimed, initiating the "
            "Tanzimat reforms. Originally part of Topkapi Palace's outer "
            "garden, it became a public park in 1912.\n\n"
            "# Notable Features\n\n"
            "The grounds contain the Column of the Goths and the Istanbul "
            "Museum of the History of Science and Technology in Islam, "
            "which opened in 2008 in the palace's former stables and "
            "displays 140 replicas of historic inventions.\n\n"
            "# Modern Character\n\n"
            "Recent renovations removed a former zoo and amusement "
            "facilities, restoring a natural landscape popular with "
            "families and revealing trees dating from the 1800s."
        ),
    ),
    # --- Accessibility (general knowledge; no schedulable entity) -------
    SourceDocument(
        source_id="wiki_en_accessible_tourism",
        title="Accessible tourism",
        language="en",
        url="https://en.wikipedia.org/wiki/Accessible_tourism",
        content_type="general_knowledge",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="accessibility",
        interest_tags=("accessibility",),
        text=(
            "# What Is Accessible Tourism\n\n"
            "Accessible tourism is the ongoing endeavor to ensure tourist "
            "destinations, products, and services are accessible to all "
            "people, regardless of physical or intellectual limitations -- "
            "a concept that extends beyond travelers with disabilities to "
            "include travelers with children and seniors.\n\n"
            "# Key Components\n\n"
            "The European Network for Accessible Tourism (ENAT) frames "
            "accessible tourism around barrier-free infrastructure and "
            "facilities, transportation suitable across air, land, and sea "
            "travel, staff training, attractions and activities enabling "
            "broad participation, and accessible digital booking "
            "systems.\n\n"
            "# Universal Design Principles\n\n"
            "Seven universal design principles, established in 1997, guide "
            "the creation of products and environments usable by anyone "
            "regardless of ability: equitable access, flexibility, "
            "intuitive use, clear information, tolerance for error, low "
            "physical effort, and appropriate size/space for approach and "
            "use.\n\n"
            "# Scale and Standards\n\n"
            "As of 2020, approximately 15 percent of the global population "
            "lives with some form of disability. The United Nations "
            "Convention on the Rights of Persons with Disabilities (CRPD, "
            "adopted 2006) provides the closest international accessibility "
            "standard, with Article 9 addressing accessibility "
            "requirements directly."
        ),
    ),
    # --- Accessibility (general knowledge; no schedulable entity) -------
    SourceDocument(
        source_id="wiki_en_universal_design",
        title="Universal design",
        language="en",
        url="https://en.wikipedia.org/wiki/Universal_design",
        content_type="general_knowledge",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="accessibility",
        interest_tags=("accessibility",),
        text=(
            "# Definition\n\n"
            "Universal design refers to the design of buildings, "
            "products, or environments to make them accessible to all "
            "people, regardless of age, disability, or other factors -- a "
            "rights-based approach aiming for designs usable by the "
            "broadest possible population.\n\n"
            "# The Seven Core Principles\n\n"
            "The Center for Universal Design at North Carolina State "
            "University established seven principles: equitable use, "
            "flexibility in use, simple and intuitive use, perceptible "
            "information, tolerance for error, low physical effort, and "
            "appropriate size and space for approach and use.\n\n"
            "# Application to Public Spaces and Transportation\n\n"
            "Practical applications include low-floor buses that kneel to "
            "ground level, level building entrances without stairs, "
            "sufficient turning space for wheelchairs, high-contrast "
            "signage, redundant visual and auditory information, lever "
            "door handles instead of twist knobs, and controls operable "
            "with low physical force.\n\n"
            "# Relevance for Travelers\n\n"
            "Accessible design removes participation barriers so people "
            "with varying abilities can navigate public environments "
            "independently -- the curb cut, originally designed for "
            "wheelchair users, is a well-known example that also benefits "
            "travelers with strollers or luggage."
        ),
    ),
    # --- Religious heritage / attractions: Blue Mosque -------------------
    SourceDocument(
        source_id="wiki_en_blue_mosque",
        title="Sultan Ahmed Mosque (Blue Mosque)",
        language="en",
        url="https://en.wikipedia.org/wiki/Sultan_Ahmed_Mosque",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_blue_mosque",
        district_id="district_fatih",
        topic_group="blue_mosque",
        interest_tags=("religious_heritage", "attractions", "history", "culture"),
        lat=41.0054, lon=28.9768, side="european", entity_kind="poi_candidate",
        text=(
            "# Construction History\n\n"
            "Construction began in 1609 and completed in 1617 under Sultan "
            "Ahmed I, built on the southeast side of the former Byzantine "
            "Hippodrome, adjacent to Hagia Sophia.\n\n"
            "# Architect and Design\n\n"
            "Sedefkar Mehmed Agha designed the mosque, synthesizing "
            "Classical Ottoman principles with Byzantine influences from "
            "Hagia Sophia.\n\n"
            "# Key Architectural Features\n\n"
            "The central dome is 23.5 metres in diameter and rises 43 "
            "metres, with a prayer hall measuring 64 by 72 metres and 260 "
            "windows. The mosque is famous for its six minarets -- "
            "according to folklore, the architect misheard 'gold "
            "minarets' as 'six minarets'. Over 21,000 handmade Iznik tiles "
            "cover the lower walls, from which the mosque's popular name "
            "derives.\n\n"
            "# Complex and Modern Status\n\n"
            "The surrounding complex includes Ahmed I's mausoleum, a "
            "madrasa, a hospital, and a market (arasta). A major "
            "restoration begun in 2018 concluded with the mosque "
            "reopening for worship in April 2023. It was included in the "
            "1985 UNESCO World Heritage designation of the Historic Areas "
            "of Istanbul and remains an active mosque and one of "
            "Istanbul's most visited sites."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_blue_mosque",
        title="Sultan Ahmet Camii",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Sultan_Ahmet_Camii",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_blue_mosque",
        district_id="district_fatih",
        topic_group="blue_mosque",
        interest_tags=("religious_heritage", "attractions", "history", "culture"),
        lat=41.0054, lon=28.9768, side="european", entity_kind="poi_candidate",
        text=(
            "# İnşa Tarihi\n\n"
            "Cami, I. Ahmed döneminde 1609-1617 yılları arasında inşa "
            "edilmiştir.\n\n"
            "# Mimar\n\n"
            "Sedefkâr Mehmed Ağa, yapının mimarıdır; tasarımında "
            "'azamet, ihtişam ve görkem' ilkelerini benimsemiştir.\n\n"
            "# Mimari Önem\n\n"
            "Merkezi kubbe 23,5 metre çapında ve 43 metre yüksekliğindedir, "
            "dört büyük ayak tarafından desteklenir.\n\n"
            "# Ayırt Edici Özellikler\n\n"
            "Cami, Türkiye'de altı minareye sahip ilk camidir; bu durum "
            "döneminde tartışma yaratmış, sorun Mekke'deki Mescid-i "
            "Haram'a yedinci bir minare eklenerek çözülmüştür. İç mekânda "
            "20.000'den fazla İznik çinisi bulunur ve 200'den fazla vitray "
            "pencere ışığın içeri dolmasını sağlar.\n\n"
            "# Dini ve Kültürel Önem\n\n"
            "Ayasofya'nın 1935'te müzeye dönüştürülmesinin ardından "
            "Sultanahmet Camii İstanbul'un başlıca camisi haline gelmiştir. "
            "1985 yılında UNESCO Dünya Mirası listesine dahil edilmiştir."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_blue_mosque",
        title="جامع السلطان أحمد (المسجد الأزرق)",
        language="ar",
        url="https://ar.wikipedia.org/wiki/%D8%A7%D9%84%D9%85%D8%B3%D8%AC%D8%AF_%D8%A7%D9%84%D8%A3%D8%B2%D8%B1%D9%82",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_blue_mosque",
        district_id="district_fatih",
        topic_group="blue_mosque",
        interest_tags=("religious_heritage", "attractions", "history", "culture"),
        lat=41.0054, lon=28.9768, side="european", entity_kind="poi_candidate",
        text=(
            "# تاريخ البناء\n\n"
            "بُني المسجد بين عامي 1609م و1616م بأمر من السلطان أحمد "
            "الأول، وافتُتح رسمياً عام 1616.\n\n"
            "# المهندس المعماري\n\n"
            "صممه المعماري العثماني سيدفكار محمد الآغا، متبعاً أفكار "
            "معلمه معمار سنان.\n\n"
            "# الخصائص المعمارية\n\n"
            "يبلغ طول المسجد 72 متراً وعرضه 64 متراً، ويضم خمس قباب "
            "رئيسية وثماني قباب صغيرة، ويتسع لعشرة آلاف مصلٍ. يشتهر "
            "المسجد بمآذنه الست، وهو رقم استثنائي نتج - بحسب الروايات "
            "الشعبية - عن التباس في الترجمة بين 'مآذن ذهبية' و'ست "
            "مآذن'.\n\n"
            "# الأهمية الدينية\n\n"
            "يُعتبر المسجد آخر المساجد العظيمة في فترة العمارة العثمانية "
            "الكلاسيكية، ومن أهم المعالم الدينية والسياحية في إسطنبول."
        ),
    ),
    # --- Religious heritage: Suleymaniye Mosque --------------------------
    SourceDocument(
        source_id="wiki_en_suleymaniye_mosque",
        title="Suleymaniye Mosque",
        language="en",
        url="https://en.wikipedia.org/wiki/S%C3%BCleymaniye_Mosque",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_suleymaniye_mosque",
        district_id="district_fatih",
        topic_group="suleymaniye_mosque",
        interest_tags=("religious_heritage", "attractions", "history"),
        lat=41.0165, lon=28.9639, side="european", entity_kind="poi_candidate",
        text=(
            "# Construction History\n\n"
            "The Suleymaniye Mosque was commissioned by Suleiman the "
            "Magnificent and designed by architect Mimar Sinan. "
            "Construction began in 1550 and the building was inaugurated "
            "in 1557, on the site of the former Ottoman palace (Eski "
            "Saray).\n\n"
            "# Architect\n\n"
            "Sinan created the mosque after impressing Suleiman with his "
            "earlier Sehzade Mosque; Sinan's own tomb sits just outside "
            "the complex's north wall.\n\n"
            "# Architectural Significance\n\n"
            "The prayer hall measures nearly 58.5 by 57.5 metres, "
            "dominated by a central dome 53 metres high with a 26.5-metre "
            "diameter, following the Hagia Sophia model of semi-domes "
            "front and back. Four minarets occupy the courtyard corners, "
            "the tallest pair rising 76 metres, with ten balconies total "
            "-- said to reflect Suleiman as the Ottoman Empire's tenth "
            "sultan.\n\n"
            "# The Kulliye and Mausoleums\n\n"
            "The complex originally included four madrasas, a hospital, "
            "public kitchen, caravanserai, and bathhouse. An enclosed "
            "cemetery behind the mosque holds the mausoleums of Suleiman "
            "and his wife Hurrem Sultan (Roxelana). It forms part of the "
            "1985 UNESCO World Heritage designation of the Historic Areas "
            "of Istanbul."
        ),
    ),
    # --- Religious heritage / art: Chora Church ---------------------------
    SourceDocument(
        source_id="wiki_en_chora_church",
        title="Chora Church (Kariye)",
        language="en",
        url="https://en.wikipedia.org/wiki/Chora_Church",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_chora_church",
        district_id="district_fatih",
        topic_group="chora_church",
        interest_tags=("religious_heritage", "attractions", "history", "art"),
        lat=41.0311, lon=28.9391, side="european", entity_kind="poi_candidate",
        text=(
            "# Origins\n\n"
            "The Chora Church began as a 4th-century monastery complex "
            "outside Constantinople's walls, later incorporated within "
            "the city when the Theodosian walls were built in 413-414.\n\n"
            "# 14th-Century Mosaics\n\n"
            "Most of the present structure dates from 1077-1081; Byzantine "
            "statesman Theodore Metochites endowed the church with its "
            "celebrated mosaics and frescoes between roughly 1310 and "
            "1317, considered the finest example of the Palaeologian "
            "Renaissance.\n\n"
            "# Transitions: Church, Mosque, Museum\n\n"
            "Around 1500 the church was converted into a mosque (Kariye "
            "Camii) under Sultan Bayezid II; the mosaics were plastered "
            "over rather than destroyed, inadvertently preserving them. "
            "In 1945 the site was secularized and became a museum, opening "
            "to the public in 1958 after restoration by the Byzantine "
            "Institute of America. In 2020 the building was reconverted to "
            "an active mosque.\n\n"
            "# Artistic Significance\n\n"
            "The interior showcases exceptional Late Byzantine art across "
            "its narthex, naos, and parecclesion, including the renowned "
            "Koimesis (Dormition) mosaic and the Anastasis fresco. The site "
            "is part of the 1985 UNESCO World Heritage designation of the "
            "Historic Areas of Istanbul."
        ),
    ),
    # --- Religious heritage: Eyup Sultan Mosque ---------------------------
    SourceDocument(
        source_id="wiki_en_eyup_sultan_mosque",
        title="Eyup Sultan Mosque",
        language="en",
        url="https://en.wikipedia.org/wiki/Eyup_Sultan_Mosque",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_eyup_sultan_mosque",
        district_id="district_eyup",
        topic_group="eyup",
        interest_tags=("religious_heritage", "history", "attractions", "seasonal_planning"),
        lat=41.0479, lon=28.9339, side="european", entity_kind="poi_candidate",
        text=(
            "# History\n\n"
            "The mosque complex was built in 1458 by Sultan Mehmed II, "
            "five years after Constantinople's conquest, then rebuilt by "
            "Sultan Selim III between 1798 and 1800 after the original "
            "structure deteriorated.\n\n"
            "# Religious Significance\n\n"
            "The mosque honors Abu Ayyub al-Ansari, a companion of the "
            "Prophet Muhammad believed buried at this site following the "
            "first Arab siege of Constantinople in the 670s. His "
            "mausoleum sits on the north side of the courtyard. Ottoman "
            "sultans traditionally proceeded along the Culus Yolu "
            "(Accession Way) to be girded with the Sword of Osman here "
            "upon their accession.\n\n"
            "# Architecture\n\n"
            "The reconstructed mosque follows an octagonal baldaquin "
            "design with a central dome surrounded by semi-domes, using "
            "white stone and marble columns bound with brass. The "
            "mausoleum displays Iznik tile panels dating to around "
            "1580.\n\n"
            "# Modern Pilgrimage\n\n"
            "Pilgrims visit year-round, with particular gatherings during "
            "Ramadan and for boys' circumcision ceremonies."
        ),
    ),
    # --- Photography / nature: Pierre Loti Hill --------------------------
    SourceDocument(
        source_id="wiki_en_pierre_loti_hill",
        title="Pierre Loti Hill",
        language="en",
        url="https://en.wikipedia.org/wiki/Eyup",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_pierre_loti_hill",
        district_id="district_eyup",
        topic_group="eyup",
        interest_tags=("photography", "nature"),
        lat=41.0563, lon=28.9339, side="european", entity_kind="poi_candidate",
        preferred_period="afternoon",
        text=(
            "# Location\n\n"
            "Pierre Loti Hill overlooks the Golden Horn from the Eyup "
            "district, one of Istanbul's most historic and religiously "
            "significant areas, stretching from the Golden Horn to the "
            "Black Sea shore.\n\n"
            "# The Cable Car and Cafe\n\n"
            "A gondola lift carries visitors from the Golden Horn shore up "
            "to the Pierre Loti Cafe, named after 19th-century French "
            "author Julien Viaud (pen name Pierre Loti), who was known to "
            "frequent the area. The outdoor cafe offers panoramic views "
            "over the Golden Horn and has become a popular viewpoint for "
            "photography, especially toward sunset.\n\n"
            "# Historical Character of Eyup\n\n"
            "Eyup was known as Kosmidion in the Byzantine period, home to "
            "a monastery of Saints Cosmas and Damian, and later "
            "industrialized in the 17th-18th centuries with facilities "
            "such as the Feshane fez factory along the Golden Horn."
        ),
    ),
    # --- Photography / transportation: Bosphorus Bridge -------------------
    SourceDocument(
        source_id="wiki_en_bosphorus_bridge",
        title="15 July Martyrs Bridge (Bosphorus Bridge)",
        language="en",
        url="https://en.wikipedia.org/wiki/15_July_Martyrs_Bridge",
        content_type="attraction",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_bosphorus_bridge",
        district_id="district_besiktas",
        topic_group="bosphorus_bridge",
        interest_tags=("photography", "transportation", "attractions"),
        lat=41.0458, lon=29.0344, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        text=(
            "# Identification\n\n"
            "Officially the 15 July Martyrs Bridge, commonly called the "
            "Bosphorus Bridge, it is the oldest and southernmost of the "
            "three suspension bridges crossing the Bosphorus, connecting "
            "Ortakoy (Europe) with Beylerbeyi (Asia).\n\n"
            "# Engineering Specifications\n\n"
            "The bridge has a total length of 1,560 metres, a main span "
            "of 1,074 metres, and 165-metre-tall towers, with a deck "
            "clearance of 64 metres above sea level.\n\n"
            "# Construction Timeline\n\n"
            "Design was contracted in 1968 to British firm Freeman Fox & "
            "Partners; construction began in February 1970 and the bridge "
            "opened on 30 October 1973, ranking as the fourth-longest "
            "suspension bridge span in the world at the time.\n\n"
            "# Renaming and Illumination\n\n"
            "The bridge was renamed in 2016 to honor victims of the failed "
            "coup attempt of 15 July 2016. Since April 2007, a "
            "computerized LED lighting system illuminates the bridge at "
            "night with changing colors, making it a well-known nighttime "
            "photography subject from both shores."
        ),
    ),
    # --- Neighborhoods / attractions: Sultanahmet -------------------------
    SourceDocument(
        source_id="wiki_en_sultanahmet",
        title="Sultanahmet",
        language="en",
        url="https://en.wikipedia.org/wiki/Sultanahmet,_Fatih",
        content_type="neighborhood",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_sultanahmet",
        district_id="district_fatih",
        topic_group="sultanahmet",
        interest_tags=("neighborhoods", "attractions", "religious_heritage", "history"),
        lat=41.0055, lon=28.9769, side="european", entity_kind="poi_candidate",
        text=(
            "# Historical Significance\n\n"
            "Sultanahmet is the core of Istanbul's historic peninsula, "
            "coinciding with the site of old Constantinople, the Byzantine "
            "capital.\n\n"
            "# Byzantine and Ottoman Heritage\n\n"
            "Following the Ottoman conquest by Mehmed II, Islamic scholars "
            "transformed the area's major churches into mosques, "
            "reshaping the neighborhood's religious character.\n\n"
            "# Major Landmarks\n\n"
            "The district contains Hagia Sophia, the Blue Mosque, Topkapi "
            "Palace, and the Basilica Cistern, an underground Byzantine "
            "structure -- one of the densest concentrations of major "
            "historic sites anywhere in the city.\n\n"
            "# Modern Character\n\n"
            "A tramway runs from the docks at Sirkeci through Sultanahmet, "
            "and the neighborhood functions as Istanbul's primary tourist "
            "district while remaining, in local terms, emblematic of 'the "
            "real Istanbul of old times.'"
        ),
    ),
    # --- Neighborhoods / photography: Balat -------------------------------
    SourceDocument(
        source_id="wiki_en_balat",
        title="Balat",
        language="en",
        url="https://en.wikipedia.org/wiki/Balat,_Istanbul",
        content_type="neighborhood",
        retrieved_at="2026-08-20T00:00:00Z",
        poi_id="poi_balat",
        district_id="district_fatih",
        topic_group="balat",
        interest_tags=("neighborhoods", "culture", "photography", "history"),
        lat=41.0292, lon=28.9491, side="european", entity_kind="poi_candidate",
        preferred_period="morning",
        text=(
            "# Overview\n\n"
            "Balat is a neighborhood in Fatih district on the western "
            "shore of the Golden Horn, sitting between Fener and "
            "Ayvansaray.\n\n"
            "# Jewish Heritage\n\n"
            "From the late 15th century, Balat became the center of "
            "Istanbul's Jewish community after Sultan Bayezid II offered "
            "citizenship to Jews fleeing the 1492 Alhambra Decree. At its "
            "peak the neighborhood held 18 synagogues; three remain "
            "active, including the 15th-century Ahrida Synagogue.\n\n"
            "# Multicultural Past\n\n"
            "Balat also housed Greek, Armenian, and Bulgarian communities, "
            "including the Bulgarian Iron Church, a prefabricated iron "
            "structure shipped via the Danube and restored in 2018.\n\n"
            "# Contemporary Character\n\n"
            "In the 2020s Balat became one of Istanbul's most "
            "photographed districts, as many historic houses were "
            "repainted in bright colors and converted into cafes and "
            "guesthouses. It was inscribed on the UNESCO World Heritage "
            "List in 1985 as part of Istanbul's Historic Areas."
        ),
    ),
    # --- Transportation: ferries ------------------------------------------
    SourceDocument(
        source_id="wiki_en_sehir_hatlari",
        title="Sehir Hatlari (Istanbul city ferries)",
        language="en",
        url="https://en.wikipedia.org/wiki/%C5%9Eehir_Hatlar%C4%B1",
        content_type="transport",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="ferries",
        interest_tags=("transportation",),
        text=(
            "# History and Founding\n\n"
            "Sehir Hatlari ('City Lines') is Istanbul's oldest and largest "
            "ferry operator, founded in 1851 as the Sirket-i Hayriye "
            "('The Goodwill Company') during the Ottoman period, later "
            "renamed in the early republican era.\n\n"
            "# Operations and Scale\n\n"
            "As a municipally owned enterprise, Sehir Hatlari operates "
            "roughly 30 ferries across 53 piers on 32 lines, serving both "
            "shores of the Bosphorus and the Princes' Islands. In 2023 the "
            "company carried 40 million passengers.\n\n"
            "# Routes\n\n"
            "Three main route categories exist: inner-city lines, "
            "Bosphorus crossings between European and Asian districts, "
            "and seasonal service to the Princes' Islands, alongside "
            "dedicated Bosphorus sightseeing tours.\n\n"
            "# Strategic Importance\n\n"
            "Before the first bridge opened in 1973, ferries were the only "
            "way to cross between the European and Asian halves of the "
            "city, and they remain a defining part of both daily commuting "
            "and tourism."
        ),
    ),
    # --- Transportation: Istanbul Airport ----------------------------------
    SourceDocument(
        source_id="wiki_en_istanbul_airport",
        title="Istanbul Airport",
        language="en",
        url="https://en.wikipedia.org/wiki/Istanbul_Airport",
        content_type="transport",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="istanbul_airport",
        interest_tags=("transportation",),
        text=(
            "# Opening and Timeline\n\n"
            "Istanbul Airport (IST) opened on 29 October 2018, and all "
            "scheduled commercial passenger flights transferred from "
            "Ataturk Airport on 6 April 2019.\n\n"
            "# Scale\n\n"
            "It is the largest airport in Turkey and the busiest in "
            "Europe by passenger traffic, with a main terminal building "
            "of 1,440,000 square metres -- the world's third-largest "
            "airport terminal building.\n\n"
            "# Role as Primary Gateway\n\n"
            "The airport is the hub for Turkish Airlines and serves as the "
            "metropolitan area's main international gateway, located "
            "about 35 kilometres from the city center in the Arnavutkoy "
            "district on the European side.\n\n"
            "# Ground Transportation\n\n"
            "The airport connects to Istanbul's public transit system via "
            "the M11 metro line at the Istanbul Havalimani station."
        ),
    ),
    # --- Etiquette / food: Turkish tea --------------------------------------
    SourceDocument(
        source_id="wiki_en_turkish_tea",
        title="Turkish tea",
        language="en",
        url="https://en.wikipedia.org/wiki/Turkish_tea",
        content_type="etiquette",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="turkish_tea",
        interest_tags=("etiquette", "food"),
        text=(
            "# Historical Development\n\n"
            "Tea cultivation in Turkey began in Rize Province in 1912; a "
            "Central Tea Nursery was established in 1924, and large-scale "
            "production stabilized between 1939 and 1945. By the mid-20th "
            "century, tea had become Turkey's beverage of choice.\n\n"
            "# Preparation Method\n\n"
            "Turkish tea uses a two-pot system called a caydanlik: water "
            "boils in the lower pot while loose tea leaves steep in the "
            "upper pot, producing a concentrated brew that is diluted per "
            "cup to taste, either strong (koyu) or weak (acik).\n\n"
            "# Serving Traditions\n\n"
            "Tea is served in small tulip-shaped glasses called ince "
            "belli, held by the rim, typically with beet sugar cubes "
            "between three and five in the afternoon.\n\n"
            "# Social and Cultural Role\n\n"
            "Offering tea to guests is a core part of Turkish hospitality. "
            "Tea gardens and traditional kiraathane (coffeehouse-style "
            "social spaces) serve as gathering places, particularly common "
            "in Istanbul's Sultanahmet and Taksim neighborhoods."
        ),
    ),
    # --- Food: Turkish cuisine ----------------------------------------------
    SourceDocument(
        source_id="wiki_en_turkish_cuisine",
        title="Turkish cuisine",
        language="en",
        url="https://en.wikipedia.org/wiki/Turkish_cuisine",
        content_type="etiquette",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="turkish_cuisine",
        interest_tags=("food",),
        text=(
            "# Historical Influences\n\n"
            "Turkish cuisine synthesizes Central Asian culinary traditions "
            "with Mediterranean and Middle Eastern influences, and shaped "
            "the food cultures of former Ottoman territories across the "
            "Balkans, Levant, North Africa, and the South Caucasus.\n\n"
            "# Regional Varieties\n\n"
            "Istanbul, Bursa, Izmir, and central Anatolia inherit Ottoman "
            "court traditions with moderate spice use and abundant "
            "vegetable stews; the Black Sea region favors fish, "
            "particularly anchovy; the southeast (Urfa, Gaziantep, Adana) "
            "is known for kebab varieties and dough-based desserts like "
            "baklava and kunefe.\n\n"
            "# Meal Structure\n\n"
            "Turkish breakfast (kahvalti, literally 'before coffee') "
            "emphasizes variety: cheese, butter, olives, eggs, tomatoes, "
            "and bread. A typical meal begins with soup, followed by "
            "vegetable or meat dishes with rice or bulgur.\n\n"
            "# Iconic Dishes\n\n"
            "Kebab varieties include Adana kebap, Iskender kebap (created "
            "1867 in Bursa), and doner kebap. Meze appetizers include "
            "cacik and stuffed vegetables. Baklava, made with pistachios "
            "or walnuts, is the best-known dessert; borek, pide, and "
            "lahmacun are everyday staples."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_turkish_cuisine",
        title="Türk mutfağı",
        language="tr",
        url="https://tr.wikipedia.org/wiki/T%C3%BCrk_mutfa%C4%9F%C4%B1",
        content_type="etiquette",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="turkish_cuisine",
        interest_tags=("food",),
        text=(
            "# Tarihi Etkiler\n\n"
            "Türk mutfağı, Orta Asya, Selçuklu ve Beylikler ile Osmanlı "
            "kültürünün mirasçısıdır ve Balkan ile Orta Doğu "
            "mutfaklarıyla karşılıklı etkileşim içinde gelişmiştir.\n\n"
            "# Bölgesel Çeşitlilik\n\n"
            "Türk mutfağı coğrafi bölgelere göre belirgin farklılıklar "
            "gösterir: Karadeniz, Ege, Orta Anadolu, Doğu Anadolu ve "
            "Güneydoğu Anadolu mutfakları kendine özgü lezzetlere "
            "sahiptir.\n\n"
            "# İkonik Yemekler\n\n"
            "Kebaplar arasında döner kebap, cağ kebabı, Adana kebabı ve "
            "Urfa kebabı öne çıkar. Meze ve salatalarda çoban salatası, "
            "cacık ve çiğ köfte yaygındır. Baklava, kadayıf ve künefe "
            "uluslararası ün kazanmış tatlılardır; sütlü tatlılar arasında "
            "muhallebi, kazandibi ve sütlaç sayılabilir.\n\n"
            "# Yemek Gelenekleri\n\n"
            "Bayram sofraları, geniş aile üyelerinin ve komşuların davet "
            "edildiği geleneksel önemli toplumsal etkinliklerdir."
        ),
    ),
    # --- Seasonal planning: climate ----------------------------------------
    SourceDocument(
        source_id="wiki_en_climate_istanbul",
        title="Climate of Istanbul / Marmara Region",
        language="en",
        url="https://en.wikipedia.org/wiki/Climate_of_Turkey",
        content_type="general_knowledge",
        retrieved_at="2026-08-20T00:00:00Z",
        topic_group="climate",
        interest_tags=("seasonal_planning", "nature"),
        text=(
            "# Climate Classification\n\n"
            "The Marmara Sea region, which includes Istanbul, has a "
            "complex, transitional, often microclimatic climate, "
            "classified as Koppen Csa/Csb/Cfa/Cfb -- mild-temperate rather "
            "than subtropical.\n\n"
            "# Temperature Patterns\n\n"
            "Annual average temperature runs 12-15 degrees Celsius, with "
            "summer means of 20-25 degrees Celsius and winter means of "
            "2-6 degrees Celsius, cooling further inland from the "
            "coast.\n\n"
            "# Precipitation\n\n"
            "Annual rainfall totals 600-1,100 millimetres. Winters are "
            "very cloudy, with rainy days far exceeding much of Europe; "
            "snow falls occasionally, often as sea-effect snow.\n\n"
            "# Seasonal Character\n\n"
            "Summers are moderately dry but feature occasional, sometimes "
            "severe, thunderstorms, with thunderstorm activity peaking in "
            "early and late summer -- a relevant planning factor for "
            "outdoor-heavy itineraries in spring and autumn shoulder "
            "seasons."
        ),
    ),
    # === RAG-FIRST SYSTEM B R.1 CORRECTION: Arabic/Turkish/official-source
    # expansion (19 further documents, fetched and paraphrased live during
    # this correction pass) ================================================
    # --- Arabic: shopping/attractions -----------------------------------
    SourceDocument(
        source_id="wiki_ar_grand_bazaar",
        title="البازار الكبير (إسطنبول)",
        language="ar",
        url="https://ar.wikipedia.org/wiki/السوق_الكبير_(إسطنبول)",
        content_type="attraction",
        poi_id="poi_grand_bazaar",
        district_id="district_fatih",
        topic_group="grand_bazaar",
        interest_tags=("history", "shopping", "attractions", "culture"),
        lat=41.0106, lon=28.9681, side="european", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# تاريخ التأسيس\n\n"
            "بدأ بناء البازار الكبير خلال شتاء عام 1455، بعد فترة وجيزة من "
            "فتح القسطنطينية، بأمر من السلطان محمد الفاتح. انتهى التشييد "
            "عام 1460-1461، حيث وُقِف المبنى على جامع آيا صوفيا.\n\n"
            "# الحجم والمساحة\n\n"
            "يشغل البازار 30,700 متر مربع ويضم 61 شارعاً مغطى وأكثر من "
            "4,000 متجر. يستقطب يومياً ما بين 250 ألف و400 ألف زائر، وكان "
            "الأكثر زيارة بـ91,250,000 زائر سنوياً عام 2014.\n\n"
            "# السلع والبضائع\n\n"
            "تاريخياً، تخصص البازار في تجارة المنسوجات والمجوهرات. احتوى "
            "على مناطق مخصصة للملابس والكتب والسلع المستعملة، مع أسواق "
            "متنوعة في الخانات المحيطة.\n\n"
            "# الأهمية الثقافية والاقتصادية\n\n"
            "يُنظر للبازار على أنه أحد مراكز التسوق الأولى في العالم. كان "
            "مركزاً لتجارة البحر الأبيض المتوسط وعكس ازدهار إسطنبول "
            "الاقتصادي منذ العصر العثماني المبكر."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_galata_tower",
        title="برج غلطة",
        language="ar",
        url="https://ar.wikipedia.org/wiki/برج_غلطة",
        content_type="attraction",
        poi_id="poi_galata_tower",
        district_id="district_beyoglu",
        topic_group="galata_tower",
        interest_tags=("history", "attractions", "photography"),
        lat=41.0256, lon=28.9741, side="european", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# تاريخ البناء\n\n"
            "تم إنشاء البرج عام 1348 خلال توسع مستعمرة جنوة في القسطنطينية. "
            "حل محل برج بيزنطي أقدم دُمر سنة 1204 أثناء الحملة الصليبية "
            "الرابعة.\n\n"
            "# الخصائص المعمارية\n\n"
            "البرج بتصميم على الطراز الرومانسكي بشكل أسطواني مع سقف "
            "مخروطي. يبلغ ارتفاعه 66.90 متراً (62.59 متراً بدون الزخرفة)، "
            "وقطره الخارجي 16.45 متراً، ويتكون من تسعة طوابق ببناية من "
            "الحجر.\n\n"
            "# الاستخدامات عبر التاريخ\n\n"
            "في العهد العثماني استُخدم كبرج مراقبة لاكتشاف الحرائق. شهد "
            "البرج في القرن السابع عشر محاولة الطيران التاريخية لهزارفن "
            "أحمد جلبي. بين عامي 1965 و1967 أعيد ترميمه وفُتح للعامة "
            "بهيكل خرساني حديث.\n\n"
            "# الأهمية السياحية والبانورامية\n\n"
            "يوفر البرج إطلالة بانورامية على إسطنبول والبوسفور، ويحتوي "
            "على مطعم ومصعدين ينقلان الزوار إلى الأعلى."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_istiklal_avenue",
        title="شارع الاستقلال",
        language="ar",
        url="https://ar.wikipedia.org/wiki/شارع_الاستقلال",
        content_type="attraction",
        poi_id="poi_istiklal_avenue",
        district_id="district_beyoglu",
        topic_group="istiklal_avenue",
        interest_tags=("shopping", "nightlife", "neighborhoods", "culture"),
        lat=41.0328, lon=28.9784, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# النشأة والأهمية التاريخية\n\n"
            "شارع الاستقلال يُعتبر أحد أشهر شوارع إسطنبول بطول 1,600 متر "
            "تقريباً. نشأ كنقطة انطلاق لمشروع تحديث تركيا في عهد السلطان "
            "عبد المجيد.\n\n"
            "# الخصائص المعمارية والعمرانية\n\n"
            "يضم الشارع مباني أثرية ومحلات ملابس ومعارض ومكتبات وسينمات "
            "وملاهي ليلية. كما يخترقه ترام قديم أقيم منذ العهد العثماني، "
            "مما يعكس طابعه التاريخي.\n\n"
            "# الحياة التجارية والدبلوماسية\n\n"
            "يستقطب الشارع ملايين الزوار يومياً ويضم عدة قنصليات منها "
            "الأمريكية والفرنسية واليونانية والبريطانية."
        ),
    ),
    # --- Arabic: religious heritage --------------------------------------
    SourceDocument(
        source_id="wiki_ar_suleymaniye_mosque",
        title="جامع السليمانية",
        language="ar",
        url="https://ar.wikipedia.org/wiki/جامع_السليمانية",
        content_type="attraction",
        poi_id="poi_suleymaniye_mosque",
        district_id="district_fatih",
        topic_group="suleymaniye_mosque",
        interest_tags=("religious_heritage", "attractions", "history"),
        lat=41.0165, lon=28.9639, side="european", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# تاريخ البناء\n\n"
            "بدأ بناء جامع السليمانية عام 1550 واكتمل عام 1557، ويقع على "
            "التلة الثالثة من تلال إسطنبول السبعة في قلب مجمع "
            "السليمانية.\n\n"
            "# المهندس المعماري\n\n"
            "صمم المسجد معمار سنان، أبرز المهندسين في العهد العثماني. "
            "يضم المجمع أيضاً ضريح المهندس المعماري سنان.\n\n"
            "# الخصائص المعمارية الرئيسية\n\n"
            "يتميز المسجد بقبة مركزية مزودة بقبتين نصفيتين وقبب في "
            "الزوايا، استلهم تصميمها من كاتدرائية آيا صوفيا. يتكون من "
            "حجر ناري وحجر رملي ورخام وطوب ورصاص وحديد.\n\n"
            "# الأهمية الدينية والعمرانية\n\n"
            "يتضمن مجمع السليمانية مقبرة وأربع مدارس ومستشفى وفندق "
            "ومئذنة ودكاكين وحمّامات ومدرسة قرآنية."
        ),
    ),
    # --- Arabic: neighborhoods --------------------------------------------
    SourceDocument(
        source_id="wiki_ar_kadikoy",
        title="قاضي كوي",
        language="ar",
        url="https://ar.wikipedia.org/wiki/قاضي_كوي",
        content_type="neighborhood",
        poi_id="poi_kadikoy_district",
        district_id="district_kadikoy",
        topic_group="kadikoy",
        interest_tags=("neighborhoods", "culture", "food", "nightlife"),
        lat=40.9903, lon=29.0275, side="asian", entity_kind="poi_candidate",
        preferred_period="evening",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# التاريخ والأصول\n\n"
            "يُعرف حي قاضي كوي قديماً باسم خلقدون. أطلق العثمانيون عليه "
            "هذا الاسم الجديد لأنها كانت القرية التي يقيم فيها القاضي "
            "قرب القسطنطينية.\n\n"
            "# الأهمية الدينية التاريخية\n\n"
            "تحمل المنطقة أهمية دينية كبرى في تاريخ المسيحية، إذ شهدت "
            "سنة 451 ميلادية انعقاد مجمع خلقيدونية الذي ساهم في حدوث "
            "انقسامات كنسية مهمة.\n\n"
            "# الموقع والخصائص الجغرافية\n\n"
            "يقع الحي على الشاطئ الشمالي من بحر مرمرة في مواجهة المركز "
            "القديم التاريخي للمدينة على الجانب الآسيوي من إسطنبول.\n\n"
            "# الطابع الثقافي والحياة الليلية\n\n"
            "تتمتع المنطقة بحضور ثقافي واضح، حيث توصف كمركز ثقافي في "
            "الجانب الأناضولي من إسطنبول. تضم دور سينما وحانات ومكتبات، "
            "مما يجعلها وجهة ثقافية وترفيهية متميزة."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_uskudar",
        title="أسكودار",
        language="ar",
        url="https://ar.wikipedia.org/wiki/أسكودار",
        content_type="neighborhood",
        poi_id="poi_uskudar_district",
        district_id="district_uskudar",
        topic_group="uskudar",
        interest_tags=("neighborhoods", "religious_heritage", "culture"),
        lat=41.0214, lon=29.0161, side="asian", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# الموقع والأهمية الجغرافية\n\n"
            "أسكودار إحدى بلديات محافظة إسطنبول الكبرى، تقع على الضفة "
            "الشرقية لمضيق البسفور مقابل البلدة القديمة في إمينونو، مما "
            "جعلها منطقة سكنية قديمة وحيوية.\n\n"
            "# الطابع الثقافي والاجتماعي\n\n"
            "يتميز حي أسكودار برائحة البحر والأمواج والقوارب وطيور "
            "النورس التي تجعله وجهة مميزة. تتركز فيه عدة جامعات، مما "
            "يضفي عليه طابعاً تعليمياً ملحوظاً.\n\n"
            "# المساحات الخضراء والمعالم\n\n"
            "يضم الحي عدداً من الحدائق، أبرزها بستان فتحي باشا، حديقة "
            "كبيرة على التلال المطلة على شاطئ البوسفور، إضافة إلى كثافة "
            "عالية من المباني التاريخية والمساجد العثمانية."
        ),
    ),
    # --- Arabic: islands/nature --------------------------------------------
    SourceDocument(
        source_id="wiki_ar_princes_islands",
        title="جزر الأمراء",
        language="ar",
        url="https://ar.wikipedia.org/wiki/جزر_الأمراء",
        content_type="attraction",
        poi_id="poi_buyukada",
        district_id="district_adalar",
        topic_group="buyukada",
        interest_tags=("islands", "nature", "family"),
        lat=40.8767, lon=29.1206, side="asian", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# النشأة والتسمية\n\n"
            "تعود التسمية إلى العصور البيزنطية، حيث كانت تُستخدم كمكان "
            "لنفي الأمراء الذين كانوا يُعتبرون تهديداً للعرش. استمرت هذه "
            "الوظيفة بعد الفتح العثماني عام 1453.\n\n"
            "# الموقع والمساحة\n\n"
            "تقع جزر الأمراء في بحر مرمرة قبالة إسطنبول، وتشكل أرخبيلاً "
            "بمساحة إجمالية 15.85 كيلومتراً مربعاً.\n\n"
            "# الجزر الرئيسية\n\n"
            "تتضمن المنطقة تسع جزر، أبرزها بويوك أدا (الأكبر والأكثر "
            "زيارة)، وهيبيلي أدا المشهورة بمساحاتها الخضراء ومعهد "
            "البحرية العسكري، وبورغاز أدا وكينالي أدا الأصغر بطابع "
            "هادئ.\n\n"
            "# حظر السيارات\n\n"
            "منطقة خالية من السيارات العادية، حيث يتم التنقل بالدراجات "
            "الهوائية والعربات الكهربائية.\n\n"
            "# الوصول والسياحة\n\n"
            "تتوفر خدمات عبّارات منتظمة من محطات ميناء إسطنبول، والرحلة "
            "تستغرق حوالي ساعة ونصف. تجذب الجزر السياح لمعالمها التاريخية "
            "والطبيعية والمطاعم البحرية."
        ),
    ),
    # --- Arabic: food -------------------------------------------------------
    SourceDocument(
        source_id="wiki_ar_turkish_cuisine",
        title="المطبخ التركي",
        language="ar",
        url="https://ar.wikipedia.org/wiki/المطبخ_التركي",
        content_type="etiquette",
        topic_group="turkish_cuisine",
        interest_tags=("food",),
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# التأثيرات التاريخية\n\n"
            "تطور المطبخ التركي عبر فترات زمنية طويلة، متخذاً شكله "
            "الحالي بعد تداخلات ثقافية عديدة على مر القرون ابتداءً من "
            "الفترتين السلجوقية والعثمانية.\n\n"
            "# التنوع الإقليمي\n\n"
            "المناطق الساحلية تستخدم الأسماك بشكل واسع، خاصة في منطقة "
            "البحر الأسود. المناطق الغربية تعتمد على زيت الزيتون. الجنوب "
            "الشرقي يشتهر بتنوع الكباب والمقبلات والحلويات القائمة على "
            "العجين. وسط الأناضول معروف بأطباق مثل الكشكك والمنتي.\n\n"
            "# الأطباق الشهيرة\n\n"
            "من المعجنات: السميت واللحماجون والبوريك. من الكباب: الدونر "
            "والإسكندر والشيش كباب وكباب أضنة الحار وكباب أورفة. من "
            "الحلويات: البقلاوة والملبن واللوقوم والكنافة.\n\n"
            "# المكونات الأساسية\n\n"
            "يعتمد المطبخ على لحم الضأن والدجاج والأرز والخضروات (خاصة "
            "الباذنجان) والمكسرات والتوابل المتنوعة."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_turkish_tea",
        title="الشاي التركي",
        language="ar",
        url="https://ar.wikipedia.org/wiki/الشاي_التركي",
        content_type="etiquette",
        topic_group="turkish_tea",
        interest_tags=("etiquette", "food"),
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# التاريخ والنشأة\n\n"
            "بدأ التاريخ الحديث للشاي التركي في عشرينيات القرن العشرين، "
            "حين أُحضرت أول شتلات شاي من روسيا. تطورت الزراعة لاحقاً في "
            "مناطق البحر الأسود الشرقية، خاصة محافظة ريزا.\n\n"
            "# الأهمية الاقتصادية والاستهلاكية\n\n"
            "أنتجت تركيا 205,500 طن من الشاي عام 2004 (6.4٪ من إجمالي "
            "إنتاج الشاي العالمي)، واحتلت المركز الأول عالمياً في "
            "استهلاك الشاي بمعدل 2.5 كيلوغرام للفرد سنوياً.\n\n"
            "# طريقة التحضير التقليدية\n\n"
            "يستخدم الأتراك أباريق شاي مزدوجة خاصة: يوضع الماء في الإناء "
            "الكبير، والشاي في الإناء الآخر. كمية الماء تتحكم بقوة "
            "الشاي.\n\n"
            "# التقديم والثقافة\n\n"
            "يُقدَّم الشاي التركي في أكواب زجاجية ذات خصر رفيع مع السكر، "
            "ويعكس هذا الأسلوب أهمية الضيافة والمجاملة في الثقافة "
            "التركية."
        ),
    ),
    # --- Arabic: transportation / etiquette --------------------------------
    SourceDocument(
        source_id="wiki_ar_transport",
        title="مترو إسطنبول",
        language="ar",
        url="https://ar.wikipedia.org/wiki/مترو_إسطنبول",
        content_type="transport",
        topic_group="transport",
        interest_tags=("transportation",),
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# نبذة تاريخية\n\n"
            "افتتح خط إم-1، أقدم جزء من النظام الحديث، عام 1989. لكن "
            "تاريخ النقل تحت الأرض في إسطنبول أقدم بكثير: خط تونيل دخل "
            "الخدمة في 17 يناير 1875، ويُعتبر ثاني أقدم نظام من نوعه "
            "عالمياً بعد لندن.\n\n"
            "# شبكة الخطوط الحالية\n\n"
            "يضم النظام الحالي عدة خطوط رئيسية (إم-1 حتى إم-6) تغطي "
            "أكثر من 105 كيلومتر و82 محطة، تربط الجانبين الأوروبي "
            "والآسيوي عبر خط مرمراي.\n\n"
            "# الاتصالات مع وسائل النقل الأخرى\n\n"
            "يتصل مترو إسطنبول مع خط مرمراي (العابر تحت البوسفور)، وخط "
            "السكك الحديدية المعلقة إف-1، وشبكة الترام، والمتروباص.\n\n"
            "# بطاقة إسطنبول كارت\n\n"
            "يمكن للركاب استعمال آلات بيع التذاكر أو الأكشاك لدفع ثمن "
            "رحلتهم، إما بالتذاكر أو بتخزين رصيد في بطاقة كارت "
            "إسطنبول."
        ),
    ),
    SourceDocument(
        source_id="wiki_ar_culture",
        title="ثقافة تركيا",
        language="ar",
        url="https://ar.wikipedia.org/wiki/ثقافة_تركيا",
        content_type="etiquette",
        topic_group="culture",
        interest_tags=("etiquette", "culture"),
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# نظرة عامة على الثقافة التركية\n\n"
            "تجمع الثقافة التركية بين مجموعة متنوعة من التقاليد "
            "المشتقة من الدولة العثمانية وأوروبا والشرق الأوسط وآسيا "
            "الوسطى، وشهدت تحولات عميقة خلال القرن الماضي.\n\n"
            "# الأدب والفنون\n\n"
            "يعتبر الأدب التركي مزيجاً من التقاليد الشعبية والعثمانية. "
            "فاز أورهان باموك بجائزة نوبل للأدب عام 2006.\n\n"
            "# العمارة\n\n"
            "مرّت العمارة التركية بفترات متعددة: الفترة الكلاسيكية "
            "(1437-1703) تحت إشراف معمار سنان، فترة التغريب "
            "(1703-1876) بتأثيرات أوروبية، والعصر الحديث بالعمارة "
            "الوطنية المستوحاة من التراث العثماني.\n\n"
            "# المطبخ التركي\n\n"
            "يعكس المطبخ التركي التراث العثماني الذي يمزج المأكولات "
            "التركية والكردية والعربية واليونانية والأرمنية "
            "والفارسية."
        ),
    ),
    # --- Turkish: attractions/religious heritage ---------------------------
    SourceDocument(
        source_id="wiki_tr_galata_tower",
        title="Galata Kulesi",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Galata_Kulesi",
        content_type="attraction",
        poi_id="poi_galata_tower",
        district_id="district_beyoglu",
        topic_group="galata_tower",
        interest_tags=("history", "attractions", "photography"),
        lat=41.0256, lon=28.9741, side="european", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# İnşa Tarihi\n\n"
            "Galata Kulesi 1348 yılında inşa edilmiştir. Cenevizliler "
            "tarafından 'Kutsal Haç Kulesi' adıyla kurulan yapı, "
            "bölgedeki tahkimatın bir parçasıydı.\n\n"
            "# Mimari Özellikler\n\n"
            "Yapı, Romanesk tarzında silindirik bir kagir kuledir. "
            "Zeminden çatısının ucuna kadar yüksekliği 62,59 metredir. "
            "İç çapı 8,95 m, dış çapı 16,45 m olup toplam on bir katlı "
            "bir yapıdan oluşmaktadır.\n\n"
            "# Dönem Dönem Kullanımları\n\n"
            "Ceneviz döneminde (1348-1453) gözetleme ve savunma amacıyla "
            "kullanılan kule, tepesinde bir haç taşıyordu. 1453'teki "
            "fetih sonrası tepesindeki haç Osmanlı bayrağıyla "
            "değiştirildi. 16-17. yüzyıllarda savaş esirlerinin barınağı "
            "ve gözlemevi olarak hizmet etti. 18. yüzyılda yangın "
            "kulesine dönüştürüldü.\n\n"
            "# Cumhuriyet Dönemi\n\n"
            "1965-1967'de turistik bir tesise dönüştürülen kule, "
            "2020'de müzeye çevrilmiştir."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_kariye_camii",
        title="Kariye Camii",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Kariye_M%C3%BCzesi",
        content_type="attraction",
        poi_id="poi_chora_church",
        district_id="district_fatih",
        topic_group="chora_church",
        interest_tags=("religious_heritage", "attractions", "history", "art"),
        lat=41.0311, lon=28.9391, side="european", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Genel Bilgi\n\n"
            "Kariye Camii, İstanbul'un Fatih ilçesinde yer alan eski bir "
            "Rum Ortodoks kilisesidir. İstanbul'un Fethi'nden sonra elli "
            "sekiz yıl daha kilise olarak işlevini sürdürdü.\n\n"
            "# Tarihçe\n\n"
            "Yapının tarihi 536 yılında I. Justinianus dönemine dayanır. "
            "11. yüzyılda Maria Dukaina himayesinde restore edildi. "
            "14. yüzyılda Metokhites, manastırın tamirine büyük servet "
            "harcadı.\n\n"
            "# Osmanlı ve Cumhuriyet Dönemleri\n\n"
            "1511 yılında Sultan II. Bayezid'in sadrazamı Atik Ali Paşa "
            "tarafından camiye dönüştürüldü. 1945'te Bakanlar Kurulu "
            "kararıyla müzeye çevrildi. 6 Mayıs 2024'te yeniden ibadete "
            "açıldı.\n\n"
            "# Sanat ve Mimari\n\n"
            "Kariye'nin mozaikleri ve freskleri 'Bizans Rönesansı' "
            "olarak değerlendirilir. Pareklezyon bölümündeki Anastasis "
            "freski, Geç Bizans sanatının başyapıtı kabul edilir. Camiye "
            "çevrilişinde mozaik ve freskler sıva ile kaplandı, bu "
            "sayede günümüze kadar ulaşmıştır."
        ),
    ),
    # --- Turkish: neighborhoods / family -----------------------------------
    SourceDocument(
        source_id="wiki_tr_kadikoy",
        title="Kadıköy",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Kad%C4%B1k%C3%B6y",
        content_type="neighborhood",
        poi_id="poi_kadikoy_district",
        district_id="district_kadikoy",
        topic_group="kadikoy",
        interest_tags=("neighborhoods", "culture", "food", "nightlife"),
        lat=40.9903, lon=29.0275, side="asian", entity_kind="poi_candidate",
        preferred_period="evening",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Tarihçe\n\n"
            "Kadıköy'ün yerleşim geçmişi çok eski dönemlere uzanır. "
            "Fikirtepe'deki kazılarda MÖ 3000 yıllarına ait aletler "
            "bulunmuştur. Antik Çağ'da Kalkedon olarak bilinen bölge, "
            "Megaralı Yunanlılar tarafından kurulmuştur.\n\n"
            "Osmanlı döneminde Fatih Sultan Mehmet'in fethinden sonra, "
            "yönetim İstanbul Kadısı Hızır Bey Çelebi'ye verilmiş ve "
            "bölge bu tarihten itibaren Kadıköy adıyla anılmaya "
            "başlanmıştır. Cumhuriyet'te 1930 yılında ilçe statüsü "
            "kazanmıştır.\n\n"
            "# Kültürel Karakter\n\n"
            "Kadıköy, İstanbul'un yaşam dolu ve sanat odaklı "
            "ilçelerinden biridir. 1900'lü yılların başında ilk sinema "
            "ve tiyatro gösterileri burada yapılmıştır. Günümüzde "
            "Süreyya Operası, Reks Sineması ve Müjdat Gezen Sanat "
            "Merkezi gibi önemli sanat kurumları bulunmaktadır.\n\n"
            "# Gece Hayatı\n\n"
            "Bahariye Caddesi'nin yaya yolu haline getirilmesi ve "
            "ticari yoğunluğu, ilçeyi zengin sosyal ve gastronomik "
            "aktivitelerin merkezi haline getirmiştir."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_uskudar",
        title="Üsküdar",
        language="tr",
        url="https://tr.wikipedia.org/wiki/%C3%9Csk%C3%BCdar",
        content_type="neighborhood",
        poi_id="poi_uskudar_district",
        district_id="district_uskudar",
        topic_group="uskudar",
        interest_tags=("neighborhoods", "religious_heritage", "culture"),
        lat=41.0214, lon=29.0161, side="asian", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Tarihçe\n\n"
            "Üsküdar'ın antik adı Khrysopolis (Altın Şehir) olup Yunanca "
            "kökenlidir. Bugünkü adı Roma ordusunun zırhlı süvari "
            "birlikleri olan Scutarii'den türetildiği düşünülmektedir. "
            "İstanbul'un fethinden sonra önemli bir yönetim merkezi "
            "olmuş, 1926 yılına kadar il statüsünde bulunmuştur.\n\n"
            "# Dini Miras ve Camiler\n\n"
            "Osmanlı döneminde geniş ölçüde mimari faaliyete sahne "
            "olan Üsküdar'da Mihrimah Sultan Külliyesi, Yeni Valide "
            "Külliyesi, Şemsi Paşa Camii, Ayazma Camii ve Çinili "
            "Külliyesi gibi önemli yapılar bulunur. Karacaahmet "
            "Mezarlığı, Anadolu yakasındaki en büyük Müslüman mezarlığı "
            "olma özelliğini yüzyıllardır korumaktadır.\n\n"
            "# Kültürel Karakter\n\n"
            "Üsküdar çok kültürlü bir miras taşır; Rum Ortodoks "
            "kiliseleri, Ermeni kiliseleri ve sinagoglar bu çeşitliliği "
            "yansıtır. Boğaziçi kıyısındaki Osmanlı dönemi yalıları "
            "mimari zenginliğin diğer örnekleridir."
        ),
    ),
    SourceDocument(
        source_id="wiki_tr_miniaturk",
        title="Miniatürk",
        language="tr",
        url="https://tr.wikipedia.org/wiki/Miniat%C3%BCrk",
        content_type="attraction",
        poi_id="poi_miniaturk",
        district_id="district_beyoglu",
        topic_group="miniaturk",
        interest_tags=("family", "attractions"),
        lat=41.0576, lon=28.9377, side="european", entity_kind="poi_candidate",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Temel Bilgiler\n\n"
            "Miniatürk, İstanbul'un Beyoğlu ilçesinde bulunan ve "
            "'Büyük Ülkenin Küçük Bir Modeli' sloganıyla tanınan bir "
            "minyatür parkıdır. 60.000 metrekarelik alanda kurulu "
            "olan tesis, dünyanın en geniş alana sahip minyatür "
            "parklarından biridir.\n\n"
            "# Açılış ve Tarih\n\n"
            "30 Haziran 2001'de temeli atılan Miniatürk, 2 Mayıs "
            "2003'te ziyaretçilere açılmıştır. Proje, İstanbul "
            "Büyükşehir Belediyesi tarafından gerçekleştirilmiş ve "
            "işletilmektedir.\n\n"
            "# Koleksiyon\n\n"
            "Parkta Hitit, Eski Yunan, Roma, Bizans, Selçuklu, Osmanlı "
            "ve Cumhuriyet dönemi mimarisini temsil eden 137 eserin "
            "1/25 oranında küçültülmüş modelleri bulunmaktadır. "
            "Eserlerden 62'si İstanbul'dan, 64'ü Anadolu'dan ve 13'ü "
            "Osmanlı coğrafyasından seçilmiştir.\n\n"
            "# Aile Dostu Özellikler\n\n"
            "Ziyaretçiler mini stadyum, kumandalı tekneler, simülasyon "
            "helikopter turu ve gezi treni gibi eğlence alanlarından "
            "yararlanabilir. Park 2015 yılında 5 milyon ziyaretçi "
            "ağırlamıştır."
        ),
    ),
    # --- Official/primary sources -------------------------------------------
    SourceDocument(
        source_id="unesco_turkish_coffee",
        title="Turkish Coffee Culture and Tradition (UNESCO ICH)",
        language="en",
        url="https://ich.unesco.org/en/RL/turkish-coffee-culture-and-tradition-00645",
        content_type="etiquette",
        topic_group="turkish_coffee_unesco",
        interest_tags=("etiquette", "food", "culture"),
        publisher_type="official_primary",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Inscription\n\n"
            "Turkish coffee culture and tradition was inscribed on "
            "UNESCO's Representative List of the Intangible Cultural "
            "Heritage of Humanity in 2013 (8th session of the "
            "Intergovernmental Committee).\n\n"
            "# What Was Inscribed\n\n"
            "The inscription covers both the preparation and brewing "
            "technique (roasting and grinding coffee beans, slow "
            "brewing producing foam, serving in small cups with water) "
            "and the communal cultural practices built around it, "
            "including coffee-house gatherings, fortune-telling through "
            "the grounds, and ceremonial roles in engagements and "
            "religious holidays.\n\n"
            "# Official Cultural Significance\n\n"
            "UNESCO's own description states that Turkish coffee "
            "'combines special preparation and brewing techniques with "
            "a rich communal traditional culture,' functioning as 'a "
            "symbol of hospitality, friendship, refinement and "
            "entertainment that permeates all walks of life,' providing "
            "'an opportunity for intimate talk and the sharing of daily "
            "concerns' among friends.\n\n"
            "# Domains\n\n"
            "UNESCO classifies the inscription under oral traditions "
            "and expressions; social practices, rituals and festive "
            "events; and traditional craftsmanship."
        ),
    ),
    SourceDocument(
        source_id="goturkiye_istanbul_overview",
        title="Istanbul (Go Türkiye official tourism portal)",
        language="en",
        url="https://goturkiye.com/istanbul",
        content_type="general_knowledge",
        topic_group="istanbul_official_overview",
        interest_tags=("attractions", "culture", "shopping", "nightlife", "neighborhoods"),
        publisher_type="official_primary",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Official Overview\n\n"
            "Go Turkiye, the official tourism portal of the Republic of "
            "Turkiye, describes Istanbul as a city 'full of charm, "
            "where the past goes hand in hand with the present,' "
            "situated at the crossroads of Europe and Asia.\n\n"
            "# Officially Highlighted Destinations\n\n"
            "The portal groups the city into eight main areas for "
            "visitors: the Historic Peninsula, Beyoglu, the Asian Side, "
            "the Bosphorus, the Golden Horn, the historic bazaars, the "
            "Princes' Islands, and the city's nature and beaches.\n\n"
            "# Officially Promoted Experiences\n\n"
            "Featured experience categories include art, gastronomy, "
            "museums, shopping, faith-based sites, health tourism, "
            "vibrant nightlife, and art/music festivals, alongside the "
            "city's 'labyrinths of marketplaces.'\n\n"
            "# Practical Information\n\n"
            "The portal references over 515 accommodation listings and "
            "content on trekking routes, dining, and the city's parks "
            "and gardens, positioning Istanbul's most famous landmarks "
            "(including the Maiden's Tower and the Bosphorus strait) as "
            "part of an official 'Istanbul Bucket List.'"
        ),
    ),
    SourceDocument(
        source_id="metro_istanbul_official",
        title="Istanbul Metro system (Metro İstanbul official site)",
        language="en",
        url="https://www.metro.istanbul/en",
        content_type="transport",
        topic_group="metro_istanbul_official",
        interest_tags=("transportation", "accessibility"),
        publisher_type="official_primary",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Official Network Scope\n\n"
            "Metro Istanbul, the system's official operating authority, "
            "lists 18 operational lines across multiple transit types: "
            "9 metro lines (M1A, M1B, M2 through M9) covering "
            "approximately 402.6 kilometres of existing infrastructure, "
            "4 tramway lines, 2 funicular lines, and 2 cable car "
            "lines, plus night-metro service on select lines.\n\n"
            "# Official Payment System\n\n"
            "The operating authority's own site describes travel-card "
            "based fare payment (Istanbulkart) with dedicated fare and "
            "ticketing information pages.\n\n"
            "# Official Operational Information\n\n"
            "The authority publishes live service-status information "
            "covering escalators, elevators, station conditions, and "
            "restrooms, alongside real-time disruption notifications -- "
            "explicit accessibility-infrastructure status reporting "
            "(elevator/escalator condition) not present in any "
            "third-party source in this corpus.\n\n"
            "# Officially Stated Expansion Plans\n\n"
            "The authority states a target network length of "
            "approximately 717 kilometres by 2050, with roughly 65 "
            "kilometres under construction at the time of this "
            "retrieval."
        ),
    ),
    # --- Food / nightlife / shopping: Cicek Pasaji --------------------------
    SourceDocument(
        source_id="wiki_en_cicek_pasaji",
        title="Çiçek Pasajı (Flower Passage)",
        language="en",
        url="https://en.wikipedia.org/wiki/%C3%87i%C3%A7ek_Pasaj%C4%B1",
        content_type="attraction",
        poi_id="poi_cicek_pasaji",
        district_id="district_beyoglu",
        topic_group="cicek_pasaji",
        interest_tags=("food", "nightlife", "shopping", "history"),
        lat=41.0326, lon=28.9772, side="european", entity_kind="poi_candidate",
        preferred_period="evening",
        retrieved_at="2026-08-20T00:00:00Z",
        text=(
            "# Overview\n\n"
            "Cicek Pasaji (Flower Passage) is a historic covered arcade "
            "on Istiklal Avenue in Beyoglu, connecting the avenue with "
            "Sahne Street and opening onto the Balik Pazari (Fish "
            "Market).\n\n"
            "# Historical Development\n\n"
            "The site previously held the Naum Theatre, severely damaged "
            "by the Fire of Pera in 1870. Ottoman Greek banker Hristaki "
            "Zografos Efendi rebuilt the property in 1876, designed by "
            "architect Kleanthis Zannos, initially called Cite de Pera or "
            "Hristaki Pasaji; the first winehouse, Yorgo's, opened there "
            "in this period.\n\n"
            "# Origin of the 'Flower Passage' Name\n\n"
            "After the 1917 Russian Revolution, impoverished Russian "
            "nobles opened flower shops in the arcade; by the 1940s the "
            "building was mostly occupied by flower shops, giving it its "
            "current Turkish name.\n\n"
            "# Contemporary Character\n\n"
            "The passage today functions as a galleria of pubs, "
            "meyhanes (taverns), and restaurants across three floors, "
            "restored in 1988, 2005, and 2022 -- one of Beyoglu's "
            "best-known destinations for traditional Istanbul dining and "
            "evening drinking culture."
        ),
    ),
)


def by_source_id(source_id: str) -> SourceDocument:
    for doc in DOCUMENTS:
        if doc.source_id == source_id:
            return doc
    raise KeyError(source_id)
