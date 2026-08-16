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


DOCUMENTS: tuple[SourceDocument, ...] = (
    # --- Istanbul (city history/geography) -----------------------------
    SourceDocument(
        source_id="wiki_en_istanbul",
        title="Istanbul",
        language="en",
        url="https://en.wikipedia.org/wiki/Istanbul",
        content_type="history",
        topic_group="istanbul",
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
        district_id="district_beyoglu",
        topic_group="beyoglu",
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
        district_id="district_beyoglu",
        topic_group="beyoglu",
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
        district_id="district_kadikoy",
        topic_group="kadikoy",
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
        district_id="district_uskudar",
        topic_group="uskudar",
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
)


def by_source_id(source_id: str) -> SourceDocument:
    for doc in DOCUMENTS:
        if doc.source_id == source_id:
            return doc
    raise KeyError(source_id)
