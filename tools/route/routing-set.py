"""Writes the labelled routing set, as JSON lines of {"text", "lang", "want", "source", "laya"}.

    python tools/route/routing-set.py data.jsonl > crates/kime-route/tests/lang/routing-set.jsonl

The data file is what lid-data.sh writes. From its test split this takes 64 MASSIVE utterances per
language, 15 more per Latin script language with the accents stripped, 40 papluca texts per
language, 300 AG News articles and 100 LeetCode statements. Then come the cases from Laya's
routing issues (#20, #54, #130, #168, #172 and #178), Latin script languages the identifier was
not trained on, and English that shares words with other languages. `want` is `english`,
`multilingual` or `default`, and `laya` is what Laya's Router._route picks. Run it with Laya on
the path (PYTHONPATH=laya-src); no weights are loaded.
"""
import json
import random
import sys
import unicodedata
from collections import defaultdict

from laya.lang import detect_script
from laya.router import Router

ISSUES = [
    # 20: Armenian had no script range and went to the default.
    ("Հայերեն", "hy", "multilingual", "laya#20"),
    ("Ես ուզում եմ չեղարկել իմ բաժանորդագրությունը", "hy", "multilingual", "laya#20"),
    ("Please refund the charge", "en", "english", "laya#20"),
    ("12345", None, "default", "laya#20"),
    # 54 and 130: plain ASCII German taken for English.
    ("trage diesen termin in meinen kalender ein", "de", "multilingual", "laya#54"),
    ("hey wie wird heute das wetter in münchen bayern", "de", "multilingual", "laya#54"),
    ("wie lautet die temperatur in fulda in hessen", "de", "multilingual", "laya#54"),
    ("wie koche ich ein butterhähnchen", "de", "multilingual", "laya#54"),
    ("schalte das licht im wohnzimmer aus", "de", "multilingual", "laya#54"),
    ("was ist die aktuelle zeit", "de", "multilingual", "laya#130"),
    ("Mein Konto wurde zweimal belastet, bitte erstatten Sie", "de", "multilingual", "laya#130"),
    # 168: scripts with no range went to the default.
    ("ｱﾘｶﾞﾄｳ", "ja", "multilingual", "laya#168"),
    ("ㄆㄇㄈㄉ", "zh", "multilingual", "laya#168"),
    ("𛀁𛀂𛀃𛀄", "ja", "multilingual", "laya#168"),
    ("𠀀𠀁𠀂𠀃", "zh", "multilingual", "laya#168"),
    ("ꥠꥡꥢꥣ", "ko", "multilingual", "laya#168"),
    ("ᏣᎳᎩ ᎦᏬᏂᎯᏍᏗ", "chr", "multilingual", "laya#168"),
    ("ᠮᠣᠩᠭᠣᠯ ᠬᠡᠯᠡ", "mn", "multilingual", "laya#168"),
    ("ܠܫܢܐ ܣܘܪܝܝܐ", "syr", "multilingual", "laya#168"),
    ("ދިވެހިބަސް", "dv", "multilingual", "laya#168"),
    ("ⵜⴰⵎⴰⵣⵉⵖⵜ", "zgh", "multilingual", "laya#168"),
    ("ꆈꌠꁱꂷ", "ii", "multilingual", "laya#168"),
    ("ＨＥＬＬＯ ＷＯＲＬＤ, please reset my password", "en", "english", "laya#168"),
    # 172 and 178: Romance text without its accents taken for English.
    ("Mi pedido llegó roto y nadie responde", "es", "multilingual", "laya#172"),
    ("Il mio ordine è arrivato rotto e nessuno risponde", "it", "multilingual", "laya#172"),
    ("La factura tiene un error en el importe total", "es", "multilingual", "laya#178"),
    ("Quiero cancelar mi plan y pedir un reembolso", "es", "multilingual", "laya#178"),
    ("Il cliente e stato addebitato due volte e vuole un rimborso", "it", "multilingual", "laya#178"),
    ("O cliente foi cobrado duas vezes e quer o dinheiro de volta", "pt", "multilingual", "laya#178"),
    ("Le client a ete facture deux fois et demande un remboursement", "fr", "multilingual", "laya#178"),
    ("Il pacco e arrivato rotto e nessuno risponde al supporto", "it", "multilingual", "laya#178"),
    ("A encomenda chegou danificada e ninguem responde no suporte", "pt", "multilingual", "laya#178"),
    ("Cât e ora acum la Tokyo", "ro", "multilingual", "laya#178"),
]

# Latin script languages the identifier has no training data for, with and without accents.
UNSEEN = {
    "cs": ["Chci zrušit své předplatné a vrátit peníze", "Kolik stojí jízdenka do Brna",
           "Moje objednávka dorazila poškozená", "Zapni prosím světlo v obývacím pokoji"],
    "sk": ["Chcem zrušiť svoje predplatné", "Koľko stojí lístok do Bratislavy",
           "Moja objednávka prišla poškodená", "Zapni prosím svetlo v obývačke"],
    "hr": ["Želim otkazati svoju pretplatu", "Koliko košta karta do Splita",
           "Moja narudžba je stigla oštećena", "Upali svjetlo u dnevnoj sobi molim te"],
    "lt": ["Noriu atšaukti savo prenumeratą", "Kiek kainuoja bilietas į Vilnių",
           "Mano užsakymas atkeliavo sugadintas", "Įjunk šviesą svetainėje"],
    "et": ["Ma tahan oma tellimuse tühistada", "Kui palju maksab pilet Tartusse",
           "Minu pakk saabus katkisena", "Palun lülita elutoas tuli sisse"],
    "ca": ["Vull cancel·lar la meva subscripció", "Quant costa un bitllet a Girona",
           "La meva comanda ha arribat trencada", "Encén el llum del menjador, si us plau"],
    "eu": ["Nire harpidetza bertan behera utzi nahi dut", "Zenbat balio du Bilborako txartelak",
           "Nire eskaera hautsita iritsi da", "Piztu egongelako argia, mesedez"],
}

# English with words that other languages' lists hold, from the controls in Laya #178.
COLLISIONS = [
    "The ruling made it the de facto standard for the whole industry",
    "Smith et al. reported the same effect in their 2019 paper",
    "Bring a snack, e.g. an apple or a banana",
    "We ordered a la carte instead of the set menu",
    "The UN issued a statement on the ceasefire this morning",
    "MI5 opened an inquiry into the leak",
    "The old DOS machine in the basement still boots",
    "No refund was issued for the cancelled flight",
    "no refund no reply",
    "Rio de Janeiro is hosting the summit this year",
    "Set the locale to en-US before running the build",
    "Apple releases Mac OS X 10.3.7 update for Panther",
    "Van Gogh painted it in Arles in 1888",
    "The café on the corner serves a great crème brûlée",
    "Our résumé template is on the wiki, see naïve Bayes for the example",
    "I want to cancel my subscription and get a refund",
    "Please send the invoice to finance@example.com by Friday",
    "Can you book a table for two at seven",
]


def strip(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def main():
    groups = defaultdict(list)
    for line in open(sys.argv[1]):
        row = json.loads(line)
        if row["split"] == "test":
            groups[(row["source"], row["lang"])].append(row["text"])
    rng = random.Random(26)
    cases = []
    per = {"massive": 64, "papluca": 40, "agnews": 300, "leetcode": 100}
    for (source, lang) in sorted(groups):
        texts = groups[(source, lang)]
        for t in rng.sample(texts, min(per[source], len(texts))):
            cases.append((t, lang, source))
        if source == "massive" and lang != "en":
            changed = [t for t in texts if strip(t) != t and detect_script(t) == "latin"]
            for t in rng.sample(changed, min(15, len(changed))):
                cases.append((strip(t), lang, "massive-stripped"))
    router = Router()
    for text, lang, source in cases:
        want = "english" if lang == "en" else "multilingual"
        write(router, text, lang, want, source)
    for text, lang, want, source in ISSUES:
        write(router, text, lang, want, source)
    for lang, texts in UNSEEN.items():
        for t in texts:
            write(router, t, lang, "multilingual", "unseen")
            if strip(t) != t:
                write(router, strip(t), lang, "multilingual", "unseen-stripped")
    for text in COLLISIONS:
        write(router, text, "en", "english", "collision")


def write(router, text, lang, want, source):
    d = router._route(text, {})
    laya = d["model"] if "default" not in d["reason"] else "default"
    row = {"text": text, "lang": lang, "want": want, "source": source, "laya": laya}
    print(json.dumps(row, ensure_ascii=False))


main()
