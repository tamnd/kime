"""Records what Laya's `laya.lang.analyse` says about a set of states, for kime-route's tests.

    python tools/route/record.py LAYA_SRC MASSIVE_JSONL OUT.jsonl [--per-lang N] [--seed S]

LAYA_SRC is a checkout of Laya. Only `laya/lang.py` is loaded, so torch is not needed.
MASSIVE_JSONL has one {"lang": ..., "text": ...} per line (see tools/route/massive.sh). Each
output line is {"state": ..., "analysis": ...}. Besides the MASSIVE texts it writes variants
that go through the parts of the rules plain text does not reach: mixed languages and scripts,
links and emails, stripped accents, names and symbols in other scripts, dict and list states,
and states over the 4,000 character limit.
"""

import argparse
import importlib.util
import json
import random
import sys
import unicodedata


def load_lang(src):
    spec = importlib.util.spec_from_file_location("laya_lang", src + "/laya/lang.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


IDENTS = ["github.com", "user@acme.com", "v1.2.3", "U.S.A.", "api.example.co.uk/path",
          "o.com", "e.it", "la.fr", "support@la-poste.fr", "ORDER-4411.pdf"]
NAMES = ["Дмитрий Петрович", "Влади́мир", "[vlɐˈdʲimʲɪr]", "α", "β-blocker", "東京",
         "Ελλάδα", "שלום", "日本語", "ありがとう", "İstanbul", "ǅemal", "Σίσυφος"]
ENGLISH = ["Please refund the second payment on my account.",
           "The app crashes when I open the settings page.",
           "I want to cancel my subscription before the renewal.",
           "Can you check the status of order 4411 for me?"]


def variants(rng, texts):
    """States built from the MASSIVE texts that stress each rule."""
    out = []
    for _ in range(len(texts) // 4):
        a, b = rng.choice(texts), rng.choice(texts)
        kind = rng.randrange(10)
        if kind == 0:
            out.append(a + " " + b)
        elif kind == 1:
            out.append(a + " " + " ".join(rng.sample(IDENTS, 3)))
        elif kind == 2:
            out.append(strip_accents(a))
        elif kind == 3:
            out.append(rng.choice(ENGLISH) + " " + rng.choice(NAMES) + " " + a)
        elif kind == 4:
            out.append({"subject": a, "body": b, "id": 4411, "tags": [a[:10], None, True]})
        elif kind == 5:
            out.append([{"role": "user", "content": a}, {"role": "assistant", "content": b}])
        elif kind == 6:
            out.append((a + " ") * rng.randrange(40, 200))
        elif kind == 7:
            out.append(a.upper())
        elif kind == 8:
            out.append(rng.choice(ENGLISH) + " " + a[: rng.randrange(1, max(2, len(a)))])
        else:
            out.append({"a": {"b": {"c": {"d": {"e": {"f": {"g": a}}}}}}, "h": b})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laya_src")
    ap.add_argument("massive")
    ap.add_argument("out")
    ap.add_argument("--per-lang", type=int, default=0, help="texts per language, 0 for all")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    lang = load_lang(a.laya_src)
    rng = random.Random(a.seed)

    by_lang = {}
    with open(a.massive) as f:
        for line in f:
            r = json.loads(line)
            by_lang.setdefault(r["lang"], []).append(r["text"])
    texts = []
    for lg in sorted(by_lang):
        t = by_lang[lg]
        texts.extend(rng.sample(t, min(a.per_lang, len(t))) if a.per_lang else t)
    states = texts + variants(rng, texts) + ["", "   ", "1234 5678", "!!!", "ㄅㄆㄇ", "ok"]

    with open(a.out, "w") as f:
        for s in states:
            f.write(json.dumps({"state": s, "analysis": lang.analyse(s)}, ensure_ascii=False) + "\n")
    print("wrote %d states" % len(states), file=sys.stderr)


if __name__ == "__main__":
    main()
