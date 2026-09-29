"""Builds the quality suites `kime eval` reads, from the public datasets on the Hugging Face hub.

    python tools/eval/build_suites.py <out dir> [--only name,name]

The suites are the ones in Laya's benchmark notebook (research/scripts/build_benchmark_nb.py in
NandhaKishorM/laya), built the same way with the same seed, so the questions are the ones behind
Laya's published results and the numbers can be checked against them. Each suite is one file,
<name>.jsonl, with one request per line plus `id`, `gold` and `tags`, as crates/kime-eval/src/suite.rs
describes. A manifest.json next to them records the dataset, config, split and revision of each.

Needs `datasets`, and `laya` for the email suite (pip install datasets laya, laya without torch is enough).
"""

import argparse
import json
import os
import random

from datasets import load_dataset

SEED = 13
N_OPTS = 20
PER_LANG = 300
MASSIVE_LANGS = ["en", "de", "fr", "es", "pt", "ru", "tr", "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw"]
XNLI_LANGS = ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur", "vi", "th", "el", "bg", "zh", "sw"]
NLI_CRIT = {
    "entailment": "the premise implies the hypothesis is true",
    "neutral": "the premise neither implies nor contradicts the hypothesis",
    "contradiction": "the premise implies the hypothesis is false",
}

SUITES = {}
MANIFEST = {}


def register(name, lines, source):
    SUITES[name] = lines
    MANIFEST[name] = source
    print("  %-26s %5d cases %6d questions" % (name, len(lines), sum(len(l["questions"]) for l in lines)))


def data(repo, config, split):
    d = load_dataset(repo, config, split=split)
    src = {"dataset": repo, "config": config, "split": split}
    try:
        from huggingface_hub import HfApi

        src["revision"] = HfApi().dataset_info(repo).sha
    except Exception:
        src["revision"] = None
    return d, src


def choice(rng, gold, labels, n_opts, instructions):
    pool = [x for x in labels if x != gold]
    keys = [gold] + rng.sample(pool, min(n_opts - 1, len(pool)))
    rng.shuffle(keys)
    crit = {k: k.replace("_", " ").replace(".", ": ") for k in keys}
    return {"label": {"type": "choice", "instructions": instructions, "criteria": crit}}


def typed_decisions():
    d, src = data("LocalLLaMA/typed-decisions", "all", "test")
    lines = []
    for i, r in enumerate(d):
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        g, soft, score = {}, {}, {}
        for qid, q in questions.items():
            x = gold[qid]
            probs = x.get("probabilities", {})
            if q["type"] == "choice":
                keys = list(q["criteria"].keys())
                g[qid] = str(x["label"])
                soft[qid] = [float(probs.get(k, 0.0)) for k in keys]
            elif q["type"] == "noul":
                g[qid] = str(x["label"]).lower() == "true"
                pt = float(probs.get("true", x.get("noul", 0.5)))
                soft[qid] = [1 - pt, pt]
            else:
                n = len(q["criteria"])
                g[qid] = int(x["label"])
                soft[qid] = [float(probs.get(str(k), 0.0)) for k in range(n)]
                score[qid] = float(x.get("score", float(x["label"])))
        lines.append({"id": "typed_decisions-%d" % i, "state": state, "questions": questions, "gold": g,
                      "soft": soft, "gold_score": score, "tags": {"workflow": r["workflow"]}})
    register("typed_decisions", lines, src)


def massive():
    for repo, short, instr in [
        ("mteb/amazon_massive_intent", "massive_intent", "What is the user asking for in `utterance`?"),
        ("mteb/amazon_massive_scenario", "massive_scenario", "Which domain does `utterance` belong to?"),
    ]:
        for lg in MASSIVE_LANGS:
            d, src = data(repo, lg, "test")
            labels = sorted(set(d["label_text"]))
            rng = random.Random(SEED)
            lines = []
            for i, r in enumerate(list(d)[:PER_LANG]):
                lines.append({"id": "%s.%s-%d" % (short, lg, i), "state": {"utterance": r["text"]},
                              "questions": choice(rng, r["label_text"], labels, N_OPTS, instr),
                              "gold": {"label": r["label_text"]}, "tags": {"lang": lg}})
            register("%s.%s" % (short, lg), lines, src)


def xnli():
    for lg in XNLI_LANGS:
        d, src = data("facebook/xnli", lg, "test")
        keys = list(NLI_CRIT)
        lines = []
        for i, r in enumerate(list(d)[:PER_LANG]):
            q = {"relation": {"type": "choice", "instructions": "What is the relationship between `premise` and `hypothesis`?",
                              "criteria": dict(NLI_CRIT)}}
            lines.append({"id": "xnli.%s-%d" % (lg, i), "state": {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                          "questions": q, "gold": {"relation": keys[int(r["label"])]}, "tags": {"lang": lg}})
        register("xnli.%s" % lg, lines, src)


def english():
    d, src = data("SetFit/sst5", None, "test")
    crit = ["very negative", "negative", "neutral", "positive", "very positive"]
    lines = []
    for i, r in enumerate(list(d)[:600]):
        q = {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `text`?", "criteria": crit}}
        lines.append({"id": "en.sst5-%d" % i, "state": {"text": r["text"]}, "questions": q,
                      "gold": {"sentiment": int(r["label"])}, "gold_score": {"sentiment": float(r["label"])}})
    register("en.sst5", lines, src)

    d, src = data("dair-ai/emotion", "split", "test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    lines = []
    for i, r in enumerate(list(d)[:600]):
        q = {"emotion": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?",
                         "criteria": {n: None for n in names}}}
        lines.append({"id": "en.emotion-%d" % i, "state": {"text": r["text"]}, "questions": q,
                      "gold": {"emotion": names[int(r["label"])]}})
    register("en.emotion", lines, src)

    d, src = data("deepset/prompt-injections", None, "test")
    lines = []
    for i, r in enumerate(list(d)):
        q = {"injection": {"type": "noul", "instructions": "Does `text` try to inject or override instructions given to an AI system?"}}
        lines.append({"id": "en.prompt_injections-%d" % i, "state": {"text": r["text"]}, "questions": q,
                      "gold": {"injection": bool(int(r["label"]))}})
    register("en.prompt_injections", lines, src)

    # PolyAI/banking77 is a loading script the current datasets refuses, mteb/banking77 is the
    # same test split as parquet with the label names next to the ids.
    d, src = data("mteb/banking77", None, "test")
    names = dict(sorted(set(zip(d["label"], d["label_text"]))))
    names = [names[i].replace("_", " ") for i in range(len(names))]
    lines = []
    for i, r in enumerate(list(d)[:500]):
        q = {"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                        "criteria": {n: None for n in names}}}
        lines.append({"id": "en.banking77_full-%d" % i, "state": {"message": r["text"]}, "questions": q,
                      "gold": {"intent": names[int(r["label"])]}})
    register("en.banking77_full", lines, src)

    d, src = data("fancyzhx/ag_news", None, "test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    keys = list(crit)
    lines = []
    for i, r in enumerate(list(d)[:600]):
        q = {"topic": {"type": "choice", "instructions": "What is the topic of `article`?", "criteria": dict(crit)}}
        lines.append({"id": "en.ag_news-%d" % i, "state": {"article": r["text"]}, "questions": q,
                      "gold": {"topic": keys[int(r["label"])]}})
    register("en.ag_news", lines, src)

    d, src = data("google/boolq", None, "validation")
    lines = []
    for i, r in enumerate(list(d)[:600]):
        q = {"answer": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}}
        lines.append({"id": "en.boolq-%d" % i, "state": {"passage": r["passage"], "question": r["question"]},
                      "questions": q, "gold": {"answer": bool(r["answer"])}})
    register("en.boolq", lines, src)


def apps():
    # Laya's application themes, research/scripts/bench_apps.py, 400 cases each. One generator
    # runs through the sets in the script's order, so each draws what it drew there.
    rng = random.Random(SEED)
    n = 400

    def noul(name, rows, src):
        lines = []
        for i, (state, qid, q, g) in enumerate(rows):
            lines.append({"id": "%s-%d" % (name, i), "state": state, "questions": {qid: q}, "gold": {qid: bool(g)}})
        register(name, lines, src)

    queues = {"Technical Support": "technical problems, bugs, outages, integrations",
              "Product Support": "help using a product or feature",
              "Customer Service": "general account or service questions",
              "IT Support": "internal IT, devices, access, networks",
              "Billing and Payments": "invoices, charges, refunds, payment methods",
              "Returns and Exchanges": "returning or exchanging an item",
              "Service Outages and Maintenance": "downtime, outages, scheduled maintenance",
              "Sales and Pre-Sales": "pricing, quotes, buying",
              "Human Resources": "employment, payroll, leave, hiring",
              "General Inquiry": "anything else"}
    d, src = data("Tobi-Bueck/customer-support-tickets", None, "train")
    lines = []
    for r in d:
        if r.get("language") != "en" or r.get("queue") not in queues or not r.get("body"):
            continue
        q = {"type": "choice", "instructions": "Which support queue should handle this ticket?", "criteria": dict(queues)}
        lines.append({"id": "app.support_triage-%d" % len(lines),
                      "state": {"subject": r["subject"] or "", "body": r["body"].replace("\\n", "\n")[:3000]},
                      "questions": {"queue": q}, "gold": {"queue": r["queue"]}})
        if len(lines) >= n:
            break
    register("app.support_triage", lines, src)

    from laya import email_state

    d, src = data("SetFit/enron_spam", None, "test")
    q = {"type": "noul", "instructions": "Is this email unsolicited spam or bulk marketing?"}
    noul("app.email_spam", [(email_state(r.get("subject") or "", (r.get("message") or "")[:3000]), "is_spam", q,
                                   int(r["label"])) for r in list(d)[:n]], src)

    d, src = data("zefang-liu/phishing-email-dataset", None, "train")
    rows = [r for r in list(d)[:6000]
            if (r.get("Email Text") or "").strip() and r.get("Email Type") in ("Safe Email", "Phishing Email")]
    rng.shuffle(rows)
    q = {"type": "noul", "instructions": "Is this email a phishing or scam attempt to steal money, credentials, or personal data?",
         "criteria": {"true": "phishing, scam, or fraud", "false": "a legitimate email (even if promotional)"}}
    noul("app.phishing", [({"email": r["Email Text"][:3000]}, "is_phishing", q, r["Email Type"] == "Phishing Email")
                                for r in rows[:n]], src)

    d, src = data("lmsys/toxic-chat", "toxicchat0124", "test")
    rows = [r for r in d if (r.get("user_input") or "").strip()]
    for name, field, key, qid, instr in [
        ("app.guardrails_jailbreak", "jailbreaking", "prompt", "jailbreak",
         "Does `prompt` try to make an AI assistant ignore its rules, policies or system instructions?"),
        ("app.moderation_toxicity", "toxicity", "post", "toxic",
         "Is `post` toxic: rude, disrespectful or likely to make someone leave the discussion?"),
    ]:
        pos = [r for r in rows if int(r.get(field, 0)) == 1][: n // 2]
        mix = pos + [r for r in rows if int(r.get(field, 0)) == 0][: n - len(pos)]
        rng.shuffle(mix)
        q = {"type": "noul", "instructions": instr}
        noul(name, [({key: r["user_input"][:3000]}, qid, q, int(r[field])) for r in mix], src)

    d, src = data("microsoft/ms_marco", "v1.1", "validation")
    q = {"type": "noul", "instructions": "Does `passage` help answer `query`?"}
    rows = []
    for r in d:
        texts, sel = r["passages"]["passage_text"], r["passages"]["is_selected"]
        pos = [t for t, x in zip(texts, sel) if x == 1]
        neg = [t for t, x in zip(texts, sel) if x == 0]
        if not pos or not neg:
            continue
        take_pos = len(rows) % 2 == 0
        rows.append(({"query": r["query"], "passage": rng.choice(pos if take_pos else neg)}, "relevant", q, take_pos))
        if len(rows) >= n:
            break
    noul("app.rag_relevance", rows, src)

    dom = {"code": "software engineering, programming, refactoring, architecture, debugging",
           "math_or_logic": "mathematics, logic puzzles, proofs, complex calculation",
           "writing": "creative writing, essays, emails, blog posts, copywriting",
           "factual_lookup": "facts, definitions, trivia, history",
           "data_analysis": "statistics, SQL, data manipulation, metrics",
           "chitchat": "casual conversation, greetings, small talk"}
    g, src = data("openai/gsm8k", "main", "test")
    pool = [(r["question"], "math_or_logic") for r in list(g)[: n // 3]]
    m, _ = data("google-research-datasets/mbpp", "full", "test")
    pool += [(r["text"], "code") for r in list(m)[: n // 3]]
    t, _ = data("fancyzhx/ag_news", None, "test")
    pool += [(r["text"][:400], "factual_lookup") for r in list(t)[: n // 3]]
    rng.shuffle(pool)
    q = {"type": "choice", "instructions": "What domain does `request` belong to?", "criteria": dict(dom)}
    lines = [{"id": "app.model_routing_domain-%d" % i, "state": {"request": text}, "questions": {"domain": q},
              "gold": {"domain": g}} for i, (text, g) in enumerate(pool[:n])]
    register("app.model_routing_domain", lines, dict(src, dataset="openai/gsm8k, google-research-datasets/mbpp, fancyzhx/ag_news"))


def extra():
    # Suites spec/13-benchmarks.md lists that Laya's notebook does not have.
    d, src = data("SetFit/sst2", None, "validation")
    lines = []
    for i, r in enumerate(d):
        q = {"sentiment": {"type": "choice", "instructions": "What is the sentiment of `text`?",
                           "criteria": {"negative": None, "positive": None}}}
        lines.append({"id": "en.sst2-%d" % i, "state": {"text": r["text"]}, "questions": q,
                      "gold": {"sentiment": ["negative", "positive"][int(r["label"])]}})
    register("en.sst2", lines, src)

    # CLINC150 has 151 labels with out of scope, too many option names for one 512 token
    # sequence, so each question gets 20 like MASSIVE: the gold, out of scope, and 18 at random.
    d, src = data("clinc/clinc_oos", "plus", "test")
    names = [x.replace("_", " ") for x in d.features["intent"].names]
    oos = names.index("oos")
    label = lambda i: "out of scope" if i == oos else names[i]
    rng = random.Random(SEED)
    pick = sorted(rng.sample(range(len(d)), 1000))
    lines = []
    for i in pick:
        r = d[i]
        g = int(r["intent"])
        rest = [j for j in range(len(names)) if j not in (g, oos)]
        keys = [g] + ([oos] if g != oos else []) + rng.sample(rest, N_OPTS - (1 if g == oos else 2))
        rng.shuffle(keys)
        q = {"intent": {"type": "choice", "instructions": "What is the user asking for in `utterance`?",
                        "criteria": {label(k): None for k in keys}}}
        lines.append({"id": "en.clinc150-%d" % i, "state": {"utterance": r["text"]}, "questions": q,
                      "gold": {"intent": label(g)}, "tags": {"scope": "out" if g == oos else "in"}})
    register("en.clinc150", lines, src)


CLERC_FILE = "https://huggingface.co/datasets/jhu-clsp/CLERC/resolve/main/teva_train_dir/train_data.jsonl.gz"
CLERC_NOUL = {
    "type": "noul",
    "instructions": "The query excerpt comes from a US federal court opinion and was written immediately around a citation to a precedent; the citation itself has been removed. Could the candidate passage be from that cited precedent \u2014 does it establish the specific legal proposition the query excerpt invokes at its citation point?",
    "criteria": {
        "true": "The candidate passage states or establishes the specific rule, standard, holding, or fact pattern that the query excerpt attributes to its removed citation.",
        "false": "The candidate passage is merely on a similar topic or doctrine; it does not supply the specific proposition the query excerpt relies on.",
    },
}


def clerc():
    # CLERC rerank as TypeSafe's re-ranking cookbook (docs.typesafe.ai/cookbooks/rerank_typesafe)
    # builds it: 170 rows of the training file pooled into one corpus of 3,565 passages, BM25 top
    # 30 from that corpus for each query, and one noul question a candidate with the cookbook's
    # wording. The sampling follows leepokai/llm-prompt-techniques-on-jev, which rebuilds the
    # cookbook's slice and gets its BM25 numbers back. en.clerc_rerank is the cookbook's 40
    # queries and en.clerc_rerank_more the other 110 of the pool.
    import hashlib

    import bm25s

    cid = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]
    rows = []
    for row in load_dataset("json", data_files=CLERC_FILE, streaming=True, split="train"):
        if row.get("positive_passages") and len(row.get("negative_passages") or []) == 20:
            rows.append(row)
        if len(rows) >= 1000:
            break
    rng = random.Random(0)
    corpus, pool = {}, []
    for row in rng.sample(rows, 170):
        gold = row["positive_passages"][0]["text"]
        corpus[cid(gold)] = gold
        for neg in row["negative_passages"]:
            corpus[cid(neg["text"])] = neg["text"]
        pool.append({"qid": str(row["query_id"]), "query": row["query"], "gold": cid(gold)})
    cookbook = rng.sample(pool[20:], 40)
    more = [q for q in pool[20:] if q not in cookbook]
    corpus = dict(sorted(corpus.items()))
    cids = list(corpus)
    r = bm25s.BM25()
    r.index(bm25s.tokenize([corpus[c] for c in cids], stopwords="en"), show_progress=False)
    src = {"dataset": "jhu-clsp/CLERC", "config": "teva_train_dir/train_data.jsonl.gz", "split": "train",
           "revision": None, "corpus_passages": len(corpus)}
    for name, qs in [("en.clerc_rerank", cookbook), ("en.clerc_rerank_more", more)]:
        idx, _ = r.retrieve(bm25s.tokenize([q["query"] for q in qs], stopwords="en"), k=30, show_progress=False)
        lines = []
        for q, got in zip(qs, idx):
            for at, j in enumerate(got):
                c = cids[j]
                lines.append({"id": "%s-%s-%d" % (name, q["qid"], at),
                              "state": {"query_excerpt": q["query"], "candidate_passage": corpus[c]},
                              "questions": {"cited": CLERC_NOUL}, "gold": {"cited": c == q["gold"]},
                              "rank": {"query": q["qid"], "at": at}})
        register(name, lines, src)


def order(out):
    # Each line of the base suite, then 5 copies with the options of each choice in a random
    # order, marked with perm_of so the report counts how often the answer moves.
    for base in ["massive_intent.en", "en.emotion", "xnli.en"]:
        if base in SUITES:
            src_lines = SUITES[base]
        else:
            src_lines = [json.loads(l) for l in open(os.path.join(out, base + ".jsonl"), encoding="utf-8")]
        rng = random.Random(SEED)
        lines = []
        for l in src_lines:
            lines.append(l)
            for k in range(1, 6):
                c = json.loads(json.dumps(l))
                c["id"] = "%s~%d" % (l["id"], k)
                c["perm_of"] = l["id"]
                for q in c["questions"].values():
                    if q["type"] == "choice" and isinstance(q["criteria"], dict):
                        items = list(q["criteria"].items())
                        rng.shuffle(items)
                        q["criteria"] = dict(items)
                lines.append(c)
        register("order." + base, lines, MANIFEST.get(base, {"from": base}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--only", default="")
    a = p.parse_args()
    groups = {"typed_decisions": typed_decisions, "massive": massive, "xnli": xnli, "english": english,
              "apps": apps, "extra": extra, "clerc": clerc, "order": lambda: order(a.out)}
    only = [x for x in a.only.split(",") if x]
    for name, build in groups.items():
        if only and name not in only:
            continue
        try:
            build()
        except Exception as e:
            print("  FAIL %s: %s: %s" % (name, type(e).__name__, str(e)[:200]))
    os.makedirs(a.out, exist_ok=True)
    for name, lines in SUITES.items():
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for l in lines:
                f.write(json.dumps(l, ensure_ascii=False) + "\n")
    path = os.path.join(a.out, "manifest.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    old.update(MANIFEST)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(old, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
