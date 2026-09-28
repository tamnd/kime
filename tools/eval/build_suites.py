"""Builds the quality suites `kime eval` reads, from the public datasets on the Hugging Face hub.

    python tools/eval/build_suites.py <out dir> [--only name,name]

The suites are the ones in Laya's benchmark notebook (research/scripts/build_benchmark_nb.py in
NandhaKishorM/laya), built the same way with the same seed, so the questions are the ones behind
Laya's published results and the numbers can be checked against them. Each suite is one file,
<name>.jsonl, with one request per line plus `id`, `gold` and `tags`, as crates/kime-eval/src/suite.rs
describes. A manifest.json next to them records the dataset, config, split and revision of each.

Needs `datasets` (pip install datasets).
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--only", default="")
    a = p.parse_args()
    groups = {"typed_decisions": typed_decisions, "massive": massive, "xnli": xnli, "english": english}
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
