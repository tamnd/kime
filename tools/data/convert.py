"""Converts the public training sources into kime's training format, drops what is too close to a
test set, and writes zstd shards with a manifest.

    python tools/data/convert.py <out dir> [--only name,name] [--max N]
        [--kime path/to/kime --tests <dir>...]

The sources and their licenses are in tools/data/sources.json, and only the ones marked train are
converted, from the pinned revision of their train split. Each line is one state with its
questions and targets, as spec/12-training.md describes:

    {"id": "banking77-12", "source": "banking77", "split": "train", "lang": "en",
     "state": {"message": "..."}, "questions": {"intent": {...}},
     "targets": {"intent": {"probs": [0.0, 1.0, 0.0], "hard": 1, "weight": 1.0}}, "episode": null}

Every question gets one of several instruction phrasings, and a choice gets the gold and a random
number of other options in a random order, sometimes with descriptions, so the model learns the
task and not the phrasing or the position. The draws come from a generator seeded with the source
and row, so a conversion is the same bytes every time.

With --kime and --tests, every source goes through `kime contam` against the test texts (the
suites and tools/eval/split_texts.py output) before it is sharded, and the manifest records what
was dropped. Shards are about 256 MB of JSONL each before compression. manifest.json lists each
shard's blake3, source, dataset, license, split, rows and questions, and is what `kime eval
--data-manifest` reads.

Needs `datasets`, `zstandard` and `blake3`, and for Mind2Web also `ijson` and kime's Python package,
whose `agent_step` builds the agent step questions (tools/data/mind2web.py). With --reuse, the raw files already in <out dir>/raw are
checked and sharded without converting again, so the conversion can run where the datasets are and
the check where the test texts are.
"""

import argparse
import datetime
import json
import os
import random
import subprocess

import blake3
import zstandard

SEED = 13
MAX_OPTS = 20
SHARD_BYTES = 256 << 20
HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = json.load(open(os.path.join(HERE, "sources.json"), encoding="utf-8"))["sources"]

NLI = {
    "entailment": ["`{p}` implies `{h}` is true", "`{h}` follows from `{p}`", "yes"],
    "neutral": ["`{p}` neither implies nor contradicts `{h}`", "`{p}` does not settle `{h}`", "maybe"],
    "contradiction": ["`{p}` implies `{h}` is false", "`{p}` rules out `{h}`", "no"],
}
NLI_INSTR = [
    "What is the relationship between `{p}` and `{h}`?",
    "Does `{p}` support, contradict or say nothing about `{h}`?",
    "Given `{p}`, is `{h}` true, false or undetermined?",
    "How does `{h}` relate to `{p}`?",
]


def pretty(label):
    return str(label).replace("_", " ").replace(".", ": ").strip()


def rng_for(name, key):
    return random.Random("%d:%s:%s" % (SEED, name, key))


def choice(rng, instr, gold, labels, desc=None, always=()):
    """A choice with the gold, the labels in always, and a random number of others, shuffled."""
    others = [x for x in labels if x != gold and x not in always]
    fixed = [x for x in always if x != gold]
    room = min(MAX_OPTS, len(labels)) - 1 - len(fixed)
    k = rng.randint(min(3, room, len(others)), min(room, len(others)))
    keys = [gold] + fixed + rng.sample(others, k)
    rng.shuffle(keys)
    with_desc = desc is not None and rng.random() < 0.5
    crit = {pretty(x): (desc.get(x) if with_desc else None) for x in keys}
    q = {"type": "choice", "instructions": instr, "criteria": crit}
    t = {"probs": [1.0 if x == gold else 0.0 for x in keys], "hard": keys.index(gold), "weight": 1.0}
    return q, t


def noul(instr, p, criteria=None):
    q = {"type": "noul", "instructions": instr}
    if criteria:
        q["criteria"] = criteria
    return q, {"probs": [1.0 - p, p], "hard": int(p >= 0.5), "weight": 1.0}


def line(name, i, lang, state, qs):
    return {"id": "%s-%s" % (name, i), "source": name, "split": "train", "lang": lang, "state": state,
            "questions": {k: q for k, (q, _) in qs.items()}, "targets": {k: t for k, (_, t) in qs.items()},
            "episode": None}


def load(src, config=None, revision=None, dataset=None):
    from datasets import load_dataset

    return load_dataset(dataset or src["dataset"], config if config is not None else src.get("config"),
                        split=src["split"], revision=revision or src["revision"])


def banking77(src):
    d = load(src)
    labels = sorted(set(d["label_text"]))
    instr = ["Which banking intent does `{k}` express?", "What does the customer want in `{k}`?",
             "Which support topic is `{k}` about?", "Classify the request in `{k}`."]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        k = rng.choice(["message", "text", "query"])
        yield line(src["name"], i, "en", {k: r["text"]},
                   {"intent": choice(rng, rng.choice(instr).format(k=k), r["label_text"], labels)})


def clinc150(src):
    d = load(src)
    names = d.features["intent"].names
    label = lambda j: "out_of_scope" if names[j] == "oos" else names[j]
    labels = [label(j) for j in range(len(names))]
    instr = ["What is the user asking for in `{k}`?", "Which intent does `{k}` have?",
             "What does the assistant need to do for `{k}`?", "Classify `{k}`, or say it is out of scope."]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        k = rng.choice(["utterance", "text", "request"])
        yield line(src["name"], i, "en", {k: r["text"]},
                   {"intent": choice(rng, rng.choice(instr).format(k=k), label(int(r["intent"])), labels,
                                     always=["out_of_scope"])})


def massive(src):
    from datasets import get_dataset_config_names

    langs = [c for c in get_dataset_config_names(src["dataset"], revision=src["revision"]) if c not in ("all", "default")]
    intent_instr = ["What is the user asking for in `{k}`?", "Which intent does `{k}` express?",
                    "What should the assistant do for `{k}`?"]
    scen_instr = ["Which domain does `{k}` belong to?", "What area is `{k}` about?", "Which scenario fits `{k}`?"]
    for lang in sorted(langs):
        d = load(src, config=lang)
        s = load(src, config=lang, revision=src["also_revision"], dataset=src["also"])
        scen = {r["id"]: r["label_text"] for r in s}
        intents = sorted(set(d["label_text"]))
        scens = sorted(set(scen.values()))
        for r in d:
            rng = rng_for(src["name"], "%s.%s" % (lang, r["id"]))
            k = rng.choice(["utterance", "text", "request"])
            qs = {"intent": choice(rng, rng.choice(intent_instr).format(k=k), r["label_text"], intents)}
            if r["id"] in scen:
                qs["scenario"] = choice(rng, rng.choice(scen_instr).format(k=k), scen[r["id"]], scens)
            yield line(src["name"], "%s.%s" % (lang, r["id"]), lang, {k: r["text"]}, qs)


def nli(src, gold_of):
    d = load(src)
    for i, r in enumerate(d):
        gold = gold_of(r)
        if gold is None:
            continue
        rng = rng_for(src["name"], i)
        p, h = rng.choice([("premise", "hypothesis"), ("text", "claim"), ("context", "statement")])
        v = rng.randrange(3)
        keys = list(NLI)
        rng.shuffle(keys)
        # The options are the relation names, or yes, maybe and no, with a description each.
        if v == 2:
            names = {x: NLI[x][2] for x in keys}
            crit = {NLI[x][2]: NLI[x][0].format(p=p, h=h) for x in keys}
        else:
            names = {x: x for x in keys}
            crit = {x: NLI[x][v].format(p=p, h=h) for x in keys}
        q = {"type": "choice", "instructions": rng.choice(NLI_INSTR).format(p=p, h=h), "criteria": crit}
        t = {"probs": [1.0 if x == gold else 0.0 for x in keys], "hard": keys.index(gold), "weight": 1.0}
        assert list(crit) == [names[x] for x in keys]
        yield line(src["name"], i, "en", {p: r["premise"], h: r["hypothesis"]}, {"relation": (q, t)})


def multi_nli(src):
    return nli(src, lambda r: {0: "entailment", 1: "neutral", 2: "contradiction"}.get(int(r["label"])))


def wanli(src):
    return nli(src, lambda r: r["gold"] if r["gold"] in NLI else None)


def go_emotions(src):
    d = load(src)
    names = d.features["labels"].feature.names
    instr = ["Which emotion is most strongly expressed in `{k}`?", "How does the writer of `{k}` feel?",
             "What emotion does `{k}` show?"]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        k = rng.choice(["text", "comment", "message"])
        labs = [names[j] for j in r["labels"]]
        if len(labs) == 1:
            qs = {"emotion": choice(rng, rng.choice(instr).format(k=k), labs[0], names)}
        else:
            yes = rng.choice(labs)
            no = rng.choice([x for x in names if x not in labs])
            qs = {"shows_" + yes: noul("Does `%s` express %s?" % (k, pretty(yes)), 1.0),
                  "shows_" + no: noul("Does `%s` express %s?" % (k, pretty(no)), 0.0)}
        yield line(src["name"], i, "en", {k: r["text"]}, qs)


def civil_comments(src, cap):
    d = load(src)
    keep = min(1.0, cap / len(d)) if cap else 1.0
    instr = ["Is `{k}` toxic: rude, disrespectful or likely to make someone leave the discussion?",
             "Would most readers find `{k}` toxic?", "Is `{k}` a toxic comment?"]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        if rng.random() >= keep or not (r["text"] or "").strip():
            continue
        k = rng.choice(["comment", "post", "text"])
        # The share of raters who called it toxic is the soft target.
        yield line(src["name"], i, "en", {k: r["text"]},
                   {"toxic": noul(rng.choice(instr).format(k=k), float(r["toxicity"]))})


def prompt_injections(src):
    d = load(src)
    instr = ["Does `{k}` try to inject or override instructions given to an AI system?",
             "Is `{k}` a prompt injection?", "Does `{k}` try to make an assistant ignore its instructions?"]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        k = rng.choice(["text", "prompt", "input"])
        yield line(src["name"], i, "en", {k: r["text"]},
                   {"injection": noul(rng.choice(instr).format(k=k), float(int(r["label"])))})


def typed_decisions(src):
    d = load(src)
    for i, r in enumerate(d):
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        qs = {}
        for qid, q in questions.items():
            g = gold[qid]
            probs = g.get("probabilities", {})
            if q["type"] == "noul":
                pt = float(probs.get("true", g.get("noul", 0.5)))
                p = [1 - pt, pt]
            else:
                # Score levels are keyed by their index.
                keys = list(q["criteria"]) if q["type"] == "choice" else [str(j) for j in range(len(q["criteria"]))]
                p = [float(probs.get(x, 0.0)) for x in keys]
            s = sum(p) or 1.0
            p = [x / s for x in p]
            qs[qid] = (q, {"probs": p, "hard": max(range(len(p)), key=p.__getitem__), "weight": 1.0})
        yield line(src["name"], r["id"], "en", state, qs)


def boolq(src):
    d = load(src)
    instr = ["Based on `{p}`, is the answer to `{q}` yes?", "Does `{p}` say yes to `{q}`?",
             "According to `{p}`, is `{q}` true?"]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        p, q = rng.choice([("passage", "question"), ("document", "question"), ("context", "query")])
        yield line(src["name"], i, "en", {p: r["passage"], q: r["question"]},
                   {"answer": noul(rng.choice(instr).format(p=p, q=q), float(bool(r["answer"])))})


def squad_v2(src):
    d = load(src)
    instr = ["Does `{p}` answer `{q}`?", "Can `{q}` be answered from `{p}` alone?",
             "Is the answer to `{q}` in `{p}`?"]
    for i, r in enumerate(d):
        rng = rng_for(src["name"], i)
        p, q = rng.choice([("passage", "question"), ("document", "query"), ("context", "question")])
        yield line(src["name"], i, "en", {p: r["context"], q: r["question"]},
                   {"answerable": noul(rng.choice(instr).format(p=p, q=q), float(bool(r["answers"]["text"])))})


def mind2web(src):
    """Every usable step of Mind2Web's train split, one file at a time, each deleted once read."""
    import tempfile

    import ijson
    import kime
    from huggingface_hub import HfApi, hf_hub_download

    import mind2web as m2w

    files = sorted(f for f in HfApi().list_repo_files(src["dataset"], repo_type="dataset", revision=src["revision"])
                   if f.startswith("data/train/"))
    with tempfile.TemporaryDirectory() as tmp:
        for name in files:
            path = hf_hub_download(src["dataset"], name, repo_type="dataset", revision=src["revision"], local_dir=tmp)
            with open(path, "rb") as f:
                for task in ijson.items(f, "item", use_float=True):
                    for uid, body, targets, _ in m2w.steps(task, kime.agent_step):
                        yield {"id": "mind2web-%s" % uid, "source": "mind2web", "split": "train", "lang": "en",
                               "state": body["state"], "questions": body["questions"], "targets": targets,
                               "episode": {"task": task["annotation_id"], "website": task["website"], "domain": task["domain"]}}
            os.remove(path)


CONVERTERS = {
    "banking77": banking77, "clinc150": clinc150, "massive": massive, "multi_nli": multi_nli,
    "wanli": wanli, "go_emotions": go_emotions, "civil_comments": civil_comments,
    "prompt_injections": prompt_injections, "typed_decisions": typed_decisions, "boolq": boolq,
    "squad_v2": squad_v2, "mind2web": mind2web,
}


def shard(path, name, out, info):
    """Cuts a JSONL file into zstd shards of about SHARD_BYTES and returns their manifest entries."""
    shards, buf, rows, qs = [], [], 0, 0

    def flush():
        if not buf:
            return
        data = "".join(buf).encode("utf-8")
        blob = zstandard.ZstdCompressor(level=10).compress(data)
        fname = "%s-%05d.jsonl.zst" % (name, len(shards))
        with open(os.path.join(out, fname), "wb") as f:
            f.write(blob)
        shards.append(dict(info, file=fname, blake3=blake3.blake3(blob).hexdigest(), bytes=len(blob),
                           raw_bytes=len(data), rows=rows, questions=qs))

    size = 0
    for l in open(path, encoding="utf-8"):
        buf.append(l)
        size += len(l.encode("utf-8"))
        rows += 1
        qs += len(json.loads(l)["questions"])
        if size >= SHARD_BYTES:
            flush()
            buf, size, rows, qs = [], 0, 0, 0
    flush()
    return shards


def main():
    global MAX_OPTS
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--only", default="")
    p.add_argument("--max", type=int, default=300000, help="rows to sample from civil_comments")
    p.add_argument("--max-options", type=int, default=MAX_OPTS, help="most options a choice over a large label set gets")
    p.add_argument("--kime", default="")
    p.add_argument("--tests", action="append", default=[])
    p.add_argument("--reuse", action="store_true", help="keep raw files already written, for checking elsewhere")
    a = p.parse_args()
    MAX_OPTS = a.max_options
    only = [x for x in a.only.split(",") if x]
    raw, clean, shards_dir = (os.path.join(a.out, x) for x in ("raw", "clean", "shards"))
    for d in (raw, clean, shards_dir):
        os.makedirs(d, exist_ok=True)
    manifest = {"format": "kime training JSONL, spec/12-training.md", "seed": SEED,
                "created": datetime.date.today().isoformat(), "sources": [], "shards": []}
    for src in REGISTRY:
        name = src["name"]
        if src["use"] != "train" or (only and name not in only):
            continue
        path = os.path.join(raw, name + ".jsonl")
        if a.reuse and os.path.exists(path):
            n = sum(1 for _ in open(path, encoding="utf-8"))
        else:
            conv = CONVERTERS[name]
            rows = conv(src, a.max) if name == "civil_comments" else conv(src)
            n = 0
            with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
                for l in rows:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")
                    n += 1
            os.replace(path + ".part", path)
        entry = {k: src[k] for k in ("name", "dataset", "config", "revision", "split", "license", "task") if k in src}
        entry["rows"] = n
        if MAX_OPTS != 20:
            entry["max_options"] = MAX_OPTS
        kept = path
        if a.kime and a.tests:
            report = os.path.join(a.out, name + ".contam.json")
            cmd = [a.kime, "contam", path, "--out", clean, "--report", report]
            for t in a.tests:
                cmd += ["--tests", t]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            r = json.load(open(report, encoding="utf-8"))
            entry["near_duplicates"] = r["data"]["near_duplicates"]
            entry["contamination"] = {"method": r["method"], "test_texts": r["tests"]["texts"]}
            kept = os.path.join(clean, name + ".jsonl")
        info = {"source": name, "dataset": src["dataset"], "config": src.get("config"), "split": src["split"],
                "license": src["license"]}
        sh = shard(kept, name, shards_dir, info)
        entry["kept"] = sum(s["rows"] for s in sh)
        entry["questions"] = sum(s["questions"] for s in sh)
        manifest["sources"].append(entry)
        manifest["shards"].extend(sh)
        print("  %-18s %8d rows, %8d kept, %8d questions, %d shards" % (name, n, entry["kept"], entry["questions"], len(sh)),
              flush=True)
    path = os.path.join(a.out, "manifest.json")
    if only and os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        names = {s["name"] for s in manifest["sources"]}
        manifest["sources"] = [s for s in old["sources"] if s["name"] not in names] + manifest["sources"]
        manifest["shards"] = [s for s in old["shards"] if s["source"] not in names] + manifest["shards"]
    manifest["sources"].sort(key=lambda s: s["name"])
    manifest["shards"].sort(key=lambda s: s["file"])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
