"""Writes the whole of every split the quality suites are drawn from, for `kime contam --tests`.

    python tools/eval/split_texts.py <suites dir> <out dir> [--split train] [--only name,name]
        [--configs name,name] [--max-rows N]

The suites hold a sample of each test split, and the contamination rules in spec/12-training.md
are about the whole test set, so this reads the manifest.json build_suites.py wrote next to the
suites and writes each dataset, config and split in it as <dataset>.<config>.<split>.jsonl, one
{"id": ..., "state": {...}} line per row with the fields the suite's state is made of. MS MARCO
writes one line per query and passage, as the RAG suite pairs them.

With --split, the named split of each dataset is written instead, when it has one. The train
splits are what the check was measured on in tools/eval/README.md. --only keeps the datasets whose
name contains one of the given strings, and --configs the configs named, for datasets that have
one, such as the languages of MASSIVE and XNLI.
"""

import argparse
import json
import os

from datasets import load_dataset

FIELDS = {
    "LocalLLaMA/typed-decisions": ["state"],
    "mteb/amazon_massive_intent": ["text"],
    "mteb/amazon_massive_scenario": ["text"],
    "facebook/xnli": ["premise", "hypothesis"],
    "SetFit/sst5": ["text"],
    "SetFit/sst2": ["text"],
    "dair-ai/emotion": ["text"],
    "deepset/prompt-injections": ["text"],
    "mteb/banking77": ["text"],
    "fancyzhx/ag_news": ["text"],
    "google/boolq": ["passage", "question"],
    "clinc/clinc_oos": ["text"],
    "Tobi-Bueck/customer-support-tickets": ["subject", "body"],
    "SetFit/enron_spam": ["subject", "message"],
    "zefang-liu/phishing-email-dataset": ["Email Text"],
    "lmsys/toxic-chat": ["user_input"],
    "openai/gsm8k": ["question"],
    "google-research-datasets/mbpp": ["text"],
}


def rows(repo, d, max_rows):
    n = 0
    for i, r in enumerate(d):
        if repo == "microsoft/ms_marco":
            for k, t in enumerate(r["passages"]["passage_text"]):
                yield "%d.%d" % (i, k), {"query": r["query"], "passage": t}
        else:
            yield str(i), {f: r[f] for f in FIELDS[repo] if isinstance(r.get(f), str)}
        n += 1
        if max_rows and n >= max_rows:
            return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("suites")
    p.add_argument("out")
    p.add_argument("--split", default="")
    p.add_argument("--only", default="")
    p.add_argument("--configs", default="")
    p.add_argument("--max-rows", type=int, default=0)
    a = p.parse_args()
    manifest = json.load(open(os.path.join(a.suites, "manifest.json"), encoding="utf-8"))
    splits = set()
    for src in manifest.values():
        for repo in (src.get("dataset") or "").split(","):
            if repo.strip():
                splits.add((repo.strip(), src.get("config"), src["split"]))
    if any(r == "openai/gsm8k" for r, _, _ in splits):
        # The routing suite names three datasets with the config and split of the first.
        splits -= {("google-research-datasets/mbpp", "main", "test"), ("fancyzhx/ag_news", "main", "test")}
        splits |= {("google-research-datasets/mbpp", "full", "test"), ("fancyzhx/ag_news", None, "test")}
    only = [x for x in a.only.split(",") if x]
    configs = [x for x in a.configs.split(",") if x]
    os.makedirs(a.out, exist_ok=True)
    done = set()
    for repo, config, split in sorted(splits, key=str):
        if only and not any(x in repo for x in only):
            continue
        if configs and config and config not in configs:
            continue
        if repo not in FIELDS and repo != "microsoft/ms_marco":
            print("  skip %s, no fields listed" % repo)
            continue
        split = a.split or split
        if (repo, config, split) in done:
            continue
        done.add((repo, config, split))
        try:
            d = load_dataset(repo, config, split=split)
        except Exception as e:
            print("  skip %s %s %s: %s" % (repo, config, split, str(e)[:120]))
            continue
        name = "%s.%s.%s" % (repo.replace("/", "__"), config or "default", split)
        n = 0
        with open(os.path.join(a.out, name + ".jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for i, state in rows(repo, d, a.max_rows):
                f.write(json.dumps({"id": i, "state": state}, ensure_ascii=False) + "\n")
                n += 1
        print("  %-60s %8d lines" % (name, n))


if __name__ == "__main__":
    main()
