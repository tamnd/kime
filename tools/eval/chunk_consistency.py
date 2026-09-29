"""Checks that a big choice gets the same answer whatever size of chunks it is scored in, the chunk
consistency test of spec/15-testing.md.

`build` takes suites with choice questions of at least 64 options, and gives each question 64 of
them: the gold one and others drawn with a seed, kept in their order. It writes one request file
per chunk size, with `kime.chunk` set, and one with no size, which scores the choice in one pass
when it fits. `compare` reads the responses back and prints how often each size picks the same
option as the others, and how often it picks the gold one.

    python tools/eval/chunk_consistency.py build <suite.jsonl>... --out <dir> --n 300
    for f in <dir>/requests.*.jsonl; do kime predict --batch < $f > ${f/requests/responses}; done
    python tools/eval/chunk_consistency.py compare <dir>
"""

import argparse
import glob
import json
import os
import random

SIZES = ["one", "8", "16", "32"]


def build(a):
    os.makedirs(a.out, exist_ok=True)
    cases = []
    for f in a.suites:
        for line in open(f, encoding="utf-8"):
            c = json.loads(line)
            for k, q in c["questions"].items():
                crit = q["criteria"]
                if q["type"] != "choice" or len(crit) < a.options or k not in c["gold"]:
                    continue
                cases.append((c, k))
    rng = random.Random(a.seed)
    rng.shuffle(cases)
    cases = cases[: a.n]
    outs = {s: open(os.path.join(a.out, "requests.%s.jsonl" % s), "w", encoding="utf-8") for s in SIZES}
    with open(os.path.join(a.out, "gold.jsonl"), "w", encoding="utf-8") as g:
        for c, k in cases:
            q, gold = c["questions"][k], c["gold"][k]
            crit = q["criteria"]
            keys = list(crit)
            r = random.Random("%s:%s:%d" % (c["id"], k, a.seed))
            keep = set(r.sample([x for x in keys if x != gold], a.options - 1)) | {gold}
            q = dict(q, criteria={x: crit[x] for x in keys if x in keep})
            qid = "%s/%s" % (c["id"], k)
            for s in SIZES:
                req = {"state": c["state"], "questions": {qid: q}}
                if s != "one":
                    req["kime"] = {"chunk": int(s)}
                outs[s].write(json.dumps(req, ensure_ascii=False) + "\n")
            g.write(json.dumps({"id": qid, "gold": gold}) + "\n")
    for f in outs.values():
        f.close()
    print("%d questions of %d options from %d suites" % (len(cases), a.options, len(a.suites)))


def picks(path):
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "answers" not in r:
            out.append(None)
            continue
        (a,) = r["answers"].values()
        out.append(a["choice"])
    return out


def compare(a):
    gold = [json.loads(l)["gold"] for l in open(os.path.join(a.dir, "gold.jsonl"), encoding="utf-8")]
    got = {}
    for s in SIZES:
        p = os.path.join(a.dir, "responses.%s.jsonl" % s)
        if os.path.exists(p):
            v = picks(p)
            if len(v) == len(gold):
                got[s] = v
    n = len(gold)
    print("| Chunk size | Answered | Top 1 |")
    print("|---|---:|---:|")
    for s, v in got.items():
        ok = [x for x in v if x is not None]
        print("| %s | %d | %.3f |" % (s, len(ok), sum(x == y for x, y in zip(v, gold)) / n))
    print()
    print("| Sizes | Same pick |")
    print("|---|---:|")
    names = list(got)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            both = [(p, q) for p, q in zip(got[x], got[y]) if p is not None and q is not None]
            if both:
                print("| %s and %s | %.3f of %d |" % (x, y, sum(p == q for p, q in both) / len(both), len(both)))
    chunked = [s for s in names if s != "one"]
    if len(chunked) > 1:
        rows = list(zip(*(got[s] for s in chunked)))
        agree = sum(len(set(r)) == 1 for r in rows if None not in r)
        print("\nAll of %s agree on %d of %d (%.3f)." % (", ".join(chunked), agree, len(rows), agree / len(rows)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("suites", nargs="+")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=300)
    b.add_argument("--options", type=int, default=64)
    b.add_argument("--seed", type=int, default=13)
    c = sub.add_parser("compare")
    c.add_argument("dir")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else compare(a)


if __name__ == "__main__":
    main()
