"""Builds a tokenizer parity corpus and the ids Hugging Face tokenizers gives for it.

The corpus is every utterance in the train split of MASSIVE (51 languages, from the
mteb/amazon_massive_intent parquet files) plus a fuzz set of random strings built to hit the edges:
combining marks, control chars, emoji and ZWJ sequences, runs of mixed whitespace, lone
surrogates decoded lossily, and the mask literals. Each line of the output is a JSON object with
the text and the ids from each tokenizer, encoded without special tokens.

    python tools/ref/tok_corpus.py --models <dir>/laya --out corpus.jsonl [--fuzz 200000]

It also times Hugging Face on the same lines with one thread, and prints lines per second and
bytes per second, so the kime side can be compared on the same corpus.
"""
import argparse
import io
import json
import os
import random
import sys
import time
import urllib.request

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"

from tokenizers import Tokenizer  # noqa: E402

API = "https://huggingface.co/api/datasets/mteb/amazon_massive_intent/parquet"


def massive(cache):
    import pyarrow.parquet as pq

    os.makedirs(cache, exist_ok=True)
    have = sorted(f[: -len("-train.parquet")] for f in os.listdir(cache) if f.endswith("-train.parquet"))
    # A full cache is used as is, so the script also runs on a box that cannot reach the Hub.
    langs = dict.fromkeys(have) if len(have) == 51 else json.load(urllib.request.urlopen(API))
    lines = []
    for lang in sorted(langs):
        path = os.path.join(cache, "%s-train.parquet" % lang)
        if not os.path.exists(path):
            urllib.request.urlretrieve(langs[lang]["train"][0], path)
        table = pq.read_table(path)
        col = "text" if "text" in table.column_names else table.column_names[0]
        lines.extend(str(x) for x in table.column(col).to_pylist())
    return lines


def fuzz(n, seed=2144):
    rng = random.Random(seed)
    pools = [
        [chr(c) for c in range(0x20, 0x7F)],
        [chr(c) for c in range(0x00, 0x20)] + ["\u0085", " ", " ", " ", "　", "​", "﻿"],
        [chr(c) for c in range(0x300, 0x370)],
        [chr(c) for c in range(0x4E00, 0x4E00 + 3000)],
        [chr(c) for c in range(0x0600, 0x06FF)] + [chr(c) for c in range(0x0900, 0x097F)],
        ["😀", "👍🏽", "👨‍👩‍👧", "🏳️‍🌈", "🇻🇳", "‍", "️"],
        [" ", "  ", "   ", "\t", "\n", "\r\n", " \n "],
        ["[MASK]", "<mask>", "[CLS]", "<bos>", "|||EMAIL_ADDRESS|||", "'s", "'ll", "n't", "▁", "Ġ"],
        [chr(c) for c in range(0x10000, 0x10100)] + [chr(c) for c in range(0x1F300, 0x1F400)],
    ]
    out = []
    for _ in range(n):
        k = rng.randint(1, 60)
        s = "".join(rng.choice(rng.choice(pools)) for _ in range(k))
        # Lone surrogates cannot be encoded, so they go through a lossy round trip like bad input would.
        if rng.random() < 0.05:
            s = (s + "\ud800").encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="massive-cache")
    ap.add_argument("--fuzz", type=int, default=200000)
    args = ap.parse_args()

    lines = massive(args.cache) + fuzz(args.fuzz)
    toks = {
        "laya": Tokenizer.from_file(os.path.join(args.models, "tokenizer", "tokenizer.json")),
        "laya-multilingual": Tokenizer.from_file(os.path.join(args.models, "multilingual", "tokenizer", "tokenizer.json")),
    }
    total_bytes = sum(len(s.encode("utf-8")) for s in lines)
    ids = {}
    for name, tok in toks.items():
        t0 = time.perf_counter()
        encs = tok.encode_batch(lines, add_special_tokens=False)
        dt = time.perf_counter() - t0
        ids[name] = [e.ids for e in encs]
        print("%s: %d lines, %.1f MB in %.2fs with one thread, %.0f lines/s, %.1f MB/s" % (
            name, len(lines), total_bytes / 1e6, dt, len(lines) / dt, total_bytes / 1e6 / dt), file=sys.stderr)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, s in enumerate(lines):
            f.write(json.dumps({"text": s, "laya": ids["laya"][i], "laya-multilingual": ids["laya-multilingual"][i]}, ensure_ascii=False) + "\n")
    print("wrote %d lines to %s" % (len(lines), args.out), file=sys.stderr)


if __name__ == "__main__":
    main()
