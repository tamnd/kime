"""Answers quality suites with Laya, for `kime eval --answers`.

    python tools/eval/run_laya.py <laya model dir> <out dir> <suite.jsonl>... [--device cuda] [--api]

By default the questions go through the model the way Laya's benchmark notebook scores them
(`score_cases` in research/scripts/build_benchmark_nb.py): every question is one sequence, the
sequences are packed into length sorted batches under fp16 autocast, and the calibrated
temperature of the question's type and option count is applied to the logits. The probabilities
are written unrounded, so the NLL and ECE are the ones the notebook would compute. With --api each
case goes through `Agent.system_one` instead, the public API, which rounds to 4 places.

Writes <out dir>/<suite>.answers.jsonl with one {"id": ..., "answers": {...}} line per case.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, collate_items, render_options, temp_bucket


def to_internal(q):
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = q["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": q["type"], "ins": ins, "crit": crit}


@torch.no_grad()
def logits_of(agent, cases, max_tokens=16384, max_seqs=128):
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    items, index = [], []
    for ci, c in enumerate(cases):
        for qid, q in c["questions"].items():
            iq = to_internal(q)
            try:
                ids, markers = build_sequence(agent.tok, c["state"], iq, max_len, head_max_len)
            except Exception:
                ids, markers = None, []
            if ids is None or len(markers) != len(render_options(iq)):
                items.append(None)
            else:
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[iq["t"]]})
            index.append((ci, qid, QTYPES[iq["t"]], len(markers)))
    valid = [i for i, it in enumerate(items) if it is not None]
    order = sorted(valid, key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    dev = agent.model.parameters().__next__().device
    i = 0
    while i < len(order):
        j, L = i, 0
        while j < len(order) and j - i < max_seqs and max(L, len(items[order[j]]["ids"])) * (j - i + 1) <= max_tokens:
            L = max(L, len(items[order[j]]["ids"]))
            j += 1
        j = max(j, i + 1)
        sel = [items[order[t]] for t in range(i, j)]
        b = collate_items([sel], agent.tok.pad_token_id)
        with torch.autocast(dev.type, dtype=agent.dtype, enabled=dev.type == "cuda"):
            logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                                    b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
        logits = logits.float().cpu().numpy()
        for r in range(j - i):
            out[order[i + r]] = logits[r, :len(sel[r]["markers"])]
        i = j
    return out, index


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / max(1e-3, float(t))
    e = np.exp(z - z.max())
    return e / e.sum()


def answer(q, p):
    if q["type"] == "noul":
        return {"type": "noul", "noul": float(p[1])}
    if q["type"] == "score":
        return {"type": "score", "probabilities": {str(i): float(x) for i, x in enumerate(p)}}
    crit = q["criteria"]
    return {"type": "choice", "probabilities": {k: float(x) for k, x in zip(list(crit), p)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("suites", nargs="+")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--api", action="store_true")
    a = ap.parse_args()
    agent = Agent(a.model, device=a.device)
    try:
        agent.model.encoder.config.reference_compile = False
    except Exception:
        pass
    agent.model.eval()
    os.makedirs(a.out, exist_ok=True)
    for path in a.suites:
        name = os.path.basename(path)[: -len(".jsonl")]
        cases = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        t = time.time()
        answers = [{} for _ in cases]
        if a.api:
            for ci, c in enumerate(cases):
                try:
                    answers[ci] = agent.system_one(c["state"], c["questions"])["answers"]
                except Exception as e:
                    print("  %s: %s" % (c["id"], str(e)[:120]), file=sys.stderr)
        else:
            logits, index = logits_of(agent, cases)
            for (ci, qid, qt, k), z in zip(index, logits):
                if z is None:
                    continue
                temp = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
                answers[ci][qid] = answer(cases[ci]["questions"][qid], softmax(z, temp))
        with open(os.path.join(a.out, name + ".answers.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for c, ans in zip(cases, answers):
                f.write(json.dumps({"id": c["id"], "answers": ans}, ensure_ascii=False) + "\n")
        print("%s: %d cases in %.1fs" % (name, len(cases), time.time() - t), flush=True)


if __name__ == "__main__":
    main()
