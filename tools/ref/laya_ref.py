"""Dumps what Laya 0.3.7 computes for every parity case, so kime can be checked against it without PyTorch.

For each case and each question this records the rendered pieces of text Laya tokenizes, the ids of
each piece before truncation, the final token ids and marker positions, the option logits and act
logits straight out of the model, and the answer `Agent.system_one` returns. The model runs through
Laya's own `Agent`, and the logits are taken from a hook on its forward, so the numbers are the
ones a Laya user gets and not a re-implementation of them.

    python tools/ref/laya_ref.py --model models/laya --name laya --device cpu
    python tools/ref/laya_ref.py --model models/laya/multilingual --name laya-multilingual --device cpu

The default device is the CPU in FP32, which is the reference every kime backend is measured against.
`--device cuda` runs Laya's own autocast path and is only useful to see how far Laya's GPU numbers
drift from its CPU ones. `--dump-hidden DIR` also writes the hidden state after every encoder layer,
the head and the scorer to one .npz file per question, for tracking down the first op that differs.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, render_options, serialize_state

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "crates", "kime-eval", "fixtures", "parity")


def pieces(tok, state, q):
    """The strings Laya passes to the tokenizer, in order, before any truncation."""
    mask = tok.mask_token
    head = "%s question: %s" % (q["t"], str(q["ins"]).replace(mask, " "))
    opts = [" " + o.replace(mask, " ") for o in render_options(q)]
    st = serialize_state(state).replace(mask, " ")
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    return {
        "head": head,
        "options": opts,
        "state": st,
        "head_ids": enc(head),
        "option_ids": [enc(o) for o in opts],
        "state_ids": enc(st),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a local Laya checkpoint directory")
    ap.add_argument("--name", required=True, help="the kime model id, used for the output file name")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cases", default=os.path.join(FIXTURES, "cases.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-hidden", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    agent = Agent(args.model, device=args.device)
    tok = agent.tok
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)

    captured = {}
    inner = agent.model.forward

    def hooked(*a, **kw):
        logits, act = inner(*a, **kw)
        captured["logits"] = logits.detach().float().cpu().numpy()
        captured["act"] = act.detach().float().cpu().numpy()
        captured["input_ids"] = a[0].cpu().numpy()
        return logits, act

    agent.model.forward = hooked

    hidden = []
    if args.dump_hidden:
        os.makedirs(args.dump_hidden, exist_ok=True)
        mods = [("embeddings", agent.model.encoder.embeddings)]
        mods += [("layer%02d" % i, m) for i, m in enumerate(agent.model.encoder.layers)]
        mods += [("final_norm", agent.model.encoder.final_norm)]
        mods += [("head%d" % i, m) for i, m in enumerate(agent.model.head.layers)]
        for name, m in mods:
            m.register_forward_hook(lambda mod, inp, out, name=name: hidden.append(
                (name, (out[0] if isinstance(out, tuple) else out).detach().float().cpu().numpy())))

    out_path = args.out or os.path.join(FIXTURES, "%s.jsonl" % args.name)
    cases = [json.loads(line) for line in open(args.cases, encoding="utf-8")]
    t0 = time.time()
    n_q = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for case in cases:
            row = {"id": case["id"], "model": args.name}
            hidden.clear()
            try:
                res = agent.system_one(case["state"], case["questions"])
            except Exception as e:  # noqa: BLE001, the error text is part of the fixture
                row["error"] = "%s: %s" % (type(e).__name__, e)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            qs = []
            for r, (qid, qdef) in enumerate(case["questions"].items()):
                q = Agent._to_internal(qdef)
                ids, markers = build_sequence(tok, case["state"], q, max_len, head_max_len)
                assert list(captured["input_ids"][r][: len(ids)]) == ids, "ids differ from the batch"
                k = len(markers)
                entry = {
                    "qid": qid,
                    "type": q["t"],
                    "qtype": QTYPES[q["t"]],
                    "pieces": pieces(tok, case["state"], q),
                    "ids": ids,
                    "markers": markers,
                    "logits": [float(x) for x in captured["logits"][r, :k]],
                    "act_probs": [float(x) for x in captured["act"][r]],
                }
                qs.append(entry)
                if args.dump_hidden:
                    np.savez_compressed(
                        os.path.join(args.dump_hidden, "%s-%s-%s.npz" % (args.name, case["id"], qid)),
                        **{name: h[r, : len(ids)] for name, h in hidden})
            n_q += len(qs)
            row["questions"] = qs
            row["answer"] = res
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print("%s: %d cases, %d questions in %.1fs on %s, wrote %s" % (
        args.name, len(cases), n_q, dt, args.device, os.path.relpath(out_path)), file=sys.stderr)


if __name__ == "__main__":
    main()
