"""The PyTorch reference for kime-train, so a kime-train run can be checked against a second trainer.

It trains Laya's own DecisionModel, built the way laya.Agent builds it, on the questions
`kime-train --dump` writes, with Laya's own `proper_reward` as the loss. Everything else is what
kime-train does, written again here: AdamW with betas 0.9 and 0.98, weight decay 0.01 and epsilon
1e-8, linear warmup and cosine decay to 1e-6, the global gradient norm clipped at 1.0, batches of
one padded length up to a token budget, and the same freezing as `--train-layers`. Dropout is off,
as it is in kime-train. The checkpoint it writes has the base's tensor names and dtypes, so
`kime eval` scores it like any other.

    kime-train --base <laya> --data <shards> --only banking77,clinc150 --max-lines 12000 --out x --dump data
    python tools/ref/train_ref.py --base <laya> --data data --out ref --train-layers 4 --tokens 2048
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from laya.common import build_model, proper_reward

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights


def load(base, device):
    with open(os.path.join(base, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    with no_init_weights():
        model = build_model(cfg, encoder_dir=os.path.join(base, "encoder"), pretrained=False)
    sd, dtypes = {}, {}
    with safe_open(os.path.join(base, "model.safetensors"), "pt") as f:
        meta = f.metadata()
        for name in f.keys():
            t = f.get_tensor(name)
            dtypes[name] = t.dtype
            sd[name] = t.float()
    model.load_state_dict(sd, strict=True)
    return model.to(device), cfg, dtypes, meta


def freeze_below(model, train):
    """Freezes the embeddings and every encoder layer but the top `train` ones."""
    layers = model.encoder.layers
    keep = max(0, len(layers) - train)
    if keep == 0:
        return
    for p in model.encoder.embeddings.parameters():
        p.requires_grad_(False)
    for layer in layers[:keep]:
        for p in layer.parameters():
            p.requires_grad_(False)


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def plan(examples, tokens, bucket, rng):
    """Batches of one padded length, as many as fit in `tokens`, in a random order."""
    order = list(range(len(examples)))
    rng.shuffle(order)
    by_len = {}
    for i in order:
        t = -(-len(examples[i]["ids"]) // bucket) * bucket
        by_len.setdefault(t, []).append(i)
    out = []
    for t in sorted(by_len):
        idx = by_len[t]
        n = max(1, tokens // t)
        out.extend((t, idx[j:j + n]) for j in range(0, len(idx), n))
    rng.shuffle(out)
    return out


def collate(examples, t, pad, device):
    n = len(examples)
    k = max(len(e["markers"]) for e in examples)
    ids = torch.full((n, t), pad, dtype=torch.long)
    att = torch.zeros((n, t), dtype=torch.long)
    mpos = torch.zeros((n, k), dtype=torch.long)
    mmask = torch.zeros((n, k), dtype=torch.bool)
    target = torch.zeros((n, k), dtype=torch.float32)
    for i, e in enumerate(examples):
        ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
        att[i, :len(e["ids"])] = 1
        m = len(e["markers"])
        mpos[i, :m] = torch.tensor(e["markers"])
        mmask[i, :m] = True
        target[i, :m] = torch.tensor(e["probs"])
    qtype = torch.tensor([e["qtype"] for e in examples])
    weight = torch.tensor([e["weight"] for e in examples], dtype=torch.float32)
    return [x.to(device) for x in (ids, att, mpos, mmask, qtype, target, weight)]


def batch_loss(model, examples, t, pad, device):
    ids, att, mpos, mmask, qtype, target, weight = collate(examples, t, pad, device)
    logits, _ = model(ids, att, mpos, mmask, qtype)
    p = torch.softmax(logits, -1)
    r = proper_reward(p, target, qtype, mmask.float())
    return -(r * weight).sum() / weight.sum().clamp_min(1e-12), p


def lr_at(step, total, lr, min_lr, warmup):
    warm = max(1, round(warmup * total))
    if step < warm:
        return lr * (step + 1) / warm
    done = min(1.0, (step - warm) / max(1, total - warm))
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * done))


@torch.no_grad()
def evaluate(model, examples, a, pad, device):
    loss = weight = right = gold = nll = 0.0
    for t, idx in plan(examples, a.tokens, a.bucket, random.Random(0)):
        ex = [examples[i] for i in idx]
        l, p = batch_loss(model, ex, t, pad, device)
        w = sum(e["weight"] for e in ex)
        loss += l.item() * w
        weight += w
        p = p.float().cpu()
        for s, e in enumerate(ex):
            if e.get("hard") is None:
                continue
            row = p[s, :len(e["probs"])]
            right += int(int(torch.argmax(row)) == e["hard"])
            gold += 1
            nll -= math.log(max(float(row[e["hard"]]), 1e-4))
    return {"loss": loss / max(weight, 1e-12), "accuracy": right / max(gold, 1), "nll": nll / max(gold, 1)}


def export(model, base, out, dtypes, meta, note):
    tmp = out + ".part"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base, tmp, ignore=shutil.ignore_patterns("model.safetensors"))
    sd = model.state_dict()
    tensors = {name: sd[name].detach().to("cpu", dtypes[name]).contiguous() for name in dtypes}
    save_file(tensors, os.path.join(tmp, "model.safetensors"), metadata=meta)
    cfg_path = os.path.join(tmp, "rl_agent_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["kime_ref"] = note
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    shutil.rmtree(out, ignore_errors=True)
    os.rename(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True, help="the folder kime-train --dump wrote")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=float, default=0.02)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--bucket", type=int, default=64)
    ap.add_argument("--train-layers", type=int, default=0, help="0 trains every layer")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model, cfg, dtypes, meta = load(a.base, a.device)
    pad = model.encoder.config.pad_token_id
    if a.train_layers:
        freeze_below(model, a.train_layers)
    # Dropout off, as in kime-train. Gradients still flow in eval mode.
    model.eval()
    train, held = read(os.path.join(a.data, "train.jsonl")), read(os.path.join(a.data, "eval.jsonl"))
    log = open(a.log, "w") if a.log else None

    def write(v):
        if log:
            log.write(json.dumps(v) + "\n")
            log.flush()

    t0 = time.time()
    before = evaluate(model, held, a, pad, a.device)
    print(f"before: loss {before['loss']:.4f} accuracy {before['accuracy']:.4f} nll {before['nll']:.4f} on {len(held)} questions ({time.time() - t0:.0f}s)", flush=True)
    write({"eval": 0, **before})

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    batches = plan(train, a.tokens, a.bucket, random.Random(a.seed))
    total = len(batches)
    tokens = 0
    t0 = time.time()
    for step, (t, idx) in enumerate(batches):
        lr = lr_at(step, total, a.lr, a.min_lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        ex = [train[i] for i in idx]
        loss, _ = batch_loss(model, ex, t, pad, a.device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        tokens += sum(len(e["ids"]) for e in ex)
        done = step + 1
        rate = tokens / (time.time() - t0)
        write({"step": done, "lr": lr, "loss": loss.item(), "grad_norm": norm, "tokens_per_s": rate})
        if done % 10 == 0 or done in (1, total):
            print(f"step {done}/{total} lr {lr:.2e} loss {loss.item():.4f} grad norm {norm:.3f} {rate:.0f} tokens/s", flush=True)
        if a.eval_every and done % a.eval_every == 0 and done < total:
            m = evaluate(model, held, a, pad, a.device)
            print(f"eval at step {done}: loss {m['loss']:.4f} accuracy {m['accuracy']:.4f} nll {m['nll']:.4f}", flush=True)
            write({"eval": done, **m})
    secs = time.time() - t0
    after = evaluate(model, held, a, pad, a.device)
    print(f"eval at step {total}: loss {after['loss']:.4f} accuracy {after['accuracy']:.4f} nll {after['nll']:.4f}", flush=True)
    write({"eval": total, **after})
    note = {"trainer": "tools/ref/train_ref.py", "torch": torch.__version__, "device": a.device, "training_questions": len(train),
            "held_out_questions": len(held), "lr": a.lr, "tokens": a.tokens, "seed": a.seed, "train_layers": a.train_layers,
            "seconds": round(secs), "held_out_before": before, "held_out_after": after}
    export(model, a.base, a.out, dtypes, meta, note)
    print(f"trained in {secs:.0f}s, wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
