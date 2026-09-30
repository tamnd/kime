"""A reference forward of the kime-v1 split encoder, written from spec/05-model.md and spec/06-input.md
and not from kime's Rust code, so the two can be checked against each other.

It runs in float64 numpy and needs nothing else. With no weights it draws the same seeded random
weights as `kime_model::kime_v1::random_weights`, runs a fixed set of states and question rows, and
writes the logits and pooled embeddings to crates/kime-cpu/tests/fixtures/kime-v1-random.json,
which the kime-cpu test `kime_v1_parity` checks the CPU forward against.

    python tools/ref/kime_ref.py
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "crates", "kime-cpu", "tests", "fixtures", "kime-v1-random.json")
HEAD = 64
SEED = 7

CONFIG = {
    "format": "kime/1", "family": "kime-v1", "id": "kime-v1-tiny", "version": "0.0.0",
    "tokenizer": {"kind": "bpe-bytelevel", "cls": 1, "sep": 2, "pad": 0,
                  "specials": {"opt": 3, "no": 4, "yes": 5, "cls_s": 6, "cls_q": 7}},
    "dims": {"d": 256, "heads": 4, "head_dim": 64, "inter": 320, "vocab": 400},
    "state_tower": {"layers": 4, "global_every": 3, "window": 16, "rope_theta_global": 160000.0,
                    "rope_theta_local": 10000.0, "segment_top_layers": 3, "max_tokens": 1024},
    "question_tower": {"layers": 2, "kv_heads": 2, "max_options_per_chunk": 32, "max_tokens": 512},
    "gelu": "erf", "norm_eps": 1e-5,
}


def names(c):
    """Tensor names and shapes in the order the checkpoint lists them."""
    d, inter, vocab = c["dims"]["d"], c["dims"]["inter"], c["dims"]["vocab"]
    kv = c["question_tower"]["kv_heads"] * HEAD
    out = [("state.embeddings.tok_embeddings.weight", [vocab, d]), ("state.embeddings.norm.weight", [d])]
    for i in range(c["state_tower"]["layers"]):
        p = "state.layers.%d" % i
        if i > 0:
            out.append((p + ".attn_norm.weight", [d]))
        out += [(p + ".attn.Wqkv.weight", [3 * d, d]), (p + ".attn.Wo.weight", [d, d]),
                (p + ".mlp_norm.weight", [d]), (p + ".mlp.Wi.weight", [2 * inter, d]),
                (p + ".mlp.Wo.weight", [d, inter])]
    out += [("state.final_norm.weight", [d]), ("type_emb.weight", [3, d])]
    for j in range(c["question_tower"]["layers"]):
        p = "question.layers.%d" % j
        out += [(p + ".attn_norm.weight", [d]), (p + ".attn.Wqkv.weight", [3 * d, d]),
                (p + ".attn.Wo.weight", [d, d]), (p + ".cross_norm.weight", [d]),
                (p + ".cross.Wq.weight", [d, d]), (p + ".cross.Wk.weight", [kv, d]),
                (p + ".cross.Wv.weight", [kv, d]), (p + ".cross.Wo.weight", [d, d]),
                (p + ".mlp_norm.weight", [d]), (p + ".mlp.Wi.weight", [2 * inter, d]),
                (p + ".mlp.Wo.weight", [d, inter])]
    out += [("question.final_norm.weight", [d]), ("ord_emb.weight", [d, 4]), ("scorer.norm.weight", [d]),
            ("scorer.Wa.weight", [d, d]), ("scorer.Wb.weight", [1, d]), ("scorer.type_bias", [3])]
    return out


M64 = (1 << 64) - 1


def draws(seed, n):
    """n values of splitmix64 mapped to [-1, 1) as (z >> 40) / 2^24 * 2 - 1, vectorized."""
    with np.errstate(over="ignore"):
        s = (np.uint64(seed) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        z = s
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(40)).astype(np.float64) / float(1 << 24) * 2.0 - 1.0


def random_weights(c, seed):
    """The weights kime_model::kime_v1::random_weights draws, rounded to f32 then held in f64."""
    shapes = names(c)
    u = draws(seed, sum(int(np.prod(s)) for _, s in shapes))
    w, at = {}, 0
    for name, shape in shapes:
        n = int(np.prod(shape))
        x = u[at:at + n]
        at += n
        if name.endswith("norm.weight"):
            v = 1.0 + x / 10.0
        elif name == "scorer.type_bias":
            v = x / 10.0
        else:
            v = x / math.sqrt(shape[-1])
        w[name] = v.astype(np.float32).astype(np.float64).reshape(shape)
    return w


ERF = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + ERF(x / math.sqrt(2.0)))


def ln(x, w, eps):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w


def rope(x, pos, theta):
    """HF rotate_half RoPE on [t, heads, 64] at integer positions."""
    inv = 1.0 / theta ** (np.arange(0, HEAD, 2) / HEAD)
    a = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(a)[:, None, :], np.sin(a)[:, None, :]
    lo, hi = x[..., :HEAD // 2], x[..., HEAD // 2:]
    return np.concatenate([lo * cos - hi * sin, hi * cos + lo * sin], -1)


def attend(q, k, v, allowed):
    """q [t, h, 64], k and v [s, h, 64], allowed [t, s] booleans."""
    s = np.einsum("thd,shd->hts", q, k) / math.sqrt(HEAD)
    s = np.where(allowed[None], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hts,shd->thd", p, v)


class Model:
    def __init__(self, c, w):
        self.c, self.w = c, w
        self.d = c["dims"]["d"]
        self.heads = c["dims"]["heads"]
        self.inter = c["dims"]["inter"]
        self.eps = c["norm_eps"]
        st = c["state_tower"]
        self.global_ = [i % st["global_every"] == 0 for i in range(st["layers"])]
        self.half = st["window"] // 2
        self.theta_g, self.theta_l = st["rope_theta_global"], st["rope_theta_local"]
        self.kv_heads = c["question_tower"]["kv_heads"]

    def embed(self, ids):
        return ln(self.w["state.embeddings.tok_embeddings.weight"][ids], self.w["state.embeddings.norm.weight"], self.eps)

    def mlp(self, h, p):
        u = ln(h, self.w[p + ".mlp_norm.weight"], self.eps) @ self.w[p + ".mlp.Wi.weight"].T
        a, g = u[:, :self.inter], u[:, self.inter:]
        return h + (gelu(a) * g) @ self.w[p + ".mlp.Wo.weight"].T

    def self_attn(self, h, p, x, pos, theta, allowed):
        t, d = h.shape[0], self.d
        qkv = x @ self.w[p + ".attn.Wqkv.weight"].T
        q = rope(qkv[:, :d].reshape(t, self.heads, HEAD), pos, theta)
        k = rope(qkv[:, d:2 * d].reshape(t, self.heads, HEAD), pos, theta)
        v = qkv[:, 2 * d:].reshape(t, self.heads, HEAD)
        o = attend(q, k, v, allowed).reshape(t, d)
        return h + o @ self.w[p + ".attn.Wo.weight"].T

    def state(self, ids):
        t = len(ids)
        h = self.embed(ids)
        pos = np.arange(t)
        near = np.abs(pos[:, None] - pos[None, :]) <= self.half
        for i, glob in enumerate(self.global_):
            p = "state.layers.%d" % i
            x = h if i == 0 else ln(h, self.w[p + ".attn_norm.weight"], self.eps)
            allowed = np.ones((t, t), bool) if glob else near
            h = self.self_attn(h, p, x, pos, self.theta_g if glob else self.theta_l, allowed)
            h = self.mlp(h, p)
        s = ln(h, self.w["state.final_norm.weight"], self.eps)
        mem = []
        for j in range(self.c["question_tower"]["layers"]):
            p = "question.layers.%d" % j
            mem.append((s @ self.w[p + ".cross.Wk.weight"].T, s @ self.w[p + ".cross.Wv.weight"].T))
        pooled = s.mean(0)
        return mem, pooled / np.linalg.norm(pooled)

    def question(self, mem, qtype, header, options):
        d = self.d
        ids = list(header)
        pos = list(range(len(header)))
        seg = [0] * len(header)
        markers = []
        for i, o in enumerate(options):
            markers.append(len(ids))
            ids += o
            pos += range(len(header), len(header) + len(o))
            seg += [i + 1] * len(o)
        t = len(ids)
        seg = np.array(seg)
        mark = np.zeros(t, bool)
        mark[markers] = True
        # The question mask of spec/05: the header and every marker are seen by all, and an option
        # token also sees its own segment.
        allowed = (seg[None, :] == 0) | mark[None, :] | ((seg[:, None] != 0) & (seg[:, None] == seg[None, :]))
        h = self.embed(np.array(ids)) + self.w["type_emb.weight"][qtype]
        group = self.heads // self.kv_heads
        for j, (k, v) in enumerate(mem):
            p = "question.layers.%d" % j
            x = ln(h, self.w[p + ".attn_norm.weight"], self.eps)
            h = self.self_attn(h, p, x, pos, self.theta_g, allowed)
            x = ln(h, self.w[p + ".cross_norm.weight"], self.eps)
            q = (x @ self.w[p + ".cross.Wq.weight"].T).reshape(t, self.heads, HEAD)
            kh = np.repeat(k.reshape(-1, self.kv_heads, HEAD), group, axis=1)
            vh = np.repeat(v.reshape(-1, self.kv_heads, HEAD), group, axis=1)
            o = attend(q, kh, vh, np.ones((t, k.shape[0]), bool)).reshape(t, d)
            h = h + o @ self.w[p + ".cross.Wo.weight"].T
            h = self.mlp(h, p)
        hq = ln(h, self.w["question.final_norm.weight"], self.eps)
        m = hq[markers]
        if qtype == 1:
            n = len(markers)
            x = np.arange(n) / (n - 1) if n > 1 else np.zeros(1)
            f = np.stack([x, x * x, np.sin(math.pi * x), np.cos(math.pi * x)], 1)
            m = m + f @ self.w["ord_emb.weight"].T
        z = gelu(ln(m, self.w["scorer.norm.weight"], self.eps) @ self.w["scorer.Wa.weight"].T) @ self.w["scorer.Wb.weight"].T
        return z[:, 0] + self.w["scorer.type_bias"][qtype]


def cases(c):
    """States and question rows that cover the layout: a two token state, one shorter than the
    window, and ones past it, choice rows of one to twelve options of mixed length, a score and a
    noul."""
    sp = c["tokenizer"]["specials"]
    sep, vocab = c["tokenizer"]["sep"], c["dims"]["vocab"]
    rng = np.random.default_rng(20260930)
    word = lambda n: [int(x) for x in rng.integers(8, vocab, n)]
    out = []
    for n in [0, 11, 40, 97]:
        state = [sp["cls_s"]] + word(n) + [sep]
        rows = []
        for qtype, k in [(0, 3), (0, 1), (0, 12), (1, 5), (2, 2)]:
            header = [sp["cls_q"]] + word(int(rng.integers(3, 20))) + [sep]
            if qtype == 2:
                options = [[sp["no"]] + word(int(rng.integers(0, 6))) + [sep],
                           [sp["yes"]] + word(int(rng.integers(0, 6))) + [sep]]
            else:
                options = [[sp["opt"]] + word(int(rng.integers(1, 9))) + [sep] for _ in range(k)]
            rows.append({"qtype": qtype, "header": header, "options": options})
        out.append({"state": state, "rows": rows})
    return out


def main():
    w = random_weights(CONFIG, SEED)
    model = Model(CONFIG, w)
    out = []
    for case in cases(CONFIG):
        mem, pooled = model.state(np.array(case["state"]))
        for r in case["rows"]:
            r["logits"] = [float(x) for x in model.question(mem, r["qtype"], r["header"], r["options"])]
        case["pooled"] = [float(x) for x in pooled]
        out.append(case)
    first = w["state.embeddings.tok_embeddings.weight"].reshape(-1)[:4]
    doc = {"config": CONFIG, "seed": SEED, "first_weights": [float(x) for x in first], "cases": out}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
        f.write("\n")
    rows = sum(len(c["rows"]) for c in out)
    print("%d states and %d question rows to %s" % (len(out), rows, os.path.relpath(OUT)))


if __name__ == "__main__":
    main()
