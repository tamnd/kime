"""Record what Laya itself builds for each published checkpoint, for kime-model's loader test.

For each checkpoint this builds Laya's DecisionModel from the checkpoint's own configs, exactly as
laya.Agent does, and records every parameter and buffer name with its shape. It then reads
model.safetensors and records each tensor's dtype plus two checksums of its values as float32
(the sum and the first four values), so the Rust side can check both the names and its f16 reading.

It also times laya.Agent's load on CPU, the number kime's load is compared against.

    python tools/ref/laya_tensors.py <models>/laya crates/kime-model/tests/fixtures/laya-tensors.json
"""
import json
import os
import sys
import time

import torch
from safetensors import safe_open
from laya.common import build_model

try:
    from transformers.initialization import no_init_weights
except ImportError:
    from transformers.modeling_utils import no_init_weights


def record(model_dir):
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    with no_init_weights():
        model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"), pretrained=False)
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    tensors = []
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            x = t.float().flatten()
            tensors.append({
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
                "shape": list(t.shape),
                "sum": float(x.double().sum()),
                "first": [float(v) for v in x[:4]],
            })
    return {"module": shapes, "tensors": tensors}


def time_agent_load(model_dir, runs=3):
    from laya import Agent
    best = None
    for _ in range(runs):
        t = time.perf_counter()
        Agent(model_dir, device="cpu")
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best


def main():
    root, out = sys.argv[1], sys.argv[2]
    result = {}
    for name, sub in [("laya", ""), ("laya-multilingual", "multilingual")]:
        d = os.path.join(root, sub) if sub else root
        result[name] = record(d)
        load = time_agent_load(d)
        print(f"{name}: {len(result[name]['tensors'])} tensors, laya.Agent load on cpu {load:.3f}s (best of 3)")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
