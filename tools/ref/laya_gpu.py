"""Times Laya's own forward on an NVIDIA GPU on the parity questions, the way Agent.system_one runs it, and reads its energy per decision from NVML.

usage: laya_gpu.py <model dir> <fixture.jsonl> [batch] [on|off]

The last argument turns autocast on (Laya's default, bf16) or off (FP32). The numbers line up with the ones kime-cuda's cuda_bench example prints for the same fixture.
"""
import ctypes, json, sys, time
import torch
from laya.agent import Agent
from laya.common import collate_items

model_dir, fixture = sys.argv[1], sys.argv[2]
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 16
amp = sys.argv[4] if len(sys.argv) > 4 else "on"
agent = Agent(model_dir, device="cuda")
if amp == "off":
    agent.model.float()
qs = []
for line in open(fixture):
    for q in json.loads(line).get("questions", []):
        qs.append({"ids": q["ids"], "markers": q["markers"], "qtype": q["qtype"]})
tokens = sum(len(q["ids"]) for q in qs)
pad = agent.tok.pad_token_id
dev = agent.device


def run(items):
    b = collate_items([items], pad)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=agent.dtype, enabled=amp == "on"):
        logits, act = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev), b["marker_mask"].to(dev), b["qtype"].to(dev))
    logits.float().cpu(); act.float().cpu()


def one_at_a_time():
    for q in qs:
        run([q])


def batches():
    for i in range(0, len(qs), batch):
        run(qs[i:i + batch])


one_at_a_time()
batches()
lat = []
t0 = time.perf_counter()
for q in qs:
    s = time.perf_counter()
    run([q])
    lat.append((time.perf_counter() - s) * 1e3)
one = time.perf_counter() - t0
lat.sort()
p = lambda f: lat[int((len(lat) - 1) * f)]
print("laya %s amp %s %s: one at a time p50 %.3f ms p99 %.3f ms mean %.3f ms" % (torch.cuda.get_device_name(), amp, agent.dtype, p(0.5), p(0.99), one * 1e3 / len(qs)))
t0 = time.perf_counter()
batches()
b = time.perf_counter() - t0
print("batches of %d: %.3f s, %.3f ms per question, %.0f tokens/s" % (batch, b, b * 1e3 / len(qs), tokens / b))

try:
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
except OSError as e:
    sys.exit("no energy counter: %s" % e)
assert nvml.nvmlInit_v2() == 0
handle = ctypes.c_void_p()
assert nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0


def measure(f):
    mj = ctypes.c_ulonglong()
    assert nvml.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(mj)) == 0
    e, t = mj.value, time.perf_counter()
    f()
    torch.cuda.synchronize()
    s = time.perf_counter() - t
    assert nvml.nvmlDeviceGetTotalEnergyConsumption(handle, ctypes.byref(mj)) == 0
    return s, (mj.value - e) / 1e3


s, j = measure(lambda: time.sleep(5))
idle = j / s
print("idle: %.1f W" % idle)
for name, f in [("one at a time", one_at_a_time), ("batches of %d" % batch, batches)]:
    s, j = measure(f)
    n = len(qs)
    print("energy, %s: %.1f W over %.2f s, %.2f mJ per decision, %.2f mJ above idle" % (name, j / s, s, j * 1e3 / n, (j - idle * s) * 1e3 / n))
