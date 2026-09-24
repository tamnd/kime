"""In-process latency and throughput of kime's Python API or Laya's, on the parity cases.

    python tools/python/bench.py kime [--device cpu] [--precision f32]
    python tools/python/bench.py laya [--device cpu|mps|cuda]

Prints one JSON line. Run the two back to back a few times on the same machine and compare.
"""

import argparse
import json
import pathlib
import statistics
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["kime", "laya"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    cases = [json.loads(l) for l in open(ROOT / "crates/kime-eval/fixtures/parity/cases.jsonl")]
    t = time.perf_counter()
    if a.engine == "kime":
        import kime

        agent = kime.load("convaiinnovations/laya", device=a.device, precision=a.precision)
    else:
        import laya

        agent = laya.load("convaiinnovations/laya", device=a.device)
    load = time.perf_counter() - t

    for c in cases[:20]:
        agent.system_one(c["state"], c["questions"])
    lat = []
    for _ in range(a.rounds):
        for c in cases:
            t = time.perf_counter()
            agent.system_one(c["state"], c["questions"])
            lat.append(time.perf_counter() - t)
    lat.sort()

    q = cases[0]["questions"]
    states = [c["state"] for c in cases]
    agent.predict_batch(states[: a.batch], q, batch_size=a.batch)
    t = time.perf_counter()
    for _ in range(a.rounds):
        agent.predict_batch(states, q, batch_size=a.batch)
    rate = a.rounds * len(states) / (time.perf_counter() - t)

    print(json.dumps({
        "engine": a.engine,
        "device": str(getattr(agent, "device", a.device)),
        "load_s": round(load, 2),
        "p50_ms": round(1e3 * statistics.median(lat), 2),
        "p99_ms": round(1e3 * lat[int(0.99 * (len(lat) - 1))], 2),
        "mean_ms": round(1e3 * statistics.fmean(lat), 2),
        "batch_states_per_s": round(rate, 1),
    }))


if __name__ == "__main__":
    main()
