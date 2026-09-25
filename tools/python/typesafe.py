"""Runs the same System One calls through `typesafe_sdk` or `kime.typesafe` and writes the
answers and the time each call took, so the two clients can be compared answer for answer.

    uv run --no-project --with typesafe-sdk==0.7.1 python tools/python/typesafe.py sdk http://127.0.0.1:18831 texts.json out-sdk.json
    python tools/python/typesafe.py kime http://127.0.0.1:18831 texts.json out-kime.json
    python tools/python/typesafe.py local /path/to/laya texts.json out-local.json
    python tools/python/typesafe.py compare out-sdk.json out-kime.json out-local.json

`texts.json` is a list of strings. Each text is asked a noul, a choice and a score question, and
the first call of each run is left out of the timings.
"""

import json
import statistics
import sys
import time


def questions(t):
    return {
        "billing": t.Noul(instructions="Is this about billing or payments?"),
        "intent": t.Choice(
            instructions="What does the writer want?",
            criteria={"help": "they need help or support", "complain": "they are unhappy", "inform": "they are sharing information"},
        ),
        "urgency": t.Score(instructions="How urgent is this?", criteria=["not urgent", "somewhat urgent", "very urgent"]),
    }


def run(kind, target, texts, out):
    if kind == "sdk":
        import typesafe_sdk as t

        client = t.TypeSafeClient(api_key="none", base_url=target)
    else:
        import kime.typesafe as t

        if kind == "local":
            client = t.TypeSafeClient(local=target, device="cpu", precision="f32")
        else:
            client = t.TypeSafeClient(base_url=target)
    q = questions(t)
    models = [m.name for m in client.models.list().models]
    rows, took = [], []
    for i, text in enumerate(texts):
        start = time.perf_counter()
        r = client.system_one(text, q, model="jev-latest")
        if i:
            took.append(time.perf_counter() - start)
        rows.append({k: a.model_dump() for k, a in r.answers.items()})
    ms = sorted(x * 1e3 for x in took)
    summary = {"client": kind, "models": models, "calls": len(texts), "mean_ms": statistics.fmean(ms),
               "p50_ms": ms[len(ms) // 2], "p99_ms": ms[int(len(ms) * 0.99)]}
    print(json.dumps(summary))
    with open(out, "w") as f:
        json.dump({"summary": summary, "answers": rows}, f)


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    base = runs[0]
    for other in runs[1:]:
        same = sum(a == b for a, b in zip(base["answers"], other["answers"]))
        print("%s vs %s: %d of %d responses identical" % (
            base["summary"]["client"], other["summary"]["client"], same, len(base["answers"])))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2:])
    else:
        run(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4])
