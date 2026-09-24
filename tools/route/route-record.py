"""Records Laya's Router._route for states, as JSON lines of {"state", "lang", "laya"}.

    python tools/route/route-record.py data.jsonl laya-tests.jsonl > route.jsonl

The first file is what lid-data.sh writes: the test split is sampled, 20 texts per source and
language. The second is crates/kime-route/tests/lang/laya-tests.jsonl, taken whole with no label.
Run it with Laya on the path (PYTHONPATH=laya-src); nothing is loaded, since _route only reads
the state.
"""
import json
import random
import sys
from collections import defaultdict

from laya.router import Router

router = Router()
groups = defaultdict(list)
for line in open(sys.argv[1]):
    row = json.loads(line)
    if row["split"] == "test":
        groups[(row["source"], row["lang"])].append(row)
rng = random.Random(7)
cases = []
for key in sorted(groups):
    rows = groups[key]
    for row in rng.sample(rows, min(20, len(rows))):
        cases.append((row["text"], row["lang"]))
for line in open(sys.argv[2]):
    cases.append((json.loads(line)["state"], None))
for state, lang in cases:
    d = router._route(state, {})
    laya = {"model": d["model"], "reason": d["reason"], "detection": d["detection"]}
    print(json.dumps({"state": state, "lang": lang, "laya": laya}, ensure_ascii=False))
