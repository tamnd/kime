"""Answers the Mind2Web suites without a model, as a floor for the models on them.

The operation gets the share of each operation in Mind2Web's train steps (CLICK 5,661, TYPE_TEXT
875 and SELECT 210 of 6,746), and nothing for the others. A target head ranks the elements by how
many words of the goal their label holds, with probability proportional to exp(2 * shared words).

    python tools/eval/baseline_mind2web.py <suites dir> <answers dir>
    kime eval <suites dir> --answers <answers dir>
"""

import glob
import json
import math
import os
import re
import sys

PRIOR = {"CLICK": 5661, "TYPE_TEXT": 875, "SELECT": 210}
WORD = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "from", "is", "my", "me", "i"}


def words(s):
    return {w for w in WORD.findall(s.lower()) if w not in STOP}


def answer(case):
    out = {}
    for k, q in case["questions"].items():
        crit = q["criteria"]
        if k == "operation":
            total = sum(PRIOR.values())
            p = {o: PRIOR.get(o, 0) / total for o in crit}
        else:
            goal = words(q["instructions"]["goal"])
            s = {o: 2.0 * len(goal & words(v["element"].split("] ", 1)[-1])) for o, v in crit.items()}
            m = max(s.values())
            z = sum(math.exp(x - m) for x in s.values())
            p = {o: math.exp(x - m) / z for o, x in s.items()}
        out[k] = {"probabilities": p}
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        name = os.path.basename(f)[: -len(".jsonl")]
        with open(f) as fi, open(os.path.join(dst, name + ".answers.jsonl"), "w") as fo:
            for line in fi:
                case = json.loads(line)
                fo.write(json.dumps({"id": case["id"], "answers": answer(case)}) + "\n")


if __name__ == "__main__":
    main()
