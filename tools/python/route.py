"""Times Router.route over a set of states and writes the decisions, to compare kime with Laya.

    python tools/python/route.py STATES.json laya|kime|word_lists OUT.json

STATES.json is a list of {"text": ..., "lang": ...}. Each text is also routed as a dict state and
inside a list, and a few states with no letters are added. `laya` needs Laya importable, `kime`
is kime.Router() and `word_lists` is kime.Router(identifier=False), which should give Laya's
decisions exactly. Nothing is loaded or run, so neither needs a checkpoint.
"""

import json
import sys
import time


def main():
    path, which, out = sys.argv[1:4]
    states = [r["text"] for r in json.load(open(path))]
    states += [{"message": s, "id": 7} for s in states[::10]] + [[s, {"n": 1}] for s in states[5::10]]
    states += [None, "", "12345 !!!", {"a": None}, ["", 3.5]]
    if which == "laya":
        from laya import Router

        r = Router()
    else:
        import kime

        r = kime.Router(identifier=which == "kime")
    t = time.perf_counter()
    got = [dict(r.route(s)) for s in states]
    dt = time.perf_counter() - t
    print("%s: %d states in %.2f s, %.1f us each" % (which, len(states), dt, 1e6 * dt / len(states)))
    with open(out, "w") as f:
        json.dump(got, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
