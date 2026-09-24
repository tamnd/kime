"""Records what jev-ultrafast's choose() sends and returns, for kime_core::agent to match.

Usage: python record.py <jev-ultrafast checkout> [size=80] [seed=20260924] > jev-ultrafast.json

The snapshots are generated in the shape snapshot.js returns: clicks, fills (each with its
"Open" click), selects with one action per unselected option, and the scroll and wait
controls. Each one is sent to choose() with post_json replaced by a function that returns
generated answers, some of them broken the ways validate_choice checks for.
"""

import json
import random
import sys
import types

root = sys.argv[1]
size = int(sys.argv[2]) if len(sys.argv) > 2 else 80
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20260924

# model.py builds an HTTP/2 client when imported. Nothing is sent, so a stub will do.
httpx = types.ModuleType("httpx")
httpx.Client = lambda **_: None
httpx.HTTPError = Exception
sys.modules["httpx"] = httpx
# The package __init__ pulls in the browser, which choose() does not need.
package = types.ModuleType("jev_ultrafast")
package.__path__ = [root + "/jev_ultrafast"]
sys.modules["jev_ultrafast"] = package
from jev_ultrafast import model  # noqa: E402

rng = random.Random(seed)
WORDS = ["Where to?", "Departure", "Search", "Done", "Économie", "東京", "Next month", "Sort by", "2 adults", "", "Open menu", "Flights", "Round trip", "Nonstop only", "Price", "ok → go"]
ROLES = ["button", "link", "textbox", "combobox", "checkbox", "radio", "gridcell", "option", "tab", "searchbox"]


def word():
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 3)))


def snapshot():
    actions = []
    for node in range(rng.randint(0, 40)):
        base = {"node": f"n{node}", "role": rng.choice(ROLES), "label": word() or "button", "rect": {"x": 1, "y": 2, "w": 3, "h": 4}}
        for key in ("checked", "selected", "expanded"):
            if rng.random() < 0.15:
                base[key] = rng.choice(["true", "false", "mixed"])
        kind = rng.random()
        if kind < 0.15:
            current = ", ".join(word() for _ in range(rng.randint(0, 2)))
            for o in range(rng.randint(0, 5)):
                actions.append({**base, "kind": "select", "value": f"v{o}", "current_value": current, "label": base["label"] + " → " + word()})
        elif kind < 0.4:
            value = word()
            actions.append({**base, "kind": "fill", "value": value})
            actions.append({**base, "kind": "click", "value": value, "label": "Open " + base["label"]})
        else:
            actions.append({**base, "kind": "click", "value": word() if rng.random() < 0.3 else ""})
    # The same node seen again later, as happens with a combobox that is both typed into and opened.
    plain = [a for a in actions if a["kind"] != "select"]
    if plain and rng.random() < 0.3:
        actions.append({**rng.choice(plain), "kind": rng.choice(["click", "fill"])})
    rng.shuffle(actions)
    actions = actions[:250]
    for i, a in enumerate(actions):
        a["id"] = f"e{i + 1}"
    if rng.random() < 0.7:
        actions.append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560})
    if rng.random() < 0.4:
        actions.append({"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560})
    actions.append({"id": "wait", "kind": "wait", "label": "Wait for the page to update"})
    text = "\n".join(word() for _ in range(rng.randint(0, 40)))[:6000]
    return {"url": "https://www.google.com/travel/flights", "title": word(), "text": text, "actions": actions}


def history():
    out = []
    for _ in range(rng.randint(0, 14)):
        h = {"action": rng.choice(["CLICK e3", "TYPE_TEXT e1", "WAIT", "SCROLL_DOWN"]), "kind": rng.choice(["click", "fill", "wait"]), "extra": 1}
        if rng.random() < 0.5:
            h["text"] = word()
        if rng.random() < 0.7:
            h["page_changed"] = rng.random() < 0.5
        out.append(h)
    return out


def answer(ids):
    ids = list(ids)
    weights = [rng.random() for _ in ids]
    total = sum(weights) or 1
    probabilities = {k: round(w / total, 2) for k, w in zip(ids, weights)}
    choice = max(probabilities, key=probabilities.get) if ids else "x"
    a = {"choice": choice, "probabilities": probabilities, "confidence": round(rng.random(), 2)}
    broken = rng.random()
    if broken < 0.03:
        a["choice"] = "nope"
    elif broken < 0.06 and ids:
        a["probabilities"][rng.choice(ids)] = 1.5
    elif broken < 0.09:
        a["probabilities"]["extra"] = 0.0
    elif broken < 0.12 and len(ids) > 1:
        a["choice"] = min(probabilities, key=probabilities.get)
    elif broken < 0.14:
        del a["confidence"]
    elif broken < 0.16:
        a["probabilities"] = {k: 0.0 for k in ids}
    return a


cases = []
for n in range(size):
    snap, hist, goal = snapshot(), history(), f"Find a flight {word()} #{n}"
    sent = {}

    def post_json(url, key, body):
        sent["body"] = body
        answers = {name: answer(q["criteria"]) for name, q in body["questions"].items() if rng.random() < 0.97}
        sent["answers"] = answers
        return {"answers": answers, "model": "jev-latest", "usage": {"input_tokens": 1}}

    model.post_json = post_json
    try:
        out = model.choose(snap, goal, hist)
        result = {k: out[k] for k in ("choice", "operation", "target", "confidence", "probabilities", "target_confidence")}
    except ValueError as e:
        result = {"error": str(e)}
    body = sent["body"]
    del body["model"]
    cases.append({"snapshot": snap, "goal": goal, "history": hist, "body": body, "answers": sent["answers"], "result": result})

json.dump(cases, sys.stdout, ensure_ascii=False, separators=(",", ":"))
