"""Turns Mind2Web steps into browser agent steps in jev-ultrafast's format, for training and for the
agent step suites.

Mind2Web (osunlp/Mind2Web, CC BY 4.0) has people doing 2,350 tasks on 137 real websites, with the
page before every action and the element the person acted on. Each step becomes the snapshot
jev-ultrafast's snapshot.js would return for that page (url, title, text, and one action per
element: click, fill, or one select action per option of a dropdown, plus the scroll and wait
controls), and then goes through kime's own `agent_step`, so the questions are the ones a client
sends. The recorded action is the label: the operation head's target is the operation the person
did, and the target head of that operation gets the element they acted on. When Mind2Web marks
more than one element as the target (a link and the text inside it, say), the probability is split
between them.

Mind2Web's candidates include containers, such as the list item around a link, which snapshot.js
would not offer next to the link, so a candidate that holds another candidate is left out, and a
step whose only target is a container is skipped. A page still has hundreds of candidates, and
Laya's head fits about 126 target keys, so a step gets a window of up to 100 elements in page order
that holds the target, at a random offset seeded with the step's id, and a dropdown offers at most 30 options.

    from mind2web import steps
    for step in steps(task): ...
"""

import json
import random
import re
from html.parser import HTMLParser

MAX_ELEMENTS = 100
MAX_OPTIONS = 30
MAX_KEYS = 120
MAX_TEXT = 6000
LABEL = 100
OPS = {"CLICK": ("CLICK", "click"), "TYPE": ("TYPE_TEXT", "fill"), "SELECT": ("SELECT", "select")}
FILL_TYPES = {"", "text", "search", "email", "tel", "url", "number", "password", "date", "datetime-local", "month", "time", "week"}
ROLES = {"a": "link", "button": "button", "select": "combobox", "textarea": "textbox", "option": "option", "img": "img",
         "li": "listitem", "label": "label", "summary": "button", "td": "cell", "th": "columnheader", "iframe": "iframe",
         "h1": "heading", "h2": "heading", "h3": "heading", "h4": "heading", "h5": "heading", "h6": "heading"}
INPUT_ROLES = {"checkbox": "checkbox", "radio": "radio", "button": "button", "submit": "button", "reset": "button",
               "search": "searchbox", "range": "slider", "number": "spinbutton"}
VOID = {"input", "img", "br", "hr", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr", "param"}
CONTROLS = [
    {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
    {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
    {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
]
WS = re.compile(r"\s+")


class Page(HTMLParser):
    """The text of a cleaned Mind2Web page, and each node's order, attributes, text and options."""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.stack, self.nodes, self.parts = [], {}, []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        nid = a.get("backend_node_id")
        if nid is not None:
            parent = next((p for _, p in reversed(self.stack) if p is not None), None)
            self.nodes[nid] = {"tag": tag, "attrs": a, "order": len(self.nodes), "text": [], "options": [], "parent": parent}
            if tag == "option":
                for _, pid in reversed(self.stack):
                    if pid is not None and self.nodes[pid]["tag"] == "select":
                        self.nodes[pid]["options"].append(nid)
                        break
        if tag not in VOID:
            self.stack.append((tag, nid))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack:
            self.stack.pop()

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        t = WS.sub(" ", data).strip()
        if not t:
            return
        self.parts.append(t)
        for _, nid in self.stack:
            if nid is not None:
                buf = self.nodes[nid]["text"]
                if sum(len(x) for x in buf) < 4 * LABEL:
                    buf.append(t)

    def text(self, nid):
        return WS.sub(" ", " ".join(self.nodes[nid]["text"])).strip() if nid in self.nodes else ""


def short(s, n=LABEL):
    s = WS.sub(" ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def leaves(page, cands):
    """The candidates that hold no other candidate. A list item around a link, or a menu around its
    items, is a container, and snapshot.js offers the link or the item, not both."""
    containers = set()
    for n in cands:
        p = page.nodes.get(n, {}).get("parent")
        while p is not None:
            if p in cands:
                containers.add(p)
            p = page.nodes[p]["parent"]
    return containers


def describe(page, nid, raw):
    """The label, role, kind and value of a candidate element."""
    node = page.nodes.get(nid, {"tag": raw.get("tag", ""), "attrs": {}})
    tag, attrs = node["tag"], dict(json.loads(raw.get("attributes") or "{}"), **node["attrs"])
    label = page.text(nid)
    for k in ("aria_label", "aria-label", "placeholder", "title", "alt", "value", "name"):
        if not label and attrs.get(k):
            label = attrs[k]
    itype = (attrs.get("type") or "").lower()
    role = attrs.get("role") or (INPUT_ROLES.get(itype, "textbox") if tag == "input" else ROLES.get(tag, "generic"))
    if tag == "select":
        kind = "select"
    elif tag == "textarea" or (tag == "input" and itype in FILL_TYPES):
        kind = "fill"
    else:
        kind = "click"
    value = attrs.get("value", "") if tag in ("input", "textarea") else ""
    return {"label": short(label), "role": role, "kind": kind, "value": short(value)}


def history_entry(repr_):
    """A recent action from Mind2Web's `[role] text -> OP: value` string."""
    left, _, right = repr_.rpartition(" -> ")
    op, _, value = right.partition(": ")
    action, kind = OPS.get(op.strip(), ("CLICK", "click"))
    text = short(re.sub(r"^\[[^\]]*\]\s*", "", left))
    if value:
        text = short("%s: %s" % (text, value))
    return {"action": action, "kind": kind, "text": text, "page_changed": None}


def steps(task, agent_step, window=True):
    """One (step id, body, targets, gold) per usable action of a Mind2Web task. `agent_step` is
    kime.agent_step. Steps whose target is not on the cleaned page are skipped."""
    for i, act in enumerate(task["actions"]):
        pos = act.get("pos_candidates") or []
        op = act["operation"]["op"]
        if not pos or op not in OPS:
            continue
        operation, gold_kind = OPS[op]
        page = Page(act["cleaned_html"])
        cands = {}
        for c in pos + (act.get("neg_candidates") or []):
            cands.setdefault(c["backend_node_id"], c)
        containers = leaves(page, cands)
        gold_ids = [c["backend_node_id"] for c in pos if c["backend_node_id"] not in containers]
        if not gold_ids:
            continue
        cands = {n: c for n, c in cands.items() if n not in containers}
        order = sorted(cands, key=lambda n: page.nodes.get(n, {}).get("order", 1 << 30))
        if window and len(order) > MAX_ELEMENTS:
            rng = random.Random("mind2web:%s" % act["action_uid"])
            at = [order.index(g) for g in gold_ids]
            lo = max(0, max(at) - MAX_ELEMENTS + 1)
            hi = min(min(at), len(order) - MAX_ELEMENTS)
            if lo > hi:
                gold_ids = [order[min(at)]]
                lo = hi = max(0, min(min(at), len(order) - MAX_ELEMENTS))
            start = rng.randint(lo, hi)
            order = order[start: start + MAX_ELEMENTS]
        actions, gold_actions = [], []
        for nid in order:
            d = describe(page, nid, cands[nid])
            is_gold = nid in gold_ids
            if is_gold and op == "TYPE":
                d["kind"] = "fill"
            elif is_gold and op == "CLICK" and d["kind"] == "fill":
                d["kind"] = "click"
            if d["kind"] == "select":
                opts = page.nodes.get(nid, {}).get("options", [])[:MAX_OPTIONS]
                current = next((page.text(o) for o in opts if "selected" in page.nodes[o]["attrs"]), "")
                for o in opts:
                    aid = "e%d" % (len(actions) + 1)
                    label = short(page.text(o))
                    actions.append({"node": nid, "role": d["role"], "label": label, "kind": "select",
                                    "value": page.nodes[o]["attrs"].get("value", label), "current_value": current, "id": aid})
                    if is_gold and op == "SELECT" and label.lower() == short(act["operation"]["value"]).lower():
                        gold_actions.append(aid)
                continue
            aid = "e%d" % (len(actions) + 1)
            a = {"node": nid, "role": d["role"], "label": d["label"], "kind": d["kind"], "id": aid}
            if d["value"]:
                a["value"] = d["value"]
            actions.append(a)
            if is_gold and d["kind"] == gold_kind:
                gold_actions.append(aid)
        if not gold_actions:
            continue
        snapshot = {"url": "https://%s/" % task["website"], "title": "", "text": " ".join(page.parts)[:MAX_TEXT],
                    "actions": actions + CONTROLS}
        history = [history_entry(r) for r in task["action_reprs"][:i]]
        step = agent_step(snapshot, task["confirmed_task"], history)
        qs = step.questions
        head = "%s_target" % operation.lower()
        # agent_step numbers elements from 1 in the order of the actions, so the target keys of the
        # gold actions are found by building the same map here.
        index, keys = {}, {}
        for a in actions:
            n = index.setdefault(a["node"], len(index) + 1)
            if a["kind"] == "select":
                k = sum(1 for x in keys.values() if x.startswith("%d:" % n)) + 1
                keys[a["id"]] = "%d:%d" % (n, k)
            else:
                keys[a["id"]] = str(n)
        gold_keys = [keys[x] for x in gold_actions]
        crit = list(qs[head]["criteria"])
        if not all(k in crit for k in gold_keys) or len(crit) > MAX_KEYS:
            # Several dropdowns can still give a select head more keys than Laya's head fits.
            continue
        op_keys = list(qs["operation"]["criteria"])
        targets = {
            "operation": {"probs": [float(k == operation) for k in op_keys], "hard": op_keys.index(operation), "weight": 1.0},
            head: {"probs": [gold_keys.count(k) / len(gold_keys) for k in crit], "hard": crit.index(gold_keys[0]), "weight": 1.0},
        }
        body = {"state": step.state, "questions": {"operation": qs["operation"], head: qs[head]}}
        gold = {"operation": operation, head: gold_keys[0]}
        if len(crit) < 2:
            # A choice needs two options, and a lone candidate says nothing about the target.
            del body["questions"][head], targets[head], gold[head]
        yield act["action_uid"], body, targets, gold
