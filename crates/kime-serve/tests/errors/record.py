"""Records what laya-serve answers to each case in cases.json, as the `laya` field.

Run it against two laya-serves with the English checkpoint on the CPU, one without a key for the
`open` cases and one with LAYA_API_KEY=s3cret for the `laya-key` cases:

    LAYA_PORT=18801 LAYA_MODELS=english LAYA_DEVICE=cpu python -m laya.serve
    LAYA_PORT=18802 LAYA_MODELS=english LAYA_DEVICE=cpu LAYA_API_KEY=s3cret python -m laya.serve
    python record.py open=18801 laya-key=18802

Cases marked `"jev": true` are Jev shaped and laya-serve is not asked, and neither is it for a
server that is not given a port.
"""

import http.client
import json
import pathlib
import sys

ports = dict(a.split("=") for a in sys.argv[1:])
path = pathlib.Path(__file__).with_name("cases.json")
cases = json.loads(path.read_text())
for c in cases:
    port = ports.get(c.get("server", "open"))
    if c.get("jev") or port is None:
        continue
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=120)
    body = c.get("body")
    headers = {"content-type": "application/json"} if body is not None else {}
    for h in c.get("headers", []):
        k, v = h.split(": ", 1)
        headers[k] = v
    conn.request(c["method"], c["path"], body=body.encode() if body is not None else None, headers=headers)
    r = conn.getresponse()
    text = r.read().decode()
    try:
        out = json.loads(text)
        if isinstance(out, dict):
            out.pop("routing", None)
    except ValueError:
        out = text
    c["laya"] = {"status": r.status, "body": out}
    print(c["name"], r.status)
path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
