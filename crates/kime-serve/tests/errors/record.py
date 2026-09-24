"""Records what laya-serve answers to each case in cases.json, as the `laya` field.

Run it against a laya-serve with the English checkpoint on the CPU:

    LAYA_PORT=18801 LAYA_MODELS=english LAYA_DEVICE=cpu python -m laya.serve
    python record.py 18801

Cases marked `"jev": true` are Jev shaped (they carry a `kime` object) and laya-serve is not asked.
"""

import http.client
import json
import pathlib
import sys

port = int(sys.argv[1])
path = pathlib.Path(__file__).with_name("cases.json")
cases = json.loads(path.read_text())
for c in cases:
    if c.get("jev"):
        continue
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    body = c.get("body")
    headers = {"content-type": "application/json"} if body is not None else {}
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
