"""Records Laya's `embed_fn_from_agent` on a few texts, the reference kime's pooled embedding is
tested against.

    python tools/python/laya_embed.py /path/to/laya crates/kime-eval/fixtures/parity/embed.json

Run it in a venv with Laya installed. Each text is embedded at max_length 512 and 16, so both a
whole text and a cut one are covered, and values are rounded to 1e-5.
"""

import json
import sys

import torch
from laya import Agent
from laya.shortlist import embed_fn_from_agent

TEXTS = [
    "what time is it in new york city",
    "play any song by joe prsaise",
    "wake me up at five am this week",
    "i need a refund for the double charge on my card, it happened twice this month",
    "refund request",
    "card lost",
    "",
    "pin blocked: my card PIN is blocked",
    "a " * 700,
    "Ich möchte mein Abonnement kündigen, bitte bestätigen Sie das per E-Mail.",
]

agent = Agent(sys.argv[1], device="cpu")
out = {"laya": getattr(sys.modules["laya"], "__version__", "?"), "torch": torch.__version__, "texts": TEXTS, "emb": {}}
for max_length in (512, 16):
    rows = embed_fn_from_agent(agent, max_length=max_length, batch_size=4)(TEXTS)
    out["emb"][str(max_length)] = [[round(float(x), 5) for x in r] for r in rows]
with open(sys.argv[2], "w") as f:
    json.dump(out, f, separators=(",", ":"))
    f.write("\n")
