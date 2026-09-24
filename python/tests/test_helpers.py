"""The pure functions: presets, email cleaning and browser agent steps, against the Rust fixtures
recorded from Laya and jev-ultrafast."""

import json
import pathlib

import pytest

import kime

ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS = ROOT / "crates" / "kime-core" / "tests"


def test_presets():
    q = kime.triage_questions()
    assert list(q) and all("type" in v for v in q.values())
    for f in (kime.email_questions, kime.guard_questions, kime.moderation_questions,
              kime.router_questions):
        assert f()
    custom = kime.email_questions({"sales": "pricing and quotes", "other": "anything else"})
    assert json.dumps(custom).count("pricing and quotes") == 1


def test_email_against_laya():
    cases = json.loads((TESTS / "email" / "laya.json").read_text())
    assert len(cases) > 10
    for c in cases:
        assert kime.clean_email_body(c["body"], c["max_chars"]) == c["clean"], c["body"]


def test_email_state():
    s = kime.email_state("  Refund ", "Hi,\n\nPlease refund.\n\n> old", sender="a@b.c", ticket=None, team="x")
    assert s == {"subject": "Refund", "body": "Hi,\n\nPlease refund.", "from": "a@b.c", "team": "x"}


def test_agent_step_against_jev_ultrafast():
    cases = json.loads((TESTS / "agent" / "jev-ultrafast.json").read_text())
    refused = 0
    for c in cases:
        step = kime.agent_step(c["snapshot"], c["goal"], c["history"])
        assert step.body("jev-latest") == dict(model="jev-latest", **c["body"])
        assert {"state": step.state, "questions": step.questions} == c["body"]
        want = c["result"]
        if "error" in want:
            with pytest.raises(ValueError):
                step.decide(c["answers"])
            refused += 1
        else:
            assert step.decide(c["answers"]) == want
    assert 0 < refused < len(cases)
