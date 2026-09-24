"""kime's Python API against answers recorded from Laya's Agent.system_one on the same inputs.
Needs the laya checkpoint under KIME_MODELS or the Hugging Face cache; skipped without it."""

import json
import os
import pathlib

import pytest

import kime

ROOT = pathlib.Path(__file__).resolve().parents[2]
PARITY = ROOT / "crates" / "kime-eval" / "fixtures" / "parity"


def lines(name):
    with open(PARITY / name) as f:
        return [json.loads(line) for line in f]


@pytest.fixture(scope="module")
def agent():
    try:
        return kime.load("convaiinnovations/laya", device="cpu", precision="f32")
    except RuntimeError as e:
        pytest.skip("no laya checkpoint: %s" % e)


def close(got, want, path=""):
    """Equal up to one step in the fourth decimal, which is where Laya rounds."""
    if isinstance(want, dict):
        assert list(got) == list(want), path
        for k in want:
            close(got[k], want[k], path + "." + k)
    elif isinstance(want, float) and not isinstance(got, bool):
        assert abs(got - want) <= 1.5e-4, (path, got, want)
    else:
        assert got == want, path


def test_system_one_matches_laya(agent):
    laya = {c["id"]: c["answer"] for c in lines("laya.jsonl")}
    exact = 0
    for case in lines("cases.jsonl"):
        got = agent.system_one(case["state"], case["questions"])
        close(got, laya[case["id"]], case["id"])
        exact += got == laya[case["id"]]
    # Nearly all answers come out byte for byte, and the rest differ in the last rounded digit.
    assert exact >= 190


def test_predict_batch_matches_system_one(agent):
    cases = lines("cases.jsonl")[:40]
    q = cases[0]["questions"]
    states = [c["state"] for c in cases]
    batch = agent.predict_batch(states, q, batch_size=16)
    assert len(batch) == len(states)
    for s, b in zip(states, batch):
        close(b, agent.system_one(s, q))


def test_predict_is_system_one():
    assert kime.Agent.predict is kime.Agent.system_one


def test_invalid_questions(agent):
    with pytest.raises(ValueError):
        agent.system_one("hi", {"q": {"type": "choice", "criteria": {}}})


def test_unsupported_arguments(agent):
    with pytest.raises(NotImplementedError):
        agent.system_one("hi", kime.triage_questions(), max_len=64)


def test_model_names():
    name = kime._model_name
    assert name("convaiinnovations/laya", None) == "laya"
    assert name("convaiinnovations/laya", "multilingual") == "laya-multilingual"
    assert name("hf://org/repo", "sub") == "hf://org/repo/sub"
    assert name("laya-typed-decisions", None) == "laya-typed-decisions"
