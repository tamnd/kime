"""kime.Router against the decisions recorded from Laya's Router in crates/kime-route/tests/lang,
and on the checkpoints when they are on disk."""

import json
import pathlib

import pytest

import kime

ROOT = pathlib.Path(__file__).resolve().parents[2]
LANG = ROOT / "crates" / "kime-route" / "tests" / "lang"


def lines(name):
    with open(LANG / name) as f:
        return [json.loads(line) for line in f]


def test_word_lists_are_laya():
    r = kime.Router(identifier=False)
    cases = lines("route.jsonl")
    assert len(cases) > 2000
    for c in cases:
        d = r.route(c["state"])
        assert (d.model, d.reason, d["detection"]) == (
            c["laya"]["model"], c["laya"]["reason"], c["laya"]["detection"]), c["state"]


def test_detect_language_is_laya():
    for c in lines("laya.jsonl") + lines("laya-tests.jsonl"):
        assert kime.detect_language(c["state"]) == c["analysis"], c["state"]
        assert kime.is_english(c["state"]) == c["analysis"]["is_english"]
    assert kime.detect_script("Привет, как дела") == "cyrillic"


def test_identifier_on_the_labelled_set():
    r = kime.Router()
    cases = [c for c in lines("routing-set.jsonl") if c["want"] != "default"]
    right = sum(r.route(c["text"]).model == c["want"] for c in cases)
    laya = sum(c["laya"] == c["want"] for c in cases)
    assert right > laya and right / len(cases) > 0.97, (right, laya, len(cases))


def test_precedence():
    r = kime.Router()
    de = "Ich möchte mein Abonnement kündigen, bitte helfen Sie mir."
    typed = {k: {"type": "noul", "instructions": k} for k in
             ("action", "category", "churn_risk", "needs_human", "urgency")}
    assert r.route(de, model="en").reason == "explicit model='en'"
    d = r.route(de, task="typed_decisions")
    assert (d.model, d["repo"]) == ("typed-decisions", "convaiinnovations/laya/typed-decisions")
    assert r.route(de, typed)["workflow"] == "customer_service"
    assert r.route(de, typed).model == "multilingual"
    auto = kime.Router(auto_task_detection=True)
    want = "question ids match the 'customer_service' typed-decisions workflow"
    assert auto.route(de, typed).reason == want
    assert r.route(de, lang="en_US.UTF-8").model == "english"
    d = r.route(de, lang_guess=lambda s: "en")
    assert d.reason == "lang_guess: the caller identified this as English text"
    d = kime.Router(lang_guess=lambda s: None).route(de)
    assert d.model == "multilingual" and d["detection"]["language"] == "de"
    d = kime.Router(lang_guess="pt-BR").route("hello there")
    assert d.reason == "Router(lang_guess=...): the caller identified this as non-English text"
    d = kime.Router(default="typed").route("12345 !!!")
    assert (d.model, d.reason) == ("typed-decisions", "no letters detected in state; using default (typed-decisions)")
    d = kime.Router(standalone_repos=True).route("Мне нужно отменить подписку")
    assert d["repo"] == "convaiinnovations/laya-multilingual"
    with pytest.raises(ValueError):
        r.route(de, model="nope")


def test_identifier_and_word_lists_differ_where_they_should():
    text = "The café on the corner serves a great crème brûlée"
    assert kime.Router().route(text).model == "english"
    assert kime.Router(identifier=False).route(text).model == "multilingual"


def router():
    try:
        return kime.Router(device="cpu", precision="f32", max_loaded=1).preload(["english"])
    except RuntimeError as e:
        pytest.skip("no laya checkpoint: %s" % e)


def test_predict():
    r = router()
    q = kime.triage_questions()
    state = "I was charged twice for my subscription, please refund me"
    res = r.predict(state, q)
    assert res["routing"]["model"] == "english"
    assert res["routing"]["reason"] == "English Latin text"
    want = r.load("english").system_one(state, q)
    assert res["answers"] == want["answers"]
    assert r.loaded == ["english"]


def test_lru():
    r = router()
    agent = r.load("english")
    r.attach("typed", agent)
    assert r.loaded == ["english", "typed-decisions"] and r.max_loaded == 2
    r.unload("typed-decisions")
    assert r.loaded == ["english"]
    with r:
        pass
    assert r.loaded == []


def test_predict_batch():
    r = router()
    q, other = kime.triage_questions(), {"refund": {"type": "noul", "instructions": "Is this a refund request?"}}
    requests = [{"state": "I was charged twice, please refund me", "questions": q},
                {"state": "Where is my parcel?", "questions": other},
                {"state": "The app crashes when I log in", "questions": q, "lang": "en"}]
    got = r.predict_batch(requests, batch_size=2)
    for req, res in zip(requests, got):
        want = r.predict(req["state"], req["questions"], lang=req.get("lang"))
        assert res["answers"] == want["answers"] and res["routing"] == want["routing"]
    assert [d.model for d in r.route_batch(requests)] == ["english"] * 3
    assert r.predict_many([]) == []
    with pytest.raises(ValueError, match="request 0 is missing required key 'questions'"):
        r.predict_batch([{"state": "x"}])
