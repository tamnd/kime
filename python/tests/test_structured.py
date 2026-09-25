"""kime.decide and kime.predict_shortlist, ported from Laya 0.3.20, on fixed answers and on the
checkpoint when it is on disk."""

import os

import pytest

import kime
from kime.structured import SchemaError

SCHEMA = {"type": "object", "properties": {
    "topic": {"enum": ["billing", "support", None, 3], "description": "What is it about?"},
    "urgency": {"type": "integer", "minimum": 1, "maximum": 3},
    "complaint": {"type": ["boolean", "null"]},
    "flag": {"enum": [True, False]},
}}


class Fixed:
    """An agent that gives the same answers whatever it is asked, and keeps the questions."""

    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def predict(self, state, questions, **kw):
        self.asked.append((questions, kw))
        return {"answers": self.answers, "usage": {"input_tokens": 7}}


def test_questions():
    q = kime.questions_from_json_schema(SCHEMA)
    assert q["topic"] == {"type": "choice", "instructions": "What is it about?",
                          "criteria": {"billing": None, "support": None, "null": None, "3": None}}
    assert q["urgency"] == {"type": "score", "instructions": "Score `urgency` from 1 to 3", "criteria": ["1", "2", "3"]}
    assert q["complaint"] == {"type": "noul", "instructions": "Is `complaint` true?"}
    assert q["flag"]["type"] == "noul"


@pytest.mark.parametrize("prop, message", [
    ({"type": "string"}, "properties.x: a free string cannot be a fixed option set; use 'enum' or a boolean"),
    ({"type": "array"}, "properties.x: arrays are not supported; ask one field per element"),
    ({"type": "object"}, "properties.x: nested objects are not supported; flatten the schema"),
    ({"$ref": "#/defs/y"}, "properties.x: $ref/recursion is not supported; flatten the schema"),
    ({"type": "integer", "minimum": 0, "maximum": 10},
     "properties.x: 11 levels exceeds MAX_SCORE_LEVELS=10; narrow the range or use an enum"),
    ({"type": "integer", "minimum": 0}, "properties.x: a numeric field needs integer 'minimum' and 'maximum' to become a score"),
    ({"enum": []}, "properties.x: 'enum' must not be empty"),
])
def test_schema_errors(prop, message):
    with pytest.raises(SchemaError) as e:
        kime.questions_from_json_schema({"properties": {"x": prop}})
    assert str(e.value) == message


def test_projection():
    answers = {
        "topic": {"type": "choice", "choice": "3", "confidence": 0.4, "probabilities": {"billing": 0.1, "support": 0.2, "null": 0.1, "3": 0.6}},
        "urgency": {"type": "score", "score": 1.7, "confidence": 0.2, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}},
        "complaint": {"type": "noul", "noul": 0.5, "confidence": 0.0},
        "flag": {"type": "noul", "noul": 0.2, "confidence": 0.6},
    }
    agent = Fixed(answers)
    assert kime.decide(agent, "x", SCHEMA) == {"topic": 3, "urgency": 2, "complaint": True, "flag": False}
    d = kime.decide(agent, "x", SCHEMA, return_details=True, model="english")
    assert d.probabilities["flag"] == {"false": 0.8, "true": 0.2} and d.usage == {"input_tokens": 7}
    assert agent.asked[-1][1] == {"model": "english"}
    assert kime.decide(agent, "x", questions={"a": {}}) == answers
    with pytest.raises(ValueError, match="exactly one"):
        kime.decide(agent, "x")


def embed(texts):
    # One axis per word in the vocabulary, so the ranking is easy to follow.
    vocab = ["refund", "card", "pin", "transfer"]
    return [[float(w in t.lower()) for w in vocab] for t in texts]


def test_shortlist():
    labels = {"refund request": "", "card lost": "", "pin blocked": "", "transfer late": ""}
    agent = Fixed({"c": {"type": "choice", "choice": "card lost"}})
    q = {"c": {"type": "choice", "instructions": "Which?", "criteria": labels}, "n": {"type": "noul"}}
    out = kime.predict_shortlist(agent, "my card is lost, I need a new card", q, embed, k=2)
    kept = agent.asked[-1][0]["c"]["criteria"]
    assert list(kept) == ["card lost", "refund request"] and q["c"]["criteria"] is labels
    assert out["shortlist"]["c"]["labels"] == ["card lost", "refund request"]
    assert out["shortlist"]["c"]["scores"][0] == pytest.approx(1.0) and out["shortlist"]["c"]["n"] == 4
    assert kime.shortlist_choice("x", ["a", "b"], None, k=5) == ["a", "b"]
    with pytest.raises(ValueError, match=r"shape \(3, dim\)"):
        kime.shortlist_choice("x", ["a", "b"], lambda t: [[1.0]], k=1)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        kime.shortlist_choice("x", ["a"], embed, k=0)


def test_decide_on_the_checkpoint():
    models = os.environ.get("KIME_MODELS")
    try:
        agent = kime.load(models + "/laya" if models else "laya", device="cpu", precision="f32")
    except RuntimeError as e:
        pytest.skip("no laya checkpoint: %s" % e)
    v = agent.decide("I was charged twice for my subscription, please refund me", SCHEMA)
    assert v["topic"] in ("billing", "support", None, 3) and v["urgency"] in (1, 2, 3)
    assert isinstance(v["complaint"], bool)
