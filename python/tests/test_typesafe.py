"""kime.typesafe against a stub server that answers from a script, for the headers, the errors
and the retries, and in process on the checkpoint when it is on disk."""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import kime
from kime import typesafe as t

ANSWER = {
    "model": "laya",
    "usage": {"input_tokens": 12, "output_tokens": 0},
    "answers": {
        "b": {"type": "noul", "noul": 0.9},
        "c": {"type": "choice", "choice": "x", "confidence": 0.5, "probabilities": {"x": 0.75, "y": 0.25}},
        "s": {"type": "score", "score": 1.2, "confidence": 0.1, "legend": {"0": "lo", "1": "mid", "2": "hi"},
              "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4}},
        "new": {"type": "someday", "x": 1},
    },
}
QUESTIONS = {"b": t.Noul(instructions="billing?"), "c": t.Choice(criteria={"x": "", "y": ""}),
             "s": t.Score(criteria=["lo", "mid", "hi"])}


class Stub:
    """Answers each request with the next (status, headers, body) in `script`."""

    def __init__(self):
        self.script, self.seen = [], []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def handle_one(self):
                n = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(n) if n else b""
                stub.seen.append((self.command, self.path, dict(self.headers), body))
                status, headers, out = stub.script.pop(0)
                out = out if isinstance(out, bytes) else json.dumps(out).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            do_GET = do_POST = handle_one

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


FAST = t.RetryPolicy(backoff_initial=0.01, backoff_max=0.02)


def test_answers_and_headers(stub):
    stub.script = [(200, {"x-typesafe-request-id": "req_1"}, ANSWER)]
    c = t.TypeSafeClient(base_url=stub.url, api_key="k1", headers={"X-Mine": "yes"})
    r = c.system_one({"message": "hi"}, QUESTIONS, extra_body={"kime": {"precision": 3}})
    assert r.model == "laya" and r.request_id == "req_1" and r.usage.input_tokens == 12
    assert r.nouls["b"].noul == 0.9 and r.choices["c"].probabilities == {"x": 0.75, "y": 0.25}
    assert r.scores["s"].legend == {0: "lo", 1: "mid", 2: "hi"} and r.scores["s"].probabilities[2] == 0.4
    assert "new" not in r.answers and r.raw_http_response.status_code == 200
    method, path, headers, body = stub.seen[0]
    assert (method, path) == ("POST", "/v1/systemone")
    assert headers["Authorization"] == "Bearer k1" and headers["X-Mine"] == "yes"
    assert headers["X-TypeSafe-SDK"].startswith("kime-python/") and "X-TypeSafe-Retry-Count" not in headers
    sent = json.loads(body)
    assert sent["model"] == "jev-latest" and sent["kime"] == {"precision": 3}
    assert sent["questions"]["c"] == {"type": "choice", "criteria": {"x": "", "y": ""}}


def test_no_key_is_fine_off_typesafe(stub):
    stub.script = [(200, {}, {"models": [{"name": "laya", "description": "d", "release_date": "2026-09-25"}]})]
    r = t.TypeSafeClient(base_url=stub.url).models.list()
    assert r.models[0].name == "laya" and "Authorization" not in stub.seen[0][2]
    with pytest.raises(t.TypeSafeError, match="No API key"):
        t.TypeSafeClient(base_url="https://api.typesafe.ai", api_key="")


def test_errors(stub):
    c = t.TypeSafeClient(base_url=stub.url, retry=t.RetryPolicy(max_retries=0))
    cases = [
        (400, {"detail": "request body must be valid JSON"}, t.TypeSafeBadRequestError, "400 request body must be valid JSON"),
        (401, {"error": {"message": "bad key"}}, t.TypeSafeAuthenticationError, "401 bad key"),
        (404, {"detail": "model 'x' not found"}, t.TypeSafeNotFoundError, "404 model 'x' not found"),
        (422, {"detail": [{"loc": ["body", "questions", "a"], "msg": "bad"}]}, t.TypeSafeUnprocessableEntityError,
         "422 questions.a: bad"),
        (503, b"", t.TypeSafeInternalServerError, "503 status code (no body)"),
    ]
    for status, body, kind, text in cases:
        stub.script = [(status, {"x-typesafe-request-id": "r9"}, body)]
        with pytest.raises(kind) as e:
            c.system_one("x", {"a": t.Noul()})
        assert str(e.value) == "POST %s/v1/systemone: %s (request_id=r9)" % (stub.url, text)
    stub.script = [(200, {}, {"model": "m", "usage": {}, "answers": {"a": {"type": "noul", "noul": "high"}}})]
    with pytest.raises(t.TypeSafeAPIResponseValidationError) as e:
        c.system_one("x", {"a": t.Noul()})
    assert e.value.field_path == "answers.a.noul" and "Invalid response data at 'answers.a.noul'." in str(e.value)
    with pytest.raises(t.TypeSafeError, match="At least one question"):
        c.system_one("x", {})
    with pytest.raises(t.TypeSafeError, match='Score question "s" has no criteria'):
        c.system_one("x", {"s": {"type": "score", "criteria": []}})


def test_retries(stub):
    stub.script = [(503, {}, {"detail": "busy"}), (429, {"retry-after-ms": "10"}, {"detail": "slow down"}),
                   (200, {}, ANSWER)]
    r = t.TypeSafeClient(base_url=stub.url, retry=FAST).system_one("x", QUESTIONS)
    assert r.nouls["b"].noul == 0.9
    assert [s[2].get("X-TypeSafe-Retry-Count") for s in stub.seen] == [None, "1", "2"]
    stub.seen.clear()
    stub.script = [(503, {}, {"detail": "busy"})] * 3
    with pytest.raises(t.TypeSafeInternalServerError):
        t.TypeSafeClient(base_url=stub.url, retry=FAST).system_one("x", QUESTIONS)
    assert len(stub.seen) == 3
    stub.seen.clear()
    stub.script = [(422, {}, {"detail": "no"})]
    with pytest.raises(t.TypeSafeUnprocessableEntityError):
        t.TypeSafeClient(base_url=stub.url, retry=FAST).system_one("x", QUESTIONS)
    assert len(stub.seen) == 1
    # A wait the server asks for that would run past the budget is not waited for.
    stub.seen.clear()
    stub.script = [(429, {"retry-after": "60"}, {"detail": "later"})]
    with pytest.raises(t.TypeSafeRateLimitError) as e:
        t.TypeSafeClient(base_url=stub.url).system_one("x", QUESTIONS)
    assert e.value.retry_after_ms == 60000 and len(stub.seen) == 1


def test_connection_errors():
    c = t.TypeSafeClient(base_url="http://127.0.0.1:9", retry=t.RetryPolicy(max_retries=1, backoff_initial=0.01))
    with pytest.raises(t.TypeSafeAPIConnectionError):
        c.system_one("x", QUESTIONS)


def test_retry_after_and_backoff():
    assert t.parse_retry_after({"retry-after-ms": "250", "retry-after": "9"}) == 250
    assert t.parse_retry_after({"retry-after": "2"}) == 2000
    assert t.parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0
    assert t.parse_retry_after({"retry-after": "-1"}) is None
    for attempt, top in ((1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (5, 5.0), (9, 5.0)):
        d = t._backoff(attempt, 0.5, 5.0, 0.25)
        assert top * 0.75 - 1e-3 <= d <= top
    with pytest.raises(t.TypeSafeError):
        t.RetryPolicy(backoff_jitter=2)


def test_env(stub, monkeypatch):
    stub.script = [(200, {}, ANSWER)]
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("KIME_BASE_URL", stub.url)
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "laya")
    t.TypeSafeClient().system_one("x", QUESTIONS)
    assert json.loads(stub.seen[0][3])["model"] == "laya"


def test_async(stub):
    stub.script = [(200, {}, ANSWER)] * 4

    async def main():
        async with t.AsyncTypeSafeClient(base_url=stub.url) as c:
            return await asyncio.gather(*(c.system_one("x", QUESTIONS) for _ in range(4)))

    assert [r.nouls["b"].noul for r in asyncio.run(main())] == [0.9] * 4


def local():
    models = os.environ.get("KIME_MODELS")
    try:
        return kime.Agent(models + "/laya" if models else "laya", device="cpu", precision="f32")
    except RuntimeError as e:
        pytest.skip("no laya checkpoint: %s" % e)


def test_local_is_the_agent():
    agent = local()
    c = t.TypeSafeClient(local=agent)
    state = "I was charged twice for my subscription, please refund me"
    r = c.system_one(state, QUESTIONS)
    want = agent.system_one(state, {k: q.model_dump() for k, q in QUESTIONS.items()})
    assert r.nouls["b"].noul == round(want["answers"]["b"]["noul"], 2)
    assert r.choices["c"].choice == want["answers"]["c"]["choice"]
    assert r.usage.input_tokens > 0 and r.request_id.startswith("req_")
    assert [m.name for m in c.models.list().models] == ["kime-latest", "laya", "jev-latest"]
    with pytest.raises(t.TypeSafeUnprocessableEntityError, match="needs at least 1 option"):
        c.system_one("x", {"a": {"type": "choice", "criteria": {}}})
