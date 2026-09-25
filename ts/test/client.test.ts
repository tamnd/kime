// @kime/sdk against a stub server that answers from a script, for the headers, the errors and
// the retries. It runs as is under `node --test`, `bun test` and `deno test`.

import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { after, before, beforeEach, test } from "node:test";

import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  APIUserAbortError,
  AuthenticationError,
  BadRequestError,
  choice,
  InternalServerError,
  noul,
  parseRetryAfter,
  RateLimitError,
  retryDelayMs,
  score,
  TypeSafeClient,
  TypeSafeError,
  UnprocessableEntityError,
  VERSION,
} from "../src/index.ts";

type Reply = { status: number; headers?: Record<string, string>; body?: unknown; delayMs?: number };
type Seen = { method: string; url: string; headers: IncomingMessage["headers"]; body: string };

let server: Server;
let url: string;
let script: Reply[] = [];
let seen: Seen[] = [];

before(async () => {
  server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      seen.push({ method: req.method!, url: req.url!, headers: req.headers, body });
      const r = script.shift() ?? { status: 500, body: { detail: "script ran out" } };
      const out = r.body === undefined ? "" : typeof r.body === "string" ? r.body : JSON.stringify(r.body);
      setTimeout(() => {
        res.writeHead(r.status, { "content-type": "application/json", ...r.headers });
        res.end(out);
      }, r.delayMs ?? 0);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});
after(() => {
  server.closeAllConnections();
  server.close();
});
beforeEach(() => {
  script = [];
  seen = [];
});

const ANSWER = {
  model: "laya",
  usage: { input_tokens: 12, output_tokens: 0 },
  answers: {
    billing: { type: "noul", noul: 0.9 },
    intent: { type: "choice", choice: "help", confidence: 0.5, probabilities: { help: 0.75, other: 0.25 } },
    urgency: { type: "score", score: 1.2, confidence: 0.1, legend: { 0: "lo", 1: "mid", 2: "hi" }, probabilities: { 0: 0.2, 1: 0.4, 2: 0.4 } },
  },
};
const QUESTIONS = {
  billing: noul("Is this about billing?"),
  intent: choice("What do they want?", { help: "they need help", other: null }),
  urgency: score("How urgent?", ["lo", "mid", "hi"]),
};
const FAST = { backoffInitialMs: 1, backoffMaxMs: 2 };
const quiet = { debug() {}, info() {}, warn() {}, error() {} };

test("answers, headers and the kime options", async () => {
  script = [{ status: 200, headers: { "x-typesafe-request-id": "req_1" }, body: ANSWER }];
  const client = new TypeSafeClient({ baseURL: `${url}//`, apiKey: "k1", defaultHeaders: { "X-Mine": "yes" } });
  const { data, response, requestId } = await client
    .systemOne({ state: "I was charged twice", questions: QUESTIONS, kime: { precision: 3, confidence: "entropy" } })
    .withResponse();
  // These lines are also the type test: a wrong label or field does not compile.
  const b: number = data.answers.billing.noul;
  const label: "help" | "other" = data.answers.intent.choice;
  const p: number = data.answers.urgency.probabilities["2"];
  assert.deepEqual([b, label, p, requestId, response.status], [0.9, "help", 0.4, "req_1", 200]);
  const s = seen[0]!;
  assert.equal(`${s.method} ${s.url}`, "POST /v1/systemone");
  assert.equal(s.headers.authorization, "Bearer k1");
  assert.equal(s.headers["x-mine"], "yes");
  assert.equal(s.headers["x-typesafe-sdk"], `kime-sdk/${VERSION}`);
  assert.equal(s.headers["content-type"], "application/json");
  assert.equal(s.headers["x-typesafe-retry-count"], undefined);
  const sent = JSON.parse(s.body);
  assert.equal(sent.model, "jev-latest");
  assert.deepEqual(sent.kime, { precision: 3, confidence: "entropy" });
  assert.deepEqual(sent.questions.intent, { type: "choice", instructions: "What do they want?", criteria: { help: "they need help", other: null } });
});

test("no key is fine off api.typesafe.ai", async () => {
  script = [{ status: 200, body: { models: [{ name: "laya", description: "d", release_date: "2026-09-25" }] } }];
  const models = await new TypeSafeClient({ baseURL: url }).models.list();
  assert.equal(models[0]!.name, "laya");
  assert.equal(seen[0]!.headers.authorization, undefined);
  assert.equal(seen[0]!.headers["content-type"], undefined);
  assert.throws(() => new TypeSafeClient({ baseURL: "https://api.typesafe.ai", apiKey: "" }), /No API key was provided/);
  script = [{ status: 200, body: { data: [] } }];
  await assert.rejects(new TypeSafeClient({ baseURL: url }).models.list(), /Unexpected response shape/);
});

test("errors map to their classes and messages", async () => {
  const client = new TypeSafeClient({ baseURL: url, retry: { maxRetries: 0 } });
  const cases: [number, unknown, Function, string][] = [
    [400, { detail: "request body must be valid JSON" }, BadRequestError, "400 request body must be valid JSON"],
    [401, { error: { message: "bad key" } }, AuthenticationError, "401 bad key"],
    [422, { detail: [{ loc: ["body", "questions", "a"], msg: "bad" }] }, UnprocessableEntityError, "422 questions.a: bad"],
    [503, undefined, InternalServerError, "503 status code (no body)"],
    [418, { note: "x".repeat(300) }, APIError, `418 ${`{"note":"${"x".repeat(300)}"}`.slice(0, 200)}…`],
  ];
  for (const [status, body, kind, message] of cases) {
    script = [{ status, headers: { "x-typesafe-request-id": "r9" }, body }];
    const err = (await client.systemOne({ state: "x", questions: { a: noul() } }).catch((e) => e)) as APIError;
    assert.ok(err instanceof kind, `${status} gave ${err.name}`);
    assert.equal(err.message, message);
    assert.equal(err.requestId, "r9");
    assert.equal(err.name, (kind as { name: string }).name);
  }
  assert.throws(() => client.systemOne({ state: "x", questions: {} }), /At least one question is required/);
  assert.throws(
    () => client.systemOne({ state: "x", questions: { s: { type: "score", criteria: ["one"] as never } } }),
    /Score question "s" has 1 criteria; at least two scores are required/,
  );
  assert.throws(() => score("x", { a: 1 } as never), /not a map/);
  assert.throws(() => choice("x", ["a"] as never), /not a list/);
});

test("retries", async () => {
  script = [
    { status: 503, body: { detail: "busy" } },
    { status: 429, headers: { "retry-after-ms": "5" }, body: { detail: "slow down" } },
    { status: 200, body: ANSWER },
  ];
  const r = await new TypeSafeClient({ baseURL: url, retry: FAST, logger: quiet }).systemOne({ state: "x", questions: QUESTIONS });
  assert.equal(r.answers.billing.noul, 0.9);
  assert.deepEqual(seen.map((s) => s.headers["x-typesafe-retry-count"]), [undefined, "1", "2"]);

  seen = [];
  script = [{ status: 503 }, { status: 503 }, { status: 503 }];
  await assert.rejects(new TypeSafeClient({ baseURL: url, retry: FAST }).systemOne({ state: "x", questions: QUESTIONS }), InternalServerError);
  assert.equal(seen.length, 3);

  seen = [];
  script = [{ status: 422, body: { detail: "no" } }];
  await assert.rejects(new TypeSafeClient({ baseURL: url, retry: FAST }).systemOne({ state: "x", questions: QUESTIONS }), UnprocessableEntityError);
  assert.equal(seen.length, 1);

  seen = [];
  script = [{ status: 429, headers: { "retry-after": "7" }, body: { detail: "later" } }];
  const err = await new TypeSafeClient({ baseURL: url, retry: { maxRetries: 0 } })
    .systemOne({ state: "x", questions: QUESTIONS })
    .catch((e) => e);
  assert.ok(err instanceof RateLimitError);
  assert.equal(err.retryAfterMs, 7000);
});

test("timeouts, aborts and connection errors", async () => {
  script = [{ status: 200, body: ANSWER, delayMs: 300 }];
  const err = await new TypeSafeClient({ baseURL: url, timeout: 50, retry: { maxRetries: 0 } })
    .systemOne({ state: "x", questions: QUESTIONS })
    .catch((e) => e);
  assert.ok(err instanceof APITimeoutError && err instanceof APIConnectionError);
  assert.equal(err.message, "Request timed out after 50ms.");

  script = [{ status: 200, body: ANSWER, delayMs: 300 }];
  const controller = new AbortController();
  const pending = new TypeSafeClient({ baseURL: url }).systemOne({ state: "x", questions: QUESTIONS }, { signal: controller.signal });
  setTimeout(() => controller.abort(), 20);
  await assert.rejects(pending, APIUserAbortError);

  await assert.rejects(
    new TypeSafeClient({ baseURL: "http://127.0.0.1:9", retry: { maxRetries: 1, backoffInitialMs: 1 }, logger: quiet }).models.list(),
    APIConnectionError,
  );
});

test("retry-after and backoff", () => {
  const h = (o: Record<string, string>) => new Headers(o);
  assert.equal(parseRetryAfter(h({ "retry-after-ms": "250", "retry-after": "9" })), 250);
  assert.equal(parseRetryAfter(h({ "retry-after": "2" })), 2000);
  assert.equal(parseRetryAfter(h({ "retry-after": "Wed, 21 Oct 2015 07:28:00 GMT" })), 0);
  assert.equal(parseRetryAfter(h({ "retry-after": "-1" })), undefined);
  for (const [attempt, top] of [[0, 500], [1, 1000], [2, 2000], [3, 4000], [4, 5000], [9, 5000]] as const) {
    assert.equal(retryDelayMs(attempt, undefined, undefined, () => 0), top);
    assert.equal(retryDelayMs(attempt, undefined, undefined, () => 1), Math.round(top * 0.75));
  }
  assert.equal(retryDelayMs(0, h({ "retry-after": "120" }), undefined, () => 0), 500);
  assert.throws(() => new TypeSafeClient({ retry: { backoffJitter: 2 } }), /must be between 0 and 1/);
  assert.throws(() => new TypeSafeClient({ timeout: 0 }), /positive number of milliseconds/);
  assert.throws(() => new TypeSafeClient({ logLevel: "loud" as never }), /Invalid log level "loud"/);
});

test("environment, kime names first", async () => {
  const env = (globalThis as { process?: { env: Record<string, string | undefined> } }).process!.env;
  env.TYPESAFE_BASE_URL = "http://127.0.0.1:1";
  env.KIME_BASE_URL = url;
  env.TYPESAFE_DEFAULT_MODEL = "laya";
  try {
    script = [{ status: 200, body: ANSWER }];
    await new TypeSafeClient().systemOne({ state: "x", questions: QUESTIONS });
    assert.equal(JSON.parse(seen[0]!.body).model, "laya");
  } finally {
    delete env.TYPESAFE_BASE_URL;
    delete env.KIME_BASE_URL;
    delete env.TYPESAFE_DEFAULT_MODEL;
  }
});

test("debug logs redact the key", async () => {
  const lines: unknown[][] = [];
  const logger = { debug: (...a: unknown[]) => lines.push(a), info() {}, warn() {}, error() {} };
  script = [{ status: 200, body: ANSWER }];
  await new TypeSafeClient({ baseURL: url, apiKey: "sk-live-abcdefgh1234", logLevel: "debug", logger }).systemOne({ state: "x", questions: QUESTIONS });
  const sent = JSON.stringify(lines);
  assert.ok(sent.includes("Bearer ***1234") && !sent.includes("abcdefgh"));
});

test("a thrown error is a TypeSafeError", () => {
  assert.ok(new APIUserAbortError() instanceof TypeSafeError);
});
