# HTTP API

kime-serve speaks the TypeSafe System One protocol byte for byte on the fields Jev defines. It also accepts every request Laya's laya-serve accepts. Everything kime adds lives under a single `kime` key in the request and response, and appears in the response only when asked for. This keeps the unmodified TypeSafe SDKs, jev-ultrafast, OpenRouter style gateways and Laya clients working by changing only the base URL.

## Endpoints

| Method and path | Purpose |
|---|---|
| `POST /v1/systemone` | Answer questions about one state. The main endpoint. |
| `POST /v1/systemone/batch` | Answer questions about many states in one call. kime extension. |
| `GET /v1/models` | List served models and aliases. |
| `GET /v1/models/{id}` | Details for one model: family, languages, limits, calibration version. kime extension. |
| `GET /health` | Liveness. Returns `{"status":"ok"}` once every preloaded model is ready. |
| `GET /ready` | Readiness with detail: loaded models, device, queue depth. kime extension. |
| `GET /metrics` | Prometheus text format. Off unless `--metrics` is set, and can be bound to a separate port. |
| `GET /openapi.json` | OpenAPI 3.1 document generated from the Rust types. |

All bodies are UTF-8 JSON. Requests may be gzip or zstd compressed with `Content-Encoding`. Responses honour `Accept-Encoding` but are not compressed below 1 KiB, because compressing small bodies costs more time than it saves.

## Authentication

Keys are configured with `--api-keys-file` (one key per line, optionally `key name rpm tps`), or the `KIME_API_KEYS` env var. When no keys are configured, auth is off and the server logs a warning at start if it is bound to a non-loopback address. The error bodies match Jev exactly:

- Missing header: `403` with `{"detail":{"error_type":"authentication_error","message":"Must supply an API key! Check your request and try again."}}`
- Unknown key: `401` with `{"detail":{"error_type":"authentication_error","message":"Cannot authenticate with the server. Please check your API key and try again."}}`

Keys are compared in constant time against a set of blake3 hashes. Raw keys are never kept in memory after load and never logged.

## POST /v1/systemone

### Request

```json
{
  "state": "string | object | array",
  "model": "string, optional in kime, required by Jev",
  "questions": {
    "<id>": {
      "type": "choice | score | noul",
      "instructions": "string | object | array | null",
      "criteria": "see below"
    }
  },
  "kime": { "optional kime options, see Extensions" }
}
```

`model` is required by Jev's schema. kime accepts a missing `model` and uses the server's default model. This matches laya-serve, which ignores the field when it is unknown. Unknown top level fields are ignored, because the official Python SDK's `extra_body` can add arbitrary fields and we must not reject them.

Criteria per type:

| Type | Accepted criteria | Notes |
|---|---|---|
| `choice` | Object `{label: EntryType or null}`, or array of label strings | 1 to 255 options by default. Up to 4,096 with `kime.shortlist`. A null or empty description means the label alone is the meaning. A single option choice is legal and returns probability 1. |
| `score` | Array of EntryType, index is the level | 2 to 32 levels. Jev caps at 10 and we accept more. An int keyed object `{"0":...,"1":...}` is accepted for old Python SDK clients if its keys are exactly 0 to n-1. |
| `noul` | Absent, or object with optional `true` and `false` keys | Keys are matched case insensitively after converting to string, so `True` and `"TRUE"` work as in Laya. |

EntryType is a string, a number, a boolean, an object, an array or null. How each one renders into tokens is in 06.

### Response

```json
{
  "model": "kime-v1-m-en-1.0.0",
  "answers": {
    "<id>": { "type": "choice", "choice": "billing", "confidence": 0.81, "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0} },
    "<id>": { "type": "score", "score": 1.05, "confidence": 0.92, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"}, "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05} },
    "<id>": { "type": "noul", "noul": 0.95 }
  },
  "usage": { "input_tokens": 318, "output_tokens": 0 }
}
```

Rules that clients depend on:

- `model` is the concrete resolved id, never an alias. That is what Jev does.
- `answers` has exactly one entry per question id, in request order. Jev returns arbitrary order, and preserving order breaks nobody.
- `probabilities` keys are exactly the offered labels, or the level indexes as strings for score. Nothing is added or dropped. The values are finite and in [0, 1].
- By default, values are rounded to 2 decimals like Jev, and the rounding is done so the rounded values still sum to exactly 1. We use largest remainder rounding on the scaled integers, so the jev-ultrafast check `abs(sum - 1) <= 0.02` always passes. `kime.precision` (0 to 6, default 2) changes the rounding. `kime.precision: null` returns raw f32 values.
- `choice` is always the argmax of the returned probabilities. Ties after rounding are broken by the unrounded value, then by option order.
- `score` is `sum(i * p_i)` computed from unrounded probabilities, then rounded.
- `legend` echoes each criterion value exactly as sent, keyed by level index as a string.
- Noul answers have no `confidence` unless extensions are on.
- `usage.input_tokens` counts all tokens the model read for this request: state tokens once, plus question and option tokens. `usage.output_tokens` is always 0. Nothing is generated, and Jev does not bill output tokens either. Clients that divide by output tokens must handle zero, and Jev's SDK does.

### Headers

Every response carries `x-request-id: req_<32 hex>` and, for Jev compatibility, the same value in `x-typesafe-request-id`. A client supplied `x-request-id` is echoed back if it matches `[A-Za-z0-9_-]{1,64}`. `server-timing` reports `queue`, `tok`, `state`, `question` and `total` durations in microseconds with a fractional part, so latency can be split without extra tooling.

## Extensions (the `kime` object)

All fields are optional.

| Field | Type | Default | Effect |
|---|---|---|---|
| `extensions` | bool | false | Include the `kime` block in the response. |
| `precision` | int or null | 2 | Decimal places for probabilities, scores and confidences. |
| `confidence` | `"jev"` or `"entropy"` | `"jev"` | Which confidence formula to use. See 04. |
| `permutations` | int 1 to 8 | 1 | Average choice probabilities over this many option orders. Cheap, because the state is encoded once. |
| `shortlist` | int | none | For choices with more than 255 options, first rank options by embedding similarity and score only the top n. |
| `truncation` | `"head"`, `"tail"`, `"middle"`, `"error"` | `"tail"` | Where to cut an over long state. `tail` keeps the start (Laya default). `head` keeps the end, which suits conversations. `error` returns 422 instead of cutting. |
| `route` | object | none | Router overrides: `model`, `task`, `lang`, `lang_guess`. See 11. |
| `cache` | `"use"`, `"bypass"`, `"refresh"` | `"use"` | Controls the state memory cache and the answer cache. |
| `deadline_ms` | int | none | Reject with 504 if the answer cannot be produced in time. The scheduler also uses it to order work. |
| `state_segments` | bool | auto | Treat top level keys of an object state as independently cacheable segments. See 05. |
| `return_embedding` | bool | false | Return the pooled state embedding (f16, base64) for client side shortlisting or dedup. |

When `extensions` is true, the response gets:

```json
"kime": {
  "request_id": "req_...",
  "timing_us": {"queue": 41, "tokenize": 18, "state": 612, "questions": 188, "total": 874},
  "routing": {"model": "kime-v1-m-en", "reason": "script=Latin lang=en score=9", "detection": {"script": "Latin", "lang": "en", "confidence": 0.97}},
  "truncation": {"state_tokens": 5120, "state_tokens_used": 4096, "dropped": 1024, "strategy": "tail"},
  "cache": {"state": "hit", "answers": "miss"},
  "answers": {
    "<id>": {"confidence_entropy": 0.71, "noul_confidence": 0.9, "act_probability": 0.97, "logits": [1.2, -0.3], "temperature": 1.31, "options_used": 3}
  }
}
```

The Laya response shape (a top level `routing` block and `action.act_probability` inside each answer) is produced instead when the resolved model is a `laya` compat model and the request did not send a `kime` object. Laya clients therefore get the exact shape they expect, and Jev clients never see extra fields.

## POST /v1/systemone/batch

```json
{"model": "kime-v1-m-en", "items": [{"id": "a", "state": "...", "questions": {...}}, ...], "kime": {...}}
```

The response is `{"model": "...", "results": [{"id": "a", "answers": {...}, "usage": {...}} or {"id": "a", "error": {...}}], "usage": {...}}`. Items fail independently. Items with byte identical `questions` objects share the rendered and tokenized question tower input. Up to 1,024 items per call, and the whole call counts against the token rate limit.

## GET /v1/models

```json
{"models": [
  {"name": "kime-latest", "description": "Default kime model, routes by language.", "release_date": "2026-11-01"},
  {"name": "kime-v1-m-en", "description": "...", "release_date": "..."},
  {"name": "jev-latest", "description": "Alias. Served by kime-latest.", "release_date": "..."}
]}
```

The server config maps aliases to concrete ids. By default `jev-latest`, `jev-preview`, `jev` and any `jev-*` id map to `kime-latest`. This lets clients with a hard coded `jev-latest` work unchanged. Operators can turn this off with `--no-jev-aliases`. `laya`, `convaiinnovations/laya`, `laya-multilingual` and `laya-typed-decisions` map to the compat models when those are loaded, and to their native kime-v1 equivalents otherwise.

## Errors

| Status | When | Body |
|---|---|---|
| 400 | Body is not JSON, not an object, or too large (default 8 MiB) | `{"detail":"<message>"}` |
| 401, 403 | Auth, see above | Jev auth shape |
| 404 | Unknown path, or unknown `model` after alias resolution | `{"detail":"Not Found"}` or `{"detail":"model 'x' not found"}` |
| 405 | Wrong method | `{"detail":"Method Not Allowed"}` |
| 413 | Request over 64k tokens, or state plus longest question over 32k | `{"detail":[{"loc":["body","state"],"msg":"...","type":"too_long"}]}` |
| 422 | Validation | FastAPI list format, one entry per problem, with `loc`, `msg`, `type` and `input` |
| 429 | Rate limit | `{"detail":{"error_type":"rate_limit_error","message":"..."}}` with `retry-after` and `retry-after-ms` |
| 504 | `deadline_ms` would be missed | `{"detail":{"error_type":"deadline_exceeded","message":"..."}}` |
| 529 | Queue full | `{"detail":{"error_type":"overloaded_error","message":"..."}}` with `retry-after-ms` |
| 500 | Bug or device error | `{"detail":{"error_type":"internal_error","message":"...","request_id":"req_..."}}` |

Validation reports every problem at once, not only the first. Error `loc` paths use the question id, for example `["body","questions","urgency","criteria",3]`. Messages are written for the person reading them, for example "score question 'urgency' needs at least 2 levels, got 1".

Validation rules:

- `questions` must be a non-empty object. An empty object is a 422 for Jev compatibility. Laya returned an empty answer instead, and the Laya compat mode keeps that behaviour.
- `type` must be one of the three types.
- `choice` needs at least 1 option. Labels must be unique after trimming whitespace. The maximum is 255 unless shortlisting is on.
- `score` needs 2 to 32 levels.
- `noul` criteria may only have the keys true and false, in any case.
- A missing `instructions` is allowed (Jev allows it and the Jev JS builder defaults it to null), and it renders as an empty question.
- The state must not be null. An empty string is allowed and is answered from priors.

## Limits and defaults

| Limit | Default | Flag |
|---|---|---|
| Max request body | 8 MiB | `--max-body` |
| Max request tokens | 65,536 | `--max-request-tokens` |
| Max state tokens seen by the model | 32,768 | `--max-state-tokens`, capped by the model's trained context |
| Max questions per request | 256 | `--max-questions` |
| Max options per choice | 255, or 4,096 with shortlist | `--max-options` |
| Max batch items | 1,024 | `--max-batch-items` |
| Rate limit per key | off | per key in the keys file, or `--rpm`, `--tps` |

## Compatibility test targets

The API is considered compatible when all of these pass unmodified against kime-serve (see 15):

1. The TypeSafe Python SDK 0.7.1 test suite's recorded request and response fixtures, replayed with the base URL changed.
2. The TypeSafe JS SDK 0.6.0 examples.
3. jev-ultrafast's `validate_choice`, plus a full run of its Wikipedia example against a local kime-serve.
4. Laya 0.3.7's `serve.py` client examples and the impossibl curl examples.
5. The OpenAPI document of api.typesafe.ai: every request valid under it is accepted by kime, and every kime response is valid under it.
