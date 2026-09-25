# @kime/sdk

The TypeScript client for `kime serve`, and for any other System One endpoint such as Jev. It is a drop-in for `@typesafe-ai/sdk` 0.6.0: the same `TypeSafeClient`, `systemOne`, `models.list`, `.withResponse()`, `noul`, `choice` and `score` builders, answer types, errors, retry policy and logging, with no dependencies. It runs on Node 20 and newer, Bun, Deno and edge runtimes.

```ts
import { choice, TypeSafeClient } from "@kime/sdk";

const client = new TypeSafeClient(); // http://127.0.0.1:8000, where `kime serve` listens
const response = await client.systemOne({
  state: { document: "I was charged twice. Please fix this ASAP." },
  questions: {
    category: choice("What is this ticket about?", { billing: null, technical: null, other: null }),
  },
  kime: { precision: 4, extensions: true },
});

console.log(response.answers.category.choice); // typed as "billing" | "technical" | "other"
```

Code written for `@typesafe-ai/sdk` works after changing the import. What is different:

- The base URL defaults to `http://127.0.0.1:8000` in place of `https://api.typesafe.ai`.
- An API key is only required for api.typesafe.ai. `kime serve` runs without keys, so no `Authorization` header is sent when there is none.
- `KIME_API_KEY`, `KIME_BASE_URL`, `KIME_DEFAULT_MODEL` and `KIME_LOG_LEVEL` are read first, and the `TYPESAFE_` names when those are not set.
- A request may carry a typed `kime` field with kime's options (`precision`, `confidence`, `extensions`, `deadline_ms`, see [spec/03-api.md](../spec/03-api.md)). Other servers ignore it.
- The SDK header is `kime-sdk/<version>` and log lines start with `[kime-sdk]`.

The package lives in this repository for now and will move to `tamnd/kime-ts` once it needs its own release cadence.

## Development

```sh
npm ci
npm test          # type checks, then runs the tests on Node
bun test          # the same tests on Bun
deno test -A test # and on Deno
npm run build     # dist/index.js and dist/index.d.ts
```

`tools/ts/typesafe.mjs` runs the same questions through this package and `@typesafe-ai/sdk` against a running server and compares the answers.

`src/index.ts` follows `@typesafe-ai/sdk` 0.6.0, which is MIT licensed; see [THIRD_PARTY.md](THIRD_PARTY.md).
