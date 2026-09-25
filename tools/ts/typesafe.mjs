// Runs the same System One calls through @typesafe-ai/sdk or @kime/sdk and writes the answers and
// the time each call took, so the two clients can be compared answer for answer.
//
//   node tools/ts/typesafe.mjs run <sdk module> <base url> texts.json out.json
//   node tools/ts/typesafe.mjs compare out-a.json out-b.json
//
// <sdk module> is a path to the package entry, such as node_modules/@typesafe-ai/sdk/dist/index.mjs
// or ts/dist/index.js. Each text is asked a noul, a choice and a score question, the first call is
// left out of the timings, and the client overhead is timed on 2000 calls to GET /v1/models.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const pct = (xs, p) => xs[Math.min(xs.length - 1, Math.floor(xs.length * p))];

async function run(modulePath, baseURL, textsPath, out) {
  const t0 = performance.now();
  const sdk = await import(pathToFileURL(resolve(modulePath)).href);
  const importMs = performance.now() - t0;
  const client = new sdk.TypeSafeClient({ apiKey: "none", baseURL });
  const questions = {
    billing: sdk.noul("Is this about billing or payments?"),
    intent: sdk.choice("What does the writer want?", {
      help: "they need help or support",
      complain: "they are unhappy",
      inform: "they are sharing information",
    }),
    urgency: sdk.score("How urgent is this?", ["not urgent", "somewhat urgent", "very urgent"]),
  };
  const texts = JSON.parse(readFileSync(textsPath, "utf8"));
  const models = (await client.models.list()).map((m) => m.name);
  const rows = [];
  const took = [];
  for (const [i, text] of texts.entries()) {
    const start = performance.now();
    const r = await client.systemOne({ state: text, questions, model: "jev-latest" });
    if (i) took.push(performance.now() - start);
    rows.push(r.answers);
  }
  const listed = [];
  for (let i = 0; i < 2000; i++) {
    const start = performance.now();
    await client.models.list();
    listed.push(performance.now() - start);
  }
  took.sort((a, b) => a - b);
  listed.sort((a, b) => a - b);
  const summary = {
    client: modulePath,
    runtime: typeof Bun !== "undefined" ? `bun ${Bun.version}` : typeof Deno !== "undefined" ? `deno ${Deno.version.deno}` : `node ${process.version}`,
    models,
    calls: texts.length,
    import_ms: +importMs.toFixed(2),
    p50_ms: +pct(took, 0.5).toFixed(2),
    p99_ms: +pct(took, 0.99).toFixed(2),
    models_p50_us: +(pct(listed, 0.5) * 1e3).toFixed(1),
    models_p99_us: +(pct(listed, 0.99) * 1e3).toFixed(1),
    rss_mb: typeof process !== "undefined" ? +(process.memoryUsage().rss / 2 ** 20).toFixed(1) : null,
  };
  console.log(JSON.stringify(summary));
  writeFileSync(out, JSON.stringify({ summary, answers: rows }));
}

function compare(paths) {
  const [base, ...others] = paths.map((p) => JSON.parse(readFileSync(p, "utf8")));
  for (const other of others) {
    const same = base.answers.filter((a, i) => JSON.stringify(a) === JSON.stringify(other.answers[i])).length;
    console.log(`${base.summary.client} vs ${other.summary.client}: ${same} of ${base.answers.length} responses identical`);
  }
}

const [cmd, ...rest] = process.argv.slice(2);
if (cmd === "compare") compare(rest);
else await run(...rest);
