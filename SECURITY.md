# Security policy

## Supported versions

The project is pre-1.0 and nothing is supported in the sense that word usually carries. Fixes go on the default branch.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository, under Security, Report a vulnerability. Please do not open a public issue for something you believe is exploitable. Expect an acknowledgement within a few days.

## What counts

kime is a server that takes text from strangers, and a library that loads weight files people download. Both are untrusted input.

- Memory unsafety on any input. The `unsafe` is in six crates: the tensor layer, the four backends and the tokenizer. A request that makes a kernel read past a buffer, or a `.kime` or safetensors file whose header sends the loader outside its mapping, is the most serious kind of bug we can have.
- A hang or unbounded memory growth on a bounded input. Every limit in [`spec/03-api.md`](spec/03-api.md) exists so that one request cannot take the machine, and a way around one is a denial of service.
- One request seeing another's data. The state cache in [`spec/11-serving.md`](spec/11-serving.md) is keyed by content hash and shared across keys, so a way to learn whether some other caller sent a given state, or to read a cached answer that was not yours, is in scope.
- Anything that gets around authentication or the per key rate limits.

A model giving a wrong or badly calibrated answer is not a vulnerability. It is a quality bug and has its own issue template.

## What we do about it

Reports are triaged, fixed on a private branch, and released with an advisory that says what the problem was and what it affected. Reporters are credited unless they prefer not to be.
