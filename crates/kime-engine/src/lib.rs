//! Session, scheduler, batching, the state memory cache, the answer cache and the tokenization cache. This is the in process API that the server, the CLI and the language bindings all sit on. See spec/07-engine.md and spec/11-serving.md.

#![forbid(unsafe_code)]
