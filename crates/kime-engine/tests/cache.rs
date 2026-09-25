//! The answer cache on the laya checkpoint: a cached answer is the same JSON as a computed one,
//! labels included, a question is found again inside another set of questions, and
//! `kime.cache` bypass and refresh do what spec/11-serving.md says.
//!
//! It needs the laya checkpoint, under `$KIME_MODELS/laya` or in the Hugging Face cache, and
//! passes with a note when it is missing unless `KIME_REQUIRE_WEIGHTS` is set.

use kime_core::request::{Limits, Request, parse};
use kime_engine::{CacheStats, Device, Kime};
use serde_json::{Value, json};

fn open(entries: usize) -> Option<Kime> {
    let model =
        std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |m| format!("{m}/laya"));
    let b = Kime::builder().model(model).device(Device::Cpu { threads: 0 }).answer_cache(entries);
    match b.build() {
        Ok(k) => Some(k),
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn req(state: &str, questions: &Value, cache: Option<&str>) -> Request {
    let mut body = json!({"state": state, "questions": questions});
    if let Some(c) = cache {
        body["kime"] = json!({"cache": c});
    }
    parse(&body, &Limits::JEV).unwrap()
}

fn questions() -> Value {
    json!({
        "topic": {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"cancel": "", "refund": "", "upgrade": ""}},
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
            "criteria": ["low", "medium", "high"]}})
}

fn counts(k: &Kime) -> (u64, u64) {
    let CacheStats { hits, misses, .. } = k.cache_stats();
    (hits, misses)
}

#[test]
fn cached_answers_are_the_computed_ones() {
    let (Some(k), Some(plain)) = (open(1000), open(0)) else { return };
    let q = questions();
    let state = "I was charged twice for my subscription, please refund me";
    let want = plain.decide(&req(state, &q, None)).unwrap().to_json();
    assert_eq!(plain.cache_stats(), CacheStats::default());
    assert!(plain.cached(&req(state, &q, None)).is_none());

    assert!(k.cached(&req(state, &q, None)).is_none());
    let (first, t) = k.decide_batch_timed(&[req(state, &q, None)]).unwrap();
    assert_eq!((first[0].to_json(), t.cached, counts(&k)), (want.clone(), 0, (0, 3)));
    let (again, t) = k.decide_batch_timed(&[req(state, &q, None)]).unwrap();
    assert_eq!((again[0].to_json(), t.cached, t.batches), (want.clone(), 3, 0));
    assert_eq!(k.cached(&req(state, &q, None)).unwrap().to_json(), want);
    assert_eq!((counts(&k), k.cache_stats().entries), ((6, 3), 3));

    // One question of the three, alone, is found; a new one next to it is computed.
    let mut two = json!({"angry": q["angry"].clone(),
        "new": {"type": "noul", "instructions": "Does the customer mention a price?"}});
    let (r, t) = k.decide_batch_timed(&[req(state, &two, None)]).unwrap();
    assert_eq!((t.cached, t.batches), (1, 1));
    assert_eq!(r[0].to_json()["answers"]["angry"], want["answers"]["angry"]);
    assert_eq!(r[0].input_tokens, plain.decide(&req(state, &two, None)).unwrap().input_tokens);

    // Other instructions make another row, which is not in the cache.
    two["angry"]["instructions"] = json!("Is the customer upset?");
    assert!(k.cached(&req(state, &two, None)).is_none());
}

#[test]
fn bypass_and_refresh() {
    let Some(k) = open(1000) else { return };
    let q = questions();
    let state = "Where is my parcel? It is three weeks late.";
    k.decide(&req(state, &q, Some("bypass"))).unwrap();
    assert_eq!((counts(&k), k.cache_stats().entries), ((0, 0), 0));
    assert!(k.cached(&req(state, &q, Some("bypass"))).is_none());
    k.decide(&req(state, &q, Some("refresh"))).unwrap();
    assert_eq!((counts(&k), k.cache_stats().entries), ((0, 0), 3));
    let (_, t) = k.decide_batch_timed(&[req(state, &q, Some("refresh"))]).unwrap();
    assert_eq!((t.cached, counts(&k)), (0, (0, 0)));
    let (_, t) = k.decide_batch_timed(&[req(state, &q, Some("use"))]).unwrap();
    assert_eq!((t.cached, counts(&k)), (3, (3, 0)));
}
