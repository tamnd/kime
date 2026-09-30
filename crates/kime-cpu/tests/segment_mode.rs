//! Segment mode on an agent loop: the largest state of the jev-ultrafast fixture, then 17 steps
//! that each change one element and add one recent action, the W8 shape of spec/13-benchmarks.md.
//! A warm step has to run the low state layers on the two changed segments and nothing else.
//!
//! The state is tokenized with Laya's English tokenizer, the one kime-v1-s-en uses, read from
//! `$KIME_MODELS/laya`. Without it the test passes with a note unless `KIME_REQUIRE_MODELS` is set.
//! The weights are random, in a model small enough to run unoptimized, with one local layer on top:
//! segment mode here is about which work is done, not about answers.
//!
//! `KIME_SEGMENT_BENCH=1 cargo test --release -p kime-cpu --test segment_mode -- --nocapture` runs
//! the same steps on kime-v1-s-en's shapes and prints the time of a cold and a warm step.

use std::path::PathBuf;
use std::time::Instant;

use kime_core::render::{native_segments, native_state};
use kime_cpu::split::{SegmentCache, Split};
use kime_model::kime_v1::{V1Graph, V1Spec, random_weights, state_segments};
use kime_model::{Tensors, safetensors};
use kime_tensor::Blob;
use kime_tok::Tokenizer;
use serde_json::{Map, Value, json};

const STEPS: usize = 17;

fn tokenizer() -> Option<Tokenizer> {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya/tokenizer"));
    match dir.filter(|d| d.is_dir()) {
        Some(d) => Some(Tokenizer::from_dir(d).unwrap()),
        None => {
            assert!(
                std::env::var_os("KIME_REQUIRE_MODELS").is_none(),
                "KIME_REQUIRE_MODELS is set and $KIME_MODELS/laya/tokenizer is missing"
            );
            eprintln!("skipping: set KIME_MODELS to a folder holding laya/tokenizer");
            None
        }
    }
}

/// The biggest state in the fixture and the states of the steps after it.
fn trace() -> Vec<Value> {
    let path =
        format!("{}/../kime-core/tests/agent/jev-ultrafast.json", env!("CARGO_MANIFEST_DIR"));
    let all: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut state =
        all.iter().map(|c| c["body"]["state"].clone()).max_by_key(|s| s.to_string().len()).unwrap();
    let mut out = vec![state.clone()];
    for step in 0..STEPS {
        let elements = state["elements"].as_array_mut().unwrap();
        let n = elements.len();
        let e = &mut elements[(step * 7 + 3) % n];
        e["value"] = json!(format!("typed {step}"));
        state["recent_actions"].as_array_mut().unwrap().push(json!({
            "action": "TYPE_TEXT", "kind": "fill", "text": format!("typed {step}"), "page_changed": "False"
        }));
        out.push(state.clone());
    }
    out
}

fn config(
    d: usize,
    heads: usize,
    inter: usize,
    layers: usize,
    top: usize,
    question: usize,
) -> Value {
    json!({
        "format": "kime/1", "family": "kime-v1", "id": "segment-test", "version": "0.0.0",
        "tokenizer": {"kind": "bpe-bytelevel", "cls": 50281, "sep": 50282, "pad": 50283,
            "specials": {"opt": 50368, "no": 50369, "yes": 50370, "cls_s": 50371, "cls_q": 50372}},
        "dims": {"d": d, "heads": heads, "head_dim": 64, "inter": inter, "vocab": 50373},
        "state_tower": {"layers": layers, "global_every": 3, "window": 128, "rope_theta_global": 160000.0,
            "rope_theta_local": 10000.0, "segment_top_layers": top, "max_tokens": 32768, "max_segments": 512},
        "question_tower": {"layers": question, "kv_heads": 2, "max_options_per_chunk": 32, "max_tokens": 4096},
        "gelu": "erf", "norm_eps": 1e-5
    })
}

fn load(spec: &V1Spec) -> Tensors {
    let (mut header, mut data) = (Map::new(), Vec::new());
    for (name, shape, v) in random_weights(spec, 11) {
        let start = data.len();
        v.iter().for_each(|x| data.extend_from_slice(&x.to_le_bytes()));
        header.insert(
            name,
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let head = Value::Object(header).to_string().into_bytes();
    let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&head);
    bytes.extend_from_slice(&data);
    safetensors::load(Blob::owned(bytes)).unwrap().0
}

fn segments(tok: &Tokenizer, spec: &V1Spec, state: &Value) -> Vec<Vec<u32>> {
    let texts = native_segments(state);
    assert_eq!(texts.concat(), native_state(state));
    let ids: Vec<Vec<u32>> = texts.iter().map(|t| tok.encode(t)).collect();
    state_segments(spec, &ids)
}

struct Step {
    tokens: usize,
    segments: usize,
    misses: usize,
    computed: usize,
    ms: f64,
}

fn run(spec: &V1Spec, split: &Split, tok: &Tokenizer) -> Vec<Step> {
    let mut cache = SegmentCache::default();
    let mut out = Vec::new();
    for state in trace() {
        let segs = segments(tok, spec, &state);
        let segs: Vec<&[u32]> = segs.iter().map(Vec::as_slice).collect();
        cache.reset_counts();
        let t0 = Instant::now();
        let mem = split.state_segments(&segs, &mut cache);
        out.push(Step {
            tokens: mem.tokens,
            segments: segs.len(),
            misses: cache.misses,
            computed: cache.tokens_computed,
            ms: t0.elapsed().as_secs_f64() * 1e3,
        });
    }
    out
}

#[test]
fn warm_steps_recompute_only_changed_segments() {
    let Some(tok) = tokenizer() else { return };
    let spec = V1Spec::from_json(&config(128, 2, 64, 3, 1, 1)).unwrap();
    let t = load(&spec);
    let split = Split::new(&spec, &V1Graph::bind(&spec, &t).unwrap(), &t, 4);
    let steps = run(&spec, &split, &tok);
    let first = &steps[0];
    assert_eq!((first.misses, first.computed), (first.segments, first.tokens), "cold");
    for (i, s) in steps.iter().enumerate().skip(1) {
        // The changed element and the new recent action. The changed element can be one seen
        // in an earlier step only if its value came back, which this trace never does.
        assert_eq!(s.misses, 2, "step {i}");
        assert!(s.computed * 10 < s.tokens, "step {i}: {} of {} tokens", s.computed, s.tokens);
    }
    let warm: usize = steps[1..].iter().map(|s| s.computed).sum();
    let all: usize = steps[1..].iter().map(|s| s.tokens).sum();
    eprintln!(
        "cold: {} tokens in {} segments; warm steps ran the low layers on {warm} of {all} tokens ({:.1}%)",
        first.tokens,
        first.segments,
        100.0 * warm as f64 / all as f64
    );
}

#[test]
fn a_small_budget_changes_only_the_work() {
    let Some(tok) = tokenizer() else { return };
    let spec = V1Spec::from_json(&config(128, 2, 64, 3, 1, 1)).unwrap();
    let t = load(&spec);
    let split = Split::new(&spec, &V1Graph::bind(&spec, &t).unwrap(), &t, 4);
    // About a fifth of what the first state takes, so most segments are evicted between steps.
    let budget = 128 * 4 * 400;
    let (mut big, mut small) = (SegmentCache::default(), SegmentCache::with_budget(budget));
    for state in trace().iter().take(6) {
        let segs = segments(&tok, &spec, state);
        let segs: Vec<&[u32]> = segs.iter().map(Vec::as_slice).collect();
        let a = split.state_segments(&segs, &mut big);
        let b = split.state_segments(&segs, &mut small);
        assert_eq!(a.states, b.states);
        assert_eq!(a.kv, b.kv);
        assert!(small.bytes() <= budget, "{} over {budget}", small.bytes());
    }
    assert!(small.misses > big.misses);
}

#[test]
fn scopes_do_not_share_segments() {
    let Some(tok) = tokenizer() else { return };
    let spec = V1Spec::from_json(&config(128, 2, 64, 3, 1, 1)).unwrap();
    let t = load(&spec);
    let split = Split::new(&spec, &V1Graph::bind(&spec, &t).unwrap(), &t, 4);
    let segs = segments(&tok, &spec, &trace()[0]);
    let segs: Vec<&[u32]> = segs.iter().map(Vec::as_slice).collect();
    let mut cache = SegmentCache::default();
    let a = split.state_segments_in(Some(&[1; 32]), &segs, &mut cache);
    let computed = cache.tokens_computed;
    // Another scope finds none of them and computes every token again, to the same bits.
    let b = split.state_segments_in(Some(&[2; 32]), &segs, &mut cache);
    assert_eq!(cache.tokens_computed, 2 * computed);
    assert_eq!((a.states, a.kv), (b.states, b.kv));
    // The first scope finds all of its own.
    let _ = split.state_segments_in(Some(&[1; 32]), &segs, &mut cache);
    assert_eq!(cache.tokens_computed, 2 * computed);
}

#[test]
fn segment_bench() {
    if std::env::var_os("KIME_SEGMENT_BENCH").is_none() {
        return;
    }
    let Some(tok) = tokenizer() else { return };
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let spec = V1Spec::from_json(&config(512, 8, 1344, 12, 3, 4)).unwrap();
    let t = load(&spec);
    let split = Split::new(&spec, &V1Graph::bind(&spec, &t).unwrap(), &t, threads);
    let whole = trace()
        .iter()
        .map(|s| {
            let mut ids = vec![spec.specials.cls_s];
            ids.extend(tok.encode(&native_state(s)));
            ids.push(spec.specials.sep);
            ids
        })
        .collect::<Vec<_>>();
    let t0 = Instant::now();
    let _ = split.state(&whole[0]);
    let plain = t0.elapsed().as_secs_f64() * 1e3;
    let steps = run(&spec, &split, &tok);
    let warm: Vec<f64> = steps[1..].iter().map(|s| s.ms).collect();
    let mean = warm.iter().sum::<f64>() / warm.len() as f64;
    eprintln!(
        "s-en shapes, {threads} threads, {} tokens: whole state {plain:.0} ms, cold segment mode {:.0} ms, warm step mean {mean:.0} ms (min {:.0}, max {:.0})",
        steps[0].tokens,
        steps[0].ms,
        warm.iter().copied().fold(f64::INFINITY, f64::min),
        warm.iter().copied().fold(0.0, f64::max)
    );
}
