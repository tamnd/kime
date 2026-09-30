//! kime-v1 through [`Kime`] on the CPU: a checkpoint of seeded random weights with Laya's
//! tokenizer, written to a temporary directory and opened by path like any other.
//!
//! The weights are random, so the answers mean nothing. What is checked is the path: every
//! question type answers, a state is run once however many questions and calls read it, a batch
//! gives the same bits as its requests alone, and a choice too big for one row goes through
//! chunks and the rerank. It needs Laya's tokenizer, from `KIME_MODELS/laya` or the Hugging Face
//! cache, and passes with a note when it is missing.
//!
//! With `KIME_NATIVE_BENCH` set it also times W3 and W4 of spec/13-benchmarks.md on the shapes of
//! kime-v1-s-en.

use std::path::{Path, PathBuf};
use std::time::Instant;

use kime_core::answer::Answer;
use kime_core::request::{Limits, Request, parse};
use kime_engine::{Device, Error, Kime};
use kime_model::kime_v1::{V1Spec, random_weights};
use serde_json::{Map, Value, json};

fn laya_tokenizer() -> Option<PathBuf> {
    let from = |root: PathBuf| {
        let p = root.join("laya/tokenizer/tokenizer.json");
        p.exists().then_some(p)
    };
    if let Some(p) = std::env::var_os("KIME_MODELS").and_then(|m| from(PathBuf::from(m))) {
        return Some(p);
    }
    let home = std::env::var_os("HOME")?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--convaiinnovations--laya/snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().find_map(|e| {
        let p = e.path().join("tokenizer/tokenizer.json");
        p.exists().then_some(p)
    })
}

fn config(id: &str, d: usize, heads: usize, inter: usize, layers: [usize; 3]) -> Value {
    json!({
        "format": "kime/1", "family": "kime-v1", "id": id, "version": "0.0.0",
        "tokenizer": {"kind": "bpe-bytelevel", "file": "tokenizer.json", "cls": 50281, "sep": 50282,
            "pad": 50283, "specials": {"opt": 50368, "no": 50369, "yes": 50370, "cls_s": 50371, "cls_q": 50372}},
        "dims": {"d": d, "heads": heads, "head_dim": 64, "inter": inter, "vocab": 50373},
        "state_tower": {"layers": layers[0], "global_every": 3, "window": 128, "rope_theta_global": 160000.0,
            "rope_theta_local": 10000.0, "segment_top_layers": layers[1], "max_tokens": 32768, "max_segments": 512},
        "question_tower": {"layers": layers[2], "kv_heads": 2, "max_options_per_chunk": 32, "max_tokens": 4096},
        "gelu": "erf", "norm_eps": 1e-5
    })
}

/// A checkpoint directory for `config` with seeded weights, removed on drop.
struct Checkpoint(PathBuf);

impl Drop for Checkpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn checkpoint(config: &Value, tokenizer: &Path) -> Checkpoint {
    let spec = V1Spec::from_json(config).unwrap();
    let dir = std::env::temp_dir().join(format!("kime-native-{}-{}", spec.id, std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (mut header, mut data) = (Map::new(), Vec::new());
    for (name, shape, v) in random_weights(&spec, 5) {
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
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    std::fs::write(dir.join("kime.json"), config.to_string()).unwrap();
    std::fs::copy(tokenizer, dir.join("tokenizer.json")).unwrap();
    Checkpoint(dir)
}

fn open(c: &Checkpoint, threads: usize) -> Kime {
    Kime::builder().model(c.0.to_str().unwrap()).device(Device::Cpu { threads }).build().unwrap()
}

fn request(state: Value, questions: Value) -> Request {
    parse(&json!({"state": state, "questions": questions}), &Limits::LAYA).unwrap()
}

fn state(n: usize) -> Value {
    json!({
        "user": {"id": "u-1042", "plan": "pro", "country": "VN"},
        "ticket": {"subject": "Refund for a double charge", "messages": (0..n).map(|i| json!({
            "from": if i % 2 == 0 { "customer" } else { "agent" },
            "text": format!("Message {i}: I was charged twice for order {} and want one of the charges back.", 7000 + i)
        })).collect::<Vec<_>>()}
    })
}

fn questions() -> Value {
    json!({
        "route": {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": {"billing": "charges, refunds and invoices", "tech": "bugs and outages",
                "sales": "plans and upgrades", "abuse": null}},
        "urgency": {"type": "score", "criteria": ["can wait", "today", "within the hour"]},
        "refund": {"type": "noul", "instructions": "The customer asks for money back.",
            "criteria": {"true": "a refund is requested", "false": "no refund is requested"}}
    })
}

fn probabilities(a: &Answer) -> Vec<f64> {
    match a {
        Answer::Choice { probabilities, .. } => probabilities.iter().map(|p| p.1).collect(),
        Answer::Score { probabilities, .. } => probabilities.clone(),
        Answer::Noul { noul, .. } => vec![1.0 - noul, *noul],
    }
}

#[test]
fn kime_v1_end_to_end() {
    let Some(tok) = laya_tokenizer() else {
        eprintln!("no Laya tokenizer, skipped");
        return;
    };
    let c = checkpoint(&config("native-test", 128, 2, 64, [3, 1, 1]), &tok);
    let kime = open(&c, 4);
    assert_eq!(kime.model_id(), "native-test");

    let a = request(state(3), questions());
    let (res, t) = kime.decide_batch_timed(std::slice::from_ref(&a)).unwrap();
    assert_eq!((t.states, t.batches), (1, 3));
    let res = &res[0];
    assert_eq!(res.model, "native-test");
    let ids: Vec<&str> = res.answers.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(ids, ["route", "urgency", "refund"]);
    for (id, ans) in &res.answers {
        let p = probabilities(ans);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-5, "{id}: {p:?}");
    }
    assert_eq!(res.input_tokens, kime.count_tokens(&a));

    // The same state again is found in the state cache, and the answers are the same bits.
    let (again, t) = kime.decide_batch_timed(std::slice::from_ref(&a)).unwrap();
    assert_eq!(t.states, 0);
    assert_eq!(again[0].answers, res.answers);

    // In a batch with other states each request gets what it gets alone.
    let b = request(json!("a plain string state about a login bug"), questions());
    let d = request(state(9), json!({"only": {"type": "noul"}}));
    let alone: Vec<_> = [&b, &d].iter().map(|r| kime.decide(r).unwrap().answers).collect();
    let fresh = open(&c, 4);
    let (batch, t) =
        fresh.decide_batch_timed(&[b.clone(), a.clone(), d.clone(), a.clone()]).unwrap();
    assert_eq!(t.states, 3, "the repeated state runs once");
    assert_eq!(batch[0].answers, alone[0]);
    assert_eq!(batch[1].answers, res.answers);
    assert_eq!(batch[2].answers, alone[1]);
    assert_eq!(batch[3].answers, res.answers);

    // 40 options take two chunks of 20 and a rerank of the best 16.
    let many: Map<String, Value> = (0..40)
        .map(|i| (format!("intent_{i}"), json!(format!("the user wants thing {i}"))))
        .collect();
    let big = request(state(2), json!({"intent": {"type": "choice", "criteria": many}}));
    let (res, t) = kime.decide_batch_timed(std::slice::from_ref(&big)).unwrap();
    assert_eq!(t.batches, 3);
    let p = probabilities(&res[0].answers[0].1);
    assert_eq!(p.len(), 40);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-5);

    // Segment mode reads the same state through its segments, and a small change reuses most.
    let seg = |n: usize| {
        let mut r = request(state(n), json!({"refund": questions()["refund"].clone()}));
        r.kime = Some(Map::from_iter([("state_segments".to_string(), json!(true))]));
        r
    };
    let (r1, t1) = kime.decide_batch_timed(&[seg(6)]).unwrap();
    let (r2, t2) = kime.decide_batch_timed(&[seg(7)]).unwrap();
    assert_eq!((t1.states, t2.states), (1, 1));
    assert_eq!((r1[0].answers.len(), r2[0].answers.len()), (1, 1));

    let e = kime.embed(&["one text", "another one"], 64).unwrap();
    assert_eq!((e.len(), e[0].len()), (2, 128));

    let metal = Kime::builder().model(c.0.to_str().unwrap()).device(Device::Metal).build();
    assert!(matches!(metal, Err(Error::Unsupported(_))));
}

#[test]
fn native_bench() {
    if std::env::var_os("KIME_NATIVE_BENCH").is_none() {
        return;
    }
    let Some(tok) = laya_tokenizer() else { return };
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let c = checkpoint(&config("kime-v1-s-en-random", 512, 8, 1344, [12, 3, 4]), &tok);
    let t0 = Instant::now();
    let kime = open(&c, threads);
    eprintln!("load {:.0} ms on {threads} threads", t0.elapsed().as_secs_f64() * 1e3);

    // W3: one state near 512 tokens and 10 questions. W4: the same state and 50 questions.
    let mut n = 1;
    while kime.count_tokens(&request(state(n + 1), json!({}))) <= 512 {
        n += 1;
    }
    let base = questions();
    let qs = |k: usize| -> Value {
        let m: Map<String, Value> = (0..k)
            .map(|i| {
                let (name, q) = base.as_object().unwrap().iter().nth(i % 3).unwrap();
                (format!("{name}_{i}"), q.clone())
            })
            .collect();
        Value::Object(m)
    };
    for (name, k) in [("W3", 10), ("W4", 50)] {
        let req = request(state(n), qs(k));
        let mut cold = Vec::new();
        let mut warm = Vec::new();
        for i in 0..6 {
            // A fresh number in the state each time makes the state cache miss.
            let mut r = req.clone();
            r.state["ticket"]["id"] = json!(format!("{name}-{i}"));
            let t = Instant::now();
            let (_, tm) = kime.decide_batch_timed(std::slice::from_ref(&r)).unwrap();
            cold.push((t.elapsed().as_secs_f64() * 1e3, tm.states));
            let t = Instant::now();
            let (_, tm) = kime.decide_batch_timed(std::slice::from_ref(&r)).unwrap();
            warm.push((t.elapsed().as_secs_f64() * 1e3, tm.states));
        }
        let med = |v: &mut Vec<(f64, usize)>| {
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
            v[v.len() / 2].0
        };
        assert!(cold.iter().all(|c| c.1 == 1) && warm.iter().all(|w| w.1 == 0));
        eprintln!(
            "{name}: {} state tokens, {k} questions, {} input tokens: cold {:.1} ms, state cached {:.1} ms (median of 6)",
            kime.count_tokens(&request(state(n), json!({}))),
            kime.count_tokens(&req),
            med(&mut cold),
            med(&mut warm)
        );
    }
}

#[test]
fn cache_scopes_and_modes() {
    let Some(tok) = laya_tokenizer() else {
        eprintln!("no Laya tokenizer, skipped");
        return;
    };
    let c = checkpoint(&config("native-scopes", 128, 2, 64, [3, 1, 1]), &tok);
    let kime = open(&c, 4);
    let states = |r: &Request| {
        let (res, t) = kime.decide_batch_timed(std::slice::from_ref(r)).unwrap();
        (t.states, res.into_iter().next().unwrap().answers)
    };
    let scoped = |n: usize, scope: Option<u8>, cache: &str| {
        let mut r = request(state(n), questions());
        r.cache_scope = scope.map(|b| [b; 32]);
        r.kime = Some(Map::from_iter([("cache".to_string(), json!(cache))]));
        r
    };

    // Each scope runs the state once, and they all get the same answers.
    let (ran, want) = states(&scoped(4, Some(1), "use"));
    assert_eq!(ran, 1);
    assert_eq!(states(&scoped(4, Some(1), "use")), (0, want.clone()));
    assert_eq!(states(&scoped(4, Some(2), "use")), (1, want.clone()));
    assert_eq!(states(&scoped(4, None, "use")), (1, want.clone()));
    assert_eq!(states(&scoped(4, None, "use")).0, 0);

    // A batch with the same state for two scopes runs it for each.
    let (_, t) =
        kime.decide_batch_timed(&[scoped(5, Some(1), "use"), scoped(5, Some(2), "use")]).unwrap();
    assert_eq!(t.states, 2);

    // bypass neither reads nor keeps, refresh runs and keeps.
    assert_eq!(states(&scoped(4, Some(1), "bypass")).0, 1);
    assert_eq!(states(&scoped(6, Some(1), "bypass")).0, 1);
    assert_eq!(states(&scoped(6, Some(1), "use")).0, 1);
    assert_eq!(states(&scoped(7, Some(1), "refresh")).0, 1);
    assert_eq!(states(&scoped(7, Some(1), "use")).0, 0);
}
