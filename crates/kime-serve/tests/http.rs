//! The server over real sockets with the Laya checkpoint on the CPU. Every parity case is sent at
//! once, so the worker batches them, and each Laya shaped answer must be the same bits the engine
//! gives for that request alone. The Jev shape, the error table and the batch endpoint are checked
//! on the same server.
//!
//! It needs the Laya checkpoint in `$KIME_MODELS/laya` or the Hugging Face cache (`kime pull laya`),
//! and passes with a note when it is missing unless `KIME_REQUIRE_WEIGHTS` is set.

use std::net::SocketAddr;

use kime_core::request::{Limits, parse};
use kime_engine::{Device, Kime};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn cases() -> Vec<Value> {
    let path = format!("{}/../kime-eval/fixtures/parity/cases.jsonl", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// One HTTP/1.1 exchange on a fresh connection: the status, the headers in lower case and the body.
async fn call(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: &str,
    extra: &str,
) -> (u16, String, Value) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\n{extra}content-length: {}\r\n\r\n{body}",
        body.len()
    );
    // The server answers an oversized body before reading all of it and closes, so a failed write
    // still leaves a response to read.
    let _ = s.write_all(req.as_bytes()).await;
    let mut out = String::new();
    let _ = s.read_to_string(&mut out).await;
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let status = head[9..12].parse().unwrap();
    (status, head.to_ascii_lowercase(), serde_json::from_str(body).unwrap_or(Value::Null))
}

async fn post(addr: SocketAddr, path: &str, body: &Value) -> (u16, Value) {
    let (s, _, v) = call(addr, "POST", path, &body.to_string(), "").await;
    (s, v)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_laya() {
    let threads = std::env::var("KIME_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(0);
    // A local copy in $KIME_MODELS/laya, or the Hugging Face cache.
    let model =
        std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |d| format!("{d}/laya"));
    let kime = match Kime::builder().model(model).device(Device::Cpu { threads }).build() {
        Ok(k) => k,
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            return;
        }
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let mut cfg = kime_serve::Config::new(addr, vec![kime.clone()]);
    // Small enough that the server has read the whole oversized body below when it answers, so
    // the connection closes cleanly instead of with a reset that can eat the response.
    cfg.max_body = 64 << 10;
    // Every parity case at once has to be answered, not turned away.
    cfg.max_queue = std::time::Duration::ZERO;
    let server = tokio::spawn(kime_serve::serve(listener, cfg, async {
        let _ = stopped.await;
    }));

    // Every case at once, Laya shape.
    let cases = cases();
    let t = std::time::Instant::now();
    let sent: Vec<_> = cases
        .iter()
        .map(|c| {
            let mut body = c.clone();
            body.as_object_mut().unwrap().remove("id");
            tokio::spawn(async move { post(addr, "/v1/systemone", &body).await })
        })
        .collect();
    let mut got = Vec::new();
    for h in sent {
        got.push(h.await.unwrap());
    }
    eprintln!("{} concurrent requests in {:?}", cases.len(), t.elapsed());
    for (c, (status, mut v)) in cases.iter().zip(got) {
        assert_eq!(status, 200, "{}: {v}", c["id"]);
        let routing = v.as_object_mut().unwrap().shift_remove("routing").unwrap();
        assert_eq!(routing["model"], "english");
        let want = kime.decide(&parse(c, &Limits::LAYA).unwrap()).unwrap().to_json();
        assert_eq!(v, want, "{}", c["id"]);
    }

    // Jev shape: rounded to 2 places summing to exactly 1, choice is the argmax, no action.
    for c in cases.iter().take(20) {
        let mut body = c.clone();
        body["model"] = json!("jev-latest");
        let (status, v) = post(addr, "/v1/systemone", &body).await;
        assert_eq!(status, 200, "{}: {v}", c["id"]);
        assert_eq!(v["model"], "laya");
        assert!(v.get("routing").is_none());
        for (id, a) in v["answers"].as_object().unwrap() {
            assert!(a.get("action").is_none(), "{id}");
            match a["type"].as_str().unwrap() {
                "noul" => assert!(a.get("confidence").is_none()),
                t => {
                    let p = a["probabilities"].as_object().unwrap();
                    let cents: i64 =
                        p.values().map(|x| (x.as_f64().unwrap() * 100.0).round() as i64).sum();
                    assert_eq!(cents, 100, "{id}: {a}");
                    if t == "choice" {
                        let best = p.iter().fold(("", -1.0), |b, (k, x)| {
                            let x = x.as_f64().unwrap();
                            if x > b.1 { (k.as_str(), x) } else { b }
                        });
                        assert_eq!(a["choice"], best.0, "{id}: {a}");
                    }
                }
            }
        }
    }

    // The error table.
    let err = |s: u16, v: &Value| (s, v["detail"].clone());
    let (s, v) =
        post(addr, "/v1/systemone", &json!({"model": "gpt-4", "state": "x", "questions": {}}))
            .await;
    assert_eq!(err(s, &v), (404, json!("model 'gpt-4' not found")));
    let (s, v) = post(addr, "/v1/systemone", &json!([1])).await;
    assert_eq!(s, 400, "{v}");
    let (s, v) = post(addr, "/v1/systemone", &json!({"state": "x"})).await;
    assert_eq!(s, 400, "{v}");
    let (s, v) = post(
        addr,
        "/v1/systemone",
        &json!({"model": "kime-latest", "state": "x", "questions": {}}),
    )
    .await;
    assert_eq!(s, 422, "{v}");
    assert_eq!(v["detail"][0]["loc"], json!(["body", "questions"]));
    let (s, v) = post(addr, "/v1/systemone", &json!({"state": "x", "questions": {}})).await;
    assert_eq!((s, &v["answers"]), (200, &json!({})));
    let (s, _, v) = call(addr, "GET", "/v1/systemone", "", "").await;
    assert_eq!((s, v), (405, json!({"detail": "Method Not Allowed"})));
    let (s, _, v) = call(addr, "GET", "/nope", "", "").await;
    assert_eq!((s, v), (404, json!({"detail": "Not Found"})));
    let big = format!("{{\"state\":\"{}\"}}", "x".repeat(65 << 10));
    let (s, _, _) = call(addr, "POST", "/v1/systemone", &big, "").await;
    assert_eq!(s, 400);

    // Request ids.
    let (_, head, _) = call(addr, "GET", "/health", "", "x-request-id: my-id_1\r\n").await;
    assert!(
        head.contains("x-request-id: my-id_1") && head.contains("x-typesafe-request-id: my-id_1")
    );
    let (_, head, v) = call(addr, "GET", "/health", "", "x-request-id: bad id!\r\n").await;
    assert!(head.contains("x-request-id: req_"));
    assert_eq!(v["loaded"], json!(["english"]));

    // Metrics count what was just answered.
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(b"GET /metrics HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n").await.unwrap();
    let mut text = String::new();
    let _ = s.read_to_string(&mut text).await;
    assert!(text.contains("kime_requests_total{route=\"/v1/systemone\",status=\"200\"}"), "{text}");
    assert!(text.contains("kime_requests_total{route=\"other\",status=\"404\"} 1"), "{text}");
    assert!(text.contains("kime_forward_passes_total{model=\"laya\"}"), "{text}");
    assert!(text.contains("kime_truncations_total{model=\"laya\"}"), "{text}");
    assert!(text.contains("kime_truncated_tokens_total{model=\"laya\"}"), "{text}");
    let bytes = |kind: &str| -> u64 {
        let key = format!("kime_device_memory_bytes{{model=\"laya\",kind=\"{kind}\"}} ");
        let line = text.lines().find(|l| l.starts_with(&key)).unwrap_or_else(|| panic!("{text}"));
        line[key.len()..].parse().unwrap()
    };
    // Laya's 149M parameters as f32, and at least the one bucket the requests above used.
    assert!(bytes("weights") > 500 << 20, "{text}");
    assert!(bytes("plans") > 0, "{text}");
    assert!(text.contains("kime_device_seconds_bucket{model=\"laya\",le=\"+Inf\"}"), "{text}");

    // The batch endpoint: items fail alone and the good ones match the single endpoint.
    let item = |c: &Value, id: &str| {
        let mut c = c.clone();
        c["id"] = json!(id);
        c
    };
    let body = json!({"model": "kime-latest", "items": [
        item(&cases[0], "a"),
        {"id": "b", "state": "x", "questions": {}},
        item(&cases[1], "c"),
    ]});
    let (s, v) = post(addr, "/v1/systemone/batch", &body).await;
    assert_eq!(s, 200, "{v}");
    let r = v["results"].as_array().unwrap();
    assert_eq!(r[1]["error"]["status"], 422);
    for (k, c) in [(0, &cases[0]), (2, &cases[1])] {
        let mut one = c.clone();
        one["model"] = json!("kime-latest");
        let (_, single) = post(addr, "/v1/systemone", &one).await;
        assert_eq!(r[k]["answers"], single["answers"]);
    }

    let _ = stop.send(());
    server.await.unwrap().unwrap();

    overload(kime, &cases).await;
}

/// With a queue budget of a microsecond, a burst gets some 529s with Jev's body and a
/// `retry-after-ms`, every other request its answer, and the server is healthy afterwards.
async fn overload(kime: Kime, cases: &[Value]) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let mut cfg = kime_serve::Config::new(addr, vec![kime]);
    cfg.max_queue = std::time::Duration::from_micros(1);
    let server = tokio::spawn(kime_serve::serve(listener, cfg, async {
        let _ = stopped.await;
    }));
    let body = |c: &Value| {
        let mut b = c.clone();
        b.as_object_mut().unwrap().remove("id");
        b.to_string()
    };
    // One request first, so the server knows how long a request takes.
    let (s, _, _) = call(addr, "POST", "/v1/systemone", &body(&cases[0]), "").await;
    assert_eq!(s, 200);
    let sent: Vec<_> = cases
        .iter()
        .take(40)
        .map(|c| {
            let b = body(c);
            tokio::spawn(async move { call(addr, "POST", "/v1/systemone", &b, "").await })
        })
        .collect();
    let (mut ok, mut busy) = (0, 0);
    for h in sent {
        let (s, head, v) = h.await.unwrap();
        match s {
            200 => ok += 1,
            529 => {
                busy += 1;
                assert_eq!(v["detail"]["error_type"], "overloaded_error", "{v}");
                let ms = head
                    .lines()
                    .find_map(|l| l.strip_prefix("retry-after-ms:"))
                    .and_then(|v| v.trim().parse::<u64>().ok());
                assert!(ms.is_some_and(|ms| ms >= 1), "{head}");
            }
            s => panic!("status {s}: {v}"),
        }
    }
    assert!(ok >= 1 && busy >= 1, "{ok} answered and {busy} turned away");
    let (s, _, _) = call(addr, "POST", "/v1/systemone", &body(&cases[1]), "").await;
    assert_eq!(s, 200, "healthy again once the queue has drained");
    let _ = stop.send(());
    server.await.unwrap().unwrap();
}

/// With `laya` and `laya-multilingual` both loaded, a request that leaves the choice to the
/// server goes where kime-route sends it, a batch is routed item by item with each item's answer
/// the same as the single endpoint's, and `/metrics` counts the decisions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn serve_routed() {
    let threads = std::env::var("KIME_THREADS").ok().and_then(|t| t.parse().ok()).unwrap_or(0);
    let load = |name: &str| {
        let model =
            std::env::var("KIME_MODELS").map_or_else(|_| name.into(), |d| format!("{d}/{name}"));
        Kime::builder().model(model).device(Device::Cpu { threads }).build()
    };
    let (en, ml) = match (load("laya"), load("laya-multilingual")) {
        (Ok(en), Ok(ml)) => (en, ml),
        // CI keeps only laya, so the second checkpoint is never required.
        (Ok(_), Err(e)) => {
            eprintln!("skipping: {e}");
            return;
        }
        (Err(e), _) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            return;
        }
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let mut cfg = kime_serve::Config::new(addr, vec![en, ml]);
    cfg.max_queue = std::time::Duration::ZERO;
    let server = tokio::spawn(kime_serve::serve(listener, cfg, async {
        let _ = stopped.await;
    }));

    let q = json!({"q": {"type": "choice", "instructions": "What does the customer want?",
        "criteria": {"cancel": "", "refund": "", "book": ""}}});
    let states = [
        ("I was charged twice for my subscription, please refund me", "laya"),
        ("Мне нужно отменить подписку", "laya-multilingual"),
        ("saya mau pesan tiket ke jakarta besok pagi", "laya-multilingual"),
        ("The café on the corner charged me twice for a crème brûlée", "laya"),
    ];
    let mut single = Vec::new();
    for (state, want) in states {
        let (s, v) = post(addr, "/v1/systemone", &json!({"state": state, "questions": q})).await;
        assert_eq!(s, 200, "{v}");
        let key = if want == "laya" { "english" } else { "multilingual" };
        assert_eq!(v["routing"]["model"], key, "{state}: {}", v["routing"]);
        let (s, v) = post(
            addr,
            "/v1/systemone",
            &json!({"model": "kime-latest", "state": state, "questions": q}),
        )
        .await;
        assert_eq!(s, 200, "{v}");
        assert_eq!(v["model"], want, "{state}");
        single.push(v);
    }
    // A caller's lang wins over detection.
    let body = json!({"state": states[1].0, "questions": q, "kime": {"route": {"lang": "en"}}});
    let (_, v) = post(addr, "/v1/systemone", &body).await;
    assert_eq!(v["model"], "laya");

    let mut items: Vec<Value> = states
        .iter()
        .enumerate()
        .map(|(i, (state, _))| json!({"id": i.to_string(), "state": state, "questions": q}))
        .collect();
    items.push(json!({"id": "bad", "state": "x", "questions": {}}));
    let body = json!({"model": "kime-latest", "items": items});
    let (s, v) = post(addr, "/v1/systemone/batch", &body).await;
    assert_eq!(s, 200, "{v}");
    let r = v["results"].as_array().unwrap();
    for (k, (_, want)) in states.iter().enumerate() {
        assert_eq!(r[k]["id"], k.to_string());
        assert_eq!(r[k]["model"], *want, "{}", r[k]);
        assert_eq!(r[k]["answers"], single[k]["answers"]);
    }
    assert_eq!(r[4]["error"]["status"], 422);

    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(b"GET /metrics HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n").await.unwrap();
    let mut text = String::new();
    s.read_to_string(&mut text).await.unwrap();
    for line in [
        "kime_route_decisions_total{model=\"laya-multilingual\",reason=\"script\"} 3",
        "kime_route_decisions_total{model=\"laya\",reason=\"lang\"} 1",
        "kime_route_decisions_total{model=\"laya\",reason=\"identifier\"} 3",
        "kime_route_decisions_total{model=\"laya-multilingual\",reason=\"identifier\"} 3",
        "kime_route_decisions_total{model=\"laya\",reason=\"word_lists\"} 3",
    ] {
        assert!(text.contains(line), "{line} in {text}");
    }

    let _ = stop.send(());
    server.await.unwrap().unwrap();
}
