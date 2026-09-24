//! OpenTelemetry spans end to end: the server posts one span per request to a collector, here a
//! socket in the test that keeps what it is sent. A request with a `traceparent` joins that trace.
//!
//! It needs the Laya checkpoint, as `tests/http.rs` does, and passes with a note when it is missing
//! unless `KIME_REQUIRE_WEIGHTS` is set.

use std::time::Duration;

use kime_engine::{Device, Kime};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

/// A collector that answers 200 to every post and hands each body over.
async fn collector() -> (String, mpsc::UnboundedReceiver<Value>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Ok((mut s, _)) = listener.accept().await {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let body = loop {
                let n = s.read(&mut chunk).await.unwrap();
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let len: usize = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length: ")
                                .map(str::to_string)
                        })
                        .unwrap()
                        .parse()
                        .unwrap();
                    assert!(head.starts_with("POST /v1/traces HTTP/1.1"), "{head}");
                    if body.len() >= len {
                        break body.to_string();
                    }
                }
                assert!(n > 0, "the connection closed before the body was sent");
            };
            s.write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}").await.unwrap();
            tx.send(serde_json::from_str(&body).unwrap()).unwrap();
        }
    });
    (format!("http://{addr}"), rx)
}

async fn call(addr: std::net::SocketAddr, req: &str) -> String {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out).await;
    out
}

fn attr<'a>(span: &'a Value, key: &str) -> &'a Value {
    let a = span["attributes"].as_array().unwrap();
    &a.iter().find(|a| a["key"] == key).unwrap_or_else(|| panic!("no {key} in {span}"))["value"]
}

/// A local copy in `$KIME_MODELS/laya`, or the Hugging Face cache.
fn model() -> String {
    std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |d| format!("{d}/laya"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spans() {
    let kime = match Kime::builder().model(model()).device(Device::Cpu { threads: 0 }).build() {
        Ok(k) => k,
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            return;
        }
    };
    let (endpoint, mut got) = collector().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut cfg = kime_serve::Config::new(addr, vec![kime]);
    let mut otlp = kime_serve::Otlp::new(&endpoint).unwrap();
    otlp.service = "kime-test".into();
    cfg.otlp = Some(otlp);
    tokio::spawn(kime_serve::serve(listener, cfg, std::future::pending()));

    let parent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let body = json!({"state": "I was charged twice, please refund me",
        "questions": {"q": {"type": "choice", "instructions": "What does the customer want?",
            "criteria": {"refund": "", "cancel": ""}}}})
    .to_string();
    let out = call(
        addr,
        &format!(
            "POST /v1/systemone HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ntraceparent: {parent}\r\nx-request-id: traced-1\r\ncontent-length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
    .await;
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    let head = out.split_once("\r\n\r\n").unwrap().0.to_ascii_lowercase();
    let theirs = head.lines().find_map(|l| l.strip_prefix("traceparent: ")).unwrap().to_string();
    assert!(theirs.starts_with("00-4bf92f3577b34da6a3ce929d0e0e4736-"), "{theirs}");
    assert_ne!(theirs, parent);
    let out = call(addr, "GET /nope HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n").await;
    assert!(out.starts_with("HTTP/1.1 404"), "{out}");

    // Both spans, in one post or two.
    let mut spans = Vec::new();
    while spans.len() < 2 {
        let v = tokio::time::timeout(Duration::from_secs(10), got.recv()).await.unwrap().unwrap();
        let rs = &v["resourceSpans"][0];
        assert_eq!(rs["resource"]["attributes"][0]["value"]["stringValue"], "kime-test");
        spans.extend(rs["scopeSpans"][0]["spans"].as_array().unwrap().iter().cloned());
    }
    let s = spans.iter().find(|s| s["name"] == "POST /v1/systemone").unwrap();
    assert_eq!(s["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(s["parentSpanId"], "00f067aa0ba902b7");
    assert_eq!(
        theirs,
        format!("00-{}-{}-01", s["traceId"].as_str().unwrap(), s["spanId"].as_str().unwrap())
    );
    assert_eq!(s["kind"], 2);
    assert_eq!(attr(s, "http.response.status_code"), &json!({"intValue": "200"}));
    assert_eq!(attr(s, "kime.request_id"), &json!({"stringValue": "traced-1"}));
    assert_eq!(attr(s, "kime.model"), &json!({"stringValue": "laya"}));
    assert_eq!(attr(s, "kime.questions"), &json!({"intValue": "1"}));
    let start: u128 = s["startTimeUnixNano"].as_str().unwrap().parse().unwrap();
    let end: u128 = s["endTimeUnixNano"].as_str().unwrap().parse().unwrap();
    assert!(end > start, "{s}");
    let s = spans.iter().find(|s| s["name"] == "GET other").unwrap();
    assert!(s.get("parentSpanId").is_none());
    assert_ne!(s["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(attr(s, "http.response.status_code"), &json!({"intValue": "404"}));

    // The counts, once the post that carried the spans has been answered.
    let mut text = String::new();
    for _ in 0..50 {
        text = call(addr, "GET /metrics HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n").await;
        if text.contains("kime_otel_spans_total{outcome=\"exported\"} 2") {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(text.contains("kime_otel_spans_total{outcome=\"exported\"} 2"), "{text}");
    assert!(text.contains("kime_otel_spans_total{outcome=\"dropped\"} 0"), "{text}");
}
