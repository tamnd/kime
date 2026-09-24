//! The chaos tests of spec/15-testing.md on a real server with the Laya checkpoint on the CPU: a
//! worker thread killed in the middle of a pass, a panic in the engine, a device error, a request
//! over the token limit among good ones, and a queue filled past `max_pending`. Every request must
//! get the right answer or a documented error, and the server must be healthy afterwards, which
//! here means `/health` answers, the queue is empty and the same request gets the same answer as
//! before. Filling the state cache waits for the cache.
//!
//! Like `tests/http.rs` it needs the checkpoint and passes with a note when it is missing unless
//! `KIME_REQUIRE_WEIGHTS` is set.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use kime_engine::{Device, Kime};
use kime_serve::Faults;
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// One exchange on a fresh connection: the status, the headers in lower case and the body.
async fn call(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String, Value) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    let _ = s.write_all(req.as_bytes()).await;
    let mut out = String::new();
    let _ = s.read_to_string(&mut out).await;
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let status = head[9..12].parse().unwrap();
    (status, head.to_ascii_lowercase(), serde_json::from_str(body).unwrap_or(Value::Null))
}

fn request(state: &str) -> String {
    json!({"state": state, "questions": {"q": {"type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {"refund": "", "cancel": "", "upgrade": ""}}}})
    .to_string()
}

const STATES: [&str; 4] = [
    "I was charged twice this month, please refund the second payment",
    "Please cancel my subscription at the end of the billing period",
    "How do I move my team to the business plan?",
    "The app keeps logging me out and I want my money back",
];

async fn start(kime: &Kime, max_pending: usize, faults: &Arc<Faults>) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut cfg = kime_serve::Config::new(addr, vec![kime.clone()]);
    cfg.max_queue = Duration::ZERO;
    cfg.max_pending = max_pending;
    cfg.max_request_tokens = 4096;
    cfg.faults = faults.clone();
    tokio::spawn(kime_serve::serve(listener, cfg, std::future::pending()));
    addr
}

/// Sends every state `times` times at once and gives back the answers by state.
async fn burst(addr: SocketAddr, times: usize) -> Vec<(usize, u16, String, Value)> {
    let mut tasks = Vec::new();
    for k in 0..times * STATES.len() {
        let i = k % STATES.len();
        tasks.push(tokio::spawn(async move {
            let (s, h, v) = call(addr, "POST", "/v1/systemone", &request(STATES[i])).await;
            (i, s, h, v)
        }));
    }
    let mut out = Vec::new();
    for t in tasks {
        out.push(t.await.unwrap());
    }
    out
}

fn metric(text: &str, name: &str) -> u64 {
    text.lines()
        .find_map(|l| l.strip_prefix(name).and_then(|v| v.trim().parse().ok()))
        .unwrap_or_else(|| panic!("no {name} in /metrics"))
}

/// Healthy: `/health` answers, nothing is left queued, and every state gets its answer again.
async fn healthy(addr: SocketAddr, want: &[Value]) -> String {
    assert_eq!(call(addr, "GET", "/health", "").await.0, 200);
    for (i, s) in STATES.iter().enumerate() {
        let (status, _, v) = call(addr, "POST", "/v1/systemone", &request(s)).await;
        assert_eq!((status, &v["answers"]), (200, &want[i]), "{s}");
    }
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    s.write_all(b"GET /metrics HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n").await.unwrap();
    let mut text = String::new();
    let _ = s.read_to_string(&mut text).await;
    assert_eq!(metric(&text, "kime_queue_depth{model=\"laya\"}"), 0, "{text}");
    text
}

/// Each answer is right or a 500 whose message says the worker failed it, and `failed` of them
/// are 500s.
fn right_or_internal(got: &[(usize, u16, String, Value)], want: &[Value], says: &str) -> usize {
    let mut failed = 0;
    for (i, status, _, v) in got {
        match status {
            200 => assert_eq!(&v["answers"], &want[*i], "{}", STATES[*i]),
            500 => {
                assert_eq!(v["detail"]["error_type"], "internal_error", "{v}");
                let msg = v["detail"]["message"].as_str().unwrap();
                assert!(msg.contains(says), "{msg}");
                failed += 1;
            }
            s => panic!("status {s}: {v}"),
        }
    }
    failed
}

/// A local copy in `$KIME_MODELS/laya`, or the Hugging Face cache.
fn model() -> String {
    std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |d| format!("{d}/laya"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chaos() {
    let kime = match Kime::builder().model(model()).device(Device::Cpu { threads: 0 }).build() {
        Ok(k) => k,
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            return;
        }
    };
    let faults = Arc::new(Faults::default());
    let addr = start(&kime, 0, &faults).await;
    let mut want = Vec::new();
    for s in STATES {
        let (status, _, v) = call(addr, "POST", "/v1/systemone", &request(s)).await;
        assert_eq!(status, 200, "{v}");
        want.push(v["answers"].clone());
    }

    // The worker thread dies with a pass in hand. Its requests get a 500, the rest are answered,
    // and a new worker takes over.
    faults.kill.store(1, Ordering::Relaxed);
    let got = burst(addr, 4).await;
    assert!(right_or_internal(&got, &want, "worker stopped") >= 1);
    let text = healthy(addr, &want).await;
    assert_eq!(metric(&text, "kime_worker_restarts_total{model=\"laya\"}"), 1);

    // A panic inside the engine fails its pass and nothing else.
    faults.panic.store(1, Ordering::Relaxed);
    let got = burst(addr, 4).await;
    assert!(right_or_internal(&got, &want, "panicked") >= 1);
    healthy(addr, &want).await;

    // A device error fails the batch, and each request of it runs again alone and is answered.
    faults.error.store(1, Ordering::Relaxed);
    let got = burst(addr, 4).await;
    assert_eq!(right_or_internal(&got, &want, "-"), 0);
    assert_eq!(faults.error.load(Ordering::Relaxed), 0);
    healthy(addr, &want).await;

    // A request over the token limit among good ones gets its 413 and holds nobody up. Laya cuts
    // long states, so only a request in the Jev shape is held to the limit.
    let mut big: Value = serde_json::from_str(&request(&"refund ".repeat(20_000))).unwrap();
    big["model"] = json!("kime-latest");
    let big = big.to_string();
    let (a, b) = tokio::join!(call(addr, "POST", "/v1/systemone", &big), burst(addr, 2));
    assert_eq!(a.0, 413, "{}", a.2);
    assert_eq!(right_or_internal(&b, &want, "-"), 0);
    healthy(addr, &want).await;

    // A queue held to 3 requests turns the rest of a burst away with 529 and comes back.
    let small = start(&kime, 3, &faults).await;
    let got = burst(small, 8).await;
    let mut turned = 0;
    for (i, status, head, v) in &got {
        match status {
            200 => assert_eq!(&v["answers"], &want[*i]),
            529 => {
                assert_eq!(v["detail"]["error_type"], "overloaded_error", "{v}");
                assert!(head.contains("retry-after-ms: "), "{head}");
                turned += 1;
            }
            s => panic!("status {s}: {v}"),
        }
    }
    assert!(turned > 0 && turned < got.len(), "{turned} of {} turned away", got.len());
    let text = healthy(small, &want).await;
    assert_eq!(metric(&text, "kime_rejected_total{reason=\"overloaded\"}"), turned as u64);
}
