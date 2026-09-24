//! A small W9 from spec/13-benchmarks.md, as spec/15-testing.md runs it on a CPU runner: W2, the
//! triage preset on a support message of about 300 tokens, from clients that each send the next
//! request as soon as the last is answered. It checks that there are no errors, that p99 is under
//! 5x p50 and that memory is bounded. Two rounds first plan the arenas for every batch size, and
//! then the resident size of the process may not grow by more than 5% from the middle of the run
//! to its end.
//!
//! `KIME_W9_CLIENTS` (8) and `KIME_W9_ROUNDS` (4) set the size. Like `tests/http.rs` it needs the
//! checkpoint and passes with a note when it is missing unless `KIME_REQUIRE_WEIGHTS` is set.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use kime_engine::{Device, Kime};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const SENTENCES: [&str; 12] = [
    "Hi, I was charged twice for my subscription this month and I would like the second payment refunded.",
    "I have been a customer for three years and this is the second time this has happened.",
    "The invoice says the pro plan but I moved my team to the basic plan in March.",
    "Your app also keeps logging me out every time I switch between my phone and my laptop.",
    "I tried clearing the cache and reinstalling, and the support article did not help at all.",
    "We have a board meeting on Friday and I need the export feature working before then.",
    "Honestly I am getting tired of chasing this every month.",
    "If this is not sorted out soon we will look at other tools.",
    "Could you also tell me how to add two more seats for the new hires?",
    "My account email is dana@example.com and the last four digits of the card are 4242.",
    "Please let me know what you need from me to sort this out.",
    "Thanks for your help, I appreciate it.",
];

/// A support message of about 300 tokens, different for each `k`.
fn message(k: usize) -> String {
    let mut out = String::new();
    let mut i = k;
    while out.len() < 1300 {
        out.push_str(SENTENCES[i % SENTENCES.len()]);
        out.push(' ');
        i = i * 7 + 3;
    }
    out
}

fn w2(k: usize) -> String {
    json!({"state": {"message": message(k)}, "questions": kime_core::presets::triage()}).to_string()
}

async fn post(addr: SocketAddr, body: &str) -> (u16, Value) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/systemone HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out).await;
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    (head[9..12].parse().unwrap(), serde_json::from_str(body).unwrap_or(Value::Null))
}

/// The resident size of this process in MB, from `ps`, which Linux and macOS both have.
fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().parse::<f64>().unwrap() / 1024.0
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Every client sends one request and waits for it, `rounds` times. The latencies, in seconds.
async fn run(addr: SocketAddr, clients: usize, rounds: usize, from: usize) -> Vec<f64> {
    let tasks: Vec<_> = (0..clients)
        .map(|c| {
            tokio::spawn(async move {
                let mut lat = Vec::new();
                for r in 0..rounds {
                    let body = w2(from + r * clients + c);
                    let t = Instant::now();
                    let (status, v) = post(addr, &body).await;
                    assert_eq!(status, 200, "{v}");
                    assert_eq!(v["answers"].as_object().unwrap().len(), 5, "{v}");
                    lat.push(t.elapsed().as_secs_f64());
                }
                lat
            })
        })
        .collect();
    let mut all = Vec::new();
    for t in tasks {
        all.extend(t.await.unwrap());
    }
    all
}

/// A local copy in `$KIME_MODELS/laya`, or the Hugging Face cache.
fn model() -> String {
    std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |d| format!("{d}/laya"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn w9() {
    let kime = match Kime::builder().model(model()).device(Device::Cpu { threads: 0 }).build() {
        Ok(k) => k,
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            return;
        }
    };
    let (clients, rounds) = (env("KIME_W9_CLIENTS", 8), env("KIME_W9_ROUNDS", 4).max(2));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut cfg = kime_serve::Config::new(addr, vec![kime]);
    // Saturation is the point, so nothing is turned away for waiting.
    cfg.max_queue = Duration::ZERO;
    tokio::spawn(kime_serve::serve(listener, cfg, std::future::pending()));

    run(addr, clients, 2, 0).await;
    let start = rss_mb();
    let t = Instant::now();
    let mut lat = run(addr, clients, rounds / 2, 2 * clients).await;
    let mid = rss_mb();
    lat.extend(run(addr, clients, rounds - rounds / 2, (2 + rounds / 2) * clients).await);
    let took = t.elapsed().as_secs_f64();
    let end = rss_mb();
    lat.sort_by(f64::total_cmp);
    let p = |q: f64| lat[((q * lat.len() as f64) as usize).min(lat.len() - 1)] * 1e3;
    let (p50, p99) = (p(0.5), p(0.99));
    eprintln!(
        "W9: {clients} clients, {} requests in {took:.1} s, {:.2} req/s, p50 {p50:.0} ms, p99 {p99:.0} ms, rss {start:.0}, {mid:.0} and {end:.0} MB",
        lat.len(),
        lat.len() as f64 / took
    );
    assert!(p99 < 5.0 * p50, "p99 {p99:.0} ms is not under 5x p50 {p50:.0} ms");
    assert!(end < mid * 1.05, "rss grew from {mid:.0} MB to {end:.0} MB");
}
