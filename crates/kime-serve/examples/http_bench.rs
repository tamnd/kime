//! Latency and throughput of any `/v1/systemone` server, kime or laya-serve, over keep-alive
//! HTTP/1.1 on real requests.
//!
//! `cargo run --release -p kime-serve --example http_bench -- http://127.0.0.1:8000 CASES.jsonl [concurrency] [seconds] [model]`
//!
//! Requests come from a JSONL file of request bodies (the parity cases work), each sent with the
//! given `model`, default `english`. With concurrency 1 it reports the latency of one request at a
//! time. Higher concurrency shows what batching does under load. Every response must be a 200.

use std::time::{Duration, Instant};

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

struct Conn {
    r: BufReader<TcpStream>,
    host: String,
}

impl Conn {
    async fn open(addr: &str) -> Conn {
        let s = TcpStream::connect(addr).await.expect("connect");
        s.set_nodelay(true).unwrap();
        Conn { r: BufReader::new(s), host: addr.to_string() }
    }

    /// One POST, its status, body and `server-timing` header, on the same connection each time.
    async fn post(&mut self, path: &str, body: &[u8]) -> (u16, Vec<u8>, String) {
        let head = format!(
            "POST {path} HTTP/1.1\r\nhost: {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            self.host,
            body.len()
        );
        let s = self.r.get_mut();
        s.write_all(head.as_bytes()).await.unwrap();
        s.write_all(body).await.unwrap();
        let mut line = String::new();
        self.r.read_line(&mut line).await.unwrap();
        let status = line.get(9..12).and_then(|s| s.parse().ok()).unwrap_or(0);
        let (mut len, mut timing) = (0, String::new());
        loop {
            line.clear();
            self.r.read_line(&mut line).await.unwrap();
            if line == "\r\n" || line.is_empty() {
                break;
            }
            let lower = line.to_ascii_lowercase();
            if let Some(v) = lower.strip_prefix("content-length:") {
                len = v.trim().parse().unwrap();
            }
            if let Some(v) = lower.strip_prefix("server-timing:") {
                timing = v.trim().to_string();
            }
        }
        let mut out = vec![0; len];
        self.r.read_exact(&mut out).await.unwrap();
        (status, out, timing)
    }
}

/// Adds up kime's `server-timing` parts: queue, tokenize, device and pass size. laya-serve sends
/// none, and then this stays at zero.
fn add_timing(sum: &mut [f64; 4], h: &str) {
    for part in h.split(',') {
        let mut kv = part.trim().split(';');
        let name = kv.next().unwrap_or_default();
        let i = match name {
            "queue" => 0,
            "tokenize" => 1,
            "device" => 2,
            "pass" => 3,
            _ => continue,
        };
        for f in kv {
            if let Some(v) = f.strip_prefix("dur=").or_else(|| f.strip_prefix("desc=")) {
                sum[i] += v.trim_matches('"').parse::<f64>().unwrap_or(0.0);
            }
        }
    }
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i].as_secs_f64() * 1e3
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let url = args.first().expect("url");
    let addr = url.trim_start_matches("http://").trim_end_matches('/').to_string();
    let file = args.get(1).expect("cases file");
    let conc: usize = args.get(2).map_or(1, |c| c.parse().unwrap());
    let secs: f64 = args.get(3).map_or(20.0, |c| c.parse().unwrap());
    let model = args.get(4).cloned().unwrap_or_else(|| "english".into());
    let bodies: Vec<Vec<u8>> = std::fs::read_to_string(file)
        .unwrap()
        .lines()
        .map(|l| {
            let mut v: Value = serde_json::from_str(l).unwrap();
            let m = v.as_object_mut().unwrap();
            m.remove("id");
            m.insert("model".into(), Value::String(model.clone()));
            serde_json::to_vec(&v).unwrap()
        })
        .collect();
    let questions: Vec<usize> = std::fs::read_to_string(file)
        .unwrap()
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["questions"]
                .as_object()
                .map_or(0, |q| q.len())
        })
        .collect();

    // Warm up: every body once, one at a time.
    let mut c = Conn::open(&addr).await;
    for b in &bodies {
        let (s, out, _) = c.post("/v1/systemone", b).await;
        assert_eq!(s, 200, "{}", String::from_utf8_lossy(&out));
    }

    let bodies = std::sync::Arc::new(bodies);
    let questions = std::sync::Arc::new(questions);
    let start = Instant::now();
    let end = start + Duration::from_secs_f64(secs);
    let tasks: Vec<_> = (0..conc)
        .map(|w| {
            let (bodies, questions, addr) = (bodies.clone(), questions.clone(), addr.clone());
            tokio::spawn(async move {
                let mut c = Conn::open(&addr).await;
                let (mut lat, mut qs, mut sum) = (Vec::new(), 0, [0.0; 4]);
                let mut i = w * 7919;
                while Instant::now() < end {
                    let k = i % bodies.len();
                    let t = Instant::now();
                    let (s, out, timing) = c.post("/v1/systemone", &bodies[k]).await;
                    lat.push(t.elapsed());
                    add_timing(&mut sum, &timing);
                    assert_eq!(s, 200, "{}", String::from_utf8_lossy(&out));
                    qs += questions[k];
                    i += 1;
                }
                (lat, qs, sum)
            })
        })
        .collect();
    let (mut lat, mut qs, mut sum) = (Vec::new(), 0, [0.0; 4]);
    for t in tasks {
        let (l, q, s) = t.await.unwrap();
        lat.extend(l);
        qs += q;
        for (a, b) in sum.iter_mut().zip(s) {
            *a += b;
        }
    }
    let took = start.elapsed().as_secs_f64();
    lat.sort();
    println!(
        "{url} concurrency {conc}: {} requests, {qs} questions in {took:.1} s, {:.1} requests/s, {:.1} questions/s, p50 {:.2} ms, p90 {:.2} ms, p99 {:.2} ms",
        lat.len(),
        lat.len() as f64 / took,
        qs as f64 / took,
        pct(&lat, 0.5),
        pct(&lat, 0.9),
        pct(&lat, 0.99),
    );
    if sum[3] > 0.0 {
        let n = lat.len() as f64;
        println!(
            "  mean per request: queue {:.2} ms, tokenize {:.2} ms, device {:.2} ms, {:.1} requests per pass",
            sum[0] / n,
            sum[1] / n,
            sum[2] / n,
            sum[3] / n,
        );
    }
}
