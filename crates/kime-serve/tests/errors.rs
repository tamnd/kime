//! Snapshots of every documented error, and of the odd requests Laya answers anyway, from a real
//! server with the Laya checkpoint on the CPU.
//!
//! The cases are in `tests/errors/cases.json`. Each has kime's answer as `kime` and, for the Laya
//! shaped ones, what laya-serve 0.3.9 answered as `laya` (recorded by `tests/errors/record.py`).
//! kime must give its snapshot, and must give laya-serve's status unless the case says why it
//! `differs`. Where both answer, the answers must agree to the fourth decimal. Where both refuse
//! with a 400, 404 or 405, the body must be the same. Laya's 422 messages come from Python
//! exceptions, so only their shape is compared.
//!
//! The cases run in order. `jev-deadline` comes right after a forward pass, so the wait estimate
//! is fresh and a 1 ms deadline is refused.
//!
//! `KIME_BLESS=1` rewrites the `kime` snapshots. Like `tests/http.rs` it needs the checkpoint in
//! `$KIME_MODELS/laya` or the Hugging Face cache, and passes with a note when it is missing unless
//! `KIME_REQUIRE_WEIGHTS` is set.

use std::net::SocketAddr;

use kime_engine::{Device, Kime};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// One exchange on a fresh connection: the status and the body, as JSON when it parses and as a
/// string when it does not, without `routing`, which spec/10 owns and kime-route will change.
async fn call(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> (u16, Value) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = match body {
        Some(b) => format!(
            "{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{b}",
            b.len()
        ),
        None => format!("{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n"),
    };
    s.write_all(head.as_bytes()).await.unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).await.unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let mut v = serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.to_string()));
    if let Some(o) = v.as_object_mut() {
        o.remove("routing");
    }
    (head[9..12].parse().unwrap(), v)
}

/// Equal, with numbers allowed to differ by one in the fourth decimal.
fn close(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            (x.as_f64().unwrap() - y.as_f64().unwrap()).abs() <= 1.5e-4
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(a, b)| close(a, b))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, a)| y.get(k).is_some_and(|b| close(a, b)))
        }
        _ => a == b,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn error_snapshots() {
    let model =
        std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |d| format!("{d}/laya"));
    let kime = match Kime::builder().model(model).device(Device::Cpu { threads: 0 }).build() {
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
    let mut cfg = kime_serve::Config::new(addr, vec![kime]);
    cfg.max_queue = std::time::Duration::ZERO;
    let server = tokio::spawn(kime_serve::serve(listener, cfg, async {
        let _ = stopped.await;
    }));

    let path = format!("{}/tests/errors/cases.json", env!("CARGO_MANIFEST_DIR"));
    let mut cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let bless = std::env::var_os("KIME_BLESS").is_some();
    let mut wrong = Vec::new();
    for c in &mut cases {
        let name = c["name"].as_str().unwrap().to_string();
        let (status, body) = call(
            addr,
            c["method"].as_str().unwrap(),
            c["path"].as_str().unwrap(),
            c["body"].as_str(),
        )
        .await;
        let got = json!({"status": status, "body": body});
        if bless {
            c["kime"] = got.clone();
        } else if !close(&got, &c["kime"]) {
            wrong.push(format!("{name}: snapshot {} got {got}", c["kime"]));
        }

        let Some(laya) = c.get("laya") else { continue };
        let (ls, lb) = (laya["status"].as_u64().unwrap() as u16, &laya["body"]);
        if c.get("differs").is_some() {
            if status == ls {
                wrong.push(format!(
                    "{name}: marked as differing from laya-serve but both give {ls}"
                ));
            }
            continue;
        }
        let same = match status {
            _ if status != ls => false,
            200 => close(&body["answers"], &lb["answers"]) && body["usage"] == lb["usage"],
            400 | 404 | 405 => body == *lb,
            _ => std::mem::discriminant(&body["detail"]) == std::mem::discriminant(&lb["detail"]),
        };
        if !same {
            wrong.push(format!("{name}: laya-serve {ls} {lb} kime {status} {body}"));
        }
    }
    if bless {
        let text = serde_json::to_string_pretty(&cases).unwrap() + "\n";
        std::fs::write(&path, text).unwrap();
    }
    let _ = stop.send(());
    server.await.unwrap().unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
