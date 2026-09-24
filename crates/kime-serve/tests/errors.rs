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
//! Cases run against one of three servers, named by `server`: `open` with no keys (the default),
//! `keys` with Jev style keys and rate limits, `laya-key` with laya-serve's `LAYA_API_KEY`, and
//! `small` with a 64 token request limit, so a short body shows the 413.
//! `headers` adds request headers, and `retry_after` in the snapshot is the `retry-after`
//! header in seconds.
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

/// One exchange on a fresh connection: the status, the body, as JSON when it parses and as a
/// string when it does not, without `routing`, which spec/10 owns and kime-route will change,
/// or `device`, and `retry-after` in seconds when it is there.
async fn call(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: Option<&str>,
    headers: &str,
) -> (u16, Value, Option<u64>) {
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let head = match body {
        Some(b) => format!(
            "{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\n{headers}content-type: application/json\r\ncontent-length: {}\r\n\r\n{b}",
            b.len()
        ),
        None => {
            format!("{method} {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\n{headers}\r\n")
        }
    };
    s.write_all(head.as_bytes()).await.unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).await.unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let mut v = serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.to_string()));
    if let Some(o) = v.as_object_mut() {
        o.remove("routing");
        // `/health` names the device and its thread count, which depend on the machine.
        o.remove("device");
    }
    let retry = head.lines().find_map(|l| {
        let (k, v) = l.split_once(':')?;
        k.eq_ignore_ascii_case("retry-after").then(|| v.trim().parse().unwrap())
    });
    (head[9..12].parse().unwrap(), v, retry)
}

/// Starts a server on an ephemeral port that stops when the sender is dropped or sent to.
fn start(
    kime: &Kime,
    auth: kime_serve::Auth,
    max_request_tokens: usize,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>, tokio::task::JoinHandle<std::io::Result<()>>) {
    let std = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    std.set_nonblocking(true).unwrap();
    let listener = tokio::net::TcpListener::from_std(std).unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let mut cfg = kime_serve::Config::new(addr, vec![kime.clone()]);
    cfg.max_queue = std::time::Duration::ZERO;
    cfg.auth = auth;
    cfg.max_request_tokens = max_request_tokens;
    let server = tokio::spawn(kime_serve::serve(listener, cfg, async {
        let _ = stopped.await;
    }));
    (addr, stop, server)
}

/// Two requests on one connection, the first refused before its body is read. `t2` is put in
/// debt first, since `t1`'s debt from the cases may be paid off by now on a slow machine.
async fn keep_alive(addr: SocketAddr) {
    use tokio::io::AsyncBufReadExt;
    let answered = r#"{"state":"a few words to read","questions":{"q":{"type":"choice","instructions":"i","criteria":{"a":"","b":""}}}}"#;
    let (status, _, _) =
        call(addr, "POST", "/v1/systemone", Some(answered), "authorization: Bearer t2\r\n").await;
    assert_eq!(status, 200);
    let s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut s = tokio::io::BufReader::new(s);
    let body = r#"{"state":"x","questions":{"q":{"type":"noul","instructions":"i"}},"kime":{}}"#;
    for (key, want) in [("nope", "401"), ("t2", "429"), ("nope", "401")] {
        let head = format!(
            "POST /v1/systemone HTTP/1.1\r\nhost: x\r\nauthorization: Bearer {key}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n",
            body.len()
        );
        // The body comes a moment after the head, as it does from Python's http.client, so the
        // server has to wait for it rather than find it already buffered.
        s.get_mut().write_all(head.as_bytes()).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        s.get_mut().write_all(body.as_bytes()).await.unwrap();
        let (mut line, mut len) = (String::new(), 0);
        s.read_line(&mut line).await.unwrap();
        assert_eq!(line.get(9..12), Some(want), "{line}");
        loop {
            let mut h = String::new();
            s.read_line(&mut h).await.unwrap();
            if h == "\r\n" {
                break;
            }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                len = v.trim().parse().unwrap();
            }
        }
        let mut out = vec![0; len];
        s.read_exact(&mut out).await.unwrap();
    }
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
    // `open` has no keys. `keys` has Jev style keys: k1 at 2 requests a minute, t1 at 10 input
    // tokens a second and t2 at 1. `laya-key` is laya-serve's LAYA_API_KEY.
    let mut keys = kime_serve::Auth::off();
    keys.add_keys_file("k1 ops 2\nt1 tokens - 10\nt2 tokens - 1\n").unwrap();
    let servers = [
        ("open", start(&kime, kime_serve::Auth::off(), 65_536)),
        ("keys", start(&kime, keys, 65_536)),
        ("laya-key", start(&kime, kime_serve::Auth::laya("s3cret"), 65_536)),
        ("small", start(&kime, kime_serve::Auth::off(), 64)),
    ];
    let addr_of = |name: &str| servers.iter().find(|(n, _)| *n == name).map(|(_, s)| s.0).unwrap();

    let path = format!("{}/tests/errors/cases.json", env!("CARGO_MANIFEST_DIR"));
    let mut cases: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let bless = std::env::var_os("KIME_BLESS").is_some();
    let mut wrong = Vec::new();
    for c in &mut cases {
        let name = c["name"].as_str().unwrap().to_string();
        let headers: String = c["headers"]
            .as_array()
            .map(|h| h.iter().map(|l| format!("{}\r\n", l.as_str().unwrap())).collect())
            .unwrap_or_default();
        let (status, body, retry) = call(
            addr_of(c["server"].as_str().unwrap_or("open")),
            c["method"].as_str().unwrap(),
            c["path"].as_str().unwrap(),
            c["body"].as_str(),
            &headers,
        )
        .await;
        let mut got = json!({"status": status, "body": body});
        if let Some(r) = retry {
            // The wait counts down from the first request of the window, so a slow machine sees
            // less of it left. Anything from a second to the snapshot's wait is right.
            let snap =
                c["kime"]["retry_after"].as_u64().filter(|&w| !bless && (1..=w).contains(&r));
            got["retry_after"] = snap.unwrap_or(r).into();
        }
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
            400 | 401 | 404 | 405 => body == *lb,
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
    // A refused request leaves the connection open, so a client can retry on it.
    keep_alive(addr_of("keys")).await;

    for (_, (_, stop, server)) in servers {
        let _ = stop.send(());
        server.await.unwrap().unwrap();
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
