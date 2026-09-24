//! OpenTelemetry traces from spec/11: one server span per request, sent to a collector over
//! OTLP/HTTP with JSON bodies. Off unless an endpoint is set, and then the only cost per request
//! is one branch.
//!
//! A request with a W3C `traceparent` header joins that trace, and every traced response says its
//! own span in `traceparent`. Spans go through a bounded queue to one task that posts them in
//! batches, so a slow or missing collector never holds up a request: when the queue is full the
//! span is dropped and counted. Only plain `http://` endpoints are taken, which is how a
//! collector running next to the server is reached.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc;

/// Spans waiting to be sent past this are dropped.
const QUEUE: usize = 8192;

/// The most spans in one post.
const BATCH: usize = 512;

/// How long a span waits for others to fill its batch.
const LINGER: Duration = Duration::from_millis(500);

/// Where traces go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Otlp {
    /// `host:port`.
    host: String,
    /// The path traces are posted to, `/v1/traces` unless the endpoint gives one.
    path: String,
    /// `service.name` on every span.
    pub service: String,
}

impl Otlp {
    /// An endpoint as `OTEL_EXPORTER_OTLP_ENDPOINT` gives it, `http://collector:4318`. With no
    /// path, traces go to `/v1/traces` as the OTLP spec says.
    ///
    /// # Errors
    ///
    /// For anything but a plain `http://` URL with a host.
    pub fn new(endpoint: &str) -> Result<Otlp, String> {
        let rest = endpoint.trim().strip_prefix("http://").ok_or_else(|| {
            format!("OTLP endpoint {endpoint:?} must start with http://, https needs a collector in front")
        })?;
        let (host, path) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
            None => (rest, ""),
        };
        if host.is_empty() {
            return Err(format!("OTLP endpoint {endpoint:?} has no host"));
        }
        let host = if host.rsplit_once(':').is_some_and(|(_, p)| p.parse::<u16>().is_ok()) {
            host.to_string()
        } else {
            format!("{host}:80")
        };
        let path = if path.is_empty() { "/v1/traces".to_string() } else { path.to_string() };
        Ok(Otlp { host, path, service: "kime".into() })
    }
}

/// A trace id and a span id, as `traceparent` carries them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Context {
    pub(crate) trace: u128,
    pub(crate) span: u64,
}

impl Context {
    /// A W3C `traceparent` header, version 00 or a later one read the same way.
    pub(crate) fn parse(h: &str) -> Option<Context> {
        let p: Vec<&str> = h.trim().split('-').collect();
        let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
        if p.len() < 4 || !hex(p[0], 2) || p[0] == "ff" || (p[0] == "00" && p.len() != 4) {
            return None;
        }
        if !hex(p[1], 32) || !hex(p[2], 16) || !hex(p[3], 2) {
            return None;
        }
        let trace = u128::from_str_radix(p[1], 16).ok()?;
        let span = u64::from_str_radix(p[2], 16).ok()?;
        (trace != 0 && span != 0).then_some(Context { trace, span })
    }

    /// The `traceparent` header for this span, sampled.
    pub(crate) fn header(self) -> String {
        format!("00-{:032x}-{:016x}-01", self.trace, self.span)
    }
}

/// A random id that is not zero.
fn random(salt: u8) -> u64 {
    static SEED: OnceLock<RandomState> = OnceLock::new();
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    SEED.get_or_init(RandomState::new).hash_one((n, t, salt)).max(1)
}

/// The span of a new request: in the caller's trace if it sent one, a new trace if not.
pub(crate) fn start(parent: Option<Context>) -> (Context, Option<u64>) {
    let span = random(0);
    match parent {
        Some(p) => (Context { trace: p.trace, span }, Some(p.span)),
        None => {
            (Context { trace: u128::from(random(1)) << 64 | u128::from(random(2)), span }, None)
        }
    }
}

/// One finished request.
#[derive(Debug)]
pub(crate) struct Span {
    pub(crate) ctx: Context,
    pub(crate) parent: Option<u64>,
    pub(crate) route: &'static str,
    pub(crate) method: String,
    pub(crate) start: SystemTime,
    pub(crate) took: Duration,
    pub(crate) status: u16,
    pub(crate) request_id: String,
    /// The model id, questions and input tokens, for an answered request.
    pub(crate) served: Option<(String, usize, u64)>,
}

fn nanos(t: SystemTime) -> String {
    t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos()).to_string()
}

fn attr(key: &str, v: Value) -> Value {
    let v = match v {
        Value::String(s) => json!({"stringValue": s}),
        // OTLP JSON carries 64 bit integers as strings.
        Value::Number(n) => json!({"intValue": n.to_string()}),
        v => json!({"stringValue": v.to_string()}),
    };
    json!({"key": key, "value": v})
}

impl Span {
    fn to_json(&self) -> Value {
        let mut attrs = vec![
            attr("http.request.method", json!(self.method)),
            attr("http.route", json!(self.route)),
            attr("http.response.status_code", json!(self.status)),
            attr("kime.request_id", json!(self.request_id)),
        ];
        if let Some((model, questions, tokens)) = &self.served {
            attrs.push(attr("kime.model", json!(model)));
            attrs.push(attr("kime.questions", json!(questions)));
            attrs.push(attr("kime.input_tokens", json!(tokens)));
        }
        let mut v = json!({
            "traceId": format!("{:032x}", self.ctx.trace),
            "spanId": format!("{:016x}", self.ctx.span),
            "name": format!("{} {}", self.method, self.route),
            // SPAN_KIND_SERVER.
            "kind": 2,
            "startTimeUnixNano": nanos(self.start),
            "endTimeUnixNano": nanos(self.start + self.took),
            "attributes": attrs,
            // STATUS_CODE_ERROR for a 5xx, unset otherwise, as the HTTP conventions say.
            "status": if self.status >= 500 { json!({"code": 2}) } else { json!({}) },
        });
        if let Some(p) = self.parent {
            v["parentSpanId"] = json!(format!("{p:016x}"));
        }
        v
    }
}

/// The body of one post.
fn body(service: &str, spans: &[Span]) -> String {
    json!({"resourceSpans": [{
        "resource": {"attributes": [
            attr("service.name", json!(service)),
            attr("service.version", json!(env!("CARGO_PKG_VERSION"))),
        ]},
        "scopeSpans": [{
            "scope": {"name": "kime-serve", "version": env!("CARGO_PKG_VERSION")},
            "spans": spans.iter().map(Span::to_json).collect::<Vec<_>>(),
        }],
    }]})
    .to_string()
}

/// What happened to the spans, for `/metrics`.
#[derive(Debug, Default)]
pub(crate) struct Counts {
    pub(crate) exported: AtomicU64,
    pub(crate) dropped: AtomicU64,
    pub(crate) failed: AtomicU64,
}

/// The sending side, one per server.
#[derive(Debug)]
pub(crate) struct Tracer {
    tx: mpsc::Sender<Span>,
    pub(crate) counts: std::sync::Arc<Counts>,
}

impl Tracer {
    /// Starts the task that posts spans. It needs a tokio runtime.
    pub(crate) fn spawn(to: Otlp) -> Tracer {
        let (tx, rx) = mpsc::channel(QUEUE);
        let counts = std::sync::Arc::new(Counts::default());
        tokio::spawn(export(to, rx, counts.clone()));
        Tracer { tx, counts }
    }

    pub(crate) fn send(&self, span: Span) {
        if self.tx.try_send(span).is_err() {
            self.counts.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn export(to: Otlp, mut rx: mpsc::Receiver<Span>, counts: std::sync::Arc<Counts>) {
    let mut batch = Vec::with_capacity(BATCH);
    while let Some(first) = rx.recv().await {
        batch.push(first);
        let until = tokio::time::Instant::now() + LINGER;
        while batch.len() < BATCH {
            match tokio::time::timeout_at(until, rx.recv()).await {
                Ok(Some(s)) => batch.push(s),
                _ => break,
            }
        }
        let n = batch.len() as u64;
        let ok = post(&to, &body(&to.service, &batch)).await.is_ok();
        let c = if ok { &counts.exported } else { &counts.failed };
        c.fetch_add(n, Ordering::Relaxed);
        batch.clear();
    }
}

/// One POST on a fresh connection. The collector's answer is read only for its status.
async fn post(to: &Otlp, body: &str) -> Result<(), String> {
    let run = async {
        let mut s = tokio::net::TcpStream::connect(&to.host).await.map_err(|e| e.to_string())?;
        let head = format!(
            "POST {} HTTP/1.1\r\nhost: {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            to.path,
            to.host,
            body.len()
        );
        s.write_all(head.as_bytes()).await.map_err(|e| e.to_string())?;
        s.write_all(body.as_bytes()).await.map_err(|e| e.to_string())?;
        let mut buf = [0u8; 16];
        let mut n = 0;
        while n < buf.len() {
            match s.read(&mut buf[n..]).await.map_err(|e| e.to_string())? {
                0 => break,
                k => n += k,
            }
        }
        match std::str::from_utf8(&buf[..n]).ok().and_then(|l| l.split(' ').nth(1)) {
            Some(code) if code.starts_with('2') => Ok(()),
            other => Err(format!("collector answered {other:?}")),
        }
    };
    tokio::time::timeout(Duration::from_secs(10), run).await.map_err(|_| "timed out".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints() {
        let o = Otlp::new("http://collector:4318").unwrap();
        assert_eq!((o.host.as_str(), o.path.as_str()), ("collector:4318", "/v1/traces"));
        let o = Otlp::new("http://127.0.0.1:4318/custom/traces/").unwrap();
        assert_eq!((o.host.as_str(), o.path.as_str()), ("127.0.0.1:4318", "/custom/traces"));
        assert_eq!(Otlp::new("http://otel").unwrap().host, "otel:80");
        assert!(Otlp::new("https://collector:4318").is_err());
        assert!(Otlp::new("http:///v1/traces").is_err());
    }

    #[test]
    fn traceparent() {
        let h = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
        let c = Context::parse(h).unwrap();
        assert_eq!(c.header(), h);
        assert_eq!(c.span, 0x00f0_67aa_0ba9_02b7);
        for bad in [
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-x",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
            "",
        ] {
            assert_eq!(Context::parse(bad), None, "{bad}");
        }
        // A later version may add fields after the flags.
        assert!(
            Context::parse("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-x").is_some()
        );
        let (c, parent) = start(Context::parse(h));
        assert_eq!(
            (c.trace, parent),
            (0x4bf9_2f35_77b3_4da6_a3ce_929d_0e0e_4736, Some(0x00f0_67aa_0ba9_02b7))
        );
        let (a, none) = start(None);
        assert!(none.is_none() && a.trace != 0 && a.span != 0 && a != start(None).0);
    }

    #[test]
    fn span_json() {
        let s = Span {
            ctx: Context { trace: 1, span: 2 },
            parent: Some(3),
            route: "/v1/systemone",
            method: "POST".into(),
            start: UNIX_EPOCH + Duration::from_nanos(1_000),
            took: Duration::from_nanos(500),
            status: 200,
            request_id: "req_1".into(),
            served: Some(("laya".into(), 2, 96)),
        };
        let v: Value = serde_json::from_str(&body("kime", &[s])).unwrap();
        let span = &v["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(span["traceId"], "00000000000000000000000000000001");
        assert_eq!(span["parentSpanId"], "0000000000000003");
        assert_eq!(span["name"], "POST /v1/systemone");
        assert_eq!(
            (span["startTimeUnixNano"].as_str(), span["endTimeUnixNano"].as_str()),
            (Some("1000"), Some("1500"))
        );
        assert!(
            span["attributes"]
                .as_array()
                .unwrap()
                .contains(&json!({"key": "kime.input_tokens", "value": {"intValue": "96"}}))
        );
        assert_eq!(
            v["resourceSpans"][0]["resource"]["attributes"][0]["value"]["stringValue"],
            "kime"
        );
    }
}
