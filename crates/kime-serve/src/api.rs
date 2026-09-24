//! The routes, the two response shapes, and the error bodies from spec/03-api.md.

// Helpers return the error response itself, which is big, but only on the path that is about to
// answer with it anyway.
#![allow(clippy::result_large_err)]

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Path, Request as HttpRequest};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response as HttpResponse};
use axum::routing::{get, post};
use axum::{Extension, Router};
use kime_core::answer::{Answer, Response};
use kime_core::request::{Limits, Loc, Problem, Request, parse};
use kime_engine::Error;
use serde_json::{Map, Value, json};

use crate::auth::{Auth, Bucket, Denied};
use crate::log::{Line, Log, Served};
use crate::metrics::{Metrics, ModelView, route};
use crate::models::{Done, Models, Named, Refused, Resolved};

/// Everything the handlers share.
#[derive(Debug)]
pub(crate) struct State {
    pub(crate) models: Models,
    pub(crate) max_body: usize,
    pub(crate) max_request_tokens: usize,
    pub(crate) metrics: Metrics,
    pub(crate) auth: Auth,
    pub(crate) buckets: Vec<Bucket>,
    pub(crate) log: Log,
}

type Shared = Arc<State>;

/// The most items `/v1/systemone/batch` takes in one call.
const MAX_BATCH_ITEMS: usize = 1024;

pub(crate) fn router(state: Shared) -> Router {
    let api = Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/v1/systemone/batch", post(batch))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{*id}", get(one_model))
        .route_layer(middleware::from_fn(guard));
    Router::new()
        .merge(api)
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .fallback(|| async { reply(StatusCode::NOT_FOUND, &json!({"detail": "Not Found"})) })
        .method_not_allowed_fallback(|| async {
            reply(StatusCode::METHOD_NOT_ALLOWED, &json!({"detail": "Method Not Allowed"}))
        })
        .layer(middleware::from_fn(request_id))
        .layer(Extension(state))
}

/// The id every response carries, `req_` and 32 hex digits, or the client's own when it is
/// safe to echo.
#[derive(Debug, Clone)]
struct RequestId(String);

fn new_id() -> String {
    static SEED: OnceLock<RandomState> = OnceLock::new();
    static COUNT: AtomicU64 = AtomicU64::new(0);
    let seed = SEED.get_or_init(RandomState::new);
    let n = COUNT.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos());
    format!("req_{:016x}{:016x}", seed.hash_one((n, t)), seed.hash_one((t, n, 1u8)))
}

fn echoable(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

async fn request_id(mut req: HttpRequest, next: Next) -> HttpResponse {
    let id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| echoable(v))
        .map_or_else(new_id, str::to_string);
    req.extensions_mut().insert(RequestId(id.clone()));
    let (start, route) = (Instant::now(), route(req.uri().path()));
    let state = req.extensions().get::<Shared>().cloned();
    // Logging needs the body's size and hash, so it reads the body here and hands the handler a
    // copy. Off, the body goes to the handler untouched.
    let logged = match &state {
        Some(s) if s.log != Log::Off => {
            let (method, path) = (req.method().to_string(), req.uri().path().to_string());
            let (parts, body) = req.into_parts();
            let Ok(bytes) = axum::body::to_bytes(body, s.max_body).await else {
                return too_large(s.max_body);
            };
            let seen = (method, path, bytes.len(), blake3::hash(&bytes), SystemTime::now());
            req = HttpRequest::from_parts(parts, Body::from(bytes));
            Some(seen)
        }
        _ => None,
    };
    let mut res = next.run(req).await;
    let took = start.elapsed();
    if let Some(s) = &state {
        s.metrics.record(route, res.status().as_u16(), took);
        if let Some((method, path, bytes, hash, at)) = &logged {
            let served = res
                .extensions()
                .get::<Served>()
                .map(|v| (s.models.list[v.model].id.as_str(), v.questions, v.tokens));
            Line {
                at: *at,
                id: &id,
                method,
                path,
                status: res.status().as_u16(),
                took,
                bytes: *bytes,
                hash: *hash,
                served,
            }
            .write(s.log);
        }
    }
    if let Ok(v) = HeaderValue::from_str(&id) {
        res.headers_mut().insert(HeaderName::from_static("x-request-id"), v.clone());
        res.headers_mut().insert(HeaderName::from_static("x-typesafe-request-id"), v);
    }
    res
}

/// Checks the key and the rate limits in front of the `/v1` routes, and charges the tokens of the
/// answer afterwards.
async fn guard(req: HttpRequest, next: Next) -> HttpResponse {
    let Some(s) = req.extensions().get::<Shared>().cloned() else { return next.run(req).await };
    let header = req.headers().get(header::AUTHORIZATION).map(HeaderValue::as_bytes);
    let at = s.auth.check(header).and_then(|at| s.buckets[at].admit().map(|()| at));
    let at = match at {
        Ok(at) => at,
        Err(d) => {
            // Read the body before refusing. Unread, it makes the server close the connection,
            // and a client that retries a 429 on its pooled connection gets a broken pipe.
            let _ = axum::body::to_bytes(req.into_body(), s.max_body).await;
            return denied(d, s.auth.laya_style());
        }
    };
    let res = next.run(req).await;
    if let Some(v) = res.extensions().get::<Served>() {
        s.buckets[at].charge(v.tokens);
    }
    res
}

/// Jev's 403, 401 and 429, or laya-serve's 401 when the key came from `LAYA_API_KEY`.
fn denied(d: Denied, laya: bool) -> HttpResponse {
    let auth = |status, message: &str| {
        reply(
            status,
            &json!({"detail": {"error_type": "authentication_error", "message": message}}),
        )
    };
    let limited = |wait: Duration, message: String| {
        let ms = wait.as_millis().max(1) as u64;
        let mut r = reply(
            StatusCode::TOO_MANY_REQUESTS,
            &json!({"detail": {"error_type": "rate_limit_error", "message": message}}),
        );
        let h = r.headers_mut();
        h.insert(HeaderName::from_static("retry-after-ms"), HeaderValue::from(ms));
        h.insert(header::RETRY_AFTER, HeaderValue::from(ms.div_ceil(1000)));
        r
    };
    match d {
        Denied::Missing | Denied::Unknown if laya => {
            detail(StatusCode::UNAUTHORIZED, "invalid or missing bearer token")
        }
        Denied::Missing => {
            auth(StatusCode::FORBIDDEN, "Must supply an API key! Check your request and try again.")
        }
        Denied::Unknown => auth(
            StatusCode::UNAUTHORIZED,
            "Cannot authenticate with the server. Please check your API key and try again.",
        ),
        Denied::Requests(n, wait) => limited(
            wait,
            format!(
                "Rate limit exceeded: this key allows {n} requests per minute. Retry after retry-after-ms."
            ),
        ),
        Denied::Tokens(n, wait) => limited(
            wait,
            format!(
                "Rate limit exceeded: this key allows {n} input tokens per second. Retry after retry-after-ms."
            ),
        ),
    }
}

fn reply(status: StatusCode, body: &Value) -> HttpResponse {
    let mut res = (status, body.to_string()).into_response();
    res.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    res
}

fn detail(status: StatusCode, msg: impl Into<String>) -> HttpResponse {
    reply(status, &json!({"detail": msg.into()}))
}

fn invalid(problems: &[Problem]) -> HttpResponse {
    let list: Vec<Value> = problems.iter().map(Problem::to_json).collect();
    reply(StatusCode::UNPROCESSABLE_ENTITY, &json!({"detail": list}))
}

fn internal(e: &Error, id: &RequestId) -> HttpResponse {
    reply(
        StatusCode::INTERNAL_SERVER_ERROR,
        &json!({"detail": {"error_type": "internal_error", "message": e.to_string(), "request_id": id.0}}),
    )
}

/// Jev's 529, for when the queue would keep a new request waiting longer than `--max-queue-ms`.
/// `retry-after-ms` is the current wait estimate.
fn overloaded(wait: Duration) -> HttpResponse {
    let ms = wait.as_millis().max(1);
    let mut r = reply(
        StatusCode::from_u16(529).expect("529 is a valid status"),
        &json!({"detail": {"error_type": "overloaded_error", "message": format!("the server is overloaded, the queue would take about {ms} ms")}}),
    );
    r.headers_mut().insert(HeaderName::from_static("retry-after-ms"), HeaderValue::from(ms as u64));
    r
}

/// Jev's 504, for a `deadline_ms` the queue and the forward pass would miss. The estimate goes in
/// `server-timing` so the body stays the same from one request to the next.
fn late(deadline: Duration, ready: Duration) -> HttpResponse {
    let mut r = reply(
        StatusCode::GATEWAY_TIMEOUT,
        &json!({"detail": {"error_type": "deadline_exceeded", "message": format!("the answer cannot be ready within deadline_ms {}, the queue and the forward pass would take longer", deadline.as_millis())}}),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("estimate;dur={:.1}", ready.as_secs_f64() * 1e3))
    {
        r.headers_mut().insert(HeaderName::from_static("server-timing"), v);
    }
    r
}

/// Whether a body is big enough that its requests could be over the token limit. A token is at
/// least one byte of the text it comes from, and the rendered text is never much longer than the
/// JSON it came in, so a body under half the limit cannot be over it.
fn counted(s: &State, bytes: usize) -> bool {
    bytes > s.max_request_tokens / 2
}

/// The tokens in each request, counted on a blocking thread because a big body takes a while.
async fn count(
    s: &State,
    at: usize,
    reqs: Vec<Request>,
) -> Result<(Vec<usize>, Vec<Request>), HttpResponse> {
    let k = s.models.list[at].kime.clone();
    tokio::task::spawn_blocking(move || (reqs.iter().map(|r| k.count_tokens(r)).collect(), reqs))
        .await
        .map_err(|e| detail(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

fn too_long_body(n: usize, max: usize) -> Value {
    json!({"detail": [{
        "loc": ["body"],
        "msg": format!("the request holds {n} tokens, over the limit of {max}"),
        "type": "too_long",
    }]})
}

/// Jev's 413, for a request over `max_request_tokens`.
fn too_many_tokens(n: usize, max: usize) -> HttpResponse {
    reply(StatusCode::PAYLOAD_TOO_LARGE, &too_long_body(n, max))
}

fn refused(r: Refused, deadline: Option<Duration>) -> HttpResponse {
    match r {
        Refused::Overloaded(wait) => overloaded(wait),
        Refused::Deadline(ready) => late(deadline.unwrap_or_default(), ready),
    }
}

fn too_large(max: usize) -> HttpResponse {
    detail(
        StatusCode::BAD_REQUEST,
        format!("request body could not be read or is over {max} bytes"),
    )
}

/// Reads and parses a JSON object body, with its size in bytes, or the 400 laya-serve and Jev
/// both give.
async fn object(body: Body, max: usize) -> Result<(Map<String, Value>, usize), HttpResponse> {
    let bytes = axum::body::to_bytes(body, max).await.map_err(|_| too_large(max))?;
    match serde_json::from_slice(&bytes) {
        Ok(Value::Object(o)) => Ok((o, bytes.len())),
        Ok(_) => Err(detail(
            StatusCode::BAD_REQUEST,
            "request body must be an object with a 'questions' field",
        )),
        Err(e) => Err(detail(StatusCode::BAD_REQUEST, format!("request body is not JSON: {e}"))),
    }
}

fn resolve(s: &State, body: &Map<String, Value>) -> Result<Resolved, HttpResponse> {
    let name = body.get("model").and_then(Value::as_str);
    s.models.resolve(name).ok_or_else(|| {
        detail(StatusCode::NOT_FOUND, format!("model '{}' not found", name.unwrap_or_default()))
    })
}

/// The `kime` options this server acts on. The rest of spec/03's table is accepted and ignored
/// until the feature behind it lands.
#[derive(Debug, Clone, Copy)]
struct Opts {
    precision: Option<u32>,
    entropy: bool,
    extensions: bool,
    deadline: Option<Duration>,
}

fn opts(kime: Option<&Value>) -> Result<Opts, Vec<Problem>> {
    let mut o = Opts { precision: Some(2), entropy: false, extensions: false, deadline: None };
    let Some(k) = kime else { return Ok(o) };
    let problem = |field: &str, kind, msg: &str, input: &Value| Problem {
        loc: vec!["body".into(), "kime".into(), Loc::from(field)],
        msg: msg.into(),
        kind,
        input: input.clone(),
    };
    let Some(k) = k.as_object() else {
        return Err(vec![Problem {
            loc: vec!["body".into(), "kime".into()],
            msg: "kime must be an object".into(),
            kind: "dict_type",
            input: k.clone(),
        }]);
    };
    let mut bad = Vec::new();
    match k.get("precision") {
        None => {}
        Some(Value::Null) => o.precision = None,
        Some(v) => match v.as_u64().filter(|&p| p <= 6) {
            Some(p) => o.precision = Some(p as u32),
            None => {
                bad.push(problem("precision", "int_range", "precision must be 0 to 6 or null", v))
            }
        },
    }
    match k.get("confidence") {
        None => {}
        Some(v) => match v.as_str() {
            Some("jev") => {}
            Some("entropy") => o.entropy = true,
            _ => bad.push(problem(
                "confidence",
                "literal_error",
                "confidence must be \"jev\" or \"entropy\"",
                v,
            )),
        },
    }
    match k.get("extensions") {
        None => {}
        Some(Value::Bool(b)) => o.extensions = *b,
        Some(v) => {
            bad.push(problem("extensions", "bool_type", "extensions must be true or false", v))
        }
    }
    match k.get("deadline_ms") {
        None | Some(Value::Null) => {}
        Some(v) => match v.as_u64().filter(|&ms| ms >= 1) {
            Some(ms) => o.deadline = Some(Duration::from_millis(ms)),
            None => bad.push(problem(
                "deadline_ms",
                "int_range",
                "deadline_ms must be a whole number of milliseconds, 1 or more",
                v,
            )),
        },
    }
    if bad.is_empty() { Ok(o) } else { Err(bad) }
}

fn us(d: Duration) -> f64 {
    (d.as_secs_f64() * 1e7).round() / 10.0
}

/// A response in Jev's shape, plus the `kime` block when extensions are on.
fn jev(model: &str, r: &Response, o: Opts, extra: Option<Value>) -> Value {
    let mut answers = Map::new();
    for (id, a) in &r.answers {
        let mut v = a.to_jev_json(o.precision, o.entropy);
        if let (true, Answer::Noul { confidence, .. }) = (o.extensions, a)
            && let Some(m) = v.as_object_mut()
        {
            m.insert("confidence".into(), json!(round(*confidence, o.precision)));
        }
        answers.insert(id.clone(), v);
    }
    let mut out = json!({
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": r.input_tokens, "output_tokens": 0},
    });
    if let (Some(extra), Some(m)) = (extra, out.as_object_mut()) {
        m.insert("kime".into(), extra);
    }
    out
}

fn round(x: f64, precision: Option<u32>) -> f64 {
    match precision {
        Some(d) => kime_core::answer::py_round(x, d as usize),
        None => f64::from(x as f32),
    }
}

/// The per answer numbers the extensions block adds.
fn extras(r: &Response, o: Opts) -> Value {
    let mut m = Map::new();
    for (id, a) in &r.answers {
        let v = match a {
            Answer::Choice { confidence, act_probability, .. }
            | Answer::Score { confidence, act_probability, .. } => json!({
                "confidence_entropy": round(*confidence, o.precision),
                "act_probability": round(*act_probability, o.precision),
            }),
            Answer::Noul { confidence, act_probability, .. } => json!({
                "noul_confidence": round(*confidence, o.precision),
                "act_probability": round(*act_probability, o.precision),
            }),
        };
        m.insert(id.clone(), v);
    }
    Value::Object(m)
}

fn engine_error(e: &Error, laya: bool, id: &RequestId) -> HttpResponse {
    match e {
        Error::Invalid(p) if laya => detail(StatusCode::UNPROCESSABLE_ENTITY, messages(p)),
        Error::Invalid(p) => invalid(p),
        Error::TooLong { question, .. } if !laya => invalid(&[Problem {
            loc: vec![
                "body".into(),
                "questions".into(),
                Loc::Key(question.clone()),
                "criteria".into(),
            ],
            msg: e.to_string(),
            kind: "too_long",
            input: Value::Null,
        }]),
        Error::TooLong { .. } => detail(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()),
        e => internal(e, id),
    }
}

fn messages(p: &[Problem]) -> String {
    p.iter().map(|p| p.msg.as_str()).collect::<Vec<_>>().join("; ")
}

async fn systemone(
    Extension(s): Extension<Shared>,
    Extension(id): Extension<RequestId>,
    body: Body,
) -> HttpResponse {
    let start = Instant::now();
    let (body, bytes) = match object(body, s.max_body).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let resolved = match resolve(&s, &body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    // A Laya client names a Laya checkpoint or nothing, and never sends a kime object.
    let laya = resolved.named != Named::Kime && !body.contains_key("kime");
    if laya && !body.contains_key("questions") {
        return detail(
            StatusCode::BAD_REQUEST,
            "request body must be an object with a 'questions' field",
        );
    }
    let body = Value::Object(body);
    let mut req = match parse(&body, if laya { &Limits::LAYA } else { &Limits::JEV }) {
        Ok(r) => r,
        Err(p) if laya => return detail(StatusCode::UNPROCESSABLE_ENTITY, messages(&p)),
        Err(p) => return invalid(&p),
    };
    let o = match opts(body.get("kime")) {
        Ok(o) => o,
        Err(p) => return invalid(&p),
    };
    // Laya cuts the state and answers whatever the size, so only Jev requests are counted.
    if !laya && counted(&s, bytes) {
        let n;
        (n, req) = match count(&s, resolved.at, vec![req]).await {
            Ok((n, mut r)) => (n[0], r.remove(0)),
            Err(r) => return r,
        };
        if n > s.max_request_tokens {
            return too_many_tokens(n, s.max_request_tokens);
        }
    }
    let mut done = match s.models.decide(resolved.at, vec![req], o.deadline).await {
        Ok(d) => d,
        Err(r) => return refused(r, o.deadline),
    };
    let res = match done.results.pop() {
        Some(Ok(r)) => r,
        Some(Err(e)) => return engine_error(&e, laya, &id),
        None => return internal(&Error::Unsupported("no answer".into()), &id),
    };
    let total = start.elapsed();
    let out = if laya {
        let mut v = res.to_json();
        if let Some(m) = v.as_object_mut() {
            m.insert("routing".into(), s.models.routing(&resolved));
        }
        v
    } else {
        let extra = o.extensions.then(|| {
            json!({
                "request_id": id.0,
                "timing_us": {
                    "queue": us(done.queue),
                    "tokenize": us(done.pass.tokenize),
                    "device": us(done.pass.device),
                    "total": us(total),
                },
                "routing": s.models.routing(&resolved),
                "answers": extras(&res, o),
            })
        });
        jev(&s.models.list[resolved.at].id, &res, o, extra)
    };
    let mut r = reply(StatusCode::OK, &out);
    timing(r.headers_mut(), &done, total);
    r.extensions_mut().insert(Served {
        model: resolved.at,
        questions: res.answers.len(),
        tokens: res.input_tokens as u64,
    });
    r
}

/// `server-timing`: the wait for the device, the tokenize and device time of the forward pass
/// that answered, which other requests may have shared, and the whole request.
fn timing(h: &mut HeaderMap, done: &Done, total: Duration) {
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let v = format!(
        "queue;dur={:.3}, tokenize;dur={:.3}, device;dur={:.3}, total;dur={:.3}, pass;desc={}",
        ms(done.queue),
        ms(done.pass.tokenize),
        ms(done.pass.device),
        ms(total),
        done.shared
    );
    if let Ok(v) = HeaderValue::from_str(&v) {
        h.insert(HeaderName::from_static("server-timing"), v);
    }
}

async fn batch(
    Extension(s): Extension<Shared>,
    Extension(id): Extension<RequestId>,
    body: Body,
) -> HttpResponse {
    let start = Instant::now();
    let (body, bytes) = match object(body, s.max_body).await {
        Ok(b) => b,
        Err(r) => return r,
    };
    let resolved = match resolve(&s, &body) {
        Ok(r) => r,
        Err(r) => return r,
    };
    let o = match opts(body.get("kime")) {
        Ok(o) => o,
        Err(p) => return invalid(&p),
    };
    let items = match body.get("items").and_then(Value::as_array) {
        Some(items) if (1..=MAX_BATCH_ITEMS).contains(&items.len()) => items,
        other => {
            return invalid(&[Problem {
                loc: vec!["body".into(), "items".into()],
                msg: format!("items must be an array of 1 to {MAX_BATCH_ITEMS} requests"),
                kind: "value_error",
                input: other.map_or(Value::Null, |v| Value::from(v.len())),
            }]);
        }
    };
    // Each item is parsed alone and fails alone. The good ones share the forward passes.
    let mut ids = Vec::with_capacity(items.len());
    let mut slots: Vec<Result<usize, Value>> = Vec::with_capacity(items.len());
    let mut good = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let item_id = match item.get("id") {
            Some(Value::String(s)) => Value::String(s.clone()),
            Some(v @ Value::Number(_)) => v.clone(),
            _ => Value::String(i.to_string()),
        };
        ids.push(item_id);
        let parsed = match item {
            Value::Object(m) => {
                let mut one = m.clone();
                one.remove("id");
                parse(&Value::Object(one), &Limits::JEV)
            }
            _ => parse(item, &Limits::JEV),
        };
        slots.push(match parsed {
            Ok(r) => {
                good.push(r);
                Ok(good.len() - 1)
            }
            Err(p) => Err(json!({"status": 422, "detail": p.iter().map(Problem::to_json).collect::<Vec<_>>()})),
        });
    }
    if counted(&s, bytes) {
        let (n, all) = match count(&s, resolved.at, good).await {
            Ok(v) => v,
            Err(r) => return r,
        };
        // Items over the limit fail alone, like items that do not parse.
        good = Vec::with_capacity(all.len());
        let mut at = vec![None; all.len()];
        for (k, r) in all.into_iter().enumerate() {
            if n[k] <= s.max_request_tokens {
                at[k] = Some(good.len());
                good.push(r);
            }
        }
        for slot in &mut slots {
            if let Ok(k) = *slot {
                *slot = at[k].ok_or_else(|| {
                    let mut e = too_long_body(n[k], s.max_request_tokens);
                    e["status"] = 413.into();
                    e
                });
            }
        }
    }
    let mut done = match s.models.decide(resolved.at, good, None).await {
        Ok(d) => d,
        Err(r) => return refused(r, None),
    };
    let mut answered: Vec<Option<Result<Response, Error>>> =
        std::mem::take(&mut done.results).into_iter().map(Some).collect();
    let model = &s.models.list[resolved.at].id;
    let (mut tokens, mut questions) = (0, 0);
    let results: Vec<Value> = ids
        .into_iter()
        .zip(slots)
        .map(|(item_id, slot)| match slot {
            Err(e) => json!({"id": item_id, "error": e}),
            Ok(k) => match answered[k].take() {
                Some(Ok(r)) => {
                    tokens += r.input_tokens;
                    questions += r.answers.len();
                    let mut v = jev(model, &r, o, None);
                    if let Some(m) = v.as_object_mut() {
                        m.shift_remove("model");
                        let mut out = Map::new();
                        out.insert("id".into(), item_id);
                        out.extend(std::mem::take(m));
                        *m = out;
                    }
                    v
                }
                Some(Err(e)) => {
                    let r = engine_error(&e, false, &id);
                    json!({"id": item_id, "error": {"status": r.status().as_u16(), "message": e.to_string()}})
                }
                None => json!({"id": item_id, "error": {"status": 500, "message": "no answer"}}),
            },
        })
        .collect();
    let mut r = reply(
        StatusCode::OK,
        &json!({"model": model, "results": results, "usage": {"input_tokens": tokens, "output_tokens": 0}}),
    );
    timing(r.headers_mut(), &done, start.elapsed());
    r.extensions_mut().insert(Served { model: resolved.at, questions, tokens: tokens as u64 });
    r
}

fn entry(name: &str, description: String) -> Value {
    json!({"name": name, "description": description, "release_date": null})
}

fn model_entries(s: &State) -> Vec<Value> {
    let default = &s.models.list[0].id;
    let mut out = vec![entry("kime-latest", format!("Default model. Served by {default}."))];
    for m in &s.models.list {
        out.push(entry(&m.id, format!("{} on {}.", m.id, m.device)));
    }
    if s.models.jev_aliases {
        out.push(entry("jev-latest", "Alias. Served by kime-latest.".into()));
    }
    out
}

async fn list_models(Extension(s): Extension<Shared>) -> HttpResponse {
    reply(StatusCode::OK, &json!({"models": model_entries(&s)}))
}

async fn one_model(Extension(s): Extension<Shared>, Path(name): Path<String>) -> HttpResponse {
    let Some(r) = s.models.resolve(Some(&name)) else {
        return detail(StatusCode::NOT_FOUND, format!("model '{name}' not found"));
    };
    let id = &s.models.list[r.at].id;
    let e = model_entries(&s).into_iter().find(|e| e["name"] == name.as_str());
    reply(StatusCode::OK, &e.unwrap_or_else(|| entry(&name, format!("Alias. Served by {id}."))))
}

async fn health(Extension(s): Extension<Shared>) -> HttpResponse {
    let loaded: Vec<Value> = s
        .models
        .list
        .iter()
        .map(|m| match m.id.as_str() {
            "laya" => json!("english"),
            "laya-multilingual" => json!("multilingual"),
            "laya-typed-decisions" => json!("typed-decisions"),
            id => json!(id),
        })
        .collect();
    reply(
        StatusCode::OK,
        &json!({"status": "ok", "loaded": loaded, "device": s.models.list[0].device}),
    )
}

async fn metrics(Extension(s): Extension<Shared>) -> HttpResponse {
    let views: Vec<ModelView<'_>> = s.models.list.iter().map(|m| m.view()).collect();
    let mut r = s.metrics.render(&views).into_response();
    r.headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4"));
    r
}

async fn ready(Extension(s): Extension<Shared>) -> HttpResponse {
    let models: Vec<Value> =
        s.models.list.iter().map(|m| json!({"id": m.id, "device": m.device})).collect();
    reply(
        StatusCode::OK,
        &json!({"status": "ready", "version": env!("CARGO_PKG_VERSION"), "models": models}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        let a = new_id();
        assert_eq!(a.len(), 36);
        assert!(a.starts_with("req_") && a[4..].bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(a, new_id());
        assert!(echoable("abc_DEF-123"));
        assert!(!echoable(""));
        assert!(!echoable("a b"));
        assert!(!echoable(&"x".repeat(65)));
    }

    #[test]
    fn options() {
        let o = opts(None).unwrap();
        assert_eq!(o.precision, Some(2));
        let o = opts(Some(&json!({"precision": null, "confidence": "entropy"}))).unwrap();
        assert_eq!(o.precision, None);
        assert!(o.entropy);
        let bad =
            opts(Some(&json!({"precision": 7, "confidence": "x", "extensions": 1}))).unwrap_err();
        assert_eq!(bad.len(), 3);
    }
}
