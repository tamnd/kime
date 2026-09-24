//! The loaded models, their aliases, and the worker thread each one answers on.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use kime_core::answer::Response;
use kime_core::request::Request;
use kime_engine::{Error, Kime, Timing};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use crate::metrics::{ModelView, Pass, Stats};

/// The Hugging Face repo and Laya's router key for each published checkpoint, for the `routing`
/// block laya-serve adds.
const LAYA_KEYS: &[(&str, &str, &str)] = &[
    ("laya", "english", "convaiinnovations/laya"),
    ("laya-multilingual", "multilingual", "convaiinnovations/laya/multilingual"),
    ("laya-typed-decisions", "typed-decisions", "convaiinnovations/laya/typed-decisions"),
];

/// How a request named its model, which decides the response shape and the routing reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Named {
    /// No model, or `convaiinnovations/laya`, which laya-serve takes to mean "you choose".
    Default,
    /// A Laya checkpoint name, normalized to Laya's router key (`english`, `multilingual`,
    /// `typed-decisions`).
    Laya(String),
    /// A kime or Jev name, `kime-latest` or `jev-*`, or a concrete id.
    Kime,
}

/// A request's model after alias resolution.
#[derive(Debug, Clone)]
pub(crate) struct Resolved {
    /// Index into [`Models::list`].
    pub(crate) at: usize,
    pub(crate) named: Named,
    /// Whether the name leaves the choice to the server: no model, `convaiinnovations/laya`,
    /// `kime-latest` or a Jev alias.
    pub(crate) auto: bool,
    /// Why the router picked the model, when it did.
    pub(crate) route: Option<Route>,
}

/// The router's reason and Laya's `analyse` of the state, for the `routing` block.
#[derive(Debug, Clone)]
pub(crate) struct Route {
    pub(crate) reason: String,
    pub(crate) detection: Value,
}

impl Resolved {
    fn new(at: usize, named: Named, auto: bool) -> Self {
        Resolved { at, named, auto, route: None }
    }
}

/// One loaded model and the queue into its worker.
#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) id: String,
    pub(crate) device: String,
    /// The engine, for work outside the queue such as counting a big request's tokens.
    pub(crate) kime: Kime,
    queue: mpsc::Sender<Job>,
    load: Arc<Load>,
    stats: Arc<Stats>,
}

impl Model {
    /// Its name and live numbers for `/metrics`.
    pub(crate) fn view(&self) -> ModelView<'_> {
        ModelView {
            id: &self.id,
            stats: &self.stats,
            pending: self.load.pending.load(Ordering::Relaxed),
            per_request: Duration::from_nanos(self.load.per_request.load(Ordering::Relaxed)),
            memory: self.kime.memory(),
        }
    }
}

/// What a model's queue holds and how fast it drains, for the overload check.
#[derive(Debug, Default)]
struct Load {
    /// Requests queued or running, not yet answered.
    pending: AtomicUsize,
    /// Device time per request over the recent forward passes, in nanoseconds.
    per_request: AtomicU64,
    /// When the last forward pass ended, in nanoseconds since [`epoch`].
    last_pass: AtomicU64,
}

/// The clock `Load::last_pass` counts from.
fn epoch() -> Instant {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn since_epoch() -> u64 {
    u64::try_from(epoch().elapsed().as_nanos()).unwrap_or(u64::MAX)
}

impl Load {
    /// How long a request queued now would wait for the ones ahead of it.
    /// Whether a request should run whatever the estimate says. Nothing runs while every request
    /// is refused, so an estimate a burst left high would never come down. Once the device has
    /// sat idle for longer than the estimate, one request goes through and refreshes it.
    fn probe(&self) -> bool {
        let per = self.per_request.load(Ordering::Relaxed);
        self.pending.load(Ordering::Relaxed) == 0
            && since_epoch().saturating_sub(self.last_pass.load(Ordering::Relaxed)) > per
    }

    fn wait(&self) -> Duration {
        self.ready(0)
    }

    /// How long until `n` requests queued now are answered: the wait plus their own share.
    fn ready(&self, n: usize) -> Duration {
        let n = (self.pending.load(Ordering::Relaxed) + n) as u64;
        Duration::from_nanos(n.saturating_mul(self.per_request.load(Ordering::Relaxed)))
    }

    /// Folds one forward pass into the average, one eighth at a time so a single slow pass does
    /// not swing it.
    fn record(&self, took: Duration, requests: usize) {
        let now = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX) / requests.max(1) as u64;
        let old = self.per_request.load(Ordering::Relaxed);
        let new = if old == 0 { now } else { old - old / 8 + now / 8 };
        self.per_request.store(new, Ordering::Relaxed);
        self.last_pass.store(since_epoch(), Ordering::Relaxed);
    }
}

/// Faults to inject into the model workers, for the chaos tests. Each count is the forward passes
/// still to fail that way, and one is taken per pass.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct Faults {
    /// Passes whose batch fails with a device error, so each request runs again alone.
    pub error: AtomicUsize,
    /// Passes that panic inside the engine.
    pub panic: AtomicUsize,
    /// Passes that kill the worker thread with the requests in them.
    pub kill: AtomicUsize,
}

fn fault(n: &AtomicUsize) -> bool {
    n.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1)).is_ok()
}

/// The loaded models. The first one is the default.
#[derive(Debug)]
pub(crate) struct Models {
    pub(crate) list: Vec<Model>,
    pub(crate) jev_aliases: bool,
    /// The longest estimated wait a new request is queued behind. Zero turns the check off.
    max_queue: Duration,
    /// The most requests a model holds queued or running. Zero turns the check off.
    max_pending: usize,
}

/// What the worker hands back for one submission.
#[derive(Debug)]
pub(crate) struct Done {
    pub(crate) results: Vec<Result<Response, Error>>,
    /// Time between submission and the start of the forward pass that answered it.
    pub(crate) queue: Duration,
    /// Where the time of that forward pass went. It is shared with whatever else was in it.
    pub(crate) pass: Timing,
    /// Requests in that forward pass, this one included.
    pub(crate) shared: usize,
}

#[derive(Debug)]
struct Job {
    reqs: Vec<Request>,
    at: Instant,
    done: oneshot::Sender<Done>,
}

impl Models {
    /// Starts a worker thread for each engine. `max_batch` and `max_batch_tokens` cap the
    /// requests and the estimated tokens one forward pass takes from the queue, and `max_queue`
    /// and `max_pending` are the wait and the queue length
    /// past which requests are turned away. A worker that dies starts again, and the requests it
    /// held get an error.
    pub(crate) fn new(
        engines: Vec<Kime>,
        jev_aliases: bool,
        max_batch: usize,
        max_batch_tokens: usize,
        max_queue: Duration,
        max_pending: usize,
        faults: &Arc<Faults>,
    ) -> Self {
        let list = engines
            .into_iter()
            .map(|k| {
                let (tx, rx) = mpsc::channel();
                let (id, device) = (k.model_id().to_string(), k.device());
                let load = Arc::new(Load::default());
                let stats = Arc::new(Stats::default());
                let (l, st, kime, f) = (load.clone(), stats.clone(), k.clone(), faults.clone());
                let take = Take {
                    requests: max_batch.max(1),
                    tokens: max_batch_tokens,
                    row: k.max_row_tokens(),
                };
                std::thread::Builder::new()
                    .name(format!("kime-{id}"))
                    .spawn(move || {
                        let batch = |r: &[Request]| {
                            if fault(&f.error) {
                                return Err(Error::Unsupported("injected device error".into()));
                            }
                            assert!(!fault(&f.panic), "injected device panic");
                            k.decide_batch_timed(r)
                        };
                        // It returns when the server drops the queue, and panics only when the
                        // worker itself dies, never for a failed pass.
                        while catch_unwind(AssertUnwindSafe(|| {
                            worker(&rx, take, &l, &st, &f.kill, batch, |r| k.decide(r));
                        }))
                        .is_err()
                        {
                            st.restarts.fetch_add(1, Ordering::Relaxed);
                        }
                    })
                    .expect("spawning a worker thread");
                Model { id, device, kime, queue: tx, load, stats }
            })
            .collect();
        Models { list, jev_aliases, max_queue, max_pending }
    }

    /// Resolves a request's `model` field, or `None` for a name no loaded model answers to.
    pub(crate) fn resolve(&self, name: Option<&str>) -> Option<Resolved> {
        let Some(raw) = name else { return Some(Resolved::new(0, Named::Default, true)) };
        let n = raw.trim().to_ascii_lowercase();
        let find = |id: &str| self.list.iter().position(|m| m.id == id);
        let laya = |key: &str, id: &str| {
            find(id).map(|at| Resolved::new(at, Named::Laya(key.to_string()), false))
        };
        match n.as_str() {
            "" | "convaiinnovations/laya" => Some(Resolved::new(0, Named::Default, true)),
            "laya" | "english" | "en" | "default" => laya("english", "laya"),
            "laya-multilingual"
            | "multilingual"
            | "multi"
            | "ml"
            | "convaiinnovations/laya-multilingual" => laya("multilingual", "laya-multilingual"),
            "laya-typed-decisions"
            | "typed-decisions"
            | "typed"
            | "typed_decisions"
            | "decisions"
            | "convaiinnovations/laya-typed-decisions" => {
                laya("typed-decisions", "laya-typed-decisions")
            }
            "kime-latest" => Some(Resolved::new(0, Named::Kime, true)),
            n if self.jev_aliases && (n == "jev" || n.starts_with("jev-")) => {
                Some(Resolved::new(0, Named::Kime, true))
            }
            n => find(n).map(|at| Resolved::new(at, Named::Kime, false)),
        }
    }

    /// The `routing` block laya-serve puts on every response.
    pub(crate) fn routing(&self, r: &Resolved) -> Value {
        let id = &self.list[r.at].id;
        let (key, repo) = LAYA_KEYS
            .iter()
            .find(|(i, _, _)| i == id)
            .map_or((id.as_str(), id.as_str()), |(_, k, repo)| (*k, *repo));
        let (reason, detection) = match (&r.named, &r.route) {
            (Named::Laya(k), _) => (format!("explicit model='{k}'"), Value::Null),
            (_, Some(route)) => (route.reason.clone(), route.detection.clone()),
            _ => (format!("using default ({key})"), Value::Null),
        };
        json!({"model": key, "repo": repo, "reason": reason, "detection": detection, "workflow": null})
    }

    /// Picks the English or the multilingual checkpoint for a request that leaves the choice to
    /// the server, when both are loaded. `kime.route.lang` and `kime.route.lang_guess` are Laya's
    /// `lang` and `lang_guess`, and detection comes after them.
    pub(crate) fn route(&self, r: &mut Resolved, req: &Request) {
        if !r.auto {
            return;
        }
        let find = |id: &str| self.list.iter().position(|m| m.id == id);
        let (Some(en), Some(ml)) = (find("laya"), find("laya-multilingual")) else { return };
        let hints = req.kime.as_ref().and_then(|k| k.get("route"));
        let hint = |k: &str| hints.and_then(|h| h.get(k));
        let d = kime_route::router::route(&req.state, hint("lang"), hint("lang_guess"), en == 0);
        r.at = match d.english {
            Some(true) => en,
            Some(false) => ml,
            None => 0,
        };
        self.list[r.at].stats.routed(d.by);
        let detection = d.detection.map_or(Value::Null, |a| a.to_json());
        r.route = Some(Route { reason: d.reason, detection });
    }

    /// Queues requests on a model's worker and waits for their answers without blocking the
    /// async runtime. It queues nothing when the requests already queued would keep these waiting
    /// longer than `max_queue` or would pass `max_pending` with these, or when the answers would
    /// be ready after `deadline`. A request bigger than `max_pending` alone still goes in when
    /// nothing else is queued.
    pub(crate) async fn decide(
        &self,
        at: usize,
        reqs: Vec<Request>,
        deadline: Option<Duration>,
    ) -> Result<Done, Refused> {
        let m = &self.list[at];
        let n = reqs.len();
        let wait = m.load.wait();
        if !self.max_queue.is_zero() && wait > self.max_queue {
            return Err(Refused::Overloaded(wait));
        }
        let ready = m.load.ready(n);
        if deadline.is_some_and(|d| ready > d) && !m.load.probe() {
            return Err(Refused::Deadline(ready));
        }
        let (tx, rx) = oneshot::channel();
        let max = self.max_pending;
        let admit = |p: usize| (max == 0 || p == 0 || p + n <= max).then_some(p + n);
        if m.load.pending.fetch_update(Ordering::Relaxed, Ordering::Relaxed, admit).is_err() {
            return Err(Refused::Overloaded(wait));
        }
        let job = Job { reqs, at: Instant::now(), done: tx };
        if m.queue.send(job).is_err() {
            m.load.pending.fetch_sub(n, Ordering::Relaxed);
            return Ok(gone(n));
        }
        let done = rx.await.unwrap_or_else(|_| gone(n));
        m.stats.queue.time(done.queue);
        Ok(done)
    }
}

/// Why requests were turned away before they were queued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The queue ahead would take longer than `max_queue`. The estimated wait.
    Overloaded(Duration),
    /// The answers would be ready after the deadline. The estimate of when.
    Deadline(Duration),
}

fn gone(n: usize) -> Done {
    let results = (0..n)
        .map(|_| Err(Error::Unsupported("the model's worker stopped with this request".into())))
        .collect();
    Done { results, queue: Duration::ZERO, pass: Timing::default(), shared: 0 }
}

/// How much one forward pass takes from the queue: at most `requests` requests and, when `tokens`
/// is not zero, about `tokens` tokens. A job that alone is over either still runs, on its own.
#[derive(Debug, Clone, Copy)]
struct Take {
    requests: usize,
    tokens: usize,
    /// The model's longest row, which no question's row goes past.
    row: usize,
}

impl Take {
    /// The tokens a request puts through the device, guessed from its size so it need not be
    /// tokenized twice: a row per question of a third of the bytes of the state and the question,
    /// up to the model's row. English prose runs about four bytes a token, so the guess is on the
    /// high side and a pass stays under its budget.
    fn tokens(&self, req: &Request) -> usize {
        let nq = req.questions.len().max(1);
        let state = req.state.to_string().len();
        let questions = req.to_json().to_string().len().saturating_sub(state);
        nq * ((state + questions / nq) / 3).clamp(1, self.row.max(1))
    }

    fn full(&self, requests: usize, tokens: usize) -> bool {
        requests >= self.requests || (self.tokens > 0 && tokens >= self.tokens)
    }
}

/// A job taken from the queue for the next pass, which leaves the count if the worker dies first.
struct Carry<'a> {
    job: Option<Job>,
    load: &'a Load,
}

impl Drop for Carry<'_> {
    fn drop(&mut self) {
        if let Some(j) = &self.job {
            self.load.pending.fetch_sub(j.reqs.len(), Ordering::Relaxed);
        }
    }
}

type Batch<'a> = &'a dyn Fn(&[Request]) -> Result<(Vec<Response>, Timing), Error>;

/// Answers jobs until the server drops the queue. Whatever has queued up while the last forward
/// pass ran goes into the next one together, up to the limits in `take`, which is where concurrent
/// requests share the device. A job that would go over them waits for the pass after.
/// `batch` answers a whole pass and `one` a single request. A panic in either becomes an error
/// for the requests of that pass, and the worker goes on with the next. `kill` is the passes left
/// that panic out of the worker, for the chaos tests.
fn worker(
    rx: &mpsc::Receiver<Job>,
    take: Take,
    load: &Load,
    stats: &Stats,
    kill: &AtomicUsize,
    batch: impl Fn(&[Request]) -> Result<(Vec<Response>, Timing), Error>,
    one: impl Fn(&Request) -> Result<Response, Error>,
) {
    let mut carry = Carry { job: None, load };
    loop {
        let first = match carry.job.take() {
            Some(j) => j,
            None => match rx.recv() {
                Ok(j) => j,
                Err(_) => break,
            },
        };
        // A job whose caller has gone, a client that timed out or hung up, is dropped unanswered.
        // Running it would only keep the ones still waiting waiting longer.
        let (mut jobs, mut n, mut tokens, mut gone) = (Vec::new(), 0, 0, 0);
        let mut next = Some(first);
        while let Some(j) = next.take() {
            if j.done.is_closed() {
                gone += j.reqs.len();
            } else {
                let t: usize = j.reqs.iter().map(|r| take.tokens(r)).sum();
                if !jobs.is_empty() && take.full(n + j.reqs.len(), tokens + t) {
                    carry.job = Some(j);
                    break;
                }
                n += j.reqs.len();
                tokens += t;
                jobs.push(j);
            }
            if !take.full(n, tokens) {
                next = rx.try_recv().ok();
            }
        }
        if gone > 0 {
            load.pending.fetch_sub(gone, Ordering::Relaxed);
            stats.abandoned.fetch_add(gone as u64, Ordering::Relaxed);
        }
        if jobs.is_empty() {
            continue;
        }
        // However the pass ends, even with the worker dying, these leave the count.
        let held = Held { load, n };
        assert!(!fault(kill), "injected worker death");
        let start = Instant::now();
        let counts: Vec<usize> = jobs.iter().map(|j| j.reqs.len()).collect();
        let reqs: Vec<Request> =
            jobs.iter_mut().flat_map(|j| std::mem::take(&mut j.reqs)).collect();
        let (mut results, pass) = answer(&reqs, &batch, &one);
        load.record(start.elapsed(), reqs.len());
        let answered = results.iter().flatten();
        stats.pass(Pass {
            requests: reqs.len(),
            questions: answered.clone().map(|r| r.answers.len()).sum(),
            tokens: answered.map(|r| r.input_tokens).sum(),
            batches: pass.batches,
            truncated: pass.truncated,
            cut_tokens: pass.cut_tokens,
            tokenize: pass.tokenize,
            device: pass.device,
        });
        drop(held);
        for (job, k) in jobs.into_iter().zip(counts) {
            let rest = results.split_off(k);
            let mine = std::mem::replace(&mut results, rest);
            let queue = start.saturating_duration_since(job.at);
            // The client may have gone away, and then nobody needs the answer.
            let _ = job.done.send(Done { results: mine, queue, pass, shared: reqs.len() });
        }
    }
}

/// Requests a pass took from the queue, taken off [`Load::pending`] when it is dropped.
struct Held<'a> {
    load: &'a Load,
    n: usize,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.load.pending.fetch_sub(self.n, Ordering::Relaxed);
    }
}

/// One forward pass for `reqs`, with one result per request whatever goes wrong.
fn answer(
    reqs: &[Request],
    batch: Batch<'_>,
    one: &dyn Fn(&Request) -> Result<Response, Error>,
) -> (Vec<Result<Response, Error>>, Timing) {
    let panicked = || Error::Unsupported("the device worker panicked on this batch".into());
    // One bad request fails a whole engine batch, so on an error each request runs alone and gets
    // its own result. The answers are the same bits either way.
    match catch_unwind(AssertUnwindSafe(|| batch(reqs))) {
        Ok(Ok((r, t))) => (r.into_iter().map(Ok).collect(), t),
        Ok(Err(_)) => {
            let each = reqs
                .iter()
                .map(|r| {
                    catch_unwind(AssertUnwindSafe(|| one(r))).unwrap_or_else(|_| Err(panicked()))
                })
                .collect();
            (each, Timing::default())
        }
        Err(_) => (reqs.iter().map(|_| Err(panicked())).collect(), Timing::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kime_core::request::{Limits, parse};
    use std::sync::atomic::AtomicBool;

    fn req(state: &str) -> Request {
        let body = json!({"state": state, "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": "a", "b": "b"}}}});
        parse(&body, &Limits::LAYA).unwrap()
    }

    fn resp(r: &Request) -> Response {
        Response {
            model: r.state.as_str().unwrap_or_default().into(),
            answers: Vec::new(),
            input_tokens: 1,
        }
    }

    fn limits(requests: usize) -> Take {
        Take { requests, tokens: 0, row: 512 }
    }

    fn submit(tx: &mpsc::Sender<Job>, load: &Load, state: &str) -> oneshot::Receiver<Done> {
        let (done, rx) = oneshot::channel();
        load.pending.fetch_add(1, Ordering::Relaxed);
        tx.send(Job { reqs: vec![req(state)], at: Instant::now(), done }).unwrap();
        rx
    }

    #[test]
    fn a_panic_fails_its_pass_and_the_worker_goes_on() {
        let (tx, rx) = mpsc::channel();
        let load = Arc::new(Load::default());
        let l = load.clone();
        let fired = Arc::new(AtomicBool::new(false));
        let f = fired.clone();
        let t = std::thread::spawn(move || {
            let batch = |r: &[Request]| {
                if r.iter().any(|r| r.state == "boom") && !f.swap(true, Ordering::SeqCst) {
                    panic!("injected device fault");
                }
                Ok((r.iter().map(resp).collect(), Timing::default()))
            };
            worker(&rx, limits(1), &l, &Stats::default(), &AtomicUsize::new(0), batch, |r| {
                Ok(resp(r))
            });
        });
        let first = submit(&tx, &load, "boom");
        let d = first.blocking_recv().unwrap();
        let err = d.results[0].as_ref().unwrap_err().to_string();
        assert!(err.contains("panicked"), "{err}");
        // The same request again, and another, both answered by the same worker.
        for s in ["boom", "fine"] {
            let d = submit(&tx, &load, s).blocking_recv().unwrap();
            assert_eq!(d.results[0].as_ref().unwrap().model, s);
        }
        assert_eq!(load.pending.load(Ordering::Relaxed), 0);
        drop(tx);
        t.join().unwrap();
    }

    #[test]
    fn a_job_nobody_waits_for_is_not_run() {
        let (tx, rx) = mpsc::channel();
        let load = Arc::new(Load::default());
        let stats = Arc::new(Stats::default());
        let (l, st) = (load.clone(), stats.clone());
        let (go, wait) = mpsc::channel::<()>();
        let (seen_tx, seen) = mpsc::channel::<Vec<String>>();
        let t = std::thread::spawn(move || {
            let batch = |r: &[Request]| {
                let states = r.iter().map(|r| r.state.as_str().unwrap().to_string()).collect();
                seen_tx.send(states).unwrap();
                if r[0].state == "first" {
                    wait.recv().unwrap();
                }
                Ok((r.iter().map(resp).collect(), Timing::default()))
            };
            worker(&rx, limits(8), &l, &st, &AtomicUsize::new(0), batch, |r| Ok(resp(r)));
        });
        let first = submit(&tx, &load, "first");
        assert_eq!(seen.recv().unwrap(), ["first"]);
        // Queued while the first pass runs: one caller gives up, the other waits.
        drop(submit(&tx, &load, "gone"));
        let kept = submit(&tx, &load, "kept");
        go.send(()).unwrap();
        assert_eq!(seen.recv().unwrap(), ["kept"]);
        assert_eq!(kept.blocking_recv().unwrap().results[0].as_ref().unwrap().model, "kept");
        first.blocking_recv().unwrap();
        assert_eq!(load.pending.load(Ordering::Relaxed), 0);
        assert_eq!(stats.abandoned.load(Ordering::Relaxed), 1);
        drop(tx);
        t.join().unwrap();
    }

    #[test]
    fn a_pass_keeps_to_its_token_budget() {
        let t = Take { requests: 64, tokens: 300, row: 512 };
        let long = "x".repeat(300);
        let each = t.tokens(&req(&long));
        assert!((100..300).contains(&each), "{each}");
        assert_eq!(t.tokens(&req(&"x".repeat(5000))), 512, "one row is at most the model's");
        let (tx, rx) = mpsc::channel();
        let load = Arc::new(Load::default());
        let l = load.clone();
        let (go, wait) = mpsc::channel::<()>();
        let (seen_tx, seen) = mpsc::channel::<usize>();
        let th = std::thread::spawn(move || {
            let batch = |r: &[Request]| {
                seen_tx.send(r.len()).unwrap();
                if r[0].state == "first" {
                    wait.recv().unwrap();
                }
                Ok((r.iter().map(resp).collect(), Timing::default()))
            };
            worker(&rx, t, &l, &Stats::default(), &AtomicUsize::new(0), batch, |r| Ok(resp(r)));
        });
        let first = submit(&tx, &load, "first");
        assert_eq!(seen.recv().unwrap(), 1);
        let rest: Vec<_> = (0..5).map(|_| submit(&tx, &load, &long)).collect();
        go.send(()).unwrap();
        // Two of about 130 tokens fit 300, so five go as two, two and one.
        let passes: Vec<usize> = (0..3).map(|_| seen.recv().unwrap()).collect();
        assert_eq!(passes, [2, 2, 1]);
        first.blocking_recv().unwrap();
        for r in rest {
            assert!(r.blocking_recv().unwrap().results[0].is_ok());
        }
        assert_eq!(load.pending.load(Ordering::Relaxed), 0);
        drop(tx);
        th.join().unwrap();
    }

    #[test]
    fn a_failed_batch_answers_each_request_alone() {
        let reqs = vec![req("good"), req("bad"), req("good")];
        let batch = |_: &[Request]| Err(Error::Unsupported("one bad request".into()));
        let one = |r: &Request| {
            if r.state == "bad" { Err(Error::Unsupported("bad".into())) } else { Ok(resp(r)) }
        };
        let (out, _) = answer(&reqs, &batch, &one);
        assert!(out[0].is_ok() && out[1].is_err() && out[2].is_ok());
    }

    #[test]
    fn wait_estimate() {
        let l = Load::default();
        assert_eq!(l.wait(), Duration::ZERO);
        l.record(Duration::from_millis(80), 40);
        l.pending.store(300, Ordering::Relaxed);
        assert_eq!(l.wait(), Duration::from_millis(600));
        assert_eq!(l.ready(10), Duration::from_millis(620));
        // Just after a pass, with work queued, there is no probe.
        assert!(!l.probe());
        // A slow pass moves the average an eighth of the way.
        l.record(Duration::from_millis(10), 1);
        assert_eq!(l.wait(), Duration::from_nanos(300 * (2_000_000 - 250_000 + 1_250_000)));
    }

    #[test]
    fn an_idle_device_probes_once_the_estimate_has_passed() {
        let l = Load::default();
        l.record(Duration::from_secs(10), 1);
        assert!(!l.probe(), "idle for less than the 10 s estimate");
        l.per_request.store(1_000, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(2));
        assert!(l.probe(), "idle for longer than the 1 us estimate");
        l.pending.store(1, Ordering::Relaxed);
        assert!(!l.probe(), "busy");
    }
}
