//! The loaded models, their aliases, and the worker thread each one answers on.

use std::sync::mpsc;
use std::time::{Duration, Instant};

use kime_core::answer::Response;
use kime_core::request::Request;
use kime_engine::{Error, Kime};
use serde_json::{Value, json};
use tokio::sync::oneshot;

/// The Hugging Face repo and Laya's router key for each published checkpoint, for the `routing`
/// block laya-serve adds.
const LAYA_KEYS: &[(&str, &str, &str)] = &[
    ("laya", "english", "convaiinnovations/laya"),
    ("laya-multilingual", "multilingual", "convaiinnovations/laya/multilingual"),
];

/// How a request named its model, which decides the response shape and the routing reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Named {
    /// No model, or `convaiinnovations/laya`, which laya-serve takes to mean "you choose".
    Default,
    /// A Laya checkpoint name, normalized to Laya's router key (`english`, `multilingual`).
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
}

/// One loaded model and the queue into its worker.
#[derive(Debug)]
pub(crate) struct Model {
    pub(crate) id: String,
    pub(crate) device: String,
    queue: mpsc::Sender<Job>,
}

/// The loaded models. The first one is the default.
#[derive(Debug)]
pub(crate) struct Models {
    pub(crate) list: Vec<Model>,
    pub(crate) jev_aliases: bool,
}

/// What the worker hands back for one submission.
#[derive(Debug)]
pub(crate) struct Done {
    pub(crate) results: Vec<Result<Response, Error>>,
    /// Time between submission and the start of the forward pass that answered it.
    pub(crate) queue: Duration,
}

#[derive(Debug)]
struct Job {
    reqs: Vec<Request>,
    at: Instant,
    done: oneshot::Sender<Done>,
}

impl Models {
    /// Starts a worker thread for each engine. `max_batch` caps the requests one forward pass
    /// takes from the queue.
    pub(crate) fn new(engines: Vec<Kime>, jev_aliases: bool, max_batch: usize) -> Self {
        let list = engines
            .into_iter()
            .map(|k| {
                let (tx, rx) = mpsc::channel();
                let (id, device) = (k.model_id().to_string(), k.device());
                std::thread::Builder::new()
                    .name(format!("kime-{id}"))
                    .spawn(move || worker(&k, &rx, max_batch.max(1)))
                    .expect("spawning a worker thread");
                Model { id, device, queue: tx }
            })
            .collect();
        Models { list, jev_aliases }
    }

    /// Resolves a request's `model` field, or `None` for a name no loaded model answers to.
    pub(crate) fn resolve(&self, name: Option<&str>) -> Option<Resolved> {
        let Some(raw) = name else { return Some(Resolved { at: 0, named: Named::Default }) };
        let n = raw.trim().to_ascii_lowercase();
        let find = |id: &str| self.list.iter().position(|m| m.id == id);
        let laya = |key: &str, id: &str| {
            find(id).map(|at| Resolved { at, named: Named::Laya(key.to_string()) })
        };
        match n.as_str() {
            "" | "convaiinnovations/laya" => Some(Resolved { at: 0, named: Named::Default }),
            "laya" | "english" | "en" | "default" => laya("english", "laya"),
            "laya-multilingual"
            | "multilingual"
            | "multi"
            | "ml"
            | "convaiinnovations/laya-multilingual" => laya("multilingual", "laya-multilingual"),
            "kime-latest" => Some(Resolved { at: 0, named: Named::Kime }),
            n if self.jev_aliases && (n == "jev" || n.starts_with("jev-")) => {
                Some(Resolved { at: 0, named: Named::Kime })
            }
            n => find(n).map(|at| Resolved { at, named: Named::Kime }),
        }
    }

    /// The `routing` block laya-serve puts on every response.
    pub(crate) fn routing(&self, r: &Resolved) -> Value {
        let id = &self.list[r.at].id;
        let (key, repo) = LAYA_KEYS
            .iter()
            .find(|(i, _, _)| i == id)
            .map_or((id.as_str(), id.as_str()), |(_, k, repo)| (*k, *repo));
        let reason = match &r.named {
            Named::Laya(k) => format!("explicit model='{k}'"),
            _ => format!("using default ({key})"),
        };
        json!({"model": key, "repo": repo, "reason": reason, "detection": null, "workflow": null})
    }

    /// Queues requests on a model's worker and waits for their answers without blocking the
    /// async runtime.
    pub(crate) async fn decide(&self, at: usize, reqs: Vec<Request>) -> Done {
        let (tx, rx) = oneshot::channel();
        let n = reqs.len();
        let job = Job { reqs, at: Instant::now(), done: tx };
        if self.list[at].queue.send(job).is_err() {
            return gone(n);
        }
        rx.await.unwrap_or_else(|_| gone(n))
    }
}

fn gone(n: usize) -> Done {
    let results = (0..n)
        .map(|_| Err(Error::Unsupported("the model's worker thread stopped".into())))
        .collect();
    Done { results, queue: Duration::ZERO }
}

/// Answers jobs until the server drops the queue. Whatever has queued up while the last forward
/// pass ran goes into the next one together, which is where concurrent requests share the device.
fn worker(kime: &Kime, rx: &mpsc::Receiver<Job>, max: usize) {
    while let Ok(first) = rx.recv() {
        let mut n = first.reqs.len();
        let mut jobs = vec![first];
        while n < max {
            match rx.try_recv() {
                Ok(j) => {
                    n += j.reqs.len();
                    jobs.push(j);
                }
                Err(_) => break,
            }
        }
        let start = Instant::now();
        let counts: Vec<usize> = jobs.iter().map(|j| j.reqs.len()).collect();
        let reqs: Vec<Request> =
            jobs.iter_mut().flat_map(|j| std::mem::take(&mut j.reqs)).collect();
        // One bad request fails a whole engine batch, so on an error each request runs alone and
        // gets its own result. The answers are the same bits either way.
        let mut results: Vec<Result<Response, Error>> = match kime.decide_batch(&reqs) {
            Ok(r) => r.into_iter().map(Ok).collect(),
            Err(_) => reqs.iter().map(|r| kime.decide(r)).collect(),
        };
        for (job, k) in jobs.into_iter().zip(counts) {
            let rest = results.split_off(k);
            let mine = std::mem::replace(&mut results, rest);
            let queue = start.saturating_duration_since(job.at);
            // The client may have gone away, and then nobody needs the answer.
            let _ = job.done.send(Done { results: mine, queue });
        }
    }
}
