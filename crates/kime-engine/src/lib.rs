//! Session, scheduler, batching, the state memory cache, the answer cache and the tokenization cache. This is the in process API that the server, the CLI and the language bindings all sit on. See spec/07-engine.md and spec/11-serving.md.
//!
//! What is here today is the compat path end to end: open a Laya checkpoint, lay each question out
//! as Laya does, run the questions of one or many requests in shared batches on the CPU or a CUDA
//! GPU, and build Laya's answers from the logits. The scheduler that merges requests from many
//! callers arrives with the server, so for now a call runs on the caller's thread and callers
//! share the device through a lock.
//!
//! The `kime` crate re-exports all of it, and its docs hold a full example.

#![forbid(unsafe_code)]

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use kime_core::answer::{LAYA_MODEL, Response, Temperatures, laya_answer};
use kime_core::render::{compat_question, compat_state};
use kime_core::request::{Limits, Problem, Question, Request, parse};
use kime_model::Model;
use kime_tensor::{BatchBuf, Buckets, Executor, Outputs};
use kime_tok::Tokenizer;
use kime_tok::layout::{CompatBudget, CompatSequence, Cut};
use serde_json::Value;

mod cache;
pub mod hub;
mod split;

use cache::{AnswerCache, Entry};
pub use cache::{CacheMode, CacheStats};

/// Where the model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// The first CUDA GPU when there is one and this build has CUDA, the CPU otherwise.
    #[default]
    Auto,
    /// The CPU, on `threads` threads, or every core for 0.
    Cpu {
        /// Worker threads.
        threads: usize,
    },
    /// A CUDA GPU by ordinal.
    Cuda(usize),
    /// The Apple GPU. Arrives with M4.
    Metal,
    /// The Apple Neural Engine. Arrives with M4.
    Ane,
}

/// The number format. The CPU computes in FP32 for both float settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// FP16 weights and GEMM inputs with FP32 accumulation, as Laya's autocast runs on a GPU.
    #[default]
    F16,
    /// FP32 throughout, closest to Laya on the CPU.
    F32,
    /// INT8 weights and activations in the encoder and decision head GEMMs on the CPU, with the
    /// scorer in FP32. Faster, and further from Laya than FP32: see spec/10-cpu.md for the gate a
    /// checkpoint has to pass. Not on GPUs yet.
    Int8,
}

/// What can go wrong.
#[derive(Debug)]
pub enum Error {
    /// The request does not validate. Every problem is listed, in the wire format.
    Invalid(Vec<Problem>),
    /// The model name did not resolve.
    NotFound(String),
    /// The checkpoint did not load.
    Model(kime_model::Error),
    /// The checkpoint's tokenizer did not load.
    Tokenizer(String),
    /// The device failed, or the batch did not fit it.
    Backend(kime_tensor::Error),
    /// A question's head and options do not fit the checkpoint's budget, so some options have no
    /// marker. Laya raises the same error.
    TooLong {
        /// The question id.
        question: String,
        /// Its options.
        options: usize,
        /// The options that fit.
        fit: usize,
    },
    /// The device asked for is not in this build or not on this machine.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(p) => {
                write!(f, "invalid request:")?;
                for p in p {
                    write!(f, " {};", p.to_json())?;
                }
                Ok(())
            }
            Error::NotFound(m) | Error::Tokenizer(m) | Error::Unsupported(m) => f.write_str(m),
            Error::Model(e) => write!(f, "{e}"),
            Error::Backend(e) => write!(f, "{e}"),
            Error::TooLong { question, options, fit } => write!(
                f,
                "question {question:?} has {options} options but only {fit} fit the head budget"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<kime_tensor::Error> for Error {
    fn from(e: kime_tensor::Error) -> Self {
        Error::Backend(e)
    }
}

/// Settings for [`Kime`], from [`Kime::builder`].
#[derive(Debug, Clone, Default)]
pub struct Builder {
    model: Option<String>,
    device: Device,
    precision: Precision,
    preload: bool,
    answer_cache: usize,
}

impl Builder {
    /// The model: an alias such as `laya`, a checkpoint directory, a `.kime` file or an
    /// `hf://org/repo[/subfolder]` reference. See [`hub`].
    #[must_use]
    pub fn model(mut self, name: impl Into<String>) -> Self {
        self.model = Some(name.into());
        self
    }

    /// Where to run.
    #[must_use]
    pub fn device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }

    /// The number format on a GPU.
    #[must_use]
    pub fn precision(mut self, p: Precision) -> Self {
        self.precision = p;
        self
    }

    /// Builds the plans for the smallest batch shapes now, so the first requests do not pay for
    /// them.
    #[must_use]
    pub fn preload(mut self, yes: bool) -> Self {
        self.preload = yes;
        self
    }

    /// Keeps the answers to about `entries` questions, so the same question on the same state
    /// is answered again without the device. 0, the default, keeps none.
    #[must_use]
    pub fn answer_cache(mut self, entries: usize) -> Self {
        self.answer_cache = entries;
        self
    }

    /// Loads the model onto the device.
    ///
    /// # Errors
    ///
    /// When the model is not found or does not load, or the device cannot be opened.
    pub fn build(self) -> Result<Kime, Error> {
        let name = self.model.unwrap_or_else(|| "laya".into());
        let path = hub::resolve(&name).map_err(Error::NotFound)?;
        let model = Model::open(&path).map_err(Error::Model)?;
        let tok_json = model
            .file("tokenizer/tokenizer.json")
            .ok_or_else(|| Error::Tokenizer(format!("{}: no tokenizer.json", path.display())))?;
        let tok = Tokenizer::from_bytes(tok_json, model.file("tokenizer/tokenizer_config.json"))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let agent = &model.spec.agent;
        let temps = Temperatures::new(agent.temperature, &agent.temperature_by_options);
        let budget = CompatBudget { max_len: agent.max_len, head_max_len: agent.head_max_len };
        let mut runner = Runner::open(&model, self.device, self.precision)?;
        let embed = runner.add_graph(model.graph.embed_plan(&model.spec));
        if self.preload {
            for b in Buckets::default().stage("compat").iter().take(4) {
                runner.prepare(*b)?;
            }
        }
        let buckets = Buckets::default().stage("compat").to_vec();
        let embed_buckets = Buckets::default().stage(EMBED_STAGE).to_vec();
        let (weights, plans) = runner.memory();
        Ok(Kime {
            inner: Arc::new(Inner {
                id: model.spec.id.clone(),
                mask: tok.mask_text().to_string(),
                tok,
                budget,
                temps,
                buckets,
                embed_buckets,
                d: model.spec.encoder.d,
                memory: [AtomicUsize::new(weights), AtomicUsize::new(plans)],
                cache: (self.answer_cache > 0).then(|| AnswerCache::new(self.answer_cache)),
                runner: Mutex::new(Session {
                    runner,
                    embed,
                    buf: BatchBuf::default(),
                    out: Outputs::default(),
                }),
            }),
        })
    }
}

enum Runner {
    Cpu(Box<Executor<kime_cpu::CpuBackend>>),
    #[cfg(feature = "cuda")]
    Cuda(Box<Executor<kime_cuda::CudaBackend>>),
}

impl Runner {
    fn open(model: &Model, device: Device, precision: Precision) -> Result<Self, Error> {
        let cpu = |threads: usize| {
            let t = if threads == 0 { kime_cpu::par::available() } else { threads };
            let backend = kime_cpu::CpuBackend::new(t).with_int8(precision == Precision::Int8);
            Ok(Runner::Cpu(Box::new(kime_cpu::executor_with(model, backend)?)))
        };
        match device {
            Device::Cpu { threads } => cpu(threads),
            #[cfg(feature = "cuda")]
            Device::Cuda(n) => {
                Ok(Runner::Cuda(Box::new(kime_cuda::executor(model, n, cuda(precision)?)?)))
            }
            #[cfg(feature = "cuda")]
            Device::Auto => {
                match cuda(precision).and_then(|p| Ok(kime_cuda::executor(model, 0, p)?)) {
                    Ok(e) => Ok(Runner::Cuda(Box::new(e))),
                    Err(_) => cpu(0),
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Auto => cpu(0),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::Unsupported("this build has no CUDA backend".into())),
            Device::Metal | Device::Ane => {
                Err(Error::Unsupported("the Apple backends arrive with M4".into()))
            }
        }
    }

    fn add_graph(&mut self, g: kime_tensor::Graph) -> usize {
        let b = Buckets::default();
        match self {
            Runner::Cpu(e) => e.add_graph(g, &b, EMBED_STAGE),
            #[cfg(feature = "cuda")]
            Runner::Cuda(e) => e.add_graph(g, &b, EMBED_STAGE),
        }
    }

    fn prepare(&mut self, b: kime_tensor::Bucket) -> Result<(), Error> {
        match self {
            Runner::Cpu(e) => e.prepare(b)?,
            #[cfg(feature = "cuda")]
            Runner::Cuda(e) => e.prepare(b)?,
        };
        Ok(())
    }

    fn run(&mut self, buf: &BatchBuf, out: &mut Outputs) -> Result<(), Error> {
        self.run_lane(0, buf, out)
    }

    fn run_lane(&mut self, lane: usize, buf: &BatchBuf, out: &mut Outputs) -> Result<(), Error> {
        match self {
            Runner::Cpu(e) => e.run_lane(lane, &buf.batch(), out)?,
            #[cfg(feature = "cuda")]
            Runner::Cuda(e) => e.run_lane(lane, &buf.batch(), out)?,
        };
        Ok(())
    }

    fn memory(&self) -> (usize, usize) {
        match self {
            Runner::Cpu(e) => e.memory(),
            #[cfg(feature = "cuda")]
            Runner::Cuda(e) => e.memory(),
        }
    }

    fn int8(&self) -> bool {
        match self {
            Runner::Cpu(e) => e.backend().int8(),
            #[cfg(feature = "cuda")]
            Runner::Cuda(_) => false,
        }
    }

    fn describe(&self) -> String {
        match self {
            Runner::Cpu(e) => {
                let int8 = if e.backend().int8() { ", int8" } else { "" };
                format!("cpu, {} threads{int8}", kime_tensor::Backend::caps(e.backend()).threads)
            }
            #[cfg(feature = "cuda")]
            Runner::Cuda(e) => format!("cuda, {}", e.backend().name()),
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda(p: Precision) -> Result<kime_cuda::Precision, Error> {
    match p {
        Precision::F16 => Ok(kime_cuda::Precision::F16),
        Precision::F32 => Ok(kime_cuda::Precision::F32),
        Precision::Int8 => Err(Error::Unsupported("INT8 runs on the CPU only for now".into())),
    }
}

/// The buckets the pooled embedding runs in: sequences with no markers.
const EMBED_STAGE: &str = "state";

struct Session {
    runner: Runner,
    /// The lane of the pooled embedding graph on the runner.
    embed: usize,
    buf: BatchBuf,
    out: Outputs,
}

struct Inner {
    id: String,
    tok: Tokenizer,
    mask: String,
    budget: CompatBudget,
    temps: Temperatures,
    /// The compat buckets, smallest first.
    buckets: Vec<kime_tensor::Bucket>,
    /// The buckets of the pooled embedding, smallest first.
    embed_buckets: Vec<kime_tensor::Bucket>,
    /// The width of the encoder, and of an embedding.
    d: usize,
    runner: Mutex<Session>,
    /// [`Memory`], kept up to date after every forward pass so reading it needs no lock.
    memory: [AtomicUsize; 2],
    cache: Option<AnswerCache>,
}

/// A loaded model on a device. Clones share it, and it can be used from any thread.
#[derive(Clone)]
pub struct Kime {
    inner: Arc<Inner>,
}

impl fmt::Debug for Kime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Kime").field("model", &self.inner.id).finish_non_exhaustive()
    }
}

/// Where the time of one [`Kime::decide_batch_timed`] call went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timing {
    /// Validating the requests and laying their questions out as token ids.
    pub tokenize: Duration,
    /// The device batches, from the first upload to the last result copied back.
    pub device: Duration,
    /// How many device batches the questions took.
    pub batches: usize,
    /// Questions whose state was cut to fit the model's sequence length.
    pub truncated: usize,
    /// State tokens left out of those questions.
    pub cut_tokens: usize,
    /// Questions answered from the answer cache, which took no device time.
    pub cached: usize,
}

/// Bytes a model holds on its device.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Memory {
    /// The weights, in the device's layout.
    pub weights: usize,
    /// The plans built so far, one per bucket used: their arenas and staging buffers.
    pub plans: usize,
}

/// One laid out question and where its answer goes.
struct Item<'a> {
    req: usize,
    q: &'a Question,
    seq: CompatSequence,
    logits: Vec<f32>,
    act: [f32; 2],
}

impl Item<'_> {
    fn key(&self) -> cache::Key {
        AnswerCache::key(&self.seq, self.q.qtype.index() as u8)
    }

    fn fill(&mut self, e: Entry) {
        self.logits = e.logits.into_vec();
        self.act = e.act;
    }
}

/// The request's `kime.cache`, the default for anything but a mode's name.
fn mode(req: &Request) -> CacheMode {
    req.kime
        .as_ref()
        .and_then(|k| k.get("cache"))
        .and_then(Value::as_str)
        .and_then(CacheMode::parse)
        .unwrap_or_default()
}

impl Kime {
    /// Settings for a new engine.
    #[must_use]
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// The model's name, `laya` or `laya-multilingual` for the published checkpoints.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.inner.id
    }

    /// The device the model runs on, in words.
    #[must_use]
    pub fn device(&self) -> String {
        self.lock().runner.describe()
    }

    /// The most tokens one question's row can hold: the state, the question and its options.
    /// Longer states are cut to fit.
    #[must_use]
    pub fn max_row_tokens(&self) -> usize {
        self.inner.budget.max_len
    }

    /// The bytes the model holds on its device, as of the last forward pass.
    #[must_use]
    pub fn memory(&self) -> Memory {
        let m = &self.inner.memory;
        Memory { weights: m[0].load(Ordering::Relaxed), plans: m[1].load(Ordering::Relaxed) }
    }

    /// The answer cache's counts, all 0 when it is off.
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.inner.cache.as_ref().map(AnswerCache::stats).unwrap_or_default()
    }

    /// The answer to `req` when the answer cache holds every one of its questions, found without
    /// the device lock, so the server can answer it without queueing it. `None` otherwise, and
    /// then nothing is counted, since the request goes on to [`Kime::decide_batch`].
    #[must_use]
    pub fn cached(&self, req: &Request) -> Option<Response> {
        let cache = self.inner.cache.as_ref()?;
        if mode(req) != CacheMode::Use || req.questions.is_empty() {
            return None;
        }
        let parsed = [parse(&req.to_json(), &Limits::LAYA).ok()?];
        let mut items = self.lay_out(&parsed).ok()?;
        let keys: Vec<_> = items.iter().map(Item::key).collect();
        for (it, e) in items.iter_mut().zip(cache.all(&keys)?) {
            it.fill(e);
        }
        self.respond(&parsed, &items).pop()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Session> {
        self.inner.runner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The tokens a request holds before anything is cut: its state once plus every question
    /// with its options. The server refuses a request over its limit with this count, before it
    /// is queued.
    #[must_use]
    pub fn count_tokens(&self, req: &Request) -> usize {
        let inner = &*self.inner;
        let mut n = inner.tok.encode(&compat_state(&req.state, &inner.mask)).len();
        for q in &req.questions {
            let text = compat_question(q, &inner.mask);
            n += inner.tok.encode(&text.head).len();
            n += text.options.iter().map(|o| inner.tok.encode(o).len()).sum::<usize>();
        }
        n
    }

    /// Answers one request.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a request that does not validate, [`Error::TooLong`] for a question
    /// whose options do not fit, and device errors.
    pub fn decide(&self, req: &Request) -> Result<Response, Error> {
        Ok(self.decide_batch(std::slice::from_ref(req))?.remove(0))
    }

    /// Answers many requests, packing all their questions into as few device batches as fit.
    /// Each answer is the same bits it would be alone, whatever else is in the batch.
    ///
    /// # Errors
    ///
    /// As [`Kime::decide`]. One bad request fails the whole call.
    pub fn decide_batch(&self, reqs: &[Request]) -> Result<Vec<Response>, Error> {
        Ok(self.decide_batch_timed(reqs)?.0)
    }

    /// [`Kime::decide_batch`], and where the time went.
    ///
    /// # Errors
    ///
    /// As [`Kime::decide_batch`].
    pub fn decide_batch_timed(&self, reqs: &[Request]) -> Result<(Vec<Response>, Timing), Error> {
        let t0 = Instant::now();
        let mut parsed = Vec::with_capacity(reqs.len());
        for r in reqs {
            parsed.push(parse(&r.to_json(), &Limits::LAYA).map_err(Error::Invalid)?);
        }
        let mut items = self.lay_out(&parsed)?;
        let tokenize = t0.elapsed();
        let t1 = Instant::now();
        let (run, keys) = self.look_up(reqs, &mut items);
        let batches = if run.is_empty() { 0 } else { self.run(&mut items, &run)? };
        if let Some(c) = &self.inner.cache {
            let keep = run.iter().filter(|&&i| mode(&reqs[items[i].req]) != CacheMode::Bypass);
            c.insert(keep.map(|&i| {
                (keys[i], Entry { logits: items[i].logits.clone().into(), act: items[i].act })
            }));
        }
        let cached = items.len() - run.len();
        let cut = items.iter().map(|it| it.seq.state_tokens - it.seq.state_tokens_used);
        let timing = Timing {
            tokenize,
            device: t1.elapsed(),
            batches,
            truncated: cut.clone().filter(|&n| n > 0).count(),
            cut_tokens: cut.sum(),
            cached,
        };
        Ok((self.respond(&parsed, &items), timing))
    }

    /// Each question of `parsed` laid out as Laya lays it out, in request order.
    fn lay_out<'a>(&self, parsed: &'a [Request]) -> Result<Vec<Item<'a>>, Error> {
        let inner = &*self.inner;
        let mut items = Vec::new();
        for (i, r) in parsed.iter().enumerate() {
            if r.questions.is_empty() {
                continue;
            }
            // Laya keeps the end of a conversation and the start of anything else.
            let cut = if matches!(r.state, Value::Array(_)) { Cut::Head } else { Cut::Tail };
            let state = inner.tok.encode_state(&compat_state(&r.state, &inner.mask));
            for q in &r.questions {
                let text = compat_question(q, &inner.mask);
                let seq =
                    inner.tok.compat_sequence(&text.head, &text.options, &state, inner.budget, cut);
                if seq.markers.len() != q.criteria.len() {
                    return Err(Error::TooLong {
                        question: q.id.clone(),
                        options: q.criteria.len(),
                        fit: seq.markers.len(),
                    });
                }
                items.push(Item { req: i, q, seq, logits: Vec::new(), act: [0.0; 2] });
            }
        }
        Ok(items)
    }

    /// Fills the items the cache holds, for the requests that read it. It gives the items left
    /// to run, and every item's key, none when the cache is off.
    fn look_up(&self, reqs: &[Request], items: &mut [Item<'_>]) -> (Vec<usize>, Vec<cache::Key>) {
        let Some(c) = &self.inner.cache else { return ((0..items.len()).collect(), Vec::new()) };
        let keys: Vec<_> = items.iter().map(Item::key).collect();
        let look: Vec<usize> =
            (0..items.len()).filter(|&i| mode(&reqs[items[i].req]) == CacheMode::Use).collect();
        let found = c.get(&look.iter().map(|&i| keys[i]).collect::<Vec<_>>());
        let mut hit = vec![false; items.len()];
        for (&i, e) in look.iter().zip(found) {
            if let Some(e) = e {
                items[i].fill(e);
                hit[i] = true;
            }
        }
        ((0..items.len()).filter(|&i| !hit[i]).collect(), keys)
    }

    /// The responses, from items whose logits are in.
    fn respond(&self, parsed: &[Request], items: &[Item<'_>]) -> Vec<Response> {
        let inner = &*self.inner;
        let mut out: Vec<Response> = parsed
            .iter()
            .map(|_| Response { model: LAYA_MODEL.into(), answers: Vec::new(), input_tokens: 0 })
            .collect();
        for it in items {
            let res = &mut out[it.req];
            res.input_tokens += it.seq.ids.len();
            res.answers
                .push((it.q.id.clone(), laya_answer(it.q, &it.logits, it.act, &inner.temps)));
        }
        out
    }

    /// Runs the items at `which` in the batches [`split::split`] picks, and says how many it
    /// took.
    fn run(&self, items: &mut [Item<'_>], which: &[usize]) -> Result<usize, Error> {
        let sizes: Vec<(usize, usize)> =
            which.iter().map(|&i| (items[i].seq.ids.len(), items[i].seq.markers.len())).collect();
        let batches: Vec<Vec<usize>> = split::split(&self.inner.buckets, &sizes)
            .into_iter()
            .map(|b| b.into_iter().map(|j| which[j]).collect())
            .collect();
        let mut s = self.lock();
        let Session { runner, buf, out, .. } = &mut *s;
        for batch in &batches {
            buf.clear();
            for &i in batch {
                let it = &items[i];
                buf.push(&it.seq.ids, &it.seq.markers, it.q.qtype.index() as u8);
            }
            runner.run(buf, out)?;
            let mut at = 0;
            for (&i, a) in batch.iter().zip(&out.act) {
                let it = &mut items[i];
                let k = it.seq.markers.len();
                it.logits = out.logits[at..at + k].to_vec();
                it.act = *a;
                at += k;
            }
        }
        let (weights, plans) = runner.memory();
        self.inner.memory[0].store(weights, Ordering::Relaxed);
        self.inner.memory[1].store(plans, Ordering::Relaxed);
        Ok(batches.len())
    }

    /// Each text's encoder output mean pooled over its tokens, as Laya's `embed_fn_from_agent`
    /// computes it: `[CLS]`, the text cut to `max_length` tokens with the specials, `[SEP]`. It
    /// runs no decision head. Each row is as wide as the encoder.
    ///
    /// # Errors
    ///
    /// Device errors, and [`Error::Unsupported`] on a backend without the pooled graph or in
    /// INT8, which puts some rows far from Laya's.
    pub fn embed(&self, texts: &[&str], max_length: usize) -> Result<Vec<Vec<f32>>, Error> {
        let inner = &*self.inner;
        if self.lock().runner.int8() {
            // On 125 texts the worst row had a cosine of 0.53 to Laya's, too far to shortlist on.
            return Err(Error::Unsupported("embeddings need FP32 or FP16, not INT8".into()));
        }
        let sp = inner.tok.specials();
        let keep = max_length.saturating_sub(2);
        let seqs: Vec<Vec<u32>> = texts
            .iter()
            .map(|t| {
                let mut ids = Vec::with_capacity(keep.min(t.len()) + 2);
                ids.push(sp.cls);
                inner.tok.encode_into(t, &mut ids);
                ids.truncate(keep + 1);
                ids.push(sp.sep);
                ids
            })
            .collect();
        let sizes: Vec<(usize, usize)> = seqs.iter().map(|s| (s.len(), 0)).collect();
        let batches = split::split(&inner.embed_buckets, &sizes);
        let mut rows = vec![Vec::new(); texts.len()];
        let mut s = self.lock();
        let Session { runner, embed, buf, out } = &mut *s;
        for batch in &batches {
            buf.clear();
            for &i in batch {
                buf.push(&seqs[i], &[], 0);
            }
            runner.run_lane(*embed, buf, out)?;
            for (&i, row) in batch.iter().zip(out.pooled.chunks_exact(inner.d)) {
                rows[i] = row.to_vec();
            }
        }
        let (weights, plans) = runner.memory();
        inner.memory[0].store(weights, Ordering::Relaxed);
        inner.memory[1].store(plans, Ordering::Relaxed);
        Ok(rows)
    }

    /// [`Kime::decide`] on a thread of its own, for async callers. It works with any executor,
    /// since it needs nothing from one but a waker.
    #[must_use]
    pub fn decide_async(&self, req: &Request) -> Decision {
        let shared = Arc::new(Mutex::new((None, None::<Waker>)));
        let (kime, req, done) = (self.clone(), req.clone(), shared.clone());
        std::thread::spawn(move || {
            let r = kime.decide(&req);
            let mut g = done.lock().unwrap_or_else(PoisonError::into_inner);
            g.0 = Some(r);
            if let Some(w) = g.1.take() {
                w.wake();
            }
        });
        Decision { shared }
    }
}

/// The future [`Kime::decide_async`] returns.
#[derive(Debug)]
pub struct Decision {
    #[allow(clippy::type_complexity)]
    shared: Arc<Mutex<(Option<Result<Response, Error>>, Option<Waker>)>>,
}

impl Future for Decision {
    type Output = Result<Response, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut g = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        match g.0.take() {
            Some(r) => Poll::Ready(r),
            None => {
                g.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
