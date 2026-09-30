//! The native family, kime-v1, on the CPU: the state tower runs once per state and its memory is
//! cached, then every question row reads it, as spec/06-tokenization.md and spec/11-serving.md
//! describe.
//!
//! A state is looked up by the blake3 of the model id, the mode and its token ids, so a request
//! that repeats a state, or a batch where many questions share one, runs the state tower once.
//! In segment mode the low state layers are cached per segment as well, so a state that differs
//! from an earlier one in a few segments recomputes only those.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use kime_core::answer::{Response, Temperatures, laya_answer};
use kime_core::render::{native_question, native_segments, native_state};
use kime_core::request::{Criteria, Limits, QType, Question, Request, parse};
use kime_cpu::split::{Memory, Row, SegmentCache, Split};
use kime_model::kime_v1::{SEGMENT_MAX_TOKENS, V1Graph, V1Spec, state_segments};
use kime_model::safetensors;
use kime_tensor::Blob;
use kime_tok::Tokenizer;
use serde_json::Value;

use crate::{Error, Timing, chunk};

/// Tokens of a question header, `[CLS_Q]` and `[SEP]` included.
const HEADER_TOKENS: usize = 512;

/// Tokens of one option segment, its marker and `[SEP]` included.
const OPTION_TOKENS: usize = 64;

/// Bytes of state memory kept by default, over both generations.
pub(crate) const STATE_CACHE_BYTES: usize = 1 << 30;

/// Bytes of segment outputs kept by default.
pub(crate) const SEGMENT_CACHE_BYTES: usize = 1 << 30;

/// Whether `dir` holds a kime-v1 checkpoint: a `kime.json` whose family is `kime-v1`.
pub(crate) fn is_native(dir: &Path) -> bool {
    std::fs::read(dir.join("kime.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .is_some_and(|v| v.get("family").and_then(Value::as_str) == Some("kime-v1"))
}

/// State memories by key, in two generations of half the budget each, like the answer cache.
struct States {
    young: HashMap<[u8; 32], Arc<Memory>>,
    old: HashMap<[u8; 32], Arc<Memory>>,
    young_bytes: usize,
    budget: usize,
}

fn bytes(m: &Memory) -> usize {
    let pairs = |p: &(Vec<f32>, Vec<f32>)| p.0.len() + p.1.len();
    let cross: usize = m.cross.iter().flatten().map(pairs).sum();
    4 * (m.states.len() + m.pooled.len() + m.kv.iter().map(pairs).sum::<usize>() + cross)
}

impl States {
    fn get(&mut self, k: &[u8; 32]) -> Option<Arc<Memory>> {
        if let Some(m) = self.young.get(k) {
            return Some(m.clone());
        }
        let m = self.old.remove(k)?;
        self.put(*k, m.clone());
        Some(m)
    }

    fn put(&mut self, k: [u8; 32], m: Arc<Memory>) {
        let n = bytes(&m);
        if n > self.budget / 2 || self.young.contains_key(&k) {
            return;
        }
        if self.young_bytes + n > self.budget / 2 {
            self.old = std::mem::take(&mut self.young);
            self.young_bytes = 0;
        }
        self.young_bytes += n;
        self.young.insert(k, m);
    }
}

/// A kime-v1 checkpoint loaded for the CPU.
pub(crate) struct Native {
    pub(crate) id: String,
    split: Split,
    tok: Tokenizer,
    temps: Temperatures,
    threads: usize,
    weights: usize,
    states: Mutex<States>,
    segments: Mutex<SegmentCache>,
}

/// One state as the tower reads it.
struct StatePlan {
    key: [u8; 32],
    /// The whole state as one sequence, or its segments.
    ids: Result<Vec<u32>, Vec<Vec<u32>>>,
    tokens: usize,
    cut: usize,
}

/// One question tower row and where its logits go.
struct RowPlan<'a> {
    req: usize,
    slot: usize,
    q: &'a Question,
    opts: Option<Vec<usize>>,
    rerank: bool,
    header: Vec<u32>,
    options: Vec<Vec<u32>>,
    logits: Vec<f32>,
}

impl Native {
    /// Loads `kime.json`, `model.safetensors` and the tokenizer the config names from `dir`.
    pub(crate) fn open(dir: &Path, threads: usize) -> Result<Self, Error> {
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .map_err(|e| Error::NotFound(format!("{}: {e}", dir.join(name).display())))
        };
        let config: Value = serde_json::from_slice(&read("kime.json")?)
            .map_err(|e| Error::NotFound(format!("{}: kime.json: {e}", dir.display())))?;
        let spec = V1Spec::from_json(&config).map_err(Error::Model)?;
        let st = dir.join("model.safetensors");
        let blob = Blob::map(&st).map_err(|e| Error::NotFound(format!("{}: {e}", st.display())))?;
        let (tensors, _) = safetensors::load(blob).map_err(Error::Model)?;
        let graph = V1Graph::bind(&spec, &tensors).map_err(Error::Model)?;
        let file =
            config.pointer("/tokenizer/file").and_then(Value::as_str).unwrap_or("tokenizer.json");
        let tok = Tokenizer::from_bytes(&read(file)?, None)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", dir.join(file).display())))?;
        let threads = if threads == 0 {
            std::thread::available_parallelism().map_or(1, usize::from)
        } else {
            threads
        };
        let weights = tensors.entries().iter().map(|e| e.numel() * 4).sum();
        Ok(Self {
            id: spec.id.clone(),
            split: Split::new(&spec, &graph, &tensors, threads),
            tok,
            // Until the checkpoint carries calibration, spec/05's temperature 1 for every type.
            temps: Temperatures::new([1.0; 3], &[]),
            threads,
            weights,
            states: Mutex::new(States {
                young: HashMap::new(),
                old: HashMap::new(),
                young_bytes: 0,
                budget: STATE_CACHE_BYTES,
            }),
            segments: Mutex::new(SegmentCache::with_budget(SEGMENT_CACHE_BYTES)),
        })
    }

    fn spec(&self) -> &V1Spec {
        self.split.spec()
    }

    pub(crate) fn device(&self) -> String {
        format!("CPU, {} threads, kime-v1", self.threads)
    }

    pub(crate) fn max_state_tokens(&self) -> usize {
        self.spec().state_max_tokens
    }

    pub(crate) fn weights(&self) -> usize {
        self.weights
    }

    /// The state tokens once plus every question's header and options, before anything is cut.
    pub(crate) fn count_tokens(&self, req: &Request) -> usize {
        let mut n = self.tok.encode(&native_state(&req.state)).len() + 2;
        for q in &req.questions {
            let t = native_question(q);
            n += self.tok.encode(&t.head).len() + 2;
            n += t.options.iter().map(|o| self.tok.encode(o).len() + 2).sum::<usize>();
        }
        n
    }

    /// How `req` asks for its state to be read: segment mode when `kime.state_segments` is true,
    /// or when it is missing or `auto` and the state is an object of more than
    /// [`SEGMENT_MAX_TOKENS`] tokens.
    fn plan_state(&self, req: &Request) -> StatePlan {
        let s = self.spec();
        let flag = req.kime.as_ref().and_then(|k| k.get("state_segments"));
        let whole = self.tok.encode(&native_state(&req.state));
        let segmented = match flag {
            Some(Value::Bool(b)) => *b,
            _ => matches!(req.state, Value::Object(_)) && whole.len() > SEGMENT_MAX_TOKENS,
        };
        let room = s.state_max_tokens.saturating_sub(2);
        if segmented && whole.len() <= room {
            let ids: Vec<Vec<u32>> =
                native_segments(&req.state).iter().map(|t| self.tok.encode(t)).collect();
            let segs: Vec<Vec<u32>> =
                state_segments(s, &ids).into_iter().filter(|g| !g.is_empty()).collect();
            let tokens = segs.iter().map(Vec::len).sum();
            if tokens <= s.state_max_tokens {
                let key = self.key(1, segs.iter().map(Vec::as_slice));
                return StatePlan { key, ids: Err(segs), tokens, cut: 0 };
            }
        }
        // Keep the end of a conversation and the start of anything else, as the compat family
        // does, unless `kime.truncation` says which end to cut.
        let head = match req.kime.as_ref().and_then(|k| k.get("truncation")).and_then(Value::as_str)
        {
            Some("head") => true,
            Some("tail") => false,
            _ => matches!(req.state, Value::Array(_)),
        };
        let cut = whole.len().saturating_sub(room);
        let kept = if head { &whole[cut..] } else { &whole[..whole.len() - cut] };
        let mut ids = Vec::with_capacity(kept.len() + 2);
        ids.push(s.specials.cls_s);
        ids.extend_from_slice(kept);
        ids.push(s.specials.sep);
        let key = self.key(0, std::iter::once(ids.as_slice()));
        StatePlan { key, tokens: ids.len(), ids: Ok(ids), cut }
    }

    fn key<'a>(&self, mode: u8, parts: impl Iterator<Item = &'a [u32]>) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(&(self.id.len() as u64).to_le_bytes());
        h.update(self.id.as_bytes());
        h.update(&[mode]);
        for p in parts {
            h.update(&(p.len() as u64).to_le_bytes());
            for id in p {
                h.update(&id.to_le_bytes());
            }
        }
        *h.finalize().as_bytes()
    }

    /// The memory of `plan`, from the cache or the state tower. The flag says whether the tower
    /// ran.
    fn memory(&self, plan: &StatePlan) -> (Arc<Memory>, bool) {
        let lock = || self.states.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(m) = lock().get(&plan.key) {
            return (m, false);
        }
        let m = Arc::new(match &plan.ids {
            Ok(ids) => self.split.state(ids),
            Err(segs) => {
                let segs: Vec<&[u32]> = segs.iter().map(Vec::as_slice).collect();
                let mut cache = self.segments.lock().unwrap_or_else(PoisonError::into_inner);
                self.split.state_segments(&segs, &mut cache)
            }
        });
        lock().put(plan.key, m.clone());
        (m, true)
    }

    /// The token ids of one question row's header and options, each cut to its budget.
    fn row_ids(&self, q: &Question) -> (Vec<u32>, Vec<Vec<u32>>) {
        let sp = &self.spec().specials;
        let text = native_question(q);
        let mut header = vec![sp.cls_q];
        self.tok.encode_into(&text.head, &mut header);
        header.truncate(HEADER_TOKENS - 1);
        header.push(sp.sep);
        let options = text
            .options
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let marker = match (q.qtype, i) {
                    (QType::Noul, 0) => sp.no,
                    (QType::Noul, _) => sp.yes,
                    _ => sp.opt,
                };
                let mut ids = vec![marker];
                self.tok.encode_into(o, &mut ids);
                ids.truncate(OPTION_TOKENS - 1);
                ids.push(sp.sep);
                ids
            })
            .collect();
        (header, options)
    }

    pub(crate) fn decide_batch_timed(
        &self,
        reqs: &[Request],
    ) -> Result<(Vec<Response>, Timing), Error> {
        let t0 = Instant::now();
        let mut parsed = Vec::with_capacity(reqs.len());
        for r in reqs {
            parsed.push(parse(&r.to_json(), &Limits::JEV).map_err(Error::Invalid)?);
        }
        let plans: Vec<Option<StatePlan>> =
            parsed.iter().map(|r| (!r.questions.is_empty()).then(|| self.plan_state(r))).collect();
        let max_options = self.spec().max_options;
        let mut rows: Vec<RowPlan<'_>> = Vec::new();
        let mut slot = 0;
        for (i, r) in parsed.iter().enumerate() {
            let size = crate::chunk_size(r).unwrap_or(max_options).min(max_options);
            for q in &r.questions {
                let (header, options) = self.row_ids(q);
                let k = options.len();
                let parts = if matches!(q.criteria, Criteria::Choice(_)) && k > size {
                    chunk::chunks(k, size).into_iter().map(Some).collect()
                } else {
                    vec![None]
                };
                for opts in parts {
                    let options = match &opts {
                        Some(c) => c.iter().map(|&j| options[j].clone()).collect(),
                        None => options.clone(),
                    };
                    rows.push(RowPlan {
                        req: i,
                        slot,
                        q,
                        opts,
                        rerank: false,
                        header: header.clone(),
                        options,
                        logits: Vec::new(),
                    });
                }
                slot += 1;
            }
        }
        let tokenize = t0.elapsed();

        let t1 = Instant::now();
        let mut mems: Vec<Option<Arc<Memory>>> = Vec::with_capacity(plans.len());
        let mut states = 0;
        for p in &plans {
            mems.push(match p {
                Some(p) => {
                    let (m, ran) = self.memory(p);
                    states += usize::from(ran);
                    Some(m)
                }
                None => None,
            });
        }
        self.run(&mut rows, &mems);
        let joint = self.reranks(&rows);
        let reranked = !joint.is_empty();
        if reranked {
            let at = rows.len();
            rows.extend(joint);
            self.run(&mut rows[at..], &mems);
        }
        let timing = Timing {
            tokenize,
            device: t1.elapsed(),
            batches: rows.len(),
            truncated: plans.iter().flatten().filter(|p| p.cut > 0).count(),
            cut_tokens: plans.iter().flatten().map(|p| p.cut).sum(),
            cached: 0,
            states,
        };
        Ok((self.respond(&parsed, &plans, &rows), timing))
    }

    fn run(&self, rows: &mut [RowPlan<'_>], mems: &[Option<Arc<Memory>>]) {
        let jobs: Vec<(&Memory, Row<'_>)> = rows
            .iter()
            .map(|r| {
                let mem = mems[r.req].as_deref().expect("a request with questions has a state");
                (mem, Row { qtype: r.q.qtype.index(), header: &r.header, options: &r.options })
            })
            .collect();
        let logits = self.split.questions(&jobs);
        for (r, l) in rows.iter_mut().zip(logits) {
            r.logits = l;
        }
    }

    /// The joint row of every choice scored in more than one chunk with more than
    /// [`chunk::TOP`] options.
    fn reranks<'a>(&self, rows: &[RowPlan<'a>]) -> Vec<RowPlan<'a>> {
        let mut out = Vec::new();
        for group in rows.chunk_by(|a, b| a.slot == b.slot) {
            let q = group[0].q;
            if group.len() < 2 || q.criteria.len() <= chunk::TOP {
                continue;
            }
            let picked = chunk::top(&chunked(group));
            let (header, options) = self.row_ids(q);
            out.push(RowPlan {
                req: group[0].req,
                slot: group[0].slot,
                q,
                options: picked.iter().map(|&i| options[i].clone()).collect(),
                opts: Some(picked),
                rerank: true,
                header,
                logits: Vec::new(),
            });
        }
        out
    }

    fn respond(
        &self,
        parsed: &[Request],
        plans: &[Option<StatePlan>],
        rows: &[RowPlan<'_>],
    ) -> Vec<Response> {
        let mut out: Vec<Response> = parsed
            .iter()
            .zip(plans)
            .map(|(_, p)| Response {
                model: self.id.clone(),
                answers: Vec::new(),
                input_tokens: p.as_ref().map_or(0, |p| p.tokens),
            })
            .collect();
        for r in rows {
            out[r.req].input_tokens +=
                r.header.len() + r.options.iter().map(Vec::len).sum::<usize>();
        }
        let (first, joint): (Vec<&RowPlan<'_>>, Vec<&RowPlan<'_>>) =
            rows.iter().partition(|r| !r.rerank);
        // The act head arrives with the correctness head, so every answer acts for now.
        let act = [0.0; 2];
        for group in first.chunk_by(|a, b| a.slot == b.slot) {
            let (q, req) = (group[0].q, group[0].req);
            let answer = if group.len() == 1 {
                laya_answer(q, &group[0].logits, act, &self.temps)
            } else {
                let t = self.temps.get(q.qtype, q.criteria.len());
                let chunked = chunked_refs(group);
                let p = match joint.iter().find(|j| j.slot == group[0].slot) {
                    Some(j) => {
                        let picked = j.opts.as_deref().unwrap_or_default();
                        let tj = self.temps.get(q.qtype, picked.len());
                        chunk::merge(&chunked, t, picked, &j.logits, tj)
                    }
                    None => chunk::softmax(&chunked, t),
                };
                laya_answer(q, &chunk::logits(&p, t), act, &self.temps)
            };
            out[req].answers.push((q.id.clone(), answer));
        }
        out
    }

    /// The mean pooled state memory of each text read as a string state, cut to `max_length`
    /// tokens with the specials.
    pub(crate) fn embed(&self, texts: &[&str], max_length: usize) -> Vec<Vec<f32>> {
        let sp = &self.spec().specials;
        let keep = max_length.min(self.spec().state_max_tokens).saturating_sub(2);
        texts
            .iter()
            .map(|t| {
                let mut ids = vec![sp.cls_s];
                self.tok.encode_into(t, &mut ids);
                ids.truncate(keep + 1);
                ids.push(sp.sep);
                self.split.state(&ids).pooled
            })
            .collect()
    }
}

/// The logits of a chunked choice's rows in option order.
fn chunked(group: &[RowPlan<'_>]) -> Vec<f32> {
    chunked_refs(&group.iter().collect::<Vec<_>>())
}

fn chunked_refs(group: &[&RowPlan<'_>]) -> Vec<f32> {
    let mut out = vec![0.0; group[0].q.criteria.len()];
    for r in group {
        match &r.opts {
            Some(opts) => opts.iter().zip(&r.logits).for_each(|(&i, &l)| out[i] = l),
            None => out.copy_from_slice(&r.logits),
        }
    }
    out
}
