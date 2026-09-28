//! Reading the training shards tools/data/convert.py writes.
//!
//! A shard is zstd compressed JSON lines, one state with its questions and a target per question.
//! Each line goes through the same request parsing and layout the engine uses, so a question is
//! trained on the exact ids and markers it is served with.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use kime_core::render::{compat_question, compat_state};
use kime_core::request::{Limits, parse};
use kime_model::Model;
use kime_tok::Tokenizer;
use kime_tok::layout::{CompatBudget, Cut};
use serde_json::{Map, Value};

use crate::model::Question;
use crate::rng::Rng;

/// One question of a training line, laid out, with its target.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// The ids and markers the model reads.
    pub q: Question,
    /// The target distribution over the options, in the order of the criteria.
    pub probs: Vec<f32>,
    /// The gold option, when there is one.
    pub hard: Option<usize>,
    /// How much the question counts in the loss.
    pub weight: f32,
}

/// Why a line gave no examples, or fewer than it has questions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Skip {
    /// The line is not JSON, or not an object with state, questions and targets.
    Malformed(String),
    /// The request does not validate.
    Invalid(String),
    /// A question has no target, or a target whose length is not the number of options.
    Target(String),
    /// The options of a question do not fit in the sequence.
    TooLong(String),
}

/// Lays out training lines for one checkpoint, with its tokenizer and budgets.
pub struct Renderer {
    tok: Tokenizer,
    mask: String,
    budget: CompatBudget,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer").field("budget", &self.budget).finish_non_exhaustive()
    }
}

impl Renderer {
    /// The renderer for the checkpoint `model`.
    ///
    /// # Errors
    ///
    /// If the checkpoint has no tokenizer, or it does not load.
    pub fn new(model: &Model) -> Result<Self, String> {
        let json = model.file("tokenizer/tokenizer.json").ok_or("no tokenizer/tokenizer.json")?;
        let tok = Tokenizer::from_bytes(json, model.file("tokenizer/tokenizer_config.json"))
            .map_err(|e| e.to_string())?;
        let agent = &model.spec.agent;
        Ok(Self {
            mask: tok.mask_text().to_string(),
            tok,
            budget: CompatBudget { max_len: agent.max_len, head_max_len: agent.head_max_len },
        })
    }

    /// The examples of one line, in question order, and what was skipped.
    #[must_use]
    pub fn render(&self, line: &str) -> (Vec<Example>, Vec<Skip>) {
        self.render_with(line, None)
    }

    /// Like [`Renderer::render`], and with a generator, the options of every choice question
    /// come in a random order with their targets moved along, which is how training fixes the
    /// order flips of spec/12-training.md.
    #[must_use]
    pub fn render_with(&self, line: &str, rng: Option<&mut Rng>) -> (Vec<Example>, Vec<Skip>) {
        let bad = |why: &str| (Vec::new(), vec![Skip::Malformed(why.to_string())]);
        let Ok(Value::Object(v)) = serde_json::from_str::<Value>(line) else {
            return bad("not a JSON object");
        };
        let (Some(questions), Some(Value::Object(targets))) =
            (v.get("questions"), v.get("targets"))
        else {
            return bad("no questions or targets");
        };
        let (questions, targets) = match rng {
            Some(rng) => shuffle_choices(questions, targets, rng),
            None => (questions.clone(), targets.clone()),
        };
        let mut body = Map::new();
        body.insert("state".into(), v.get("state").cloned().unwrap_or(Value::Null));
        body.insert("questions".into(), questions);
        let r = match parse(&Value::Object(body), &Limits::LAYA) {
            Ok(r) => r,
            Err(p) => return (Vec::new(), vec![Skip::Invalid(format!("{p:?}"))]),
        };
        let cut = if matches!(r.state, Value::Array(_)) { Cut::Head } else { Cut::Tail };
        let state = self.tok.encode_state(&compat_state(&r.state, &self.mask));
        let (mut out, mut skipped) = (Vec::with_capacity(r.questions.len()), Vec::new());
        for q in &r.questions {
            let k = q.criteria.len();
            let t = targets.get(&q.id);
            let probs: Option<Vec<f32>> = t
                .and_then(|t| t.get("probs"))
                .and_then(Value::as_array)
                .and_then(|a| a.iter().map(|x| x.as_f64().map(|x| x as f32)).collect());
            let Some(probs) = probs.filter(|p| p.len() == k) else {
                skipped.push(Skip::Target(q.id.clone()));
                continue;
            };
            let text = compat_question(q, &self.mask);
            let seq = self.tok.compat_sequence(&text.head, &text.options, &state, self.budget, cut);
            if seq.markers.len() != k {
                skipped.push(Skip::TooLong(q.id.clone()));
                continue;
            }
            let t = t.unwrap_or(&Value::Null);
            out.push(Example {
                q: Question { ids: seq.ids, markers: seq.markers, qtype: q.qtype.index() },
                probs,
                hard: t.get("hard").and_then(Value::as_u64).map(|h| h as usize).filter(|&h| h < k),
                weight: t.get("weight").and_then(Value::as_f64).map_or(1.0, |w| w as f32),
            });
        }
        (out, skipped)
    }
}

/// Reorders the options of each choice question and its target the same way. Anything that is
/// not a well formed choice with a matching target is left as it is, for `parse` to judge.
fn shuffle_choices(
    questions: &Value,
    targets: &Map<String, Value>,
    rng: &mut Rng,
) -> (Value, Map<String, Value>) {
    let (mut questions, mut targets) = (questions.clone(), targets.clone());
    let Some(qs) = questions.as_object_mut() else { return (questions, targets) };
    for (id, q) in qs.iter_mut() {
        if q.get("type").and_then(Value::as_str) != Some("choice") {
            continue;
        }
        let Some(t) = targets.get_mut(id).and_then(Value::as_object_mut) else { continue };
        let Some(probs) = t.get("probs").and_then(Value::as_array).cloned() else { continue };
        let Some(criteria) = q.get_mut("criteria") else { continue };
        let k = match criteria {
            Value::Object(m) => m.len(),
            Value::Array(a) => a.len(),
            _ => continue,
        };
        if k != probs.len() || k < 2 {
            continue;
        }
        let mut order: Vec<usize> = (0..k).collect();
        rng.shuffle(&mut order);
        *criteria = match criteria {
            Value::Object(m) => {
                let entries: Vec<(String, Value)> =
                    m.iter().map(|(a, b)| (a.clone(), b.clone())).collect();
                Value::Object(order.iter().map(|&i| entries[i].clone()).collect())
            }
            Value::Array(a) => Value::Array(order.iter().map(|&i| a[i].clone()).collect()),
            _ => unreachable!(),
        };
        t.insert("probs".into(), Value::Array(order.iter().map(|&i| probs[i].clone()).collect()));
        if let Some(h) = t.get("hard").and_then(Value::as_u64)
            && let Some(at) = order.iter().position(|&i| i as u64 == h)
        {
            t.insert("hard".into(), at.into());
        }
    }
    (questions, targets)
}

/// The lines of a shard, decompressed as they are read, or of a plain JSON lines file when the
/// name does not end in `.zst`.
///
/// # Errors
///
/// If the file does not open or is not zstd.
pub fn lines(path: &Path) -> io::Result<Box<dyn Iterator<Item = io::Result<String>>>> {
    let f = File::open(path)?;
    if path.extension().is_some_and(|e| e == "zst") {
        Ok(Box::new(BufReader::with_capacity(1 << 20, zstd::Decoder::new(f)?).lines()))
    } else {
        Ok(Box::new(BufReader::with_capacity(1 << 20, f).lines()))
    }
}
