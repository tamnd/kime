//! Quality suites: JSON lines of requests with their gold answers.
//!
//! A suite line is a request as `/v1/systemone` takes it, plus the fields the scorer reads:
//!
//! ```json
//! {"id": "ag_news-0", "state": {"article": "..."},
//!  "questions": {"topic": {"type": "choice", "instructions": "...", "criteria": {...}}},
//!  "gold": {"topic": "business"}, "tags": {"lang": "en"}}
//! ```
//!
//! The gold of a choice is the option label, of a score the level index, and of a noul `true` or
//! `false`. `soft` holds soft labels in option order and `gold_score` a score that can fall
//! between levels, both keyed by question like `gold`. `tags` are strings the report groups by.

use std::collections::BTreeMap;

use kime_core::answer::Answer;
use kime_core::request::{Criteria, Limits, QType, Question, Request, parse};
use serde_json::Value;

use crate::metrics::Row;

/// One line of a suite.
#[derive(Debug, Clone)]
pub struct Case {
    /// The line's `id`, or its line number.
    pub id: String,
    /// The request, validated with Laya's limits.
    pub request: Result<Request, String>,
    /// The line as read, for the question definitions when the request did not validate.
    pub body: Value,
    /// The tags, sorted by name.
    pub tags: BTreeMap<String, String>,
}

/// Reads a suite. A line that is not JSON is an error; a request that does not validate is kept,
/// and its questions count as dropped.
pub fn load(text: &str) -> Result<Vec<Case>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let body: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| (i + 1).to_string(), str::to_string);
        let mut req = body.clone();
        if let Some(m) = req.as_object_mut() {
            for k in ["id", "gold", "soft", "gold_score", "tags"] {
                m.remove(k);
            }
        }
        let request = parse(&req, &Limits::LAYA)
            .map_err(|p| p.iter().map(|p| p.to_json().to_string()).collect::<Vec<_>>().join(", "));
        let tags = body
            .get("tags")
            .and_then(Value::as_object)
            .map(|t| {
                t.iter()
                    .map(|(k, v)| {
                        (k.clone(), v.as_str().map_or_else(|| v.to_string(), str::to_string))
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.push(Case { id, request, body, tags });
    }
    Ok(out)
}

/// The question type's name as the report writes it.
#[must_use]
pub fn type_name(t: QType) -> &'static str {
    match t {
        QType::Choice => "choice",
        QType::Score => "score",
        QType::Noul => "noul",
    }
}

/// The index of the gold option of `q`, from the suite line's `gold`.
fn gold_index(q: &Question, gold: &Value) -> Option<usize> {
    match &q.criteria {
        Criteria::Choice(opts) => {
            let g = gold.as_str().map_or_else(|| gold.to_string(), str::to_string);
            opts.iter().position(|o| o.label == g)
        }
        Criteria::Score(levels) => gold.as_u64().map(|g| g as usize).filter(|g| *g < levels.len()),
        Criteria::Noul { .. } => gold.as_bool().map(usize::from),
    }
}

/// The probabilities of an answer in option order, at full precision.
#[must_use]
pub fn probs(a: &Answer) -> Vec<f64> {
    match a {
        Answer::Choice { probabilities, .. } => probabilities.iter().map(|p| p.1).collect(),
        Answer::Score { probabilities, .. } => probabilities.clone(),
        Answer::Noul { noul, .. } => vec![1.0 - noul, *noul],
    }
}

/// The probabilities of an answer in Laya's JSON shape, in the option order of `q`. `None` when
/// the answer does not fit the question.
#[must_use]
pub fn probs_json(q: &Question, a: &Value) -> Option<Vec<f64>> {
    let p = a.get("probabilities");
    match &q.criteria {
        Criteria::Choice(opts) => opts.iter().map(|o| p?.get(&o.label)?.as_f64()).collect(),
        Criteria::Score(levels) => {
            (0..levels.len()).map(|i| p?.get(i.to_string())?.as_f64()).collect()
        }
        Criteria::Noul { .. } => {
            let t = a.get("noul")?.as_f64()?;
            Some(vec![1.0 - t, t])
        }
    }
}

/// One scored question: the row the metrics read, and where it came from.
#[derive(Debug, Clone)]
pub struct Scored {
    /// The case id.
    pub case: String,
    /// The question id.
    pub question: String,
    /// The question type.
    pub qtype: QType,
    /// The case's tags.
    pub tags: BTreeMap<String, String>,
    /// The gold, the probabilities and the extras.
    pub row: Row,
}

/// The rows of one case, from `answer`, which gives the probabilities of a question in option
/// order or `None` when the engine did not answer it. The second value counts the questions with
/// a gold that were dropped, because the request did not validate, the engine did not answer, or
/// the gold is not one of the options.
pub fn score(
    case: &Case,
    mut answer: impl FnMut(&Question) -> Option<Vec<f64>>,
) -> (Vec<Scored>, usize) {
    let gold = case.body.get("gold").and_then(Value::as_object);
    let Ok(req) = &case.request else {
        return (Vec::new(), gold.map_or(0, serde_json::Map::len));
    };
    let (mut out, mut dropped) = (Vec::new(), 0);
    for q in &req.questions {
        let Some(g) = gold.and_then(|g| g.get(&q.id)) else { continue };
        let (Some(gi), Some(p)) = (gold_index(q, g), answer(q)) else {
            dropped += 1;
            continue;
        };
        let field = |name: &str| case.body.get(name).and_then(|s| s.get(&q.id));
        let soft = field("soft")
            .and_then(Value::as_array)
            .map(|s| s.iter().filter_map(Value::as_f64).collect());
        let gold_score = field("gold_score").and_then(Value::as_f64);
        out.push(Scored {
            case: case.id.clone(),
            question: q.id.clone(),
            qtype: q.qtype,
            tags: case.tags.clone(),
            row: Row { gold: gi, probs: p, soft, gold_score },
        });
    }
    (out, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_a_line_from_json_answers() {
        let text = r#"{"id": "a", "state": "x", "questions": {
            "t": {"type": "choice", "instructions": "i", "criteria": {"b": null, "a": null}},
            "s": {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]},
            "n": {"type": "noul", "instructions": "i"}},
            "gold": {"t": "a", "s": 2, "n": false}, "gold_score": {"s": 1.5}, "tags": {"lang": "en"}}"#
            .replace('\n', " ");
        let cases = load(&text).unwrap();
        let answers: Value = serde_json::json!({
            "t": {"probabilities": {"a": 0.25, "b": 0.75}},
            "s": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
            "n": {"noul": 0.9}});
        let (rows, dropped) = score(&cases[0], |q| probs_json(q, &answers[&q.id]));
        assert_eq!(dropped, 0);
        assert_eq!(rows.len(), 3);
        assert_eq!((rows[0].row.gold, rows[0].row.probs.clone()), (1, vec![0.75, 0.25]));
        assert_eq!(rows[1].row.gold, 2);
        assert_eq!(rows[1].row.gold_score, Some(1.5));
        assert_eq!(rows[2].row.gold, 0);
        assert!((rows[2].row.probs[0] - 0.1).abs() < 1e-12);
        assert_eq!(rows[0].tags["lang"], "en");
    }
}
