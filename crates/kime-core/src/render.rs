//! The text the model reads for a question, before tokenizing.
//!
//! Only the compat family is here for now. It reproduces what Laya 0.3.7 feeds its tokenizer
//! (`build_sequence`, `render_options`, `render_criterion` and `serialize_state` in
//! `laya/common.py`, and `Agent._to_internal`), because a single differing space changes the ids.
//! The native family renders differently (spec/06-input.md) and arrives with the native models.

use serde_json::Value;

use crate::pyjson::Dumps;
use crate::request::{Criteria, Question};

const RAW: Dumps = Dumps { ensure_ascii: false };
const ASCII: Dumps = Dumps { ensure_ascii: true };

/// The default noul descriptions Laya uses when a side is missing or empty.
pub const NOUL_FALSE: &str = "no, the statement does not hold";
/// See [`NOUL_FALSE`].
pub const NOUL_TRUE: &str = "yes, the statement holds";

/// The three pieces of text for one compat question. Every piece has had the tokenizer's mask
/// literal replaced by a space, so user text cannot forge a marker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatText {
    /// `"<type> question: <instructions>"`.
    pub head: String,
    /// One per option, in option order, each with a leading space.
    pub options: Vec<String>,
}

/// Renders a question the way Laya does. `mask` is the tokenizer's mask literal, `[MASK]` for
/// ModernBERT and `<mask>` for mmBERT.
#[must_use]
pub fn compat_question(q: &Question, mask: &str) -> CompatText {
    // Laya turns non string instructions into json.dumps(ins) with the defaults, so ensure_ascii
    // is on here and nowhere else, and null becomes the word null. Laya rejects a missing field and
    // kime renders it as nothing.
    let ins = match &q.instructions {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v) => ASCII.to_string(v),
    };
    let head = format!("{} question: {}", q.qtype.as_str(), ins.replace(mask, " "));
    let options = compat_options(&q.criteria)
        .into_iter()
        .map(|o| format!(" {}", o.replace(mask, " ")))
        .collect();
    CompatText { head, options }
}

/// Laya's `render_options`, without the leading space and mask replacement.
#[must_use]
pub fn compat_options(c: &Criteria) -> Vec<String> {
    match c {
        Criteria::Choice(opts) => opts
            .iter()
            .map(|o| match &o.description {
                // Only null and "" mean no description. 0 and false are real values.
                None => o.label.clone(),
                Some(Value::String(s)) if s.is_empty() => o.label.clone(),
                Some(v) => format!("{}: {}", o.label, criterion(v)),
            })
            .collect(),
        Criteria::Score(levels) => {
            levels.iter().enumerate().map(|(i, v)| format!("level {i}: {}", criterion(v))).collect()
        }
        Criteria::Noul { when_false, when_true, labels } => {
            let side = |v: &Option<Value>, default: &str| match v {
                None | Some(Value::Null) => default.to_string(),
                Some(Value::String(s)) if s.is_empty() => default.to_string(),
                Some(v) => criterion(v),
            };
            let (f, t) =
                labels.as_ref().map_or(("false", "true"), |(f, t)| (f.as_str(), t.as_str()));
            vec![
                format!("{f}: {}", side(when_false, NOUL_FALSE)),
                format!("{t}: {}", side(when_true, NOUL_TRUE)),
            ]
        }
    }
}

/// Laya's `render_criterion`: strings as they are, anything else as JSON with raw Unicode.
#[must_use]
pub fn criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => RAW.to_string(v),
    }
}

/// Laya's `serialize_state` followed by the mask replacement.
#[must_use]
pub fn compat_state(state: &Value, mask: &str) -> String {
    let s = match state {
        Value::String(s) => s.clone(),
        v => RAW.to_string(v),
    };
    s.replace(mask, " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{Limits, parse};
    use serde_json::json;

    fn render(q: Value) -> CompatText {
        let r = parse(&json!({"state": "", "questions": {"q": q}}), &Limits::LAYA).unwrap();
        compat_question(&r.questions[0], "[MASK]")
    }

    #[test]
    fn choice_score_noul() {
        let t = render(
            json!({"type": "choice", "instructions": "Pick [MASK] one", "criteria": {"a": "", "b": null, "c": 0, "d": {"x": [1, 2.5]}}}),
        );
        assert_eq!(t.head, "choice question: Pick   one");
        assert_eq!(t.options, [" a", " b", " c: 0", " d: {\"x\": [1, 2.5]}"]);
        let t = render(
            json!({"type": "score", "instructions": {"ask": "é"}, "criteria": ["low", null, true]}),
        );
        assert_eq!(t.head, "score question: {\"ask\": \"\\u00e9\"}");
        assert_eq!(t.options, [" level 0: low", " level 1: null", " level 2: true"]);
        let t = render(
            json!({"type": "noul", "instructions": null, "criteria": {"TRUE": "", "false": "nope"}}),
        );
        assert_eq!(t.head, "noul question: null");
        assert_eq!(t.options, [" false: nope", " true: yes, the statement holds"]);
    }

    #[test]
    fn state() {
        assert_eq!(
            compat_state(&json!({"k": "ü", "n": [1e-5]}), "[MASK]"),
            "{\"k\": \"ü\", \"n\": [1e-05]}"
        );
        assert_eq!(compat_state(&json!("a<mask>b"), "<mask>"), "a b");
    }
}
