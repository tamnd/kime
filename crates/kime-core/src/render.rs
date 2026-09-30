//! The text the model reads for a question, before tokenizing.
//!
//! Only the compat family is here for now. It reproduces what Laya 0.3.7 feeds its tokenizer
//! (`build_sequence`, `render_options`, `render_criterion` and `serialize_state` in
//! `laya/common.py`, and `Agent._to_internal`), because a single differing space changes the ids.
//! The native family renders differently (spec/06-input.md). Its state rendering is here, whole
//! and in segments. Its questions arrive with the native models.

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

/// The native state text: a string as it is, anything else as compact JSON with raw Unicode.
#[must_use]
pub fn native_state(state: &Value) -> String {
    match state {
        Value::String(s) => s.clone(),
        v => json(v),
    }
}

fn json(v: &Value) -> String {
    serde_json::to_string(v).expect("a Value always serializes")
}

/// The native state text cut into segments for segment mode, each tokenized and cached on its
/// own. Every top level key of an object is a segment, and so is every element of an array,
/// whether the array is the state or the value of a top level key. The punctuation between
/// segments goes at the start of the next one, and the closing brackets at the end form the last
/// segment, so the segments join to exactly [`native_state`], and appending an element or a key
/// leaves every earlier segment as it was. A string or scalar state is one segment.
#[must_use]
pub fn native_segments(state: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let mut pending = String::new();
    let array = |a: &[Value], pending: &mut String, out: &mut Vec<String>| {
        pending.push('[');
        for (j, e) in a.iter().enumerate() {
            if j > 0 {
                pending.push(',');
            }
            pending.push_str(&json(e));
            out.push(std::mem::take(pending));
        }
        pending.push(']');
    };
    match state {
        Value::Object(m) if !m.is_empty() => {
            for (i, (k, v)) in m.iter().enumerate() {
                pending.push(if i == 0 { '{' } else { ',' });
                pending.push_str(&json(&Value::String(k.clone())));
                pending.push(':');
                match v {
                    Value::Array(a) if !a.is_empty() => array(a, &mut pending, &mut out),
                    v => {
                        pending.push_str(&json(v));
                        out.push(std::mem::take(&mut pending));
                    }
                }
            }
            pending.push('}');
        }
        Value::Array(a) if !a.is_empty() => array(a, &mut pending, &mut out),
        v => pending = native_state(v),
    }
    out.push(pending);
    out
}

/// The texts of one question tower row of the native family, before the markers go in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeText {
    /// `"<type_text> <instructions>"`, where the type text is `choose:`, `rate:` or
    /// `true or false:`.
    pub head: String,
    /// One per option, in option order: a choice's `label` or `label: description`, a score
    /// level's description, and a noul's false then true criterion, empty when not given.
    pub options: Vec<String>,
}

/// A question as the native family lays it out, spec/06-tokenization.md. Values that are not
/// strings render as compact JSON, like the state.
#[must_use]
pub fn native_question(q: &Question) -> NativeText {
    let text = |v: &Value| match v {
        Value::String(s) => s.clone(),
        v => json(v),
    };
    let type_text = match q.qtype {
        crate::request::QType::Choice => "choose:",
        crate::request::QType::Score => "rate:",
        crate::request::QType::Noul => "true or false:",
    };
    let head = match q.instructions.as_ref().map(text) {
        Some(ins) if !ins.is_empty() => format!("{type_text} {ins}"),
        _ => type_text.to_string(),
    };
    let options = match &q.criteria {
        Criteria::Choice(opts) => opts
            .iter()
            .map(|o| match &o.description {
                None => o.label.clone(),
                Some(Value::String(s)) if s.is_empty() => o.label.clone(),
                Some(v) => format!("{}: {}", o.label, text(v)),
            })
            .collect(),
        Criteria::Score(levels) => levels.iter().map(text).collect(),
        Criteria::Noul { when_false, when_true, .. } => [when_false, when_true]
            .into_iter()
            .map(|v| match v {
                None | Some(Value::Null) => String::new(),
                Some(v) => text(v),
            })
            .collect(),
    };
    NativeText { head, options }
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
    fn segments_join_to_the_state() {
        let state = json!({"page": {"url": "u", "text": "東京 \"a\""}, "elements": [{"id": 1}, "x", [2]],
            "empty": [], "recent_actions": ["a", "b"], "n": null});
        let segs = native_segments(&state);
        assert_eq!(
            segs,
            [
                r#"{"page":{"url":"u","text":"東京 \"a\""}"#,
                r#","elements":[{"id":1}"#,
                r#","x""#,
                r#",[2]"#,
                r#"],"empty":[]"#,
                r#","recent_actions":["a""#,
                r#","b""#,
                r#"],"n":null"#,
                "}",
            ]
        );
        assert_eq!(segs.concat(), native_state(&state));
        let mut more = state.clone();
        more["recent_actions"].as_array_mut().unwrap().push(json!("c"));
        let after = native_segments(&more);
        assert_eq!(after.len(), segs.len() + 1);
        assert!(segs.iter().all(|s| after.contains(s)), "an old segment changed");
        for (v, want) in [
            (json!("plain"), vec!["plain"]),
            (json!([1, 2]), vec!["[1", ",2", "]"]),
            (json!({}), vec!["{}"]),
            (json!([]), vec!["[]"]),
            (json!(3.5), vec!["3.5"]),
        ] {
            assert_eq!(native_segments(&v), want, "{v}");
        }
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

    #[test]
    fn native_rows() {
        let q = |q: Value| {
            let r = parse(&json!({"state": "", "questions": {"q": q}}), &Limits::LAYA).unwrap();
            native_question(&r.questions[0])
        };
        let c = q(json!({"type": "choice", "instructions": "Pick one",
            "criteria": {"a": "", "b": null, "c": {"x": 1}}}));
        assert_eq!(c.head, "choose: Pick one");
        assert_eq!(c.options, ["a", "b", "c: {\"x\":1}"]);
        let s = q(json!({"type": "score", "criteria": ["bad", "good"]}));
        assert_eq!((s.head.as_str(), s.options), ("rate:", vec!["bad".to_string(), "good".into()]));
        let n = q(json!({"type": "noul", "instructions": {"k": [1, 2]},
            "criteria": {"true": "spam"}}));
        assert_eq!(n.head, "true or false: {\"k\":[1,2]}");
        assert_eq!(n.options, ["", "spam"]);
    }
}
