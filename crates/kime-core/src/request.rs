//! The `/v1/systemone` request, parsed from JSON and validated the way spec/03-api.md describes.
//!
//! Validation collects every problem instead of stopping at the first, and reports each one in
//! FastAPI's shape (`loc`, `msg`, `type`, `input`), because that is what Jev returns and what its
//! SDKs parse. Parsing works on a `serde_json::Value` built with `preserve_order`, so questions and
//! choice labels keep the order they were sent in.

use serde_json::{Map, Value};

/// The three question types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    /// Pick one of a set of labels.
    Choice,
    /// Pick a level on an ordered scale, answered as an expected value.
    Score,
    /// Yes or no, answered as the probability of yes.
    Noul,
}

impl QType {
    /// The name used on the wire and in Laya's rendered text.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }

    /// The index Laya's type embedding uses: choice 0, score 1, noul 2.
    #[must_use]
    pub fn index(self) -> usize {
        match self {
            QType::Choice => 0,
            QType::Score => 1,
            QType::Noul => 2,
        }
    }
}

/// One choice option. The description is `None` when it was null.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceOption {
    /// The label, exactly as sent.
    pub label: String,
    /// The description, or `None` for null or when the criteria were a list of labels.
    pub description: Option<Value>,
}

/// The criteria of one question, in the shape its type takes.
#[derive(Debug, Clone, PartialEq)]
pub enum Criteria {
    /// Options in the order they were sent.
    Choice(Vec<ChoiceOption>),
    /// Levels, index 0 first. Each is echoed back in the `legend` exactly as sent.
    Score(Vec<Value>),
    /// The optional descriptions of the false and true outcomes.
    Noul {
        /// What false means, when given.
        when_false: Option<Value>,
        /// What true means, when given.
        when_true: Option<Value>,
    },
}

impl Criteria {
    /// The number of options the model scores for this question.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Criteria::Choice(o) => o.len(),
            Criteria::Score(l) => l.len(),
            Criteria::Noul { .. } => 2,
        }
    }

    /// Never true for a validated question, but clippy wants it next to `len`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One validated question.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// The key it was sent under.
    pub id: String,
    /// Its type.
    pub qtype: QType,
    /// The instructions. `None` when the field was left out, which is not the same as null: Laya
    /// renders null as the text `null`.
    pub instructions: Option<Value>,
    /// Its criteria.
    pub criteria: Criteria,
}

/// A validated request.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// The state, as sent. Never null.
    pub state: Value,
    /// The requested model, if any.
    pub model: Option<String>,
    /// The questions in request order.
    pub questions: Vec<Question>,
    /// The `kime` extension object, not yet interpreted.
    pub kime: Option<Map<String, Value>>,
}

/// The limits and the few rules where Jev and Laya disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The most questions in one request.
    pub max_questions: usize,
    /// The most options in one choice question.
    pub max_options: usize,
    /// The fewest levels a score question may have. Jev needs 2 and Laya takes 1.
    pub min_levels: usize,
    /// The most levels a score question may have.
    pub max_levels: usize,
    /// Whether an empty `questions` object is allowed. Jev rejects it and Laya answers it with no
    /// answers.
    pub allow_no_questions: bool,
}

impl Limits {
    /// The defaults from spec/03-api.md.
    pub const JEV: Limits = Limits {
        max_questions: 256,
        max_options: 255,
        min_levels: 2,
        max_levels: 32,
        allow_no_questions: false,
    };

    /// What Laya accepts, for requests to a compat model that did not send a `kime` object.
    pub const LAYA: Limits = Limits {
        max_questions: 256,
        max_options: 255,
        min_levels: 1,
        max_levels: 32,
        allow_no_questions: true,
    };
}

impl Default for Limits {
    fn default() -> Self {
        Limits::JEV
    }
}

/// One element of an error location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loc {
    /// An object key.
    Key(String),
    /// An array index.
    Index(usize),
}

impl From<&str> for Loc {
    fn from(s: &str) -> Self {
        Loc::Key(s.to_string())
    }
}

/// One validation problem, in the shape FastAPI reports it.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    /// Where the problem is, starting with `body`.
    pub loc: Vec<Loc>,
    /// A sentence for the person reading it.
    pub msg: String,
    /// A short machine readable kind, such as `missing` or `too_short`.
    pub kind: &'static str,
    /// The value that was rejected, or null when the field was missing.
    pub input: Value,
}

impl Problem {
    /// The FastAPI JSON form: `{"loc": [...], "msg": "...", "type": "...", "input": ...}`.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let loc: Vec<Value> = self
            .loc
            .iter()
            .map(|l| match l {
                Loc::Key(k) => Value::String(k.clone()),
                Loc::Index(i) => Value::from(*i),
            })
            .collect();
        serde_json::json!({"loc": loc, "msg": self.msg, "type": self.kind, "input": self.input})
    }
}

struct Problems(Vec<Problem>);

impl Problems {
    fn add(&mut self, loc: &[Loc], kind: &'static str, msg: impl Into<String>, input: &Value) {
        let mut full = vec![Loc::from("body")];
        full.extend_from_slice(loc);
        self.0.push(Problem { loc: full, msg: msg.into(), kind, input: input.clone() });
    }
}

/// Parse and validate a request body.
///
/// # Errors
///
/// Every problem found, in the order the fields appear.
pub fn parse(body: &Value, limits: &Limits) -> Result<Request, Vec<Problem>> {
    let mut p = Problems(Vec::new());
    let Some(obj) = body.as_object() else {
        p.add(&[], "dict_type", "the request body must be a JSON object", body);
        return Err(p.0);
    };

    let state = match obj.get("state") {
        None => {
            p.add(&["state".into()], "missing", "Field required", &Value::Null);
            Value::Null
        }
        Some(Value::Null) => {
            p.add(
                &["state".into()],
                "value_error",
                "state must not be null, send \"\" for an empty state",
                &Value::Null,
            );
            Value::Null
        }
        Some(s) => s.clone(),
    };

    let model = match obj.get("model") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            p.add(&["model".into()], "string_type", "Input should be a valid string", other);
            None
        }
    };

    let kime = match obj.get("kime") {
        None | Some(Value::Null) => None,
        Some(Value::Object(m)) => Some(m.clone()),
        Some(other) => {
            p.add(&["kime".into()], "dict_type", "Input should be a valid dictionary", other);
            None
        }
    };

    let mut questions = Vec::new();
    match obj.get("questions") {
        None => p.add(&["questions".into()], "missing", "Field required", &Value::Null),
        Some(Value::Object(qs)) => {
            if qs.is_empty() && !limits.allow_no_questions {
                p.add(
                    &["questions".into()],
                    "too_short",
                    "questions must have at least one question",
                    &Value::Object(qs.clone()),
                );
            }
            if qs.len() > limits.max_questions {
                p.add(
                    &["questions".into()],
                    "too_long",
                    format!(
                        "a request can have at most {} questions, got {}",
                        limits.max_questions,
                        qs.len()
                    ),
                    &Value::Null,
                );
            }
            for (id, q) in qs {
                if let Some(q) = question(id, q, limits, &mut p) {
                    questions.push(q);
                }
            }
        }
        Some(other) => {
            p.add(&["questions".into()], "dict_type", "Input should be a valid dictionary", other)
        }
    }

    if p.0.is_empty() { Ok(Request { state, model, questions, kime }) } else { Err(p.0) }
}

fn question(id: &str, q: &Value, limits: &Limits, p: &mut Problems) -> Option<Question> {
    let at = |rest: &[Loc]| {
        let mut loc = vec![Loc::from("questions"), Loc::Key(id.to_string())];
        loc.extend_from_slice(rest);
        loc
    };
    let Some(obj) = q.as_object() else {
        p.add(&at(&[]), "dict_type", "Input should be a valid dictionary", q);
        return None;
    };
    let qtype = match obj.get("type") {
        None => {
            p.add(&at(&["type".into()]), "missing", "Field required", &Value::Null);
            return None;
        }
        Some(Value::String(t)) if t == "choice" => QType::Choice,
        Some(Value::String(t)) if t == "score" => QType::Score,
        Some(Value::String(t)) if t == "noul" => QType::Noul,
        Some(other) => {
            p.add(
                &at(&["type".into()]),
                "literal_error",
                "Input should be 'choice', 'score' or 'noul'",
                other,
            );
            return None;
        }
    };
    let instructions = obj.get("instructions").cloned();
    let crit_loc = at(&["criteria".into()]);
    let crit = obj.get("criteria");
    let before = p.0.len();
    let criteria = match qtype {
        QType::Choice => choice(id, crit, limits, &crit_loc, p),
        QType::Score => score(id, crit, limits, &crit_loc, p),
        QType::Noul => noul(id, crit, &crit_loc, p),
    };
    if p.0.len() > before {
        return None;
    }
    Some(Question { id: id.to_string(), qtype, instructions, criteria })
}

fn with(loc: &[Loc], last: Loc) -> Vec<Loc> {
    let mut v = loc.to_vec();
    v.push(last);
    v
}

fn choice(
    id: &str,
    crit: Option<&Value>,
    limits: &Limits,
    loc: &[Loc],
    p: &mut Problems,
) -> Criteria {
    let mut options = Vec::new();
    match crit {
        None => p.add(loc, "missing", "Field required", &Value::Null),
        Some(Value::Object(m)) => {
            for (label, desc) in m {
                let description = if desc.is_null() { None } else { Some(desc.clone()) };
                options.push(ChoiceOption { label: label.clone(), description });
            }
        }
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(label) => options.push(ChoiceOption { label: label.clone(), description: None }),
                    other => p.add(&with(loc, Loc::Index(i)), "string_type", "Input should be a valid string", other),
                }
            }
        }
        Some(other) => p.add(
            loc,
            "choice_criteria_type",
            format!("choice question '{id}' takes criteria as an object of label to description, or an array of labels"),
            other,
        ),
    }
    if let Some(c) = crit.filter(|c| c.is_object() || c.is_array()) {
        let n = match c {
            Value::Object(m) => m.len(),
            Value::Array(a) => a.len(),
            _ => 0,
        };
        if n == 0 {
            p.add(
                loc,
                "too_short",
                format!("choice question '{id}' needs at least 1 option, got 0"),
                c,
            );
        }
        if n > limits.max_options {
            p.add(
                loc,
                "too_long",
                format!(
                    "choice question '{id}' has {n} options, the limit is {}",
                    limits.max_options
                ),
                &Value::Null,
            );
        }
    }
    let mut seen = std::collections::HashSet::new();
    for o in &options {
        let trimmed = o.label.trim();
        if !seen.insert(trimmed) {
            p.add(
                loc,
                "duplicate_label",
                format!("choice question '{id}' has the label '{trimmed}' more than once"),
                &Value::String(o.label.clone()),
            );
        }
    }
    Criteria::Choice(options)
}

fn score(
    id: &str,
    crit: Option<&Value>,
    limits: &Limits,
    loc: &[Loc],
    p: &mut Problems,
) -> Criteria {
    let levels: Vec<Value> = match crit {
        None => {
            p.add(loc, "missing", "Field required", &Value::Null);
            return Criteria::Score(Vec::new());
        }
        Some(Value::Array(items)) => items.clone(),
        // Old Python SDK clients send {"0": ..., "1": ...}. Accepted when the keys are exactly 0 to n-1.
        Some(Value::Object(m)) if (0..m.len()).all(|i| m.contains_key(&i.to_string())) => {
            (0..m.len()).map(|i| m[&i.to_string()].clone()).collect()
        }
        Some(other) => {
            p.add(
                loc,
                "score_criteria_type",
                format!("score question '{id}' takes criteria as an array of level descriptions, level 0 first"),
                other,
            );
            return Criteria::Score(Vec::new());
        }
    };
    let n = levels.len();
    if n < limits.min_levels {
        let unit = if limits.min_levels == 1 { "level" } else { "levels" };
        p.add(
            loc,
            "too_short",
            format!("score question '{id}' needs at least {} {unit}, got {n}", limits.min_levels),
            crit.unwrap_or(&Value::Null),
        );
    }
    if n > limits.max_levels {
        p.add(
            loc,
            "too_long",
            format!("score question '{id}' has {n} levels, the limit is {}", limits.max_levels),
            &Value::Null,
        );
    }
    Criteria::Score(levels)
}

fn noul(id: &str, crit: Option<&Value>, loc: &[Loc], p: &mut Problems) -> Criteria {
    let mut when_false = None;
    let mut when_true = None;
    match crit {
        None | Some(Value::Null) => {}
        Some(Value::Object(m)) => {
            // Keys match case insensitively, as in Laya, which lower cases str(key). A later key
            // wins over an earlier one that lower cases the same, as it does in a Python dict.
            for (k, v) in m {
                match k.to_lowercase().as_str() {
                    "true" => when_true = Some(v.clone()),
                    "false" => when_false = Some(v.clone()),
                    _ => p.add(
                        &with(loc, Loc::Key(k.clone())),
                        "noul_key",
                        format!("noul question '{id}' takes only the keys true and false, got '{k}'"),
                        v,
                    ),
                }
            }
        }
        Some(other) => p.add(
            loc,
            "noul_criteria_type",
            format!("noul question '{id}' takes criteria as an object with optional true and false descriptions, or none"),
            other,
        ),
    }
    Criteria::Noul { when_false, when_true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_spec_example_parses() {
        let body = json!({
            "state": {"ticket": {"subject": "Refund", "messages": [{"text": "charged twice"}]}},
            "model": "jev-latest",
            "questions": {
                "topic": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "money", "technical": null, "sales": ""}},
                "urgency": {"type": "score", "instructions": "How upset?", "criteria": ["Calm", "Frustrated", "Very angry"]},
                "refund": {"type": "noul", "instructions": "Wants a refund."}
            }
        });
        let r = parse(&body, &Limits::JEV).unwrap();
        assert_eq!(r.model.as_deref(), Some("jev-latest"));
        let ids: Vec<&str> = r.questions.iter().map(|q| q.id.as_str()).collect();
        assert_eq!(ids, ["topic", "urgency", "refund"]);
        let Criteria::Choice(o) = &r.questions[0].criteria else { panic!() };
        assert_eq!(o[1].description, None);
        assert_eq!(r.questions[2].criteria, Criteria::Noul { when_false: None, when_true: None });
    }

    #[test]
    fn every_problem_is_reported() {
        let body = json!({
            "questions": {
                "a": {"type": "score", "criteria": ["only"]},
                "b": {"type": "maybe"},
                "c": {"type": "choice", "criteria": {"x": 1, " x": 2}},
                "d": {"type": "noul", "criteria": {"True": "yes", "perhaps": "?"}},
                "e": {"type": "choice", "criteria": []}
            }
        });
        let errs = parse(&body, &Limits::JEV).unwrap_err();
        let msgs: Vec<&str> = errs.iter().map(|e| e.msg.as_str()).collect();
        assert_eq!(
            msgs,
            [
                "Field required",
                "score question 'a' needs at least 2 levels, got 1",
                "Input should be 'choice', 'score' or 'noul'",
                "choice question 'c' has the label 'x' more than once",
                "noul question 'd' takes only the keys true and false, got 'perhaps'",
                "choice question 'e' needs at least 1 option, got 0",
            ]
        );
        assert_eq!(errs[1].to_json()["loc"], json!(["body", "questions", "a", "criteria"]));
        assert_eq!(
            errs[4].to_json()["loc"],
            json!(["body", "questions", "d", "criteria", "perhaps"])
        );
    }

    #[test]
    fn laya_rules() {
        let body = json!({"state": "", "questions": {}});
        assert!(parse(&body, &Limits::JEV).is_err());
        assert!(parse(&body, &Limits::LAYA).unwrap().questions.is_empty());
        let one = json!({"state": "x", "questions": {"s": {"type": "score", "criteria": ["a"]}}});
        assert!(parse(&one, &Limits::JEV).is_err());
        assert!(parse(&one, &Limits::LAYA).is_ok());
    }

    #[test]
    fn old_sdk_score_objects() {
        let body = json!({"state": "x", "questions": {"s": {"type": "score", "criteria": {"1": "b", "0": "a"}}}});
        let r = parse(&body, &Limits::JEV).unwrap();
        assert_eq!(r.questions[0].criteria, Criteria::Score(vec![json!("a"), json!("b")]));
        let bad = json!({"state": "x", "questions": {"s": {"type": "score", "criteria": {"0": "a", "2": "b"}}}});
        assert!(parse(&bad, &Limits::JEV).is_err());
    }
}
