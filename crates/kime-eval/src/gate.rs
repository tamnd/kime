//! The model behaviour tests of the release gate, spec/12-training.md and spec/15-testing.md: the
//! change each test makes to a question, the numbers read from the answers, and the report.
//! `kime eval --gate` in kime-cli runs them.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use kime_core::request::{ChoiceOption, Criteria, Question};
use serde_json::{Value, json};

use crate::suite::Scored;

/// The most score questions the mirror test reads, and the chunk test's questions.
pub const MIRROR_QUESTIONS: usize = 1_000;
/// Options in each question of the chunk consistency test.
pub const CHUNK_OPTIONS: usize = 64;
/// The chunk sizes it compares.
pub const CHUNK_SIZES: [usize; 3] = [8, 16, 32];
/// Questions it reads.
pub const CHUNK_QUESTIONS: usize = 300;

/// `q` with its score levels reversed, so level `i` becomes level `k - 1 - i`.
#[must_use]
pub fn mirror(q: &Question) -> Option<Question> {
    let Criteria::Score(levels) = &q.criteria else { return None };
    let levels = levels.iter().rev().cloned().collect();
    Some(Question { criteria: Criteria::Score(levels), ..q.clone() })
}

/// `q` with what true and false mean swapped, for a noul that says what they mean. A bare noul
/// has nothing to swap.
#[must_use]
pub fn swap(q: &Question) -> Option<Question> {
    let Criteria::Noul { when_false, when_true, labels } = &q.criteria else { return None };
    if when_false.is_none() && when_true.is_none() && labels.is_none() {
        return None;
    }
    let criteria = Criteria::Noul {
        when_false: when_true.clone(),
        when_true: when_false.clone(),
        labels: labels.as_ref().map(|(f, t)| (t.clone(), f.clone())),
    };
    Some(Question { criteria, ..q.clone() })
}

/// `q` with its choice labels renamed `option 1` to `option k`, for a choice whose every option
/// has a description, so the description is all the model has to go on.
#[must_use]
pub fn neutral(q: &Question) -> Option<Question> {
    let Criteria::Choice(opts) = &q.criteria else { return None };
    let described = |o: &ChoiceOption| match &o.description {
        None => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(v) => !v.is_null(),
    };
    if !opts.iter().all(described) {
        return None;
    }
    let opts = opts
        .iter()
        .enumerate()
        .map(|(i, o)| ChoiceOption { label: format!("option {}", i + 1), ..o.clone() })
        .collect();
    Some(Question { criteria: Criteria::Choice(opts), ..q.clone() })
}

/// `q` cut to `n` of its choice options, the gold one and others drawn with `seed`, in their
/// order, and where the gold is now. `None` for a choice with fewer options.
#[must_use]
pub fn sample(q: &Question, gold: usize, n: usize, seed: u64) -> Option<(Question, usize)> {
    let Criteria::Choice(opts) = &q.criteria else { return None };
    if opts.len() < n || gold >= opts.len() {
        return None;
    }
    let mut rest: Vec<usize> = (0..opts.len()).filter(|&i| i != gold).collect();
    let mut s = seed;
    // A partial Fisher Yates shuffle, drawing the n - 1 others.
    for i in 0..n - 1 {
        let j = i + (splitmix(&mut s) % (rest.len() - i) as u64) as usize;
        rest.swap(i, j);
    }
    let mut keep: Vec<usize> = rest[..n - 1].to_vec();
    keep.push(gold);
    keep.sort_unstable();
    let at = keep.iter().position(|&i| i == gold)?;
    let opts = keep.iter().map(|&i| opts[i].clone()).collect();
    Some((Question { criteria: Criteria::Choice(opts), ..q.clone() }, at))
}

/// A seed from a text, FNV-1a.
#[must_use]
pub fn seed(text: &str) -> u64 {
    text.bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The index of the largest value, the first on a tie.
#[must_use]
pub fn argmax(p: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > p[best] {
            best = i;
        }
    }
    best
}

/// The share of pairs where the mirrored answer's argmax is the mirror of the original's.
#[must_use]
pub fn mirrored(pairs: &[(Vec<f64>, Vec<f64>)]) -> f64 {
    let ok = pairs.iter().filter(|(a, b)| argmax(b) == a.len() - 1 - argmax(a)).count();
    share(ok, pairs.len())
}

/// The share of pairs of true probabilities where the swapped one is within `tol` of one minus
/// the original.
#[must_use]
pub fn polarity(pairs: &[(f64, f64)], tol: f64) -> f64 {
    let ok = pairs.iter().filter(|(a, b)| (b - (1.0 - a)).abs() <= tol).count();
    share(ok, pairs.len())
}

/// The score levels that are the gold of at least 1% of the score questions in `rows` and the
/// argmax of none, for each question id and number of levels.
#[must_use]
pub fn uncovered(rows: &[Scored]) -> BTreeMap<(String, usize), Vec<usize>> {
    let mut by_q: BTreeMap<(String, usize), (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for s in rows.iter().filter(|s| s.perm_of.is_none() && s.qtype.as_str() == "score") {
        let k = s.row.probs.len();
        let e = by_q.entry((s.question.clone(), k)).or_insert_with(|| (vec![0; k], vec![0; k]));
        e.0[s.row.gold] += 1;
        e.1[argmax(&s.row.probs)] += 1;
    }
    by_q.into_iter()
        .map(|(q, (gold, won))| {
            let n: usize = gold.iter().sum();
            let miss = (0..gold.len())
                .filter(|&i| gold[i] > 0 && gold[i] * 100 >= n && won[i] == 0)
                .collect();
            (q, miss)
        })
        .collect()
}

/// The share of rows where every pick is the same.
#[must_use]
pub fn agreement(picks: &[Vec<usize>]) -> f64 {
    let ok = picks.iter().filter(|p| p.windows(2).all(|w| w[0] == w[1])).count();
    share(ok, picks.len())
}

fn share(ok: usize, n: usize) -> f64 {
    if n == 0 { 0.0 } else { ok as f64 / n as f64 }
}

/// How a check came out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Within its threshold.
    Pass,
    /// Outside it.
    Fail,
    /// Not run, because its suites were not given or what it tests is not built yet.
    NotRun,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Fail => "fail",
            Status::NotRun => "not run",
        }
    }
}

/// One line of the gate.
#[derive(Debug, Clone)]
pub struct Check {
    /// The test.
    pub name: String,
    /// The suite it read, when it reads one.
    pub suite: String,
    /// What it measured.
    pub value: Option<f64>,
    /// The threshold, as the spec gives it.
    pub target: String,
    /// The questions or pairs behind the value.
    pub n: usize,
    /// How it came out.
    pub status: Status,
    /// Why it was not run, or what failed.
    pub note: String,
}

impl Check {
    /// A check that ran: it passes when `ok` holds.
    #[must_use]
    pub fn ran(name: &str, suite: &str, value: f64, target: &str, n: usize, ok: bool) -> Check {
        let status = if n == 0 {
            Status::NotRun
        } else if ok {
            Status::Pass
        } else {
            Status::Fail
        };
        let note = if n == 0 { "no questions to test".into() } else { String::new() };
        Check {
            name: name.into(),
            suite: suite.into(),
            value: (n > 0).then_some(value),
            target: target.into(),
            n,
            status,
            note,
        }
    }

    /// A check that did not run, and why.
    #[must_use]
    pub fn skipped(name: &str, target: &str, why: &str) -> Check {
        Check {
            name: name.into(),
            suite: String::new(),
            value: None,
            target: target.into(),
            n: 0,
            status: Status::NotRun,
            note: why.into(),
        }
    }
}

/// Whether no check failed.
#[must_use]
pub fn passed(checks: &[Check]) -> bool {
    checks.iter().all(|c| c.status != Status::Fail)
}

/// The gate as markdown.
#[must_use]
pub fn markdown(title: &str, checks: &[Check]) -> String {
    let mut s = format!(
        "# {title}\n\n| Test | Suite | Value | Target | n | Result | Note |\n|---|---|---:|---|---:|---|---|\n"
    );
    for c in checks {
        let v = c.value.map_or_else(String::new, |v| format!("{v:.4}"));
        let _ = writeln!(
            s,
            "| {} | {} | {v} | {} | {} | {} | {} |",
            c.name,
            c.suite,
            c.target,
            c.n,
            c.status.as_str(),
            c.note
        );
    }
    let failed = checks.iter().filter(|c| c.status == Status::Fail).count();
    let ran = checks.iter().filter(|c| c.status != Status::NotRun).count();
    let _ = writeln!(
        s,
        "\n{} of {} checks ran and {failed} failed, so the checkpoint {}.",
        ran,
        checks.len(),
        if failed == 0 { "passes" } else { "does not pass" }
    );
    s
}

/// The gate as JSON.
#[must_use]
pub fn to_json(checks: &[Check]) -> Value {
    let rows: Vec<Value> = checks
        .iter()
        .map(|c| {
            json!({"test": c.name, "suite": c.suite, "value": c.value, "target": c.target, "n": c.n,
                "result": c.status.as_str(), "note": c.note})
        })
        .collect();
    json!({"passed": passed(checks), "checks": rows})
}

#[cfg(test)]
mod tests {
    use super::*;
    use kime_core::request::{Limits, parse};

    fn question(body: &Value) -> Question {
        let req = parse(&json!({"state": "x", "questions": {"q": body}}), &Limits::LAYA).unwrap();
        req.questions.into_iter().next().unwrap()
    }

    #[test]
    fn mirror_reverses_levels() {
        let q = question(
            &json!({"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}),
        );
        let m = mirror(&q).unwrap();
        assert_eq!(m.criteria, Criteria::Score(vec![json!("hi"), json!("mid"), json!("lo")]));
        assert!(mirror(&question(&json!({"type": "noul", "instructions": "i"}))).is_none());
        let pairs = vec![
            (vec![0.1, 0.2, 0.7], vec![0.6, 0.3, 0.1]),
            (vec![0.8, 0.1, 0.1], vec![0.8, 0.1, 0.1]),
        ];
        assert!((mirrored(&pairs) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn swap_needs_meanings() {
        let bare = question(&json!({"type": "noul", "instructions": "i"}));
        assert!(swap(&bare).is_none());
        let q = question(&json!({"type": "noul", "instructions": "i",
            "criteria": {"true": "it is spam", "false": "it is not"}}));
        let Criteria::Noul { when_true, when_false, .. } = swap(&q).unwrap().criteria else {
            panic!()
        };
        assert_eq!((when_true, when_false), (Some(json!("it is not")), Some(json!("it is spam"))));
        assert!((polarity(&[(0.9, 0.12), (0.9, 0.3)], 0.05) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn neutral_needs_descriptions() {
        let q = question(&json!({"type": "choice", "instructions": "i",
            "criteria": {"yes": "they agree", "no": "they do not"}}));
        let Criteria::Choice(o) = neutral(&q).unwrap().criteria else { panic!() };
        assert_eq!(o[1].label, "option 2");
        assert_eq!(o[1].description, Some(json!("they do not")));
        let bare = question(
            &json!({"type": "choice", "instructions": "i", "criteria": {"yes": null, "no": "x"}}),
        );
        assert!(neutral(&bare).is_none());
    }

    #[test]
    fn sample_keeps_the_gold() {
        let crit: serde_json::Map<String, Value> =
            (0..77).map(|i| (format!("l{i}"), Value::Null)).collect();
        let q = question(&json!({"type": "choice", "instructions": "i", "criteria": crit}));
        for g in [0, 40, 76] {
            let (s, at) = sample(&q, g, 64, seed("case")).unwrap();
            let Criteria::Choice(o) = &s.criteria else { panic!() };
            assert_eq!(o.len(), 64);
            assert_eq!(o[at].label, format!("l{g}"));
            let idx: Vec<usize> = o.iter().map(|x| x.label[1..].parse().unwrap()).collect();
            assert!(idx.windows(2).all(|w| w[0] < w[1]), "options keep their order");
        }
        assert_eq!(sample(&q, 3, 64, 7).unwrap().0, sample(&q, 3, 64, 7).unwrap().0);
        assert!(sample(&q, 3, 78, 7).is_none());
    }

    #[test]
    fn agreement_and_report() {
        assert!((agreement(&[vec![1, 1, 1], vec![1, 2, 1]]) - 0.5).abs() < 1e-12);
        let checks = vec![
            Check::ran("Score mirror", "en.sst5", 0.99, ">= 0.98", 600, true),
            Check::skipped("Routing", ">= 0.99", "the router is not built yet"),
        ];
        assert!(passed(&checks));
        assert!(markdown("gate", &checks).contains("1 of 2 checks ran and 0 failed"));
        let fail = [Check::ran("Noul polarity", "app.phishing", 0.5, ">= 0.97", 400, false)];
        assert!(!passed(&fail));
        assert_eq!(to_json(&fail)["passed"], json!(false));
    }
}
