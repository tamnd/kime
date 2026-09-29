//! The results of a run: one summary per suite, as JSON, Markdown and a table of rows.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde_json::{Map, Value, json};

use crate::metrics::{Extra, Hard, Row, bootstrap, extra, hard};
use crate::parquet::{self, Column};
use crate::suite::{Scored, type_name};

/// Bootstrap resamples for the intervals, and their seed.
const RESAMPLES: usize = 1000;
const SEED: u64 = 13;

/// How often the answer moves when the options are reordered, over the questions of lines with
/// `perm_of` and the lines they copy.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Flips {
    /// Questions with at least one reordered copy.
    pub questions: usize,
    /// Reordered copies compared against their original.
    pub copies: usize,
    /// The share of copies whose predicted option differs from the original's.
    pub rate: f64,
    /// The share of questions where any copy differs.
    pub any: f64,
}

/// The flip rate of `scored`, `None` when no line is a reordered copy.
#[must_use]
pub fn flips(scored: &[Scored]) -> Option<Flips> {
    let original: BTreeMap<(&str, &str), &str> = scored
        .iter()
        .filter(|s| s.perm_of.is_none())
        .map(|s| ((s.case.as_str(), s.question.as_str()), s.pred.as_str()))
        .collect();
    let mut by_q: BTreeMap<(&str, &str), (usize, bool)> = BTreeMap::new();
    let mut f = Flips::default();
    for s in scored {
        let Some(of) = &s.perm_of else { continue };
        let key = (of.as_str(), s.question.as_str());
        let Some(p0) = original.get(&key) else { continue };
        let moved = *p0 != s.pred;
        let e = by_q.entry(key).or_default();
        e.0 += 1;
        e.1 |= moved;
        f.copies += 1;
        f.rate += f64::from(u8::from(moved));
    }
    if f.copies == 0 {
        return None;
    }
    f.questions = by_q.len();
    f.rate /= f.copies as f64;
    f.any = by_q.values().filter(|v| v.1).count() as f64 / f.questions as f64;
    Some(f)
}

/// One suite's numbers.
#[derive(Debug, Clone)]
pub struct Summary {
    /// The suite's name, its file name without `.jsonl`.
    pub suite: String,
    /// Every question.
    pub all: Hard,
    /// The 95 percent intervals of accuracy and ECE.
    pub accuracy_ci: [f64; 2],
    pub ece_ci: [f64; 2],
    /// Soft label and score metrics.
    pub extra: Extra,
    /// Questions with a gold that got no row.
    pub dropped: usize,
    /// By question type, when the suite has more than one.
    pub by_type: BTreeMap<String, Hard>,
    /// By tag name, then value.
    pub by_tag: BTreeMap<String, BTreeMap<String, Hard>>,
    /// The flip rate, when the suite has reordered copies.
    pub flips: Option<Flips>,
}

/// Summarizes one suite's rows.
#[must_use]
pub fn summarize(suite: &str, scored: &[Scored], dropped: usize) -> Summary {
    let rows: Vec<Row> = scored.iter().map(|s| s.row.clone()).collect();
    let (accuracy_ci, ece_ci) = bootstrap(&rows, RESAMPLES, SEED);
    let mut types: BTreeMap<String, Vec<Row>> = BTreeMap::new();
    let mut tags: BTreeMap<String, BTreeMap<String, Vec<Row>>> = BTreeMap::new();
    for s in scored {
        types.entry(type_name(s.qtype).into()).or_default().push(s.row.clone());
        for (k, v) in &s.tags {
            tags.entry(k.clone()).or_default().entry(v.clone()).or_default().push(s.row.clone());
        }
    }
    let by_type = match types.len() {
        0 | 1 => BTreeMap::new(),
        _ => types.iter().map(|(k, r)| (k.clone(), hard(r))).collect(),
    };
    let by_tag = tags
        .iter()
        .map(|(k, vs)| (k.clone(), vs.iter().map(|(v, r)| (v.clone(), hard(r))).collect()))
        .collect();
    Summary {
        suite: suite.into(),
        all: hard(&rows),
        accuracy_ci,
        ece_ci,
        extra: extra(&rows),
        dropped,
        by_type,
        by_tag,
        flips: flips(scored),
    }
}

fn num(x: f64) -> Value {
    if x.is_finite() { json!((x * 1e6).round() / 1e6) } else { Value::Null }
}

fn hard_json(h: &Hard) -> Value {
    json!({
        "n": h.n, "accuracy": num(h.accuracy), "macro_f1": num(h.macro_f1), "ece": num(h.ece),
        "brier": num(h.brier), "nll": num(h.nll), "aurc": num(h.aurc),
        "mean_confidence": num(h.mean_confidence), "acc_at_50_coverage": num(h.acc_at_50),
        "acc_at_80_coverage": num(h.acc_at_80), "auroc": num(h.auroc),
        "gold_zero_rate": num(h.gold_zero),
    })
}

impl Summary {
    /// The summary as JSON, with the metric names of Laya's results file.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut v = hard_json(&self.all);
        let Some(m) = v.as_object_mut() else { return v };
        m.insert("accuracy_ci".into(), json!([num(self.accuracy_ci[0]), num(self.accuracy_ci[1])]));
        m.insert("ece_ci".into(), json!([num(self.ece_ci[0]), num(self.ece_ci[1])]));
        m.insert("dropped_questions".into(), json!(self.dropped));
        for (k, x) in [
            ("soft_acc", self.extra.soft_accuracy),
            ("brier_soft", self.extra.soft_brier),
            ("score_mae", self.extra.score_mae),
            ("within_1", self.extra.within_1),
        ] {
            if let Some(x) = x {
                m.insert(k.into(), num(x));
            }
        }
        if let Some(f) = &self.flips {
            m.insert(
                "option_order".into(),
                json!({"questions": f.questions, "copies": f.copies, "flip_rate": num(f.rate),
                       "any_flip_rate": num(f.any)}),
            );
        }
        if !self.by_type.is_empty() {
            m.insert(
                "by_question_type".into(),
                self.by_type
                    .iter()
                    .map(|(k, h)| (k.clone(), hard_json(h)))
                    .collect::<Map<_, _>>()
                    .into(),
            );
        }
        if !self.by_tag.is_empty() {
            m.insert(
                "by_tag".into(),
                self.by_tag
                    .iter()
                    .map(|(k, vs)| {
                        (
                            k.clone(),
                            vs.iter()
                                .map(|(v, h)| (v.clone(), hard_json(h)))
                                .collect::<Map<String, Value>>()
                                .into(),
                        )
                    })
                    .collect::<Map<_, _>>()
                    .into(),
            );
        }
        json!({ self.suite.clone(): v })
    }
}

/// The Markdown report of a run over several suites.
#[must_use]
pub fn markdown(title: &str, summaries: &[Summary]) -> String {
    let mut s = format!("# {title}\n\n");
    s.push_str("| Suite | Questions | Dropped | Accuracy | 95% interval | Macro F1 | ECE | Brier | NLL | Acc at 50% coverage | AUROC |\n");
    s.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for x in summaries {
        let h = &x.all;
        let _ = writeln!(
            s,
            "| {} | {} | {} | {:.4} | {:.4} to {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |",
            x.suite,
            h.n,
            x.dropped,
            h.accuracy,
            x.accuracy_ci[0],
            x.accuracy_ci[1],
            h.macro_f1,
            h.ece,
            h.brier,
            h.nll,
            h.acc_at_50,
            h.auroc
        );
    }
    let flipped: Vec<(&Summary, &Flips)> =
        summaries.iter().filter_map(|x| x.flips.as_ref().map(|f| (x, f))).collect();
    if !flipped.is_empty() {
        s.push_str("\n| Suite | Questions | Reordered copies | Flip rate | Questions with any flip |\n|---|---|---|---|---|\n");
        for (x, f) in flipped {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {:.4} | {:.4} |",
                x.suite, f.questions, f.copies, f.rate, f.any
            );
        }
    }
    let extras: Vec<&Summary> = summaries.iter().filter(|x| x.extra != Extra::default()).collect();
    if !extras.is_empty() {
        s.push_str("\n| Suite | Soft accuracy | Soft Brier | Score MAE | Within 1 |\n|---|---|---|---|---|\n");
        let f = |x: Option<f64>| x.map_or_else(String::new, |x| format!("{x:.4}"));
        for x in extras {
            let e = &x.extra;
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |",
                x.suite,
                f(e.soft_accuracy),
                f(e.soft_brier),
                f(e.score_mae),
                f(e.within_1)
            );
        }
    }
    for x in summaries.iter().filter(|x| !x.by_type.is_empty() || !x.by_tag.is_empty()) {
        let _ = write!(
            s,
            "\n## {}\n\n| Group | Questions | Accuracy | ECE | Brier | NLL |\n|---|---|---|---|---|---|\n",
            x.suite
        );
        let groups = x.by_type.iter().map(|(k, h)| (format!("type {k}"), h)).chain(
            x.by_tag
                .iter()
                .flat_map(|(k, vs)| vs.iter().map(move |(v, h)| (format!("{k} {v}"), h))),
        );
        for (g, h) in groups {
            let _ = writeln!(
                s,
                "| {g} | {} | {:.4} | {:.4} | {:.4} | {:.4} |",
                h.n, h.accuracy, h.ece, h.brier, h.nll
            );
        }
    }
    s
}

/// One line per scored question, tab separated, with a header: the suite, case, question, type,
/// gold index, predicted index, confidence, whether it was right, and the probabilities.
#[must_use]
pub fn rows_tsv(suite: &str, scored: &[Scored], header: bool) -> String {
    let mut s = String::new();
    if header {
        s.push_str("suite\tcase\tquestion\ttype\tgold\tpred\tconfidence\tcorrect\tprobs\n");
    }
    for x in scored {
        let r = &x.row;
        let probs: Vec<String> = r.probs.iter().map(|p| format!("{p:.6}")).collect();
        let _ = writeln!(
            s,
            "{suite}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}",
            x.case,
            x.question,
            type_name(x.qtype),
            r.gold,
            r.pred(),
            r.confidence(),
            u8::from(r.correct()),
            probs.join(",")
        );
    }
    s
}

/// The rows of every suite as a Parquet file, with the columns of [`rows_tsv`] and the
/// probabilities as a list of doubles.
#[must_use]
pub fn rows_parquet(suites: &[(String, Vec<Scored>)]) -> Vec<u8> {
    let all = || suites.iter().flat_map(|(n, s)| s.iter().map(move |x| (n, x)));
    let text =
        |f: &dyn Fn(&str, &Scored) -> String| Column::Str(all().map(|(n, x)| f(n, x)).collect());
    let int = |f: &dyn Fn(&Scored) -> usize| Column::I64(all().map(|(_, x)| f(x) as i64).collect());
    parquet::write(&[
        ("suite", text(&|n, _| n.to_string())),
        ("case", text(&|_, x| x.case.clone())),
        ("question", text(&|_, x| x.question.clone())),
        ("type", text(&|_, x| type_name(x.qtype).to_string())),
        ("gold", int(&|x| x.row.gold)),
        ("pred", int(&|x| x.row.pred())),
        ("confidence", Column::F64(all().map(|(_, x)| x.row.confidence()).collect())),
        ("correct", Column::Bool(all().map(|(_, x)| x.row.correct()).collect())),
        ("probs", Column::F64List(all().map(|(_, x)| x.row.probs.clone()).collect())),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::{load, probs_json, score};

    #[test]
    fn counts_flips_by_option_name() {
        let q = |a: &str, b: &str| {
            format!(
                r#""questions": {{"t": {{"type": "choice", "instructions": "i", "criteria": {{"{a}": null, "{b}": null}}}}}}, "gold": {{"t": "a"}}"#
            )
        };
        let text = [
            format!(r#"{{"id": "x", "state": "s", {}}}"#, q("a", "b")),
            format!(r#"{{"id": "x~1", "perm_of": "x", "state": "s", {}}}"#, q("b", "a")),
            format!(r#"{{"id": "x~2", "perm_of": "x", "state": "s", {}}}"#, q("b", "a")),
        ]
        .join("\n");
        let cases = load(&text).unwrap();
        // The original and the first copy pick a, the second copy picks b.
        let ans = [
            serde_json::json!({"t": {"probabilities": {"a": 0.7, "b": 0.3}}}),
            serde_json::json!({"t": {"probabilities": {"a": 0.6, "b": 0.4}}}),
            serde_json::json!({"t": {"probabilities": {"a": 0.4, "b": 0.6}}}),
        ];
        let scored: Vec<Scored> = cases
            .iter()
            .zip(&ans)
            .flat_map(|(c, a)| score(c, |q| probs_json(q, &a[&q.id])).0)
            .collect();
        let f = flips(&scored).unwrap();
        assert_eq!((f.questions, f.copies), (1, 2));
        assert!((f.rate - 0.5).abs() < 1e-12 && (f.any - 1.0).abs() < 1e-12);
        assert!(flips(&scored[..1]).is_none());
    }
}
