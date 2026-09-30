//! `kime eval --gate`: the model behaviour tests of spec/15-testing.md, run after the quality
//! suites on the same model, each checked against its threshold.

use std::collections::HashMap;

use kime::Kime;
use kime::request::{QType, Question, Request};
use kime_eval::gate::{
    CHUNK_OPTIONS, CHUNK_QUESTIONS, CHUNK_SIZES, Check, MIRROR_QUESTIONS, agreement, argmax,
    mirror, mirrored, neutral, polarity, sample, seed, swap, uncovered,
};
use kime_eval::report::Summary;
use kime_eval::suite::{Case, Scored, probs};
use serde_json::{Map, Value, json};

/// The suite the chunk test draws its questions from.
const CHUNK_SUITE: &str = "en.banking77_full";

/// One suite as the quality run left it.
pub(crate) struct Ran<'a> {
    pub name: &'a str,
    pub cases: &'a [Case],
    pub scored: &'a [Scored],
    pub summary: &'a Summary,
}

/// Every check, in the order of spec/15-testing.md. `baseline` is the results.json of the
/// previous release, for the calibration check.
pub(crate) fn checks(
    kime: &Kime,
    ran: &[Ran<'_>],
    baseline: Option<&Value>,
) -> Result<Vec<Check>, String> {
    let mut out = Vec::new();
    let order: Vec<&Ran<'_>> = ran.iter().filter(|r| r.summary.flips.is_some()).collect();
    for r in &order {
        let f = r.summary.flips.as_ref().expect("filtered on flips");
        out.push(Check::ran("Order flips", r.name, f.rate, "<= 0.02", f.copies, f.rate <= 0.02));
    }
    if order.is_empty() {
        out.push(Check::skipped("Order flips", "<= 0.02", "no order.* suite was given"));
    }
    let before = out.len();
    for r in ran {
        let (orig, reqs) = changed(r, QType::Score, mirror);
        if orig.is_empty() {
            continue;
        }
        let got = ask(kime, &reqs)?;
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = orig
            .into_iter()
            .zip(got)
            .filter_map(|(a, b)| Some((a.row.probs.clone(), b?)))
            .collect();
        let v = mirrored(&pairs);
        out.push(Check::ran("Score mirror", r.name, v, ">= 0.98", pairs.len(), v >= 0.98));
    }
    if out.len() == before {
        out.push(Check::skipped("Score mirror", ">= 0.98", "no suite has score questions"));
    }
    let before = out.len();
    for r in ran {
        let (orig, reqs) = changed(r, QType::Noul, swap);
        if orig.is_empty() {
            continue;
        }
        let got = ask(kime, &reqs)?;
        let pairs: Vec<(f64, f64)> =
            orig.into_iter().zip(got).filter_map(|(a, b)| Some((a.row.probs[1], b?[1]))).collect();
        let v = polarity(&pairs, 0.05);
        out.push(Check::ran(
            "Noul polarity",
            r.name,
            v,
            ">= 0.97 within 0.05",
            pairs.len(),
            v >= 0.97,
        ));
    }
    if out.len() == before {
        out.push(Check::skipped("Noul polarity", ">= 0.97", "no suite has nouls with criteria"));
    }
    let before = out.len();
    for r in ran.iter().filter(|r| !r.name.starts_with("order.")) {
        let n = r.scored.iter().filter(|s| s.qtype == QType::Score && s.perm_of.is_none()).count();
        if n == 0 {
            continue;
        }
        let miss: Vec<String> = uncovered(r.scored)
            .into_iter()
            .filter(|(_, m)| !m.is_empty())
            .map(|((q, k), m)| format!("{q} of {k} levels never picks {m:?}"))
            .collect();
        let mut c = Check::ran(
            "Level coverage",
            r.name,
            miss.len() as f64,
            "0 questions",
            n,
            miss.is_empty(),
        );
        c.note = miss.join(", ");
        out.push(c);
    }
    if out.len() == before {
        out.push(Check::skipped("Level coverage", "0 questions", "no suite has score questions"));
    }
    let before = out.len();
    for r in ran.iter().filter(|r| english(r.name)) {
        let (orig, reqs) = changed(r, QType::Choice, neutral);
        if orig.is_empty() {
            continue;
        }
        let got = ask(kime, &reqs)?;
        let (mut a, mut b, mut n) = (0, 0, 0);
        for (s, p) in orig.into_iter().zip(got) {
            let Some(p) = p else { continue };
            n += 1;
            a += usize::from(argmax(&s.row.probs) == s.row.gold);
            b += usize::from(argmax(&p) == s.row.gold);
        }
        let d = if n == 0 { 0.0 } else { (b as f64 - a as f64) / n as f64 };
        out.push(Check::ran("Label neutrality", r.name, d, "within 0.01", n, d.abs() <= 0.01));
    }
    if out.len() == before {
        out.push(Check::skipped(
            "Label neutrality",
            "within 0.01",
            "no English suite has choices with every option described",
        ));
    }
    out.push(match ran.iter().find(|r| r.name == CHUNK_SUITE) {
        Some(r) => chunk_check(kime, r)?,
        None => {
            Check::skipped("Chunk consistency", ">= 0.99", &format!("{CHUNK_SUITE} was not given"))
        }
    });
    out.push(Check::skipped(
        "Segment consistency",
        ">= 0.99",
        "segment mode is not built yet (M4)",
    ));
    out.push(Check::skipped("Routing", ">= 0.99", "the routing set is not built yet"));
    match baseline {
        Some(b) => out.extend(calibration(ran, b)),
        None => out.push(Check::skipped(
            "Calibration regression",
            "ECE up by <= 0.01",
            "no --baseline was given",
        )),
    }
    out.push(Check::skipped(
        "Quality targets",
        "spec/13-benchmarks.md",
        "read report.md against the targets of the model's tier",
    ));
    Ok(out)
}

/// Suites in English, by name, which the label neutrality test reads.
fn english(name: &str) -> bool {
    !name.starts_with("order.")
        && (name.starts_with("en.")
            || name.starts_with("app.")
            || name == "typed_decisions"
            || name.ends_with(".en"))
}

/// The scored questions of type `t` that `change` applies to, up to [`MIRROR_QUESTIONS`], and a
/// request for each with only the changed question.
fn changed<'a>(
    r: &Ran<'a>,
    t: QType,
    change: fn(&Question) -> Option<Question>,
) -> (Vec<&'a Scored>, Vec<Request>) {
    let by_id: HashMap<&str, &Request> =
        r.cases.iter().filter_map(|c| Some((c.id.as_str(), c.request.as_ref().ok()?))).collect();
    let (mut orig, mut reqs) = (Vec::new(), Vec::new());
    for s in r.scored.iter().filter(|s| s.qtype == t && s.perm_of.is_none()) {
        let Some(req) = by_id.get(s.case.as_str()) else { continue };
        let Some(q) = req.questions.iter().find(|q| q.id == s.question).and_then(change) else {
            continue;
        };
        orig.push(s);
        reqs.push(Request {
            state: req.state.clone(),
            model: None,
            questions: vec![q],
            kime: None,
        });
        if orig.len() == MIRROR_QUESTIONS {
            break;
        }
    }
    (orig, reqs)
}

/// The probabilities of the one question of each request, `None` where there is no answer.
fn ask(kime: &Kime, reqs: &[Request]) -> Result<Vec<Option<Vec<f64>>>, String> {
    let mut out = Vec::with_capacity(reqs.len());
    for chunk in reqs.chunks(crate::eval::CHUNK) {
        let res = kime.decide_batch(chunk).map_err(|e| e.to_string())?;
        for (req, r) in chunk.iter().zip(res) {
            out.push(r.get(&req.questions[0].id).map(probs));
        }
    }
    Ok(out)
}

/// The chunk consistency test: [`CHUNK_QUESTIONS`] choices of the suite cut to
/// [`CHUNK_OPTIONS`] options, each scored with every size in [`CHUNK_SIZES`].
fn chunk_check(kime: &Kime, r: &Ran<'_>) -> Result<Check, String> {
    let by_id: HashMap<&str, &Request> =
        r.cases.iter().filter_map(|c| Some((c.id.as_str(), c.request.as_ref().ok()?))).collect();
    let mut base = Vec::new();
    for s in r.scored.iter().filter(|s| s.qtype == QType::Choice && s.perm_of.is_none()) {
        let Some(req) = by_id.get(s.case.as_str()) else { continue };
        let Some(q) = req.questions.iter().find(|q| q.id == s.question) else { continue };
        let key = format!("{}/{}", s.case, s.question);
        if let Some((q, _)) = sample(q, s.row.gold, CHUNK_OPTIONS, seed(&key)) {
            base.push((req.state.clone(), q));
        }
        if base.len() == CHUNK_QUESTIONS {
            break;
        }
    }
    let target = ">= 0.99";
    if base.is_empty() {
        return Ok(Check::skipped(
            "Chunk consistency",
            target,
            &format!("{} has no choice of {CHUNK_OPTIONS} or more options", r.name),
        ));
    }
    let mut picks = vec![Vec::new(); base.len()];
    for size in CHUNK_SIZES {
        let mut kime_ext = Map::new();
        kime_ext.insert("chunk".into(), json!(size));
        let reqs: Vec<Request> = base
            .iter()
            .map(|(state, q)| Request {
                state: state.clone(),
                model: None,
                questions: vec![q.clone()],
                kime: Some(kime_ext.clone()),
            })
            .collect();
        for (p, got) in picks.iter_mut().zip(ask(kime, &reqs)?) {
            p.push(got.map_or(usize::MAX, |g| argmax(&g)));
        }
    }
    picks.retain(|p| !p.contains(&usize::MAX));
    let v = agreement(&picks);
    Ok(Check::ran("Chunk consistency", r.name, v, target, picks.len(), v >= 0.99))
}

/// The ECE of each suite against the same suite in the previous release's results.json.
fn calibration(ran: &[Ran<'_>], baseline: &Value) -> Vec<Check> {
    let mut out = Vec::new();
    for r in ran {
        let Some(old) =
            baseline.pointer(&format!("/suites/{}/ece", r.name)).and_then(Value::as_f64)
        else {
            continue;
        };
        let d = r.summary.all.ece - old;
        let mut c = Check::ran(
            "Calibration regression",
            r.name,
            d,
            "ECE up by <= 0.01",
            r.summary.all.n,
            d <= 0.01,
        );
        c.note = format!("{old:.4} to {:.4}", r.summary.all.ece);
        out.push(c);
    }
    if out.is_empty() {
        out.push(Check::skipped(
            "Calibration regression",
            "ECE up by <= 0.01",
            "the baseline has none of these suites",
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_suites() {
        for n in
            ["en.ag_news", "app.support_triage", "massive_intent.en", "xnli.en", "typed_decisions"]
        {
            assert!(english(n), "{n}");
        }
        for n in ["xnli.de", "order.xnli.en", "massive_scenario.zh-CN", "mind2web.task"] {
            assert!(!english(n), "{n}");
        }
    }
}
