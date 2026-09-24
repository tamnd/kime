//! `kime_route::router` against Laya's `Router._route` on the states `tools/route/route-record.py`
//! recorded in `tests/lang/route.jsonl`: 20 test texts per source and language from the language
//! id data, plus every string in Laya's language and routing tests. The detection is always
//! Laya's. The checkpoint and the reason are Laya's too, except where the language identifier
//! decided, and there the checkpoint has to match the label when there is one.

use std::io::{BufRead, BufReader};

use kime_route::router::detect;
use serde_json::Value;

#[test]
fn matches_laya_router() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lang/route.jsonl");
    let (mut n, mut same, mut lid, mut lid_right, mut lid_labelled) = (0, 0, 0, 0, 0);
    let mut bad = Vec::new();
    for line in BufReader::new(std::fs::File::open(path).unwrap()).lines() {
        let case: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let (state, laya) = (&case["state"], &case["laya"]);
        let d = detect(state, true);
        n += 1;
        let detection = d.detection.as_ref().map_or(Value::Null, |a| a.to_json());
        if detection != laya["detection"] {
            bad.push(format!("detection {state}: {detection} vs {}", laya["detection"]));
            continue;
        }
        let model = match d.english {
            Some(false) => "multilingual",
            _ => "english",
        };
        if d.reason.contains("language identifier") {
            lid += 1;
            if let Some(lang) = case["lang"].as_str() {
                lid_labelled += 1;
                lid_right += usize::from((lang == "en") == (model == "english"));
            }
        } else if model == laya["model"] && d.reason == laya["reason"] {
            same += 1;
        } else {
            bad.push(format!(
                "{state}: {model} {:?} vs {} {}",
                d.reason, laya["model"], laya["reason"]
            ));
        }
    }
    eprintln!(
        "{n} states: {same} as Laya's router, {lid} decided by the identifier, {lid_right} of {lid_labelled} labelled ones right"
    );
    for b in bad.iter().take(10) {
        eprintln!("{b}");
    }
    assert!(bad.is_empty(), "{} of {n} differ", bad.len());
    assert!(n > 2000);
    assert!(lid_right * 100 >= lid_labelled * 95);
}

/// The labelled routing set `tools/route/routing-set.py` writes: MASSIVE, papluca, AG News and
/// LeetCode test texts, MASSIVE with the accents stripped, the cases from Laya's routing issues,
/// Latin script languages the identifier never saw and English that shares words with other
/// languages. A server whose default is English answers `default` with the English checkpoint.
#[test]
fn routing_set() {
    use std::collections::BTreeMap;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lang/routing-set.jsonl");
    let mut by: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    let mut wrong = Vec::new();
    for line in BufReader::new(std::fs::File::open(path).unwrap()).lines() {
        let case: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let want = case["want"].as_str().unwrap();
        let got = match detect(&case["text"], true).english {
            Some(true) => "english",
            Some(false) => "multilingual",
            None => "default",
        };
        let right = |got: &str| got == want || (want == "english" && got == "default");
        let laya = case["laya"].as_str().unwrap();
        let source = case["source"].as_str().unwrap();
        let group = if source.starts_with("laya#") { "laya issues" } else { source };
        let e = by.entry(group.to_string()).or_default();
        e[0] += 1;
        e[1] += usize::from(right(laya));
        e[2] += usize::from(right(got));
        if !right(got) {
            wrong.push(format!("{source} {want} {got}: {}", case["text"]));
        }
    }
    let mut total = [0; 3];
    for (group, c) in &by {
        let pct = |n: usize| 100.0 * n as f64 / c[0] as f64;
        eprintln!("{group:<18} {:>5} laya {:>6.2}% kime {:>6.2}%", c[0], pct(c[1]), pct(c[2]));
        for k in 0..3 {
            total[k] += c[k];
        }
    }
    let pct = |n: usize| 100.0 * n as f64 / total[0] as f64;
    eprintln!(
        "{:<18} {:>5} laya {:>6.2}% kime {:>6.2}%",
        "all",
        total[0],
        pct(total[1]),
        pct(total[2])
    );
    for w in &wrong {
        eprintln!("{w}");
    }
    assert!(total[0] >= 5000);
    assert!(total[2] * 100 >= total[0] * 99, "{} of {} right", total[2], total[0]);
    for group in ["laya issues", "collision", "leetcode"] {
        assert_eq!(by[group][2], by[group][0], "{group}");
    }
}
