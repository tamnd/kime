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
