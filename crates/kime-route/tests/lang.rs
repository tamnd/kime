//! `kime_route::lang::analyse` against Laya's `laya.lang.analyse`. `tests/lang/laya.jsonl` has
//! MASSIVE texts and variants of them recorded by `tools/route/record.py`, and
//! `tests/lang/laya-tests.jsonl` every string in Laya's language and routing tests. Set
//! `KIME_ROUTE_CASES` to check a bigger recording the same way.

use std::io::{BufRead, BufReader};

use serde_json::Value;

fn check(path: &str) -> usize {
    let file = std::fs::File::open(path).unwrap();
    let (mut n, mut bad) = (0, Vec::new());
    for line in BufReader::new(file).lines() {
        let case: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let got = kime_route::lang::analyse(&case["state"]).to_json();
        if got != case["analysis"] {
            bad.push((case["state"].to_string(), got, case["analysis"].clone()));
        }
        n += 1;
    }
    for (state, got, want) in bad.iter().take(10) {
        let end = state.char_indices().nth(300).map_or(state.len(), |(i, _)| i);
        eprintln!("state {}\n  kime {got}\n  laya {want}", &state[..end]);
    }
    assert!(bad.is_empty(), "{path}: {} of {n} differ", bad.len());
    n
}

#[test]
fn massive_matches_laya() {
    assert!(check(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lang/laya.jsonl")) > 3000);
}

#[test]
fn laya_test_strings_match_laya() {
    assert!(check(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/lang/laya-tests.jsonl")) > 700);
}

#[test]
fn recording_matches_laya() {
    if let Ok(path) = std::env::var("KIME_ROUTE_CASES") {
        eprintln!("{} cases", check(&path));
    }
}
