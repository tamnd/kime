//! `clean_email_body` against Laya's own: every body Laya's `tests/test_email.py` cleans, and a
//! generated corpus of emails built from the pieces the cleaner looks for, each with what Laya
//! answered. `tools/email/record.py` writes `tests/email/laya.json`.

use kime_core::email::clean_email_body;
use serde_json::Value;

#[test]
fn same_as_laya() {
    // KIME_EMAIL_CASES points at a bigger recording than the committed one.
    let path = std::env::var("KIME_EMAIL_CASES")
        .unwrap_or_else(|_| format!("{}/tests/email/laya.json", env!("CARGO_MANIFEST_DIR")));
    let cases: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut wrong = Vec::new();
    for (i, c) in cases.iter().enumerate() {
        let body = c["body"].as_str().unwrap();
        let max = c["max_chars"].as_u64().unwrap() as usize;
        let got = clean_email_body(body, max);
        if got != c["clean"].as_str().unwrap() {
            wrong.push(format!(
                "case {i} ({}): {body:?}\n  laya {:?}\n  kime {got:?}",
                c["source"], c["clean"]
            ));
        }
    }
    assert!(wrong.is_empty(), "{} of {} differ:\n{}", wrong.len(), cases.len(), wrong.join("\n"));
}
