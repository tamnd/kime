//! Token ids against the ones Hugging Face tokenizers gave for the parity fixtures, which
//! `tools/ref/laya_ref.py` recorded. The tokenizer files are not in the repository, so the test
//! reads them from `$KIME_MODELS/laya`, a download of `convaiinnovations/laya`. Without it the test
//! says so and passes, unless `KIME_REQUIRE_MODELS` is set, which CI sets.

use std::path::PathBuf;

use kime_tok::Tokenizer;
use serde_json::Value;

fn models() -> Option<PathBuf> {
    let dir = std::env::var_os("KIME_MODELS").map(PathBuf::from)?;
    dir.join("laya").is_dir().then_some(dir.join("laya"))
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../kime-eval/fixtures/parity")
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| u32::try_from(x.as_u64().unwrap()).unwrap()).collect()
}

fn check(model: &str, tok_dir: &str) {
    let Some(root) = models() else {
        assert!(std::env::var_os("KIME_REQUIRE_MODELS").is_none(), "KIME_MODELS is not set or has no laya folder");
        eprintln!("skipping {model}: set KIME_MODELS to a folder holding laya/");
        return;
    };
    let tok = Tokenizer::from_dir(root.join(tok_dir)).unwrap();
    let dump = std::fs::read_to_string(fixtures().join(format!("{model}.jsonl"))).unwrap();
    let (mut pieces, mut bad) = (0usize, Vec::new());
    for line in dump.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let Some(qs) = row["questions"].as_array() else { continue };
        for q in qs {
            let p = &q["pieces"];
            let mut check_one = |text: &str, want: Vec<u32>| {
                pieces += 1;
                let got = tok.encode(text);
                if got != want && bad.len() < 5 {
                    bad.push(format!("{} {}: {text:?}\n  want {want:?}\n  got  {got:?}", row["id"], q["qid"]));
                }
            };
            check_one(p["head"].as_str().unwrap(), ids(&p["head_ids"]));
            check_one(p["state"].as_str().unwrap(), ids(&p["state_ids"]));
            for (o, want) in p["options"].as_array().unwrap().iter().zip(p["option_ids"].as_array().unwrap()) {
                check_one(o.as_str().unwrap(), ids(want));
            }
        }
    }
    assert!(bad.is_empty(), "{model}: ids differ from Hugging Face\n{}", bad.join("\n"));
    assert!(pieces > 1000, "{model}: only {pieces} pieces checked");
    eprintln!("{model}: {pieces} pieces match Hugging Face");
}

#[test]
fn modernbert_ids_match_hugging_face() {
    check("laya", "tokenizer");
}

#[test]
fn mmbert_ids_match_hugging_face() {
    check("laya-multilingual", "multilingual/tokenizer");
}

#[test]
fn special_tokens_come_from_the_config() {
    let Some(root) = models() else { return };
    let en = Tokenizer::from_dir(root.join("tokenizer")).unwrap();
    let s = en.specials();
    assert_eq!((s.cls, s.sep, s.mask, s.pad), (50281, 50282, 50284, 50283));
    assert_eq!(en.mask_text(), "[MASK]");
    let ml = Tokenizer::from_dir(root.join("multilingual/tokenizer")).unwrap();
    let s = ml.specials();
    assert_eq!((s.cls, s.sep, s.mask, s.pad), (2, 1, 4, 0));
    assert_eq!(ml.mask_text(), "<mask>");
}
