//! The compat input pipeline against Laya 0.3.7 on the 200 parity cases: request parsing and
//! rendering in kime-core, then tokenizing and layout in kime-tok, compared with the text, ids and
//! markers `tools/ref/laya_ref.py` recorded. The rendered text is checked always. The ids need the
//! tokenizer files from `$KIME_MODELS/laya`, and are skipped without them unless
//! `KIME_REQUIRE_MODELS` is set.

use std::path::PathBuf;

use kime_core::render::{compat_question, compat_state};
use kime_core::request::{Limits, parse};
use kime_tok::Tokenizer;
use kime_tok::layout::{CompatBudget, Cut};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/parity")
}

fn tokenizer(dir: &str) -> Option<Tokenizer> {
    let root = std::env::var_os("KIME_MODELS").map(PathBuf::from).map(|d| d.join("laya"));
    match root.filter(|r| r.is_dir()) {
        Some(r) => Some(Tokenizer::from_dir(r.join(dir)).unwrap()),
        None => {
            assert!(
                std::env::var_os("KIME_REQUIRE_MODELS").is_none(),
                "KIME_MODELS is not set or has no laya folder"
            );
            None
        }
    }
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| u32::try_from(x.as_u64().unwrap()).unwrap()).collect()
}

fn check(model: &str, tok_dir: &str, mask: &str, budget: CompatBudget) {
    let tok = tokenizer(tok_dir);
    if let Some(t) = &tok {
        assert_eq!(t.mask_text(), mask);
    }
    let cases: Vec<Value> = std::fs::read_to_string(fixtures().join("cases.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let dumps: Vec<Value> = std::fs::read_to_string(fixtures().join(format!("{model}.jsonl")))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(cases.len(), dumps.len());
    let (mut texts, mut seqs, mut bad) = (0usize, 0usize, Vec::new());
    for (case, dump) in cases.iter().zip(&dumps) {
        assert_eq!(case["id"], dump["id"]);
        let req = parse(case, &Limits::LAYA).unwrap_or_else(|e| panic!("{}: {e:?}", case["id"]));
        let state = compat_state(&req.state, mask);
        let state_ids = tok.as_ref().map(|t| t.encode_state(&state));
        let Some(qs) = dump["questions"].as_array() else { continue };
        for (q, want) in req.questions.iter().zip(qs) {
            assert_eq!(q.id, want["qid"].as_str().unwrap());
            let p = &want["pieces"];
            let text = compat_question(q, mask);
            texts += 1;
            let want_opts: Vec<&str> =
                p["options"].as_array().unwrap().iter().map(|o| o.as_str().unwrap()).collect();
            if text.head != p["head"] || text.options != want_opts || state != p["state"] {
                bad.push(format!(
                    "{} {}: rendered text differs\n  head {:?}\n  want {}",
                    case["id"], q.id, text.head, p["head"]
                ));
            }
            let (Some(tok), Some(state_ids)) = (&tok, &state_ids) else { continue };
            let seq = tok.compat_sequence(&text.head, &text.options, state_ids, budget, Cut::Tail);
            seqs += 1;
            if seq.ids != u32s(&want["ids"]) || seq.markers != u32s(&want["markers"]) {
                bad.push(format!("{} {}: ids or markers differ", case["id"], q.id));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{model}: {} questions differ from Laya\n{}",
        bad.len(),
        bad.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(texts > 600, "{model}: only {texts} questions checked");
    eprintln!(
        "{model}: {texts} questions render as Laya does, {seqs} laid out to the same ids and markers"
    );
}

#[test]
fn laya_english() {
    check("laya", "tokenizer", "[MASK]", CompatBudget { max_len: 512, head_max_len: 192 });
}

#[test]
fn laya_multilingual() {
    check(
        "laya-multilingual",
        "multilingual/tokenizer",
        "<mask>",
        CompatBudget { max_len: 1024, head_max_len: 256 },
    );
}
