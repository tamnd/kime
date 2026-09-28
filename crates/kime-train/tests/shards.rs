//! Rendering training lines with Laya's tokenizer, and reading whole shards.
//!
//! The tokenizer comes with the weights, so the test reads `$KIME_MODELS/laya` and passes with a
//! note when it is missing. With `KIME_SHARDS` set to the shards/ folder convert.py wrote, it also
//! reads every shard and prints how many examples it gives and how many it skips.

use std::path::PathBuf;
use std::time::Instant;

use kime_model::Model;
use kime_train::data::{Renderer, Skip, lines};

fn renderer() -> Option<Renderer> {
    let dir = std::env::var_os("KIME_MODELS").map(|m| PathBuf::from(m).join("laya"));
    let Some(dir) = dir.filter(|d| d.join("model.safetensors").is_file()) else {
        assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "no weights for laya");
        eprintln!("skipping: set KIME_MODELS to a folder holding laya/ with its weights");
        return None;
    };
    Some(Renderer::new(&Model::open(&dir).unwrap()).unwrap())
}

#[test]
fn renders_a_line() {
    let Some(r) = renderer() else { return };
    let line = r#"{"id": "x", "state": {"text": "I am still waiting on my card?"},
        "questions": {"intent": {"type": "choice", "instructions": "Which intent does `text` express?",
            "criteria": {"card_arrival": "the card has not arrived", "lost_card": "the card is lost", "top_up": "a top up"}},
          "urgent": {"type": "noul", "instructions": "Is it urgent?"}},
        "targets": {"intent": {"probs": [1.0, 0.0, 0.0], "hard": 0, "weight": 1.0},
          "urgent": {"probs": [0.7, 0.3], "weight": 0.5}}}"#;
    let (ex, skipped) = r.render(&line.replace('\n', " "));
    assert!(skipped.is_empty(), "{skipped:?}");
    assert_eq!(ex.len(), 2);
    assert_eq!((ex[0].q.markers.len(), ex[0].q.qtype, ex[0].hard), (3, 0, Some(0)));
    assert_eq!((ex[1].q.markers.len(), ex[1].q.qtype, ex[1].hard, ex[1].weight), (2, 2, None, 0.5));
    // A target of the wrong length is skipped, the other question still renders.
    let (ex, skipped) = r.render(&line.replace('\n', " ").replace("[0.7, 0.3]", "[1.0]"));
    assert_eq!((ex.len(), skipped), (1, vec![Skip::Target("urgent".into())]));
}

#[test]
fn reads_shards() {
    let Some(dir) = std::env::var_os("KIME_SHARDS").map(PathBuf::from) else {
        eprintln!("skipping: set KIME_SHARDS to the shards/ folder convert.py wrote");
        return;
    };
    let Some(r) = renderer() else { return };
    let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    let start = Instant::now();
    let (mut rows, mut examples, mut tokens, mut too_long, mut other) = (0, 0, 0usize, 0, 0);
    for f in &files {
        let (mut fr, mut fe) = (0, 0);
        for line in lines(f).unwrap() {
            let (ex, skipped) = r.render(&line.unwrap());
            fr += 1;
            fe += ex.len();
            tokens += ex.iter().map(|e| e.q.ids.len()).sum::<usize>();
            for s in skipped {
                match s {
                    Skip::TooLong(_) => too_long += 1,
                    s => {
                        if other < 5 {
                            eprintln!("{}: {s:?}", f.display());
                        }
                        other += 1;
                    }
                }
            }
        }
        eprintln!("{}: {fr} rows, {fe} examples", f.file_name().unwrap().to_string_lossy());
        rows += fr;
        examples += fe;
    }
    eprintln!(
        "{} shards, {rows} rows, {examples} examples, {tokens} tokens, {too_long} too long, {other} other skips, {:.0}s",
        files.len(),
        start.elapsed().as_secs_f64()
    );
    assert_eq!(other, 0);
}
