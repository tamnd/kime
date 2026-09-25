//! [`Kime::embed`] against Laya's `embed_fn_from_agent` on the texts in
//! `kime-eval/fixtures/parity/embed.json`, recorded by `tools/python/laya_embed.py`. On the CPU in
//! FP32 every value is within 5e-3 of Laya's and every row has a cosine of at least 0.99999 to it.
//! A text gets the same bits alone as in a batch, and INT8 is refused.
//!
//! It needs the laya checkpoint, under `$KIME_MODELS/laya` or in the Hugging Face cache, and
//! passes with a note when it is missing unless `KIME_REQUIRE_WEIGHTS` is set.

use kime_engine::{Device, Error, Kime, Precision};
use serde_json::Value;

fn model() -> String {
    std::env::var("KIME_MODELS").map_or_else(|_| "laya".into(), |m| format!("{m}/laya"))
}

fn open(precision: Precision) -> Option<Kime> {
    let b = Kime::builder().model(model()).device(Device::Cpu { threads: 0 }).precision(precision);
    match b.build() {
        Ok(k) => Some(k),
        Err(e) => {
            assert!(std::env::var_os("KIME_REQUIRE_WEIGHTS").is_none(), "{e}");
            eprintln!("skipping: {e}");
            None
        }
    }
}

fn cosine(a: &[f32], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * y).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
    let nb: f64 = b.iter().map(|y| y * y).sum();
    dot / (na * nb).sqrt()
}

#[test]
fn matches_laya() {
    let Some(kime) = open(Precision::F32) else { return };
    let path = format!("{}/../kime-eval/fixtures/parity/embed.json", env!("CARGO_MANIFEST_DIR"));
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let texts: Vec<&str> =
        fixture["texts"].as_array().unwrap().iter().map(|t| t.as_str().unwrap()).collect();
    for (max_length, rows) in fixture["emb"].as_object().unwrap() {
        let got = kime.embed(&texts, max_length.parse().unwrap()).unwrap();
        assert_eq!(got.len(), texts.len());
        let (mut worst_abs, mut worst_cos) = (0f64, 1f64);
        for (i, (g, w)) in got.iter().zip(rows.as_array().unwrap()).enumerate() {
            let w: Vec<f64> = w.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
            assert_eq!(g.len(), w.len());
            let abs = g.iter().zip(&w).map(|(a, b)| (f64::from(*a) - b).abs()).fold(0.0, f64::max);
            let cos = cosine(g, &w);
            assert!(
                abs < 5e-3 && cos > 0.99999,
                "text {i} at {max_length}: abs {abs:.2e}, cosine {cos}"
            );
            (worst_abs, worst_cos) = (worst_abs.max(abs), worst_cos.min(cos));
        }
        eprintln!(
            "max_length {max_length}: worst abs {worst_abs:.2e}, worst cosine {worst_cos:.7}"
        );
        // Alone or batched with the rest, a text gets the same bits.
        for (i, t) in texts.iter().enumerate() {
            let alone = kime.embed(&[t], max_length.parse().unwrap()).unwrap();
            assert!(
                alone[0].iter().zip(&got[i]).all(|(a, b)| a.to_bits() == b.to_bits()),
                "text {i}"
            );
        }
    }
    assert!(kime.embed(&[], 512).unwrap().is_empty());
}

#[test]
fn int8_is_refused() {
    let Some(kime) = open(Precision::Int8) else { return };
    assert!(matches!(kime.embed(&["card lost"], 512), Err(Error::Unsupported(_))));
}
