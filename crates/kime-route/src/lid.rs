//! A small language identifier for Latin script text. It answers the one question routing needs,
//! whether the English checkpoint can read the text, with a logistic regression over hashed
//! character n-grams and words. Laya's word lists in [`crate::lang`] know eleven languages, so
//! Latin script text in any other language without diacritics goes to the English checkpoint;
//! this is trained on 38 Latin script languages and does not need a list per language.
//!
//! The weights are in `lid.bin`, written by `examples/train_lid.rs` from the data that
//! `tools/route/lid-data.sh` builds. The file is the bucket count, the bias, the threshold and a
//! scale as little endian `u32`, `f32`, `f32` and `f32`, then one `i16` per bucket.

use std::sync::OnceLock;

use serde_json::Value;

use crate::lang::{MAX_CHARS, analyse_text, state_text};

/// Hash buckets for the features.
pub const BUCKETS: usize = 1 << 18;

/// Texts with fewer words than this keep the answer of Laya's rules: one word says too little.
pub const MIN_WORDS: usize = 2;

/// The English probability from which the identifier overrules Laya's rules when they take Latin
/// script text for another language, which they do for English with a word like `os` or `van`.
pub const OVERRULE: f32 = 0.99;

/// The most words of a text that are read. Past this the answer does not change.
pub const MAX_WORDS: usize = 256;

/// The words of a text as [`features`] splits it: runs of letters.
#[must_use]
pub fn words(text: &str) -> usize {
    text.split(|c: char| !crate::lang::is_alpha(c)).filter(|w| !w.is_empty()).count()
}

fn fnv(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325 ^ seed.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The features of a text: bucket and value pairs, sorted by bucket, with unit length. Words are
/// runs of letters after lowercasing. Each word gives its character 1 to 4 grams with a space on
/// either side and the whole word.
#[must_use]
pub fn features(text: &str) -> Vec<(u32, f32)> {
    let lower = text.to_lowercase();
    let mut ids: Vec<u32> = Vec::new();
    let mut padded: Vec<u8> = Vec::with_capacity(64);
    let mut ends: Vec<usize> = Vec::with_capacity(64);
    let words = lower.split(|c: char| !crate::lang::is_alpha(c)).filter(|w| !w.is_empty());
    for w in words.take(MAX_WORDS) {
        ids.push((fnv(9, w.as_bytes()) as usize % BUCKETS) as u32);
        padded.clear();
        ends.clear();
        padded.push(b' ');
        ends.push(1);
        for c in w.chars() {
            let mut b = [0; 4];
            padded.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
            ends.push(padded.len());
        }
        padded.push(b' ');
        ends.push(padded.len());
        // ends[i] is where character i of " w " ends; character 0 starts at 0.
        let n_chars = ends.len();
        for n in 1..=4 {
            for i in 0..n_chars.saturating_sub(n - 1) {
                let start = if i == 0 { 0 } else { ends[i - 1] };
                let end = ends[i + n - 1];
                let gram = &padded[start..end];
                if gram == b" " {
                    continue;
                }
                ids.push((fnv(n as u64, gram) as usize % BUCKETS) as u32);
            }
        }
    }
    ids.sort_unstable();
    let mut out: Vec<(u32, f32)> = Vec::new();
    for id in ids {
        match out.last_mut() {
            Some((last, v)) if *last == id => *v += 1.0,
            _ => out.push((id, 1.0)),
        }
    }
    let mut norm = 0.0f32;
    for (_, v) in &mut out {
        *v = v.ln_1p();
        norm += *v * *v;
    }
    let norm = norm.sqrt().max(1e-6);
    for (_, v) in &mut out {
        *v /= norm;
    }
    out
}

/// Trained weights.
#[derive(Debug, Clone)]
pub struct Model {
    /// One weight per bucket. Positive means English.
    pub weights: Vec<f32>,
    pub bias: f32,
    /// The English probability under which a text is taken as not English.
    pub threshold: f32,
}

impl Model {
    /// The log odds that the text is English.
    #[must_use]
    pub fn logit(&self, feats: &[(u32, f32)]) -> f32 {
        self.bias + feats.iter().map(|&(i, v)| self.weights[i as usize] * v).sum::<f32>()
    }

    /// The probability that the English checkpoint can read the text.
    #[must_use]
    pub fn p_english(&self, text: &str) -> f32 {
        let z = self.logit(&features(text));
        1.0 / (1.0 + (-z).exp())
    }

    /// Whether the text reads as English.
    #[must_use]
    pub fn is_english(&self, text: &str) -> bool {
        self.p_english(text) >= self.threshold
    }

    /// The model in the `lid.bin` format, with the weights rounded to 16 bits.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let max = self.weights.iter().fold(0.0f32, |m, w| m.max(w.abs())).max(1e-12);
        let scale = max / f32::from(i16::MAX);
        let mut out = Vec::with_capacity(16 + 2 * self.weights.len());
        out.extend_from_slice(&(self.weights.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.bias.to_le_bytes());
        out.extend_from_slice(&self.threshold.to_le_bytes());
        out.extend_from_slice(&scale.to_le_bytes());
        for w in &self.weights {
            out.extend_from_slice(&((w / scale).round() as i16).to_le_bytes());
        }
        out
    }

    /// Reads the `lid.bin` format.
    ///
    /// # Errors
    ///
    /// When the bytes are not a model for [`BUCKETS`] buckets.
    pub fn from_bytes(b: &[u8]) -> Result<Model, String> {
        let word = |i: usize| -> [u8; 4] { b[i..i + 4].try_into().unwrap_or_default() };
        if b.len() < 16 {
            return Err("language id model is truncated".into());
        }
        let n = u32::from_le_bytes(word(0)) as usize;
        if n != BUCKETS || b.len() != 16 + 2 * n {
            return Err(format!("language id model has {n} buckets, want {BUCKETS}"));
        }
        let scale = f32::from_le_bytes(word(12));
        let weights = b[16..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from(i16::from_le_bytes(*c)) * scale)
            .collect();
        Ok(Model {
            weights,
            bias: f32::from_le_bytes(word(4)),
            threshold: f32::from_le_bytes(word(8)),
        })
    }
}

/// The model that ships with kime.
///
/// # Panics
///
/// Never: the embedded file is checked by the tests.
#[must_use]
pub fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| {
        Model::from_bytes(include_bytes!("lid.bin")).unwrap_or_else(|e| unreachable!("{e}"))
    })
}

/// Whether a request's state goes to the English checkpoint. For Latin script text of two words
/// or more the identifier has to say English, and Laya's rules too unless the identifier is sure
/// past [`OVERRULE`]. Everything else keeps the answer of Laya's rules.
#[must_use]
pub fn english_model(state: &Value) -> bool {
    english_model_text(&state_text(state, MAX_CHARS))
}

/// [`english_model`] for text already flattened by [`state_text`].
#[must_use]
pub fn english_model_text(text: &str) -> bool {
    let a = analyse_text(text);
    if a.script != "latin" || words(text) < MIN_WORDS {
        return a.is_english;
    }
    let m = model();
    let p = m.p_english(text);
    p >= m.threshold && (a.is_english || p >= OVERRULE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_unit_length() {
        let f = features("Hello, World! hello");
        let norm: f32 = f.iter().map(|(_, v)| v * v).sum();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(f.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(features("123 !!").is_empty());
    }

    #[test]
    fn round_trip() {
        let m = model();
        let back = Model::from_bytes(&m.to_bytes()).unwrap();
        assert!((back.bias - m.bias).abs() < 1e-9);
        assert!(m.weights.iter().zip(&back.weights).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn routes() {
        for t in [
            "I was charged twice for my subscription, please refund the second payment",
            "wake me up at seven tomorrow",
            "EUR",
            "Apple releases Mac OS X 10.3.7 Update, a maintenance release for its operating system.",
            "Given an array of integers nums, return the indices of the two numbers that add up to target.",
        ] {
            assert!(english_model_text(t), "{t}");
        }
        for t in [
            "saya mau pesan tiket ke jakarta besok pagi",
            "maak my wakker om sewe uur more",
            "ninataka kununua tiketi ya ndege kesho",
            "gusto kong mag-order ng pizza ngayong gabi",
            "wek mij morgen om zeven uur",
            "vekk meg klokken syv i morgen",
        ] {
            assert!(!english_model_text(t), "{t}");
        }
    }
}
