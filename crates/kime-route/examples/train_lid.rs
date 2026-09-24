//! Trains the language identifier in `src/lid.rs` and reports how routing does with it.
//!
//! ```sh
//! tools/route/lid-data.sh                      # writes ~/data/kime/lid/data.jsonl
//! cargo run --release -p kime-route --example train_lid -- ~/data/kime/lid/data.jsonl src/lid.bin
//! cargo run --release -p kime-route --example train_lid -- ~/data/kime/lid/data.jsonl
//! ```
//!
//! With an output path it trains on the train split, picks the threshold on the val split and
//! writes the model. Without one it scores the model built into the crate. Either way it prints
//! routing on the test split next to Laya's rules alone.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use kime_route::lang::analyse_text;
use kime_route::lid::{BUCKETS, MIN_WORDS, Model, OVERRULE, features, model, words};
use serde_json::Value;

struct Row {
    lang: String,
    source: String,
    split: String,
    english: bool,
    /// What Laya's rules say.
    laya_english: bool,
    /// Whether Laya's rules take it for Latin script text.
    latin: bool,
    words: usize,
    feats: Vec<(u32, f32)>,
}

fn row(text: &str, lang: &str, source: &str, split: &str) -> Row {
    let a = analyse_text(text);
    Row {
        english: lang == "en",
        lang: lang.to_string(),
        source: source.to_string(),
        split: split.to_string(),
        laya_english: a.is_english,
        latin: a.script == "latin",
        words: words(text),
        feats: features(text),
    }
}

fn load(path: &str) -> Vec<Row> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let lines: Vec<String> = BufReader::new(file).lines().map_while(Result::ok).collect();
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = lines.len().div_ceil(threads).max(1);
    std::thread::scope(|s| {
        let parts: Vec<_> = lines
            .chunks(chunk)
            .map(|part| {
                s.spawn(move || {
                    part.iter()
                        .enumerate()
                        .flat_map(|(i, l)| {
                            let v: Value = serde_json::from_str(l).unwrap_or_default();
                            let text = v["text"].as_str().unwrap_or_default();
                            let lang = v["lang"].as_str().unwrap_or_default();
                            let source = v["source"].as_str().unwrap_or_default();
                            let split = v["split"].as_str().unwrap_or_default();
                            let mut out = vec![row(text, lang, source, split)];
                            // Short English is mostly MASSIVE commands otherwise, so the longer
                            // texts also give a window of 2 to 6 words for training.
                            if split == "train" && source != "massive" {
                                let w: Vec<&str> = text.split_whitespace().collect();
                                let len = 2 + i % 5;
                                if w.len() > len {
                                    let at = (i * 7919) % (w.len() - len);
                                    let window = w[at..at + len].join(" ");
                                    out.push(row(
                                        &window,
                                        lang,
                                        &format!("{source}-window"),
                                        split,
                                    ));
                                }
                            }
                            out
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        parts.into_iter().flat_map(|p| p.join().unwrap_or_default()).collect()
    })
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Adagrad on the weighted log loss. Every source and class pair weighs the same in total, so
/// the 60,000 news articles do not drown the short English commands.
fn train(rows: &[&Row], epochs: usize) -> Model {
    let mut groups: BTreeMap<(&str, bool), usize> = BTreeMap::new();
    for r in rows {
        *groups.entry((r.source.as_str(), r.english)).or_default() += 1;
    }
    let per_group = rows.len() as f32 / groups.len() as f32;
    let weight: Vec<f32> =
        rows.iter().map(|r| per_group / groups[&(r.source.as_str(), r.english)] as f32).collect();

    let (mut w, mut g2) = (vec![0.0f32; BUCKETS], vec![1e-6f32; BUCKETS]);
    let (mut bias, mut b2) = (0.0f32, 1e-6f32);
    let (lr, l2) = (0.2f32, 1e-7f32);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            order.swap(i, (seed % (i as u64 + 1)) as usize);
        }
        let mut loss = 0.0f64;
        for &i in &order {
            let r = rows[i];
            let z = bias + r.feats.iter().map(|&(j, v)| w[j as usize] * v).sum::<f32>();
            let p = sigmoid(z);
            let y = if r.english { 1.0 } else { 0.0 };
            loss -= f64::from(weight[i])
                * f64::from(if r.english { p } else { 1.0 - p }).max(1e-9).ln();
            let g = weight[i] * (p - y);
            for &(j, v) in &r.feats {
                let j = j as usize;
                let gj = g * v + l2 * w[j];
                g2[j] += gj * gj;
                w[j] -= lr * gj / g2[j].sqrt();
            }
            b2 += g * g;
            bias -= lr * g / b2.sqrt();
        }
        eprintln!("epoch {epoch}: loss {:.5}", loss / rows.len() as f64);
    }
    Model { weights: w, bias, threshold: 0.5 }
}

/// The router's answer, as `lid::english_model` gives it.
fn routes_english(m: &Model, r: &Row, threshold: f32) -> bool {
    if r.latin && r.words >= MIN_WORDS {
        let p = sigmoid(m.logit(&r.feats));
        p >= threshold && (r.laya_english || p >= OVERRULE)
    } else {
        r.laya_english
    }
}

fn report(m: &Model, rows: &[&Row]) {
    let mut by: BTreeMap<(String, String), [usize; 3]> = BTreeMap::new();
    for r in rows {
        let key = (r.source.clone(), r.lang.clone());
        let e = by.entry(key).or_default();
        e[0] += 1;
        e[1] += usize::from(r.laya_english == r.english);
        e[2] += usize::from(routes_english(m, r, m.threshold) == r.english);
    }
    let mut tot = [0usize; 3];
    let mut tot_src: BTreeMap<String, [usize; 3]> = BTreeMap::new();
    println!("{:<10} {:<6} {:>7} {:>8} {:>8}", "source", "lang", "texts", "laya %", "kime %");
    for ((src, lang), c) in &by {
        let pct = |n: usize| 100.0 * n as f64 / c[0] as f64;
        println!("{src:<10} {lang:<6} {:>7} {:>8.2} {:>8.2}", c[0], pct(c[1]), pct(c[2]));
        let t = tot_src.entry(src.clone()).or_default();
        for k in 0..3 {
            t[k] += c[k];
            tot[k] += c[k];
        }
    }
    for (src, c) in &tot_src {
        let pct = |n: usize| 100.0 * n as f64 / c[0] as f64;
        println!("{src:<10} {:<6} {:>7} {:>8.2} {:>8.2}", "all", c[0], pct(c[1]), pct(c[2]));
    }
    let pct = |n: usize| 100.0 * n as f64 / tot[0] as f64;
    println!("{:<10} {:<6} {:>7} {:>8.2} {:>8.2}", "all", "all", tot[0], pct(tot[1]), pct(tot[2]));
    let en: Vec<_> = rows.iter().filter(|r| r.english).collect();
    let miss = en.iter().filter(|r| !routes_english(m, r, m.threshold)).count();
    let laya_miss = en.iter().filter(|r| !r.laya_english).count();
    println!(
        "English sent to the multilingual model: laya {laya_miss}, kime {miss} of {}",
        en.len()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let data = args.first().map_or("data.jsonl", String::as_str);
    let t = Instant::now();
    let rows = load(data);
    eprintln!("{} texts featurized in {:.1} s", rows.len(), t.elapsed().as_secs_f64());
    let split = |s: &str| rows.iter().filter(|r| r.split == s).collect::<Vec<_>>();

    let m = if let Some(out) = args.get(1) {
        let t = Instant::now();
        let mut m = train(&split("train"), 5);
        eprintln!("trained in {:.1} s", t.elapsed().as_secs_f64());
        m = Model::from_bytes(&m.to_bytes()).unwrap_or_else(|e| panic!("{e}"));

        // The threshold with the fewest mistakes on the val split, where English sent to the
        // multilingual model counts a hundred times. AG News has some French, German and Spanish
        // articles labelled English, so this cannot ask for no English mistakes at all.
        let val = split("val");
        let cost = |th: f32| {
            let (mut en, mut other) = (0, 0);
            for r in &val {
                if routes_english(&m, r, th) != r.english {
                    if r.english { en += 1 } else { other += 1 }
                }
            }
            (en, other, 100 * en + other)
        };
        let grid: Vec<f32> = (1..100).map(|i| i as f32 / 100.0).collect();
        let best = grid.iter().copied().min_by_key(|&th| cost(th).2).unwrap_or(0.5);
        for th in [0.9, 0.5, 0.2, 0.1, 0.05, 0.01, best] {
            let (en, other, _) = cost(th);
            eprintln!("val threshold {th:.2}: {en} English and {other} other texts misrouted");
        }
        m.threshold = best;
        std::fs::write(out, m.to_bytes()).unwrap_or_else(|e| panic!("{out}: {e}"));
        eprintln!("wrote {out}, threshold {:.2}", m.threshold);
        m
    } else {
        model().clone()
    };
    report(&m, &split("test"));
}
