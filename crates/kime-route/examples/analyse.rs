//! Times `lang::analyse` over the states of a recording from `tools/route/record.py`.
//!
//!     cargo run --release -p kime-route --example analyse -- cases.jsonl

use std::io::{BufRead, BufReader};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: analyse <cases.jsonl>");
    let states: Vec<serde_json::Value> = BufReader::new(std::fs::File::open(path).unwrap())
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(&l.unwrap()).unwrap()["state"].take())
        .collect();
    let mut english = 0;
    let t = Instant::now();
    for s in &states {
        english += usize::from(kime_route::lang::analyse(s).is_english);
    }
    let took = t.elapsed();
    println!(
        "{} states, {english} English, {:.2} s, {:.2} us per state",
        states.len(),
        took.as_secs_f64(),
        took.as_secs_f64() * 1e6 / states.len() as f64
    );
}
