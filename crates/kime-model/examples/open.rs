//! Times opening a compat model, from its Laya directory and from a `.kime` pack of it, and the
//! `.kime` hash check, and building the tokenizer from the packed files. The pack is written next to the given path, so it needs that much free disk.
//!
//!     cargo run --release -p kime-model --example open -- <models>/laya [scratch dir]

use std::time::Instant;

use kime_model::Model;

fn best<T>(runs: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut last = None;
    for _ in 0..runs {
        let t = Instant::now();
        let v = f();
        best = best.min(t.elapsed().as_secs_f64());
        last = Some(v);
    }
    (best * 1e3, last.expect("runs > 0"))
}

fn main() {
    let dir = std::env::args().nth(1).expect("usage: open <laya dir> [scratch dir]");
    let scratch =
        std::env::args().nth(2).unwrap_or_else(|| std::env::temp_dir().display().to_string());
    let packed =
        std::path::Path::new(&scratch).join(format!("open-bench-{}.kime", std::process::id()));
    let (dir_ms, m) = best(10, || Model::open(&dir).unwrap());
    let mb = m.tensors.data_bytes() as f64 / 1e6;
    let t = Instant::now();
    let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&packed).unwrap());
    m.pack(&mut w).unwrap();
    drop(w);
    let pack_ms = t.elapsed().as_secs_f64() * 1e3;
    let (kime_ms, k) = best(10, || Model::open(&packed).unwrap());
    let (verify_ms, ()) = best(3, || k.verify().unwrap());
    let (tok_ms, _) = best(3, || {
        let json = k.file("tokenizer/tokenizer.json").unwrap();
        kime_tok::Tokenizer::from_bytes(json, k.file("tokenizer/tokenizer_config.json")).unwrap()
    });
    println!(
        "{}: {} tensors, {mb:.0} MB, open dir {dir_ms:.1} ms, pack {pack_ms:.0} ms, open .kime {kime_ms:.2} ms, hash check {verify_ms:.0} ms ({:.1} GB/s), tokenizer {tok_ms:.0} ms",
        m.spec.id,
        m.tensors.entries().len(),
        mb / verify_ms,
    );
    std::fs::remove_file(packed).unwrap();
}
