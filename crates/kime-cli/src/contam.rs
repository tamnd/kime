//! `kime contam`: find the training lines that are near duplicates of a test text, and write the
//! data without them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use kime_eval::contam::{BANDS, Index, PERMS, SHINGLE, THRESHOLD, text_of};
use serde_json::{Value, json};

const USAGE: &str =
    "usage: kime contam --tests <file or directory>... <data.jsonl>... [--out <dir>]
       [--report <file>] [--threshold 0.5]
Indexes the text of every line of the test files (the strings of its state, or its text), then
checks the text of every line of the data files against them with MinHash over 5-gram shingles.
A line at or over the threshold in Jaccard similarity with some test text is a near duplicate.
With --out, each data file is written to <dir> without its near duplicates. --tests can be
given more than once, and a directory means every .jsonl in it.";

/// Near duplicates listed in the report.
const EXAMPLES: usize = 50;

struct Opts {
    tests: Vec<PathBuf>,
    data: Vec<PathBuf>,
    out: Option<PathBuf>,
    report: Option<PathBuf>,
    threshold: f64,
}

fn jsonl_in(p: &Path) -> Result<Vec<PathBuf>, String> {
    if !p.is_dir() {
        return Ok(vec![p.to_path_buf()]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(p)
        .map_err(|e| format!("{}: {e}", p.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|f| f.extension().is_some_and(|x| x == "jsonl") && !stem(f).ends_with(".answers"))
        .collect();
    files.sort();
    Ok(files)
}

fn opts(args: &[String]) -> Result<Opts, String> {
    let mut o =
        Opts { tests: Vec::new(), data: Vec::new(), out: None, report: None, threshold: THRESHOLD };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--tests" => o.tests.extend(jsonl_in(Path::new(&val()?))?),
            "--out" => o.out = Some(val()?.into()),
            "--report" => o.report = Some(val()?.into()),
            "--threshold" => {
                o.threshold = val()?
                    .parse()
                    .ok()
                    .filter(|t: &f64| *t > 0.0 && *t <= 1.0)
                    .ok_or("--threshold takes a number over 0 and at most 1")?;
            }
            "--help" | "-h" => return Err(USAGE.into()),
            p if !p.starts_with('-') => o.data.extend(jsonl_in(Path::new(p))?),
            other => return Err(format!("unknown option {other:?}\n{USAGE}")),
        }
    }
    if o.tests.is_empty() || o.data.is_empty() {
        return Err(USAGE.into());
    }
    Ok(o)
}

fn stem(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

pub(crate) fn run(args: &[String]) -> ExitCode {
    match contam(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kime contam: {e}");
            ExitCode::FAILURE
        }
    }
}

fn read(p: &Path) -> Result<String, String> {
    std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))
}

fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty())
}

fn id_of(v: &Value, line: usize) -> String {
    match v.get("id") {
        Some(Value::String(s)) => s.clone(),
        Some(x) => x.to_string(),
        None => (line + 1).to_string(),
    }
}

fn contam(args: &[String]) -> Result<(), String> {
    let o = opts(args)?;
    let t = Instant::now();
    let mut ix = Index::new(o.threshold);
    for p in &o.tests {
        let text = read(p)?;
        let source = stem(p);
        let mut items = Vec::new();
        for (i, line) in lines(&text) {
            let v: Value = serde_json::from_str(line)
                .map_err(|e| format!("{} line {}: {e}", p.display(), i + 1))?;
            items.push((id_of(&v, i), text_of(&v)));
        }
        ix.add_all(&source, &items);
    }
    let index_secs = t.elapsed().as_secs_f64();
    eprintln!(
        "{} test texts from {} files, {} distinct, indexed in {index_secs:.1}s",
        ix.added,
        o.tests.len(),
        ix.len()
    );
    if let Some(dir) = &o.out {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let t = Instant::now();
    let (mut files, mut examples, mut rows, mut dropped) = (Vec::new(), Vec::new(), 0, 0);
    for p in &o.data {
        let text = read(p)?;
        let (mut raw, mut vals) = (Vec::new(), Vec::new());
        for (i, line) in lines(&text) {
            let v: Value = serde_json::from_str(line)
                .map_err(|e| format!("{} line {}: {e}", p.display(), i + 1))?;
            raw.push(line);
            vals.push((i, v));
        }
        let texts: Vec<String> = vals.iter().map(|(_, v)| text_of(v)).collect();
        let hits = ix.query_all(&texts);
        let (mut kept, mut by_test, mut exact, mut n) =
            (String::new(), BTreeMap::<String, usize>::new(), 0, 0);
        for ((line, (i, v)), hit) in raw.iter().zip(&vals).zip(&hits) {
            let Some(h) = hit else {
                kept.push_str(line);
                kept.push('\n');
                continue;
            };
            n += 1;
            exact += usize::from(h.jaccard >= 1.0);
            let doc = ix.doc(h.doc);
            for s in &doc.sources {
                *by_test.entry(s.clone()).or_default() += 1;
            }
            if examples.len() < EXAMPLES {
                examples.push(json!({"file": stem(p), "id": id_of(v, *i), "test": doc.sources[0],
                    "test_id": doc.id, "jaccard": (h.jaccard * 1e4).round() / 1e4}));
            }
        }
        if let Some(dir) = &o.out {
            let name = p.file_name().ok_or_else(|| format!("{}: no file name", p.display()))?;
            let dst = dir.join(name);
            std::fs::write(&dst, kept).map_err(|e| format!("{}: {e}", dst.display()))?;
        }
        println!(
            "{:<44} {:>9} lines {:>7} near duplicates ({:.3}%), {exact} exact",
            stem(p),
            vals.len(),
            n,
            100.0 * n as f64 / vals.len().max(1) as f64
        );
        rows += vals.len();
        dropped += n;
        files.push(json!({"file": stem(p), "lines": vals.len(), "near_duplicates": n,
            "exact": exact, "by_test": by_test}));
    }
    let query_secs = t.elapsed().as_secs_f64();
    println!(
        "{rows} lines, {dropped} near duplicates, checked in {query_secs:.1}s ({:.0} lines/s)",
        rows as f64 / query_secs.max(1e-9)
    );
    if let Some(path) = &o.report {
        let report = json!({
            "kime": env!("CARGO_PKG_VERSION"),
            "method": {"shingle_words": SHINGLE, "perms": PERMS, "bands": BANDS, "threshold": o.threshold},
            "tests": {"files": o.tests.iter().map(|p| stem(p)).collect::<Vec<_>>(),
                      "texts": ix.added, "distinct": ix.len(), "index_secs": index_secs},
            "data": {"lines": rows, "near_duplicates": dropped, "check_secs": query_secs, "files": files},
            "examples": examples,
        });
        let text = serde_json::to_string_pretty(&report).unwrap_or_default() + "\n";
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}
