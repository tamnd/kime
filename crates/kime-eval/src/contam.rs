//! Contamination checks, per the rules in spec/12-training.md.
//!
//! Training text is compared with every benchmark test set by MinHash over 5-gram word shingles,
//! and a line whose Jaccard similarity with some test text is 0.5 or more is a near duplicate and
//! is dropped. Scripts without spaces between words (Chinese, Japanese, Thai and the like) count
//! each character as a word. Candidates come from locality sensitive hashing of the signatures, 64
//! bands of 2, which misses a pair at Jaccard 0.5 with probability 0.75^64, about 1e-8, and every
//! candidate is then checked with the exact Jaccard of the shingle sets, so nothing under the
//! threshold is ever dropped. 32 bands of 4 missed 3 to 7 percent of the pairs an exhaustive search
//! finds between the train and test splits of MASSIVE, Banking77 and CLINC150, and 2 rows per band
//! was no slower on 1.46 million lines, since checking a candidate is cheap.
//!
//! [`manifest_conflicts`] is the other rule: a model whose data manifest lists a test split is
//! not scored.

use std::collections::HashMap;

use serde_json::Value;

/// Words per shingle.
pub const SHINGLE: usize = 5;
/// Hash functions in a signature.
pub const PERMS: usize = 128;
/// Bands the signature is cut into for candidate lookup.
pub const BANDS: usize = 64;
const ROWS: usize = PERMS / BANDS;
/// The Jaccard similarity at which a training line is a near duplicate of a test text.
pub const THRESHOLD: f64 = 0.5;

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3))
}

/// Characters of scripts written without spaces, each of which counts as a word.
fn unspaced(c: char) -> bool {
    matches!(u32::from(c),
        0x0E00..=0x0EFF      // Thai, Lao
        | 0x1000..=0x109F    // Myanmar
        | 0x1780..=0x17FF    // Khmer
        | 0x3040..=0x30FF    // Hiragana, Katakana
        | 0x3400..=0x4DBF    // CJK extension A
        | 0x4E00..=0x9FFF    // CJK
        | 0xF900..=0xFAFF    // CJK compatibility
        | 0x20000..=0x2FA1F) // CJK extensions
}

/// The hashes of the lowercased words of a text.
pub fn words(text: &str) -> Vec<u64> {
    let (mut out, mut w) = (Vec::new(), String::new());
    for c in text.chars().flat_map(char::to_lowercase) {
        if unspaced(c) {
            if !w.is_empty() {
                out.push(fnv(w.as_bytes()));
                w.clear();
            }
            let mut b = [0; 4];
            out.push(fnv(c.encode_utf8(&mut b).as_bytes()));
        } else if c.is_alphanumeric() {
            w.push(c);
        } else if !w.is_empty() {
            out.push(fnv(w.as_bytes()));
            w.clear();
        }
    }
    if !w.is_empty() {
        out.push(fnv(w.as_bytes()));
    }
    out
}

/// The sorted, distinct shingle hashes of a text. A text shorter than a shingle is one shingle,
/// so it matches only a text with the same words; a text with no words has none and matches
/// nothing.
pub fn shingles(text: &str) -> Vec<u64> {
    let w = words(text);
    let mut out: Vec<u64> = w
        .windows(SHINGLE.min(w.len()).max(1))
        .map(|g| g.iter().fold(0x9e37_79b9_7f4a_7c15, |h, x| mix(h ^ x)))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The Jaccard similarity of two sorted, distinct sets.
pub fn jaccard(a: &[u64], b: &[u64]) -> f64 {
    let (mut i, mut j, mut both) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                both += 1;
                i += 1;
                j += 1;
            }
        }
    }
    let union = a.len() + b.len() - both;
    if union == 0 { 0.0 } else { both as f64 / union as f64 }
}

fn seeds() -> [u64; PERMS] {
    let mut s = [0; PERMS];
    let mut x = 0x6b69_6d65_u64;
    for v in &mut s {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        *v = mix(x);
    }
    s
}

/// The MinHash signature of a shingle set.
pub fn signature(sh: &[u64], seeds: &[u64; PERMS]) -> [u32; PERMS] {
    let mut sig = [u32::MAX; PERMS];
    for &s in sh {
        for (m, seed) in sig.iter_mut().zip(seeds) {
            let h = (mix(s ^ seed) >> 32) as u32;
            if h < *m {
                *m = h;
            }
        }
    }
    sig
}

fn band_keys(sig: &[u32; PERMS]) -> impl Iterator<Item = u64> + '_ {
    sig.chunks(ROWS)
        .enumerate()
        .map(|(b, rows)| rows.iter().fold(mix(b as u64 + 1), |h, r| mix(h ^ u64::from(*r))))
}

fn set_key(sh: &[u64]) -> u64 {
    sh.iter().fold(sh.len() as u64, |h, x| mix(h ^ x))
}

/// A test text in the index.
#[derive(Debug)]
pub struct Doc {
    /// The test set it came from, and the other test sets holding the same text.
    pub sources: Vec<String>,
    /// Its id in the first test set.
    pub id: String,
    shingles: Vec<u64>,
}

/// The closest test text to a training line, at or over the threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Its index, for [`Index::doc`].
    pub doc: usize,
    /// Its Jaccard similarity with the line.
    pub jaccard: f64,
}

/// The test texts, ready to be queried.
#[derive(Debug)]
pub struct Index {
    seeds: [u64; PERMS],
    docs: Vec<Doc>,
    exact: HashMap<u64, u32>,
    /// Band keys and the text each came from, sorted by key.
    bands: Vec<(u64, u32)>,
    /// Texts added, before merging the ones with the same shingles.
    pub added: usize,
    /// The threshold [`Index::query`] reports at.
    pub threshold: f64,
}

impl Default for Index {
    fn default() -> Self {
        Self::new(THRESHOLD)
    }
}

impl Index {
    /// An empty index that reports near duplicates at `threshold`.
    pub fn new(threshold: f64) -> Self {
        Index {
            seeds: seeds(),
            docs: Vec::new(),
            exact: HashMap::new(),
            bands: Vec::new(),
            added: 0,
            threshold,
        }
    }

    /// Adds a test text. A text with no words is skipped, and one with the same shingles as a text
    /// already in the index only adds its test set to that text. Each call sorts the band keys, so
    /// many texts go in faster through [`Index::add_all`].
    pub fn add(&mut self, source: &str, id: &str, text: &str) {
        self.add_all(source, &[(id.to_string(), text.to_string())]);
    }

    /// [`Index::add`] for many `(id, text)` pairs of one test set, with the shingles and
    /// signatures worked out on every core.
    pub fn add_all(&mut self, source: &str, items: &[(String, String)]) {
        let seeds = &self.seeds;
        let prepared = par_map(items, |(_, t)| {
            let sh = shingles(t);
            let sig = signature(&sh, seeds);
            (sh, sig)
        });
        for ((id, _), (sh, sig)) in items.iter().zip(prepared) {
            self.insert(source, id, sh, &sig);
        }
        // The keys already in the index are one sorted run and the new ones are appended, which
        // a stable sort merges in about linear time.
        self.bands.sort_by_key(|b| b.0);
    }

    fn insert(&mut self, source: &str, id: &str, sh: Vec<u64>, sig: &[u32; PERMS]) {
        if sh.is_empty() {
            return;
        }
        self.added += 1;
        let key = set_key(&sh);
        if let Some(&d) = self.exact.get(&key) {
            let doc = &mut self.docs[d as usize];
            if doc.shingles == sh {
                if !doc.sources.iter().any(|s| s == source) {
                    doc.sources.push(source.to_string());
                }
                return;
            }
        }
        let d = self.docs.len() as u32;
        for k in band_keys(sig) {
            self.bands.push((k, d));
        }
        self.exact.insert(key, d);
        self.docs.push(Doc { sources: vec![source.to_string()], id: id.to_string(), shingles: sh });
    }

    /// Distinct test texts.
    pub fn len(&self) -> usize {
        self.docs.len()
    }

    /// Whether no test text has been added.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// A test text.
    pub fn doc(&self, i: usize) -> &Doc {
        &self.docs[i]
    }

    /// The most similar test text to `text` if it is at or over the threshold.
    pub fn query(&self, text: &str) -> Option<Hit> {
        let sh = shingles(text);
        if sh.is_empty() {
            return None;
        }
        if let Some(&d) = self.exact.get(&set_key(&sh))
            && self.docs[d as usize].shingles == sh
        {
            return Some(Hit { doc: d as usize, jaccard: 1.0 });
        }
        let mut seen: Vec<u32> = Vec::new();
        for k in band_keys(&signature(&sh, &self.seeds)) {
            let from = self.bands.partition_point(|b| b.0 < k);
            seen.extend(self.bands[from..].iter().take_while(|b| b.0 == k).map(|b| b.1));
        }
        seen.sort_unstable();
        seen.dedup();
        let mut best: Option<Hit> = None;
        for d in seen {
            let j = jaccard(&sh, &self.docs[d as usize].shingles);
            if j >= self.threshold && best.is_none_or(|b| j > b.jaccard) {
                best = Some(Hit { doc: d as usize, jaccard: j });
            }
        }
        best
    }

    /// [`Index::query`] for many texts, on every core.
    pub fn query_all(&self, texts: &[String]) -> Vec<Option<Hit>> {
        par_map(texts, |t| self.query(t))
    }
}

/// `f` over `items` on every core, in order.
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = items.len().div_ceil(threads).max(1);
    let f = &f;
    std::thread::scope(|s| {
        let parts: Vec<_> = items
            .chunks(per)
            .map(|c| s.spawn(move || c.iter().map(f).collect::<Vec<_>>()))
            .collect();
        parts.into_iter().flat_map(|p| p.join().unwrap_or_default()).collect()
    })
}

/// The text of a line: every string in its `state`, or in its `text` when it has no state, in
/// order and one per line. Keys are left out, so the same fields under other names still match.
pub fn text_of(line: &Value) -> String {
    fn walk(v: &Value, out: &mut String) {
        match v {
            Value::String(s) => {
                // A state can be a JSON object written as a string.
                match serde_json::from_str::<Value>(s) {
                    Ok(inner @ (Value::Object(_) | Value::Array(_))) => walk(&inner, out),
                    _ => {
                        out.push_str(s);
                        out.push('\n');
                    }
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(m) => m.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = String::new();
    walk(line.get("state").or_else(|| line.get("text")).unwrap_or(&Value::Null), &mut out);
    out
}

/// A dataset, its config and a split.
type Split = (String, Option<String>, String);

fn split_of(v: &Value) -> Option<Split> {
    let s = |k| v.get(k).and_then(Value::as_str).map(str::to_string);
    let dataset = s("dataset").or_else(|| s("source"))?;
    Some((dataset, s("config"), s("split")?))
}

/// Why a model with the data manifest `data` may not be scored on suites built from `tests`, a
/// suite manifest as tools/eval/build_suites.py writes it (suite name to dataset, config and
/// split). The data manifest lists its shards or sources, each with a `source` or `dataset`, an
/// optional `config` and a `split`. A split named test, or the split a suite was drawn from, is a
/// conflict; an empty list means none.
pub fn manifest_conflicts(data: &Value, tests: &Value) -> Vec<String> {
    let entries = ["shards", "sources"]
        .iter()
        .filter_map(|k| data.get(*k).and_then(Value::as_array))
        .flatten()
        .filter_map(split_of);
    let suites: Vec<(&String, Split)> = tests
        .as_object()
        .map(|m| m.iter().filter_map(|(k, v)| Some((k, split_of(v)?))).collect())
        .unwrap_or_default();
    let mut out = Vec::new();
    for (ds, cfg, split) in entries {
        if split.to_lowercase().starts_with("test") {
            out.push(format!("{ds} lists the {split} split"));
            continue;
        }
        for (suite, (tds, tcfg, tsplit)) in &suites {
            // A suite drawn from several datasets names them joined with commas, and the config
            // of the first, so its config says nothing about the others.
            let several = tds.contains(',');
            let same_cfg = several || cfg.is_none() || tcfg.is_none() || cfg == *tcfg;
            let named = tds.split(',').any(|t| ds.eq_ignore_ascii_case(t.trim()));
            if named && split == *tsplit && same_cfg {
                out.push(format!("{ds} {split} is the split suite {suite} is drawn from"));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn words_split_scripts() {
        assert_eq!(words("Hello, WORLD! it's").len(), 4);
        // Each Han character is a word, the Latin run beside it one more.
        assert_eq!(words("我爱北京abc").len(), 5);
        assert!(words("  ...  ").is_empty());
    }

    #[test]
    fn jaccard_of_shingles() {
        let a = shingles("the quick brown fox jumps over the lazy dog");
        assert_eq!(a.len(), 5);
        assert!((jaccard(&a, &a) - 1.0).abs() < 1e-12);
        let b = shingles("the quick brown fox jumps over the lazy cat");
        // Four of the five shingles are shared, and the union has six.
        assert!((jaccard(&a, &b) - 4.0 / 6.0).abs() < 1e-12);
        assert_eq!(shingles("wake me up").len(), 1);
    }

    #[test]
    fn index_finds_near_duplicates_only() {
        let mut ix = Index::default();
        let base = "the service was slow and the food arrived cold but the staff apologised and took it off the bill";
        ix.add("reviews", "r1", base);
        ix.add("reviews2", "x", &base.to_uppercase());
        ix.add(
            "reviews",
            "r2",
            "a completely different sentence about the weather in hanoi during the rainy season",
        );
        assert_eq!((ix.added, ix.len()), (3, 2));
        assert_eq!(ix.doc(0).sources, ["reviews", "reviews2"]);
        let hit = ix.query("The service was slow, and the food arrived cold! But the staff apologised and took it off the bill.");
        assert_eq!(hit, Some(Hit { doc: 0, jaccard: 1.0 }));
        let near = ix.query("the service was slow and the food arrived cold but the staff apologised and took it off the check");
        assert!(near.is_some_and(|h| h.doc == 0 && h.jaccard >= 0.5 && h.jaccard < 1.0));
        assert_eq!(
            ix.query("the service was quick and the food arrived hot so we tipped the staff well"),
            None
        );
        assert_eq!(ix.query(""), None);
        let all =
            ix.query_all(&[base.to_string(), "nothing like it at all in this one here".into()]);
        assert_eq!(all.len(), 2);
        assert!(all[0].is_some() && all[1].is_none());
        let mut many = Index::default();
        many.add_all("reviews", &[("r1".into(), base.into()), ("r2".into(), base.to_uppercase())]);
        assert_eq!((many.added, many.len()), (2, 1));
        assert_eq!(many.query(base), Some(Hit { doc: 0, jaccard: 1.0 }));
    }

    #[test]
    fn text_of_reads_state() {
        let l = json!({"id": "a", "state": {"premise": "p one", "hypothesis": "h two"}, "questions": {"q": {"instructions": "ignored"}}});
        assert_eq!(text_of(&l), "p one\nh two\n");
        let l = json!({"state": "{\"subject\": \"hi\", \"body\": [\"x\", 3]}"});
        assert_eq!(text_of(&l), "hi\nx\n");
        assert_eq!(text_of(&json!({"text": "t"})), "t\n");
    }

    #[test]
    fn manifest_conflicts_name_test_splits() {
        let tests = json!({
            "en.boolq": {"dataset": "google/boolq", "config": null, "split": "validation"},
            "xnli.en": {"dataset": "facebook/xnli", "config": "en", "split": "test"},
            "app.routing": {"dataset": "openai/gsm8k, google/mbpp", "config": "main", "split": "train"},
        });
        let ok = json!({"shards": [{"source": "google/boolq", "split": "train"}, {"source": "facebook/xnli", "config": "en", "split": "train"}]});
        assert!(manifest_conflicts(&ok, &tests).is_empty());
        let bad = json!({"sources": [{"dataset": "google/boolq", "split": "validation"}, {"source": "mine", "split": "test_a"}]});
        let c = manifest_conflicts(&bad, &tests);
        assert_eq!(c.len(), 2);
        assert!(c[0].contains("en.boolq"));
        assert!(c[1].contains("test_a"));
        let bad =
            json!({"shards": [{"source": "google/mbpp", "config": "full", "split": "train"}]});
        assert!(manifest_conflicts(&bad, &tests)[0].contains("app.routing"));
    }
}
