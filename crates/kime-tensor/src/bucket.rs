//! The bucket table: which padded shapes plans are built for.
//!
//! The table is data. The default is `buckets.txt` next to this file, compiled in, and an operator
//! can load another with [`Buckets::parse`] without touching code.

use std::fmt;

use crate::plan::Rows;

/// The shape a plan is built for. A batch fits a bucket when it has no more of each.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bucket {
    /// Tokens.
    pub tokens: usize,
    /// Sequences.
    pub seqs: usize,
    /// Markers.
    pub markers: usize,
}

impl Bucket {
    /// The row count of `rows` in this bucket.
    #[must_use]
    pub fn rows(&self, rows: Rows) -> usize {
        match rows {
            Rows::Tokens => self.tokens,
            Rows::Seqs => self.seqs,
            Rows::Markers => self.markers,
        }
    }

    /// Whether a batch of this size fits.
    #[must_use]
    pub fn holds(&self, tokens: usize, seqs: usize, markers: usize) -> bool {
        tokens <= self.tokens && seqs <= self.seqs && markers <= self.markers
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} tokens, {} seqs, {} markers", self.tokens, self.seqs, self.markers)
    }
}

/// A bucket table line that does not parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// 1 based.
    pub line: usize,
    /// What is wrong with it.
    pub reason: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bucket table line {}: {}", self.line, self.reason)
    }
}

impl std::error::Error for ParseError {}

/// Buckets per stage, each stage sorted smallest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Buckets {
    stages: Vec<(String, Vec<Bucket>)>,
}

/// The table kime ships with.
pub const DEFAULT: &str = include_str!("buckets.txt");

impl Default for Buckets {
    fn default() -> Self {
        Self::parse(DEFAULT).expect("the default bucket table parses")
    }
}

impl Buckets {
    /// Reads a table: one bucket per line as `stage tokens seqs markers`, with `#` comments.
    ///
    /// # Errors
    ///
    /// A line without four fields, a count that is not a number, or a bucket that repeats.
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let mut stages: Vec<(String, Vec<Bucket>)> = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let err = |reason: String| ParseError { line: i + 1, reason };
            let line = line.split('#').next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            let [stage, tokens, seqs, markers] = f[..] else {
                return Err(err(format!("expected 4 fields, found {}", f.len())));
            };
            let num =
                |s: &str| s.parse::<usize>().map_err(|_| err(format!("{s:?} is not a count")));
            let b = Bucket { tokens: num(tokens)?, seqs: num(seqs)?, markers: num(markers)? };
            let at = match stages.iter().position(|s| s.0 == stage) {
                Some(at) => at,
                None => {
                    stages.push((stage.to_string(), Vec::new()));
                    stages.len() - 1
                }
            };
            if stages[at].1.contains(&b) {
                return Err(err(format!("{stage} {b} is listed twice")));
            }
            stages[at].1.push(b);
        }
        for s in &mut stages {
            s.1.sort_by_key(|b| (b.tokens, b.seqs, b.markers));
        }
        Ok(Self { stages })
    }

    /// The buckets of `stage`, smallest first, empty for an unknown stage.
    #[must_use]
    pub fn stage(&self, stage: &str) -> &[Bucket] {
        self.stages.iter().find(|s| s.0 == stage).map_or(&[], |s| &s.1)
    }

    /// The smallest bucket of `stage` that holds the batch.
    #[must_use]
    pub fn pick(&self, stage: &str, tokens: usize, seqs: usize, markers: usize) -> Option<Bucket> {
        self.stage(stage).iter().copied().find(|b| b.holds(tokens, seqs, markers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_table() {
        let t = Buckets::default();
        assert_eq!(t.stage("state").len(), 11);
        assert_eq!(t.stage("question").len(), 9);
        let b = t.pick("compat", 119, 1, 4).unwrap();
        assert_eq!(b, Bucket { tokens: 128, seqs: 32, markers: 128 });
        assert_eq!(t.stage("compat").len(), 20);
        // Many short sequences push past the sequence limit before the token limit.
        assert_eq!(t.pick("compat", 100, 41, 80).unwrap().tokens, 192);
        assert_eq!(t.pick("compat", 20000, 1, 1), None);
        assert_eq!(t.pick("nothing", 1, 1, 1), None);
    }

    #[test]
    fn another_table_changes_the_choice() {
        let t = Buckets::parse("compat 100 1 8 # one\n\ncompat 50 2 8\n").unwrap();
        assert_eq!(t.stage("compat")[0].tokens, 50);
        assert_eq!(t.pick("compat", 60, 1, 1).unwrap().tokens, 100);
        assert_eq!(t.pick("compat", 60, 2, 1), None);
    }

    #[test]
    fn bad_tables() {
        assert_eq!(Buckets::parse("compat 1 2").unwrap_err().line, 1);
        assert!(Buckets::parse("# c\ncompat 1 x 3").unwrap_err().reason.contains("\"x\""));
        assert!(Buckets::parse("a 1 1 1\na 1 1 1").unwrap_err().reason.contains("twice"));
    }
}
