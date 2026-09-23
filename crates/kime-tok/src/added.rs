//! Added tokens: the special tokens and the extra entries a tokenizer defines on top of its BPE
//! vocabulary. They are matched in the text before BPE runs, leftmost first and longest on a tie,
//! the way the `aho-corasick` crate's leftmost longest mode does in Hugging Face tokenizers.

use std::collections::HashMap;

use crate::hash::FxMap;

#[derive(Debug, Clone)]
pub(crate) struct AddedToken {
    pub(crate) id: u32,
    pub(crate) content: String,
    pub(crate) single_word: bool,
    pub(crate) lstrip: bool,
    pub(crate) rstrip: bool,
    pub(crate) normalized: bool,
    pub(crate) special: bool,
}

/// What a split produces: a run of text still to be tokenized, or an added token's id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Piece<'a> {
    Text(&'a str),
    Token(u32),
}

/// Patterns bucketed by first byte, longest first within a bucket, so the first pattern that
/// matches at a position is the longest one there.
#[derive(Debug, Default)]
struct Matcher {
    patterns: Vec<(Box<[u8]>, usize)>,
    by_first: Vec<Vec<usize>>,
}

impl Matcher {
    fn new(patterns: Vec<(Box<[u8]>, usize)>) -> Matcher {
        let mut by_first = vec![Vec::new(); 256];
        for (i, (p, _)) in patterns.iter().enumerate() {
            if let Some(&b) = p.first() {
                by_first[usize::from(b)].push(i);
            }
        }
        for bucket in &mut by_first {
            bucket.sort_by_key(|&i| std::cmp::Reverse(patterns[i].0.len()));
        }
        Matcher { patterns, by_first }
    }

    fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// The longest pattern that matches at `pos`, as (length, token index).
    fn at(&self, hay: &[u8], pos: usize) -> Option<(usize, usize)> {
        let rest = &hay[pos..];
        for &i in &self.by_first[usize::from(rest[0])] {
            let (p, tok) = &self.patterns[i];
            if rest.starts_with(p) {
                return Some((p.len(), *tok));
            }
        }
        None
    }
}

#[derive(Debug, Default)]
pub(crate) struct Added {
    tokens: Vec<AddedToken>,
    raw: Matcher,
    normalized: Matcher,
    by_content: HashMap<String, u32>,
    by_id: FxMap<u32, usize>,
}

impl Added {
    /// `normalize` is the tokenizer's normalizer, applied to the content of every token marked
    /// `normalized`, because those are matched against normalized text.
    pub(crate) fn new(tokens: Vec<AddedToken>, normalize: impl Fn(&str) -> String) -> Added {
        let mut raw = Vec::new();
        let mut normalized = Vec::new();
        let mut by_content = HashMap::new();
        let mut by_id = FxMap::default();
        for (i, t) in tokens.iter().enumerate() {
            by_content.insert(t.content.clone(), t.id);
            by_id.insert(t.id, i);
            if t.content.is_empty() {
                continue;
            }
            if t.normalized {
                normalized.push((normalize(&t.content).into_bytes().into_boxed_slice(), i));
            } else {
                raw.push((t.content.clone().into_bytes().into_boxed_slice(), i));
            }
        }
        Added { tokens, raw: Matcher::new(raw), normalized: Matcher::new(normalized), by_content, by_id }
    }

    pub(crate) fn len(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn max_id(&self) -> Option<u32> {
        self.tokens.iter().map(|t| t.id).max()
    }

    pub(crate) fn id_of(&self, content: &str) -> Option<u32> {
        self.by_content.get(content).copied()
    }

    pub(crate) fn content_of(&self, id: u32) -> Option<&str> {
        self.by_id.get(&id).map(|&i| self.tokens[i].content.as_str())
    }

    pub(crate) fn is_special(&self, id: u32) -> bool {
        self.by_id.get(&id).is_some_and(|&i| self.tokens[i].special)
    }

    pub(crate) fn split_raw<'a>(&self, text: &'a str, f: &mut dyn FnMut(Piece<'a>)) {
        self.split(&self.raw, text, f);
    }

    pub(crate) fn split_normalized<'a>(&self, text: &'a str, f: &mut dyn FnMut(Piece<'a>)) {
        self.split(&self.normalized, text, f);
    }

    /// Hugging Face's `AddedVocabulary::find_matches`, including the strip and single word rules.
    /// Empty text pieces are dropped, as `PreTokenizedString::split` drops them.
    fn split<'a>(&self, m: &Matcher, text: &'a str, f: &mut dyn FnMut(Piece<'a>)) {
        if m.is_empty() {
            if !text.is_empty() {
                f(Piece::Text(text));
            }
            return;
        }
        let hay = text.as_bytes();
        let mut offset = 0;
        let mut pos = 0;
        while pos < hay.len() {
            let Some((len, idx)) = m.at(hay, pos) else {
                pos += 1;
                continue;
            };
            let tok = &self.tokens[idx];
            let (mut start, mut stop) = (pos, pos + len);
            pos = stop;
            if tok.single_word {
                let start_space = start == 0 || !text[..start].chars().next_back().is_some_and(is_word_char);
                let stop_space = stop == hay.len() || !text[stop..].chars().next().is_some_and(is_word_char);
                if !start_space || !stop_space {
                    continue;
                }
            }
            if tok.lstrip {
                let new_start = text[..start]
                    .char_indices()
                    .rev()
                    .find(|(_, c)| !c.is_whitespace())
                    .map_or(0, |(i, c)| i + c.len_utf8());
                start = new_start.max(offset);
            }
            if tok.rstrip {
                stop += text[stop..].char_indices().find(|(_, c)| !c.is_whitespace()).map_or(hay.len() - stop, |(i, _)| i);
                pos = stop;
            }
            if offset < start {
                f(Piece::Text(&text[offset..start]));
            }
            f(Piece::Token(tok.id));
            offset = stop;
        }
        if offset < hay.len() {
            f(Piece::Text(&text[offset..]));
        }
    }
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}
