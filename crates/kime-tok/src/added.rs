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

/// A byte trie over the patterns. Walking it from a position and keeping the last node that ends
/// a pattern gives the longest match there, in time bounded by the length of that match. The root
/// is a full table because it is visited at every byte, and inner nodes keep a short sorted list.
#[derive(Debug, Default)]
struct Matcher {
    root: Vec<u32>,
    nodes: Vec<Node>,
}

#[derive(Debug, Default)]
struct Node {
    children: Vec<(u8, u32)>,
    /// The token index of the pattern that ends here.
    token: Option<usize>,
}

impl Matcher {
    fn new(patterns: Vec<(Box<[u8]>, usize)>) -> Matcher {
        if patterns.is_empty() {
            return Matcher::default();
        }
        // Node 0 is a placeholder so that 0 can mean no child in the root table.
        let mut m = Matcher { root: vec![0; 256], nodes: vec![Node::default()] };
        for (p, tok) in patterns {
            let Some((&first, rest)) = p.split_first() else { continue };
            let mut at = m.root[usize::from(first)];
            if at == 0 {
                at = m.nodes.len() as u32;
                m.nodes.push(Node::default());
                m.root[usize::from(first)] = at;
            }
            for &b in rest {
                at = match m.nodes[at as usize].children.binary_search_by_key(&b, |c| c.0) {
                    Ok(i) => m.nodes[at as usize].children[i].1,
                    Err(i) => {
                        let id = m.nodes.len() as u32;
                        m.nodes.push(Node::default());
                        m.nodes[at as usize].children.insert(i, (b, id));
                        id
                    }
                };
            }
            // Two tokens with the same pattern: the first one listed wins.
            m.nodes[at as usize].token.get_or_insert(tok);
        }
        m
    }

    fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// The longest pattern that matches at `pos`, as (length, token index).
    #[inline]
    fn at(&self, hay: &[u8], pos: usize) -> Option<(usize, usize)> {
        let mut at = self.root[usize::from(hay[pos])];
        if at == 0 {
            return None;
        }
        let mut best = None;
        let mut len = 1;
        loop {
            let node = &self.nodes[at as usize];
            if let Some(tok) = node.token {
                best = Some((len, tok));
            }
            let Some(&b) = hay.get(pos + len) else { break };
            match node.children.iter().find(|c| c.0 == b) {
                Some(&(_, next)) => at = next,
                None => break,
            }
            len += 1;
        }
        best
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
        Added {
            tokens,
            raw: Matcher::new(raw),
            normalized: Matcher::new(normalized),
            by_content,
            by_id,
        }
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
            // Most bytes cannot start a pattern, so jump to the next one that can.
            match hay[pos..].iter().position(|&b| m.root[usize::from(b)] != 0) {
                Some(skip) => pos += skip,
                None => break,
            }
            let Some((len, idx)) = m.at(hay, pos) else {
                pos += 1;
                continue;
            };
            let tok = &self.tokens[idx];
            let (mut start, mut stop) = (pos, pos + len);
            pos = stop;
            if tok.single_word {
                let start_space =
                    start == 0 || !text[..start].chars().next_back().is_some_and(is_word_char);
                let stop_space =
                    stop == hay.len() || !text[stop..].chars().next().is_some_and(is_word_char);
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
                stop += text[stop..]
                    .char_indices()
                    .find(|(_, c)| !c.is_whitespace())
                    .map_or(hay.len() - stop, |(i, _)| i);
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
