//! Merge rank BPE, following Hugging Face's `Word::merge_all`: a heap of candidate pairs ordered
//! by rank and then position, over a linked list of symbols. Any other order can give different
//! ids on words where two merges compete, so this one is kept exactly.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::bytelevel::BYTE_TO_CHAR;
use crate::hash::FxMap;

#[derive(Debug)]
pub(crate) struct Bpe {
    vocab: HashMap<Box<str>, u32>,
    vocab_r: Vec<Option<Box<str>>>,
    /// Single char tokens, which is where every word starts.
    chars: FxMap<u32, u32>,
    /// `(left << 32) | right` to `(rank, merged id)`.
    merges: FxMap<u64, (u32, u32)>,
    /// For byte level models, the id of the token for each byte's GPT-2 char.
    byte_ids: [Option<u32>; 256],
    /// For byte fallback models, the id of `<0xNN>` for each byte. Gemma leaves a few out, such
    /// as `<0x09>`, because the char has a token of its own.
    fallback: Option<[Option<u32>; 256]>,
    unk: Option<u32>,
    fuse_unk: bool,
}

#[derive(Debug, Clone, Copy)]
struct Symbol {
    id: u32,
    prev: i32,
    next: i32,
    alive: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max heap, so both keys are reversed: lowest rank first, then leftmost.
        other.rank.cmp(&self.rank).then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn pair(a: u32, b: u32) -> u64 {
    (u64::from(a) << 32) | u64::from(b)
}

impl Bpe {
    pub(crate) fn new(
        vocab: HashMap<Box<str>, u32>,
        merges: &[(String, String)],
        unk: Option<&str>,
        byte_fallback: bool,
        fuse_unk: bool,
    ) -> Result<Bpe, String> {
        let size = vocab.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut vocab_r = vec![None; size];
        let mut chars = FxMap::default();
        for (tok, &id) in &vocab {
            vocab_r[id as usize] = Some(tok.clone());
            let mut it = tok.chars();
            if let (Some(c), None) = (it.next(), it.next()) {
                chars.insert(c as u32, id);
            }
        }
        let mut table = FxMap::with_capacity_and_hasher(merges.len(), Default::default());
        let mut joined = String::new();
        for (rank, (a, b)) in merges.iter().enumerate() {
            let ia = *vocab.get(a.as_str()).ok_or_else(|| format!("merge {rank} uses {a:?}, which is not in the vocabulary"))?;
            let ib = *vocab.get(b.as_str()).ok_or_else(|| format!("merge {rank} uses {b:?}, which is not in the vocabulary"))?;
            joined.clear();
            joined.push_str(a);
            joined.push_str(b);
            let new_id = *vocab.get(joined.as_str()).ok_or_else(|| format!("merge {rank} makes {joined:?}, which is not in the vocabulary"))?;
            let rank = u32::try_from(rank).map_err(|_| "more than 4 billion merges".to_string())?;
            // The first merge of a pair wins, as it does when Hugging Face builds its map.
            table.entry(pair(ia, ib)).or_insert((rank, new_id));
        }
        let mut byte_ids = [None; 256];
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            *slot = chars.get(&(BYTE_TO_CHAR[b] as u32)).copied();
        }
        let fallback = if byte_fallback {
            let mut f = [None; 256];
            for (b, slot) in f.iter_mut().enumerate() {
                *slot = vocab.get(format!("<0x{b:02X}>").as_str()).copied();
            }
            Some(f)
        } else {
            None
        };
        let unk = match unk {
            Some(u) => Some(*vocab.get(u).ok_or_else(|| format!("the unknown token {u:?} is not in the vocabulary"))?),
            None => None,
        };
        Ok(Bpe { vocab, vocab_r, chars, merges: table, byte_ids, fallback, unk, fuse_unk })
    }

    pub(crate) fn vocab_len(&self) -> usize {
        self.vocab_r.len()
    }

    pub(crate) fn merges_len(&self) -> usize {
        self.merges.len()
    }

    pub(crate) fn id_of(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub(crate) fn token_of(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(id as usize).and_then(|t| t.as_deref())
    }

    /// A byte level word: each byte starts as the token of its GPT-2 char. A byte whose char is
    /// not in the vocabulary is dropped, which is what Hugging Face does with no unknown token.
    pub(crate) fn encode_bytelevel(&self, word: &[u8], out: &mut Vec<u32>) {
        let mut syms: Vec<Symbol> = Vec::with_capacity(word.len());
        for &b in word {
            if let Some(id) = self.byte_ids[usize::from(b)] {
                push(&mut syms, id);
            }
        }
        self.merge(&mut syms, out);
    }

    /// A word of chars, as SentencePiece style models see it. A char outside the vocabulary becomes
    /// its UTF-8 bytes as `<0xNN>` tokens with byte fallback, or the unknown token without.
    pub(crate) fn encode_chars(&self, word: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<Symbol> = Vec::with_capacity(word.len());
        let mut unk_pending = false;
        for c in word.chars() {
            if let Some(&id) = self.chars.get(&(c as u32)) {
                if unk_pending {
                    push(&mut syms, self.unk.expect("pending only when there is an unknown token"));
                    unk_pending = false;
                }
                push(&mut syms, id);
                continue;
            }
            if let Some(fb) = &self.fallback {
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                // Only when every byte has a token, otherwise the char is unknown, as in Hugging Face.
                if bytes.iter().all(|&b| fb[usize::from(b)].is_some()) {
                    for &b in bytes {
                        push(&mut syms, fb[usize::from(b)].expect("checked above"));
                    }
                    continue;
                }
            }
            if let Some(unk) = self.unk {
                if unk_pending && !self.fuse_unk {
                    push(&mut syms, unk);
                }
                unk_pending = true;
            }
        }
        if unk_pending {
            push(&mut syms, self.unk.expect("pending only when there is an unknown token"));
        }
        self.merge(&mut syms, out);
    }

    fn merge(&self, syms: &mut [Symbol], out: &mut Vec<u32>) {
        let n = syms.len();
        let mut heap = BinaryHeap::with_capacity(n);
        for i in 0..n.saturating_sub(1) {
            if let Some(&(rank, new_id)) = self.merges.get(&pair(syms[i].id, syms[i + 1].id)) {
                heap.push(Merge { pos: i, rank, new_id });
            }
        }
        while let Some(top) = heap.pop() {
            let cur = syms[top.pos];
            if !cur.alive || cur.next < 0 {
                continue;
            }
            let next_pos = cur.next as usize;
            let right = syms[next_pos];
            // An entry is stale when either side has merged since it was pushed.
            match self.merges.get(&pair(cur.id, right.id)) {
                Some(&(_, id)) if id == top.new_id => {}
                _ => continue,
            }
            syms[top.pos].id = top.new_id;
            syms[top.pos].next = right.next;
            syms[next_pos].alive = false;
            if right.next >= 0 {
                syms[right.next as usize].prev = top.pos as i32;
            }
            let cur = syms[top.pos];
            if cur.prev >= 0 {
                let p = cur.prev as usize;
                if let Some(&(rank, new_id)) = self.merges.get(&pair(syms[p].id, cur.id)) {
                    heap.push(Merge { pos: p, rank, new_id });
                }
            }
            if cur.next >= 0 {
                let q = cur.next as usize;
                if let Some(&(rank, new_id)) = self.merges.get(&pair(cur.id, syms[q].id)) {
                    heap.push(Merge { pos: top.pos, rank, new_id });
                }
            }
        }
        out.extend(syms.iter().filter(|s| s.alive).map(|s| s.id));
    }
}

fn push(syms: &mut Vec<Symbol>, id: u32) {
    let i = syms.len() as i32;
    if let Some(last) = syms.last_mut() {
        last.next = i;
    }
    syms.push(Symbol { id, prev: i - 1, next: -1, alive: true });
}
