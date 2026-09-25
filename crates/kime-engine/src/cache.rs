//! The answer cache from spec/11-serving.md. A question's answer depends only on its own row of
//! token ids, its option markers and its type, since every question is its own sequence and the
//! engine gives the same bits whatever else is in the batch. So one entry is the logits and the
//! act head output of one row, keyed by the blake3 of what went to the device. The answer JSON is
//! built from them again on a hit, which keeps the labels the request sent.
//!
//! Each cache belongs to one loaded model, so the model, its version and its precision are part
//! of the key without being hashed. Eviction keeps two generations: new entries go in the young
//! one, and when it is full the old one is dropped and the young one takes its place. A hit in
//! the old generation moves the entry back to the young one, so what is used stays in, as with
//! an LRU, for one hash map insert rather than a linked list.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use kime_tok::layout::CompatSequence;

/// The blake3 hash of one row.
pub(crate) type Key = [u8; 32];

/// What the device gave for one row.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Entry {
    pub(crate) logits: Box<[f32]>,
    pub(crate) act: [f32; 2],
}

/// How a request uses the cache, from `kime.cache`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheMode {
    /// Answers from the cache when it can and keeps what it computes.
    #[default]
    Use,
    /// Neither reads nor writes the cache.
    Bypass,
    /// Computes every answer and overwrites the cache with it.
    Refresh,
}

impl CacheMode {
    /// The mode `kime.cache` names: `"use"`, `"bypass"` or `"refresh"`.
    #[must_use]
    pub fn parse(s: &str) -> Option<CacheMode> {
        match s {
            "use" => Some(CacheMode::Use),
            "bypass" => Some(CacheMode::Bypass),
            "refresh" => Some(CacheMode::Refresh),
            _ => None,
        }
    }
}

/// Hits and misses since the model loaded, and what the cache holds now.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// The most answers it holds, 0 when it is off.
    pub capacity: usize,
    /// Answers it holds.
    pub entries: usize,
    /// Questions answered from it.
    pub hits: u64,
    /// Questions looked up and not found, then computed.
    pub misses: u64,
}

#[derive(Default)]
struct Generations {
    young: HashMap<Key, Entry>,
    old: HashMap<Key, Entry>,
}

pub(crate) struct AnswerCache {
    /// The most entries in one generation, half the capacity.
    half: usize,
    maps: Mutex<Generations>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl AnswerCache {
    /// A cache for about `capacity` answers. It holds between half of that and all of it.
    pub(crate) fn new(capacity: usize) -> Self {
        AnswerCache {
            half: capacity.div_ceil(2).max(1),
            maps: Mutex::default(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub(crate) fn key(seq: &CompatSequence, qtype: u8) -> Key {
        let mut h = blake3::Hasher::new();
        h.update(&[qtype]);
        h.update(&(seq.ids.len() as u64).to_le_bytes());
        for id in &seq.ids {
            h.update(&id.to_le_bytes());
        }
        for m in &seq.markers {
            h.update(&m.to_le_bytes());
        }
        *h.finalize().as_bytes()
    }

    /// The entries for `keys`, in order, with `None` for each miss. The counts go up.
    pub(crate) fn get(&self, keys: &[Key]) -> Vec<Option<Entry>> {
        let mut g = self.maps.lock().unwrap_or_else(PoisonError::into_inner);
        let out: Vec<Option<Entry>> = keys
            .iter()
            .map(|k| {
                if let Some(e) = g.young.get(k) {
                    return Some(e.clone());
                }
                let e = g.old.remove(k)?;
                self.put(&mut g, *k, e.clone());
                Some(e)
            })
            .collect();
        drop(g);
        let hits = out.iter().filter(|e| e.is_some()).count() as u64;
        self.hits.fetch_add(hits, Ordering::Relaxed);
        self.misses.fetch_add(keys.len() as u64 - hits, Ordering::Relaxed);
        out
    }

    /// The entries for `keys` when every one is there, counted as hits. When one is missing
    /// nothing is counted, since the caller looks them up again with [`AnswerCache::get`].
    pub(crate) fn all(&self, keys: &[Key]) -> Option<Vec<Entry>> {
        let g = self.maps.lock().unwrap_or_else(PoisonError::into_inner);
        let out = keys
            .iter()
            .map(|k| g.young.get(k).or_else(|| g.old.get(k)).cloned())
            .collect::<Option<Vec<_>>>()?;
        drop(g);
        self.hits.fetch_add(keys.len() as u64, Ordering::Relaxed);
        Some(out)
    }

    /// Keeps `entries`, replacing any already there.
    pub(crate) fn insert(&self, entries: impl IntoIterator<Item = (Key, Entry)>) {
        let mut g = self.maps.lock().unwrap_or_else(PoisonError::into_inner);
        for (k, e) in entries {
            g.old.remove(&k);
            self.put(&mut g, k, e);
        }
    }

    fn put(&self, g: &mut Generations, k: Key, e: Entry) {
        if g.young.len() >= self.half && !g.young.contains_key(&k) {
            g.old = std::mem::take(&mut g.young);
        }
        g.young.insert(k, e);
    }

    pub(crate) fn stats(&self) -> CacheStats {
        let g = self.maps.lock().unwrap_or_else(PoisonError::into_inner);
        CacheStats {
            capacity: self.half * 2,
            entries: g.young.len() + g.old.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(x: f32) -> Entry {
        Entry { logits: vec![x].into(), act: [x, 0.0] }
    }

    #[test]
    fn keeps_what_is_used_and_counts() {
        let c = AnswerCache::new(4);
        let k = |i: u8| [i; 32];
        c.insert((0..2).map(|i| (k(i), entry(f32::from(i)))));
        // The young generation is full at 2, so 2 and 3 push 0 and 1 to the old one.
        c.insert((2..4).map(|i| (k(i), entry(f32::from(i)))));
        assert_eq!(c.stats().entries, 4);
        // 0 is used, so it moves back to the young one, which turns the generations over and
        // drops 1.
        assert_eq!(c.get(&[k(0)]), vec![Some(entry(0.0))]);
        c.insert([(k(4), entry(4.0))]);
        let got = c.get(&[k(0), k(1), k(4), k(9)]);
        assert_eq!(got, vec![Some(entry(0.0)), None, Some(entry(4.0)), None]);
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.capacity), (3, 2, 4));
        assert!(s.entries <= 4, "{s:?}");
    }

    #[test]
    fn insert_overwrites() {
        let c = AnswerCache::new(10);
        c.insert([([1; 32], entry(1.0))]);
        c.insert([([1; 32], entry(2.0))]);
        assert_eq!(c.get(&[[1; 32]]), vec![Some(entry(2.0))]);
        assert_eq!(c.stats().entries, 1);
    }

    #[test]
    fn modes() {
        assert_eq!(CacheMode::parse("bypass"), Some(CacheMode::Bypass));
        assert_eq!(CacheMode::parse("refresh"), Some(CacheMode::Refresh));
        assert_eq!(CacheMode::parse("use"), Some(CacheMode::Use));
        assert_eq!(CacheMode::parse("off"), None);
    }
}
