//! API keys and per key rate limits, from spec/03-api.md and spec/11-serving.md.
//!
//! Keys are kept only as blake3 hashes. A presented key is hashed and compared against every
//! stored hash, with blake3's constant time equality and no early exit, so the time taken says
//! nothing about which key came close.
//!
//! Each key has two limits, requests per minute and input tokens per second. Each one is a token
//! bucket kept as a single atomic, the time at which the bucket would be full again (GCRA, the
//! generic cell rate algorithm). Admitting a request is one compare and swap, with no lock.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Who may call the API and how often. Off by default.
#[derive(Debug, Default)]
pub struct Auth {
    keys: Vec<Key>,
    laya: bool,
    rpm: Option<u32>,
    tps: Option<u32>,
}

#[derive(Debug)]
struct Key {
    hash: blake3::Hash,
    rpm: Option<u32>,
    tps: Option<u32>,
}

/// Why a request was not let in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Denied {
    /// No `Authorization: Bearer` header.
    Missing,
    /// A key that is not configured.
    Unknown,
    /// Over the requests per minute limit, with the time until the next one fits.
    Requests(u32, Duration),
    /// Over the tokens per second limit, with the time until the debt is paid down.
    Tokens(u32, Duration),
}

impl Auth {
    /// No keys: every request is let in, and `rpm` and `tps` apply to all of them together.
    #[must_use]
    pub fn off() -> Self {
        Auth::default()
    }

    /// Adds the keys in a keys file: one key per line as `key [name [rpm [tps]]]`, where `-`
    /// for rpm or tps means the server default. Blank lines and lines starting with `#` are
    /// skipped. The name is for the operator and is not used yet.
    ///
    /// # Errors
    ///
    /// A line that does not parse, or a key given twice, with its line number.
    pub fn add_keys_file(&mut self, text: &str) -> Result<(), String> {
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut f = line.split_whitespace();
            let key = f.next().unwrap_or_default();
            let _name = f.next();
            let limit = |v: Option<&str>, what: &str| match v {
                None | Some("-") => Ok(None),
                Some(v) => v.parse::<u32>().ok().filter(|&n| n > 0).map(Some).ok_or_else(|| {
                    format!("line {}: {what} must be a whole number above 0, got '{v}'", i + 1)
                }),
            };
            let rpm = limit(f.next(), "rpm")?;
            let tps = limit(f.next(), "tps")?;
            if f.next().is_some() {
                return Err(format!("line {}: expected key [name [rpm [tps]]]", i + 1));
            }
            self.add(key, rpm, tps).map_err(|e| format!("line {}: {e}", i + 1))?;
        }
        Ok(())
    }

    /// Adds one key, with its own limits or the server defaults.
    ///
    /// # Errors
    ///
    /// When the key is already there.
    pub fn add(&mut self, key: &str, rpm: Option<u32>, tps: Option<u32>) -> Result<(), String> {
        let hash = blake3::hash(key.as_bytes());
        if self.keys.iter().any(|k| k.hash == hash) {
            return Err("the same key is given twice".into());
        }
        self.keys.push(Key { hash, rpm, tps });
        Ok(())
    }

    /// laya-serve's `LAYA_API_KEY`: one key, and laya-serve's single 401 for a missing or wrong
    /// one, so its clients see no change.
    #[must_use]
    pub fn laya(key: &str) -> Self {
        let mut a = Auth { laya: true, ..Auth::default() };
        let _ = a.add(key, None, None);
        a
    }

    /// The limits for keys that set none, and for every request when there are no keys.
    #[must_use]
    pub fn with_defaults(mut self, rpm: Option<u32>, tps: Option<u32>) -> Self {
        self.rpm = rpm;
        self.tps = tps;
        self
    }

    /// How many keys there are.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether there are no keys, so auth is off.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Whether requests need a key.
    #[must_use]
    pub fn is_on(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Whether a missing or wrong key gets laya-serve's 401 rather than Jev's 403 and 401.
    pub(crate) fn laya_style(&self) -> bool {
        self.laya
    }

    /// The bucket index for an `Authorization` header value. With auth off everyone shares
    /// bucket 0.
    pub(crate) fn check(&self, header: Option<&[u8]>) -> Result<usize, Denied> {
        if self.keys.is_empty() {
            return Ok(0);
        }
        let key = header.and_then(|h| {
            // laya-serve compares the whole header with "Bearer <key>", so its scheme is case
            // sensitive. HTTP says the scheme is not, and Jev follows HTTP.
            let (scheme, key) = (h.get(..7)?, h.get(7..)?);
            let ok = if self.laya {
                scheme == b"Bearer "
            } else {
                scheme.eq_ignore_ascii_case(b"bearer ")
            };
            ok.then_some(key)
        });
        let Some(key) = key else { return Err(Denied::Missing) };
        let hash = blake3::hash(key);
        let mut found = None;
        for (i, k) in self.keys.iter().enumerate() {
            if k.hash == hash {
                found = Some(i);
            }
        }
        found.ok_or(Denied::Unknown)
    }

    /// The buckets, one per key or one for everyone.
    pub(crate) fn buckets(&self) -> Vec<Bucket> {
        let one =
            |rpm: Option<u32>, tps: Option<u32>| Bucket::new(rpm.or(self.rpm), tps.or(self.tps));
        if self.keys.is_empty() {
            vec![one(None, None)]
        } else {
            self.keys.iter().map(|k| one(k.rpm, k.tps)).collect()
        }
    }
}

/// The two limits of one key.
#[derive(Debug)]
pub(crate) struct Bucket {
    rpm: Option<u32>,
    tps: Option<u32>,
    /// When the requests bucket is full again, in nanoseconds since [`epoch`].
    requests: AtomicU64,
    /// When the tokens bucket is full again, in nanoseconds since [`epoch`].
    tokens: AtomicU64,
}

const MINUTE: u64 = 60_000_000_000;
const SECOND: u64 = 1_000_000_000;

impl Bucket {
    fn new(rpm: Option<u32>, tps: Option<u32>) -> Self {
        Bucket { rpm, tps, requests: AtomicU64::new(0), tokens: AtomicU64::new(0) }
    }

    /// Lets one request in, or says how long until one fits. A minute of requests may come at
    /// once. Tokens are only known after the answer, so they are charged afterwards with
    /// [`Bucket::charge`], and a key in debt of more than a second of tokens waits it out.
    pub(crate) fn admit(&self) -> Result<(), Denied> {
        self.admit_at(now())
    }

    fn admit_at(&self, now: u64) -> Result<(), Denied> {
        if let Some(tps) = self.tps {
            let full = self.tokens.load(Ordering::Relaxed);
            if full > now + SECOND {
                return Err(Denied::Tokens(tps, Duration::from_nanos(full - SECOND - now)));
            }
        }
        if let Some(rpm) = self.rpm {
            let step = MINUTE / u64::from(rpm);
            let mut full = self.requests.load(Ordering::Relaxed);
            loop {
                let next = full.max(now) + step;
                if next > now + MINUTE {
                    return Err(Denied::Requests(rpm, Duration::from_nanos(next - MINUTE - now)));
                }
                match self.requests.compare_exchange_weak(
                    full,
                    next,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(seen) => full = seen,
                }
            }
        }
        Ok(())
    }

    /// Charges the input tokens of an answered request.
    pub(crate) fn charge(&self, tokens: u64) {
        self.charge_at(tokens, now());
    }

    fn charge_at(&self, tokens: u64, now: u64) {
        if let Some(tps) = self.tps {
            let cost = tokens.saturating_mul(SECOND) / u64::from(tps);
            let _ = self.tokens.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |full| {
                Some(full.max(now).saturating_add(cost))
            });
        }
    }
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn now() -> u64 {
    // Starts a minute in, so a bucket that has never been used counts as full.
    u64::try_from(epoch().elapsed().as_nanos()).unwrap_or(u64::MAX / 2) + MINUTE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(text: &str) -> Auth {
        let mut a = Auth::off();
        a.add_keys_file(text).unwrap();
        a
    }

    #[test]
    fn keys_file() {
        let a = keys("# ops\n\nk1\nk2 billing 60\nk3 batch - 1000\n");
        assert_eq!(a.keys.len(), 3);
        assert_eq!((a.keys[1].rpm, a.keys[1].tps), (Some(60), None));
        assert_eq!((a.keys[2].rpm, a.keys[2].tps), (None, Some(1000)));
        let mut b = Auth::off();
        assert_eq!(b.add_keys_file("k1\nk1").unwrap_err(), "line 2: the same key is given twice");
        assert_eq!(
            Auth::off().add_keys_file("k1 n 0").unwrap_err(),
            "line 1: rpm must be a whole number above 0, got '0'"
        );
        assert!(Auth::off().add_keys_file("k1 n 1 2 3").is_err());
    }

    #[test]
    fn bearer() {
        let a = keys("k1\nk2");
        assert_eq!(a.check(Some(b"Bearer k2")), Ok(1));
        assert_eq!(a.check(Some(b"bearer k1")), Ok(0));
        assert_eq!(a.check(Some(b"Bearer k3")), Err(Denied::Unknown));
        assert_eq!(a.check(Some(b"k1")), Err(Denied::Missing));
        assert_eq!(a.check(None), Err(Denied::Missing));
        let l = Auth::laya("s3cret");
        assert_eq!(l.check(Some(b"Bearer s3cret")), Ok(0));
        assert_eq!(l.check(Some(b"bearer s3cret")), Err(Denied::Missing));
        assert_eq!(Auth::off().check(None), Ok(0));
    }

    #[test]
    fn requests_per_minute() {
        let b = Bucket::new(Some(2), None);
        let t = MINUTE;
        assert_eq!(b.admit_at(t), Ok(()));
        assert_eq!(b.admit_at(t), Ok(()));
        // Both slots are used. The next one frees up after half a minute.
        assert_eq!(b.admit_at(t), Err(Denied::Requests(2, Duration::from_secs(30))));
        assert_eq!(b.admit_at(t + 10 * SECOND), Err(Denied::Requests(2, Duration::from_secs(20))));
        assert_eq!(b.admit_at(t + 30 * SECOND), Ok(()));
    }

    #[test]
    fn tokens_per_second() {
        let b = Bucket::new(None, Some(100));
        let t = MINUTE;
        assert_eq!(b.admit_at(t), Ok(()));
        // 250 tokens at 100 a second is 2.5 s of debt, a second of which is allowed.
        b.charge_at(250, t);
        assert_eq!(b.admit_at(t), Err(Denied::Tokens(100, Duration::from_millis(1500))));
        assert_eq!(b.admit_at(t + 1500 * 1_000_000), Ok(()));
    }

    #[test]
    fn unlimited() {
        let b = Bucket::new(None, None);
        for _ in 0..1000 {
            assert_eq!(b.admit_at(MINUTE), Ok(()));
        }
    }
}
