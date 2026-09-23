//! The Metaspace pre-tokenizer and decoder, as mmBERT's Gemma 2 tokenizer uses them.

use std::borrow::Cow;

/// Replace spaces, put the replacement char in front when `prepend` is set and the text does not
/// already start with it, then cut before every replacement char. The replacement char is merged
/// with what follows it, so `▁a▁▁b` gives `▁a`, `▁` and `▁b`.
pub(crate) fn split(
    text: &str,
    replacement: char,
    prepend: bool,
    split: bool,
    mut f: impl FnMut(&str),
) {
    let mut s: Cow<'_, str> = if text.contains(' ') {
        let mut buf = [0u8; 4];
        Cow::Owned(text.replace(' ', replacement.encode_utf8(&mut buf)))
    } else {
        Cow::Borrowed(text)
    };
    if prepend && !s.starts_with(replacement) {
        let mut owned = String::with_capacity(s.len() + replacement.len_utf8());
        owned.push(replacement);
        owned.push_str(&s);
        s = Cow::Owned(owned);
    }
    if !split {
        f(&s);
        return;
    }
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if c == replacement && i > start {
            f(&s[start..i]);
            start = i;
        }
    }
    if start < s.len() {
        f(&s[start..]);
    }
}

fn byte_token(t: &str) -> Option<u8> {
    let hex = t.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Replace, ByteFallback and Fuse, in that order, as the mmBERT `decoder` lists them. A run of
/// byte tokens that is not valid UTF-8 becomes one U+FFFD per byte, as Hugging Face does.
pub(crate) fn decode<'a>(tokens: impl Iterator<Item = &'a str>, replacement: char) -> String {
    let mut out = String::new();
    let mut bytes: Vec<u8> = Vec::new();
    let flush = |bytes: &mut Vec<u8>, out: &mut String| {
        if bytes.is_empty() {
            return;
        }
        match std::str::from_utf8(bytes) {
            Ok(s) => out.push_str(s),
            Err(_) => {
                for _ in 0..bytes.len() {
                    out.push('\u{fffd}');
                }
            }
        }
        bytes.clear();
    };
    for t in tokens {
        if let Some(b) = byte_token(t) {
            bytes.push(b);
            continue;
        }
        flush(&mut bytes, &mut out);
        for c in t.chars() {
            out.push(if c == replacement { ' ' } else { c });
        }
    }
    flush(&mut bytes, &mut out);
    out
}
