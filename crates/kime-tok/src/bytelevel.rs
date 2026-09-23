//! The byte level pre-tokenizer and decoder from GPT-2, as ModernBERT uses them.
//!
//! The split pattern is `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
//! written out as a scanner instead of a regex. Each branch below is one alternative of the
//! pattern, tried in the same order, so the first one that matches wins exactly as it does in
//! Oniguruma.

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    c.general_category_group() == GeneralCategoryGroup::Letter
}

fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    c.general_category_group() == GeneralCategoryGroup::Number
}

fn is_space(c: char) -> bool {
    c.is_whitespace()
}

fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// The end of the run of chars from `i` that satisfy `pred`.
fn run(text: &str, i: usize, pred: impl Fn(char) -> bool) -> usize {
    text[i..].char_indices().find(|&(_, c)| !pred(c)).map_or(text.len(), |(j, _)| i + j)
}

fn char_at(text: &str, i: usize) -> Option<char> {
    text[i..].chars().next()
}

/// The length of the pre-token that starts at `i`.
fn next_token(text: &str, i: usize) -> usize {
    let rest = &text.as_bytes()[i..];
    if rest[0] == b'\'' && rest.len() >= 2 {
        match rest[1] {
            b's' | b't' | b'm' | b'd' => return 2,
            b'r' if rest.get(2) == Some(&b'e') => return 3,
            b'v' if rest.get(2) == Some(&b'e') => return 3,
            b'l' if rest.get(2) == Some(&b'l') => return 3,
            _ => {}
        }
    }
    let c = char_at(text, i).expect("i is a char boundary inside the text");
    // ` ?\p{L}+`, ` ?\p{N}+` and ` ?[^\s\p{L}\p{N}]+`, with the optional leading space.
    let (after_space, next) = if c == ' ' {
        match char_at(text, i + 1) {
            Some(n) => (i + 1, n),
            None => (i, c),
        }
    } else {
        (i, c)
    };
    for pred in [is_letter as fn(char) -> bool, is_number, is_other] {
        if pred(next) {
            return run(text, after_space, pred) - i;
        }
        if pred(c) {
            return run(text, i, pred) - i;
        }
    }
    // `\s+(?!\S)` and then `\s+`. A run of spaces followed by more text gives back its last char,
    // so that the next pre-token can start with it, unless the run is a single char.
    let end = run(text, i, is_space);
    if end == text.len() {
        return end - i;
    }
    let last = text[i..end].chars().next_back().expect("the run is not empty");
    let shorter = end - last.len_utf8();
    if shorter > i { shorter - i } else { end - i }
}

/// Cut `text` into pre-tokens and hand each one to `f`.
pub(crate) fn split(text: &str, mut f: impl FnMut(&[u8])) {
    let mut i = 0;
    while i < text.len() {
        let n = next_token(text, i);
        f(&text.as_bytes()[i..i + n]);
        i += n;
    }
}

/// GPT-2's map from bytes to printable chars. Printable bytes map to themselves and the others
/// to code points from 256 up, in byte order.
pub(crate) const fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    let mut b = 0usize;
    while b < 256 {
        let printable = (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
        table[b] = if printable {
            match char::from_u32(b as u32) {
                Some(c) => c,
                None => '\0',
            }
        } else {
            let c = match char::from_u32(256 + n) {
                Some(c) => c,
                None => '\0',
            };
            n += 1;
            c
        };
        b += 1;
    }
    table
}

pub(crate) const BYTE_TO_CHAR: [char; 256] = byte_to_char_table();

fn char_to_byte(c: char) -> Option<u8> {
    let u = c as u32;
    let direct = (0x21..=0x7e).contains(&u) || (0xa1..=0xac).contains(&u) || (0xae..=0xff).contains(&u);
    if direct {
        return Some(u as u8);
    }
    if (256..256 + 68).contains(&u) {
        return BYTE_TO_CHAR.iter().position(|&x| x == c).map(|b| b as u8);
    }
    None
}

/// Hugging Face's byte level decoder: every token's chars go back to bytes, and a token with a
/// char outside the alphabet, which only added tokens have, is taken as its own UTF-8.
pub(crate) fn decode<'a>(tokens: impl Iterator<Item = &'a str>) -> String {
    let mut bytes = Vec::new();
    for t in tokens {
        let start = bytes.len();
        let mut ok = true;
        for c in t.chars() {
            match char_to_byte(c) {
                Some(b) => bytes.push(b),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok {
            bytes.truncate(start);
            bytes.extend_from_slice(t.as_bytes());
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
