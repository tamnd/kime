//! The byte level pre-tokenizer and decoder from GPT-2, as ModernBERT uses them.
//!
//! The split pattern is `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
//! written out as a scanner instead of a regex. Each branch below is one alternative of the
//! pattern, tried in the same order, so the first one that matches wins exactly as it does in
//! Oniguruma.

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

/// Which of the pattern's char classes a char is in. Every char is in exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

/// Classes of the ASCII chars, so the common case never decodes UTF-8 or looks at Unicode tables.
const ASCII_CLASS: [Class; 128] = {
    let mut t = [Class::Other; 128];
    let mut b = 0;
    while b < 128 {
        let c = b as u8;
        t[b] = if c.is_ascii_alphabetic() {
            Class::Letter
        } else if c.is_ascii_digit() {
            Class::Number
        } else if matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
            // `char::is_whitespace` on ASCII, which unlike Python leaves out 0x1c to 0x1f.
            Class::Space
        } else {
            Class::Other
        };
        b += 1;
    }
    t
};

fn class(c: char) -> Class {
    if c.is_ascii() {
        return ASCII_CLASS[c as usize];
    }
    if c.is_whitespace() {
        return Class::Space;
    }
    match c.general_category_group() {
        GeneralCategoryGroup::Letter => Class::Letter,
        GeneralCategoryGroup::Number => Class::Number,
        _ => Class::Other,
    }
}

/// The end of the run of chars from `i` in class `k`.
fn run(text: &str, i: usize, k: Class) -> usize {
    let bytes = text.as_bytes();
    let mut j = i;
    while j < bytes.len() {
        let b = bytes[j];
        if b < 0x80 {
            if ASCII_CLASS[usize::from(b)] != k {
                return j;
            }
            j += 1;
            continue;
        }
        let c = char_at(text, j).expect("j is a char boundary inside the text");
        if class(c) != k {
            return j;
        }
        j += c.len_utf8();
    }
    j
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
    let kc = class(c);
    // ` ?\p{L}+`, ` ?\p{N}+` and ` ?[^\s\p{L}\p{N}]+`, with the optional leading space. The
    // alternatives are tried in order, and for each one the form with the space comes first.
    if c == ' '
        && let Some(next) = char_at(text, i + 1)
    {
        let kn = class(next);
        if kn != Class::Space {
            return run(text, i + 1, kn) - i;
        }
    }
    if kc != Class::Space {
        return run(text, i, kc) - i;
    }
    // `\s+(?!\S)` and then `\s+`. A run of spaces followed by more text gives back its last char,
    // so that the next pre-token can start with it, unless the run is a single char.
    let end = run(text, i, Class::Space);
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
        let printable =
            (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
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
    let direct =
        (0x21..=0x7e).contains(&u) || (0xa1..=0xac).contains(&u) || (0xae..=0xff).contains(&u);
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
