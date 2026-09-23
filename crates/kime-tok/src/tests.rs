use crate::bytelevel;
use crate::metaspace;

fn gpt2_split(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    bytelevel::split(text, |w| out.push(String::from_utf8(w.to_vec()).unwrap()));
    out
}

#[test]
fn bytelevel_split_follows_the_gpt2_pattern() {
    assert_eq!(gpt2_split("Hello world"), ["Hello", " world"]);
    assert_eq!(
        gpt2_split("I'm here, they'll see"),
        ["I", "'m", " here", ",", " they", "'ll", " see"]
    );
    assert_eq!(gpt2_split("a  b"), ["a", " ", " b"]);
    assert_eq!(gpt2_split("a\n\nb"), ["a", "\n", "\n", "b"]);
    assert_eq!(gpt2_split("end   "), ["end", "   "]);
    assert_eq!(gpt2_split("x 123 4.5"), ["x", " 123", " 4", ".", "5"]);
    assert_eq!(gpt2_split("{\"a\": 1}"), ["{\"", "a", "\":", " 1", "}"]);
    assert_eq!(gpt2_split("Straße 東京"), ["Straße", " 東京"]);
    assert_eq!(gpt2_split(" "), [" "]);
    assert_eq!(gpt2_split("'x"), ["'", "x"]);
}

#[test]
fn bytelevel_split_covers_every_byte_once() {
    let text = "mixed \t text\u{a0}with\u{2003}odd  spaces\r\n123abc!!?  ";
    assert_eq!(gpt2_split(text).concat(), text);
}

#[test]
fn byte_table_is_a_bijection() {
    let t = bytelevel::BYTE_TO_CHAR;
    let mut seen = std::collections::HashSet::new();
    for c in t {
        assert!(seen.insert(c), "{c:?} appears twice");
    }
    assert_eq!(t[b' ' as usize], 'Ġ');
    assert_eq!(t[b'\n' as usize], 'Ċ');
    assert_eq!(t[b'a' as usize], 'a');
    let all: Vec<&str> = t.iter().map(|_| "").collect();
    assert_eq!(all.len(), 256);
}

#[test]
fn metaspace_merges_the_marker_with_what_follows() {
    let mut out = Vec::new();
    metaspace::split("▁a▁▁b", '▁', true, true, |w| out.push(w.to_string()));
    assert_eq!(out, ["▁a", "▁", "▁b"]);
    out.clear();
    metaspace::split("hi there", '▁', true, true, |w| out.push(w.to_string()));
    assert_eq!(out, ["▁hi", "▁there"]);
}

#[test]
fn metaspace_decode_joins_bytes_and_replaces_the_marker() {
    let toks = ["▁caf", "<0xC3>", "<0xA9>", "▁ok", "<0xFF>"];
    assert_eq!(metaspace::decode(toks.into_iter(), '▁'), " café ok\u{fffd}");
}
