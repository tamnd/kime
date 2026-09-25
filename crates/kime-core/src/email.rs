//! Laya's email cleaner, `clean_email_body` and `email_state` from `laya/email.py`, with the same
//! output. It drops quoted history, signatures, device footers and confidentiality disclaimers in
//! English, Portuguese and Spanish, so the model reads the new message and not the thread under it.
//!
//! The patterns are Laya's, copied as written. Python's regexes have lookarounds and Rust's do
//! not, so the two lookaheads and the sentence split's lookbehind are done by hand. Python's `\s`
//! also takes the four separators U+001C to U+001F, so `py` adds them wherever a pattern says
//! `\s`, and `space` is Python's `str.isspace`.

use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

/// How much of the cleaned body Laya keeps by default.
pub const MAX_CHARS: usize = 3000;

/// Python's `str.isspace`.
fn space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

fn strip(s: &str) -> &str {
    s.trim_matches(space)
}

/// A Python pattern as a Rust one: `\s` and `\S` take Python's four extra separators.
fn py(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() + 32);
    let mut chars = pattern.chars();
    let mut depth = 0;
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('s') if depth > 0 => out.push_str(r"\s\x1c-\x1f"),
                Some('s') => out.push_str(r"[\s\x1c-\x1f]"),
                Some('S') => out.push_str(r"[^\s\x1c-\x1f]"),
                Some(n) => {
                    out.push('\\');
                    out.push(n);
                }
                None => out.push('\\'),
            },
            '[' => {
                depth += 1;
                out.push(c);
            }
            ']' if depth > 0 => {
                depth -= 1;
                out.push(c);
            }
            _ => out.push(c),
        }
    }
    out
}

fn re(pattern: &str) -> Regex {
    regex::RegexBuilder::new(&py(pattern))
        .dfa_size_limit(64 << 20)
        .build()
        .expect("Laya's email patterns compile")
}

/// A pattern with a lookahead for a digit anywhere after `prefix`, which Laya writes as
/// `(?=.*\d)`.
struct Dated {
    prefix: Regex,
    full: Regex,
}

impl Dated {
    fn is_match(&self, line: &str) -> bool {
        self.prefix.find(line).is_some_and(|m| digit().is_match(&line[m.end()..]))
            && self.full.is_match(line)
    }
}

fn digit() -> &'static Regex {
    static D: OnceLock<Regex> = OnceLock::new();
    D.get_or_init(|| Regex::new(r"\d").expect("digit"))
}

struct Patterns {
    quote: Vec<Regex>,
    quote_dated: Vec<Dated>,
    attribution_tail: Regex,
    attribution_head: Dated,
    header_from_name: Regex,
    header_next: Regex,
    signature: Vec<Regex>,
    device_footer: Regex,
    disclaimer: Regex,
    paragraph: Regex,
    blanks: Regex,
}

fn patterns() -> &'static Patterns {
    static P: OnceLock<Patterns> = OnceLock::new();
    P.get_or_init(|| {
        let device = "iphone|ipad|android|ios|celular|telemóvel|móvil|galaxy|smartphone|samsung|tablet|outlook|yahoo|mail|e-?mail|gmail|windows";
        Patterns {
            quote: vec![
                re(r"(?i)^\s*On .{0,300}wrote:\s*$"),
                re(r"(?i)^\s*-{2,}\s*(Original|Forwarded) Message\s*-{2,}"),
                re(
                    r"(?i)^\s*-{2,}\s*(Mensagem (original|encaminhada)|Mensaje (original|reenviado))\s*-{2,}",
                ),
                re(r"^\s*_{8,}\s*$"),
                re(r"(?i)^\s*From:\s.+$"),
                re(r"(?i)^\s*De:\s.*[@<]"),
            ],
            quote_dated: vec![
                Dated {
                    prefix: re(r"(?i)^\s*Em "),
                    full: re(r"(?i)^\s*Em .{0,300}escreveu:\s*$"),
                },
                Dated {
                    prefix: re(r"(?i)^\s*El "),
                    full: re(r"(?i)^\s*El .{0,300}escribi[óo]:\s*$"),
                },
            ],
            attribution_tail: re(r"(?i)^.{0,120}\S@\S+\s+(wrote|escreveu|escribi[óo]):\s*$"),
            attribution_head: Dated {
                prefix: re(r"(?i)^\s*(On|Em|El) "),
                full: re(r"(?i)^\s*(On|Em|El) "),
            },
            header_from_name: re(r"(?i)^\s*De:\s+\S"),
            header_next: re(r"(?i)^\s*(Enviad[oa]( em| el)?:\s|(Data|Fecha):\s.*\d{4})"),
            signature: vec![
                re(r"^\s*--\s*$"),
                re(concat!(
                    r"^\s*(?i:best|kind|warmest|warm|many thanks|thanks|thank you|regards|cheers|sincerely)",
                    r"(?i:\s+(?:and|&)\s+regards|\s+(?:regards|wishes|again|in advance|a lot|so much|very much))?",
                    r"[\s,;:!.]*(?:[^\W\d_a-zß-öø-ÿ][\w'-]*[\s,.]*){0,3}$",
                )),
                re(r"(?i)^\s*sent from my (iphone|android|mobile|ipad)"),
                re(concat!(
                    r"(?i)^\s*(atenciosamente|att|abraços?|abs|um abraço|cordialmente|grat[oa]|(muito )?obrigad[oa]s?",
                    r"( desde já| pela atenção)?|(com os melhores )?cumprimentos|saudações|",
                    r"(un )?saludos?( cordiales)?|atentamente|(muchas )?gracias( de antemano)?)[\s,!.]*$",
                )),
            ],
            device_footer: re(&format!(
                r"(?i)^\s*((enviad[oa] (do|pelo|pela|via|desde|a partir do)( meu| minha| mi)?|sent from( my)?) ({device})( ({device}|para|for|no|na|\d+))*|(obter o|get) outlook (para|for) (ios|android))[\s.!]*$"
            )),
            disclaimer: re(concat!(
                r"(?i)(\b(e-?mail|message|information|communication|transmission|contents?)\b[^.]{0,60}",
                r"\bconfidential\b[^.]{0,60}\b(intended|solely|addressee|recipient|privileged|",
                r"disclos|unauthori[sz]ed)|",
                r"\bconfidential\b[^.]{0,60}\b(and (may|is) (also )?privileged)|",
                r"if you (have )?received this (e-?mail|message) in error|",
                r"\b(esta|este) (mensagem|e-?mail|mensaje|correo)\b[^.]{0,80}(confidencia|sigilos|privilegiad)|",
                r"\b(uso exclusivo|exclusivamente|únicamente|unicamente)\b[^.]{0,30}",
                r"(destinatári|destinatari|pessoa|persona|entidade|entidad)|",
                r"\b(recebeu|recebido|receber) (esta|este) (mensagem|e-?mail)\b[^.]{0,20} por (engano|erro)|",
                r"\b(ha recibido|recibió|recibe) (este|esta) (mensaje|correo)\b[^.]{0,20} por error|",
                r"\bantes de imprimir\b[^.]{0,100}(meio ambiente|medio ambiente|natureza|planeta|realmente necess)|",
                r"\b(meio|medio) ambiente\b[^.]{0,30}antes de imprimir)",
            )),
            paragraph: re(r"\n\s*\n"),
            blanks: re(r"[ \t]+"),
        }
    })
}

/// Python's `re.split(r"(?<=[.!?])\s+", text)`: splits at every run of whitespace that follows a
/// full stop, question mark or exclamation mark.
fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut prev) = (0, None);
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if space(c) && matches!(prev, Some('.' | '!' | '?')) {
            out.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            while let Some(&(j, n)) = it.peek() {
                if !space(n) {
                    break;
                }
                end = j + n.len_utf8();
                it.next();
            }
            start = end;
            prev = None;
            continue;
        }
        prev = Some(c);
    }
    out.push(&text[start..]);
    out
}

/// Python's `str.isalpha` and `str.isupper` on the first letter: a fresh sentence, not a wrapped
/// line. Lines in uncased scripts never start one.
fn starts_new_sentence(line: &str) -> bool {
    line.chars().find(|c| c.is_alphabetic()).is_some_and(char::is_uppercase)
}

/// Splits a sentence the disclaimer matched at each newline that starts a new sentence, so a
/// request line glued to a footer line survives and a wrapped footer still drops whole.
fn split_fused_lines(sentence: &str) -> Vec<String> {
    if !sentence.contains('\n') {
        return vec![sentence.to_string()];
    }
    let (mut pieces, mut buf) = (Vec::new(), String::new());
    for line in sentence.split('\n').map(strip) {
        if line.is_empty() {
            continue;
        }
        if !buf.is_empty() && starts_new_sentence(line) {
            pieces.push(std::mem::replace(&mut buf, line.to_string()));
        } else if buf.is_empty() {
            buf = line.to_string();
        } else {
            buf.push(' ');
            buf.push_str(line);
        }
    }
    if !buf.is_empty() {
        pieces.push(buf);
    }
    pieces
}

/// Drops the disclaimer sentences from one paragraph, and keeps the rest.
fn strip_disclaimer(paragraph: &str) -> String {
    let d = &patterns().disclaimer;
    if !d.is_match(paragraph) {
        return paragraph.to_string();
    }
    let mut pieces = Vec::new();
    for p in sentences(paragraph).into_iter().map(strip).filter(|p| !p.is_empty()) {
        if d.is_match(p) {
            pieces.extend(split_fused_lines(p));
        } else {
            pieces.push(p.to_string());
        }
    }
    pieces.retain(|p| !d.is_match(p));
    pieces.join(" ")
}

/// The first `n` characters of `s`, as Python's `s[:n]`.
fn head(s: &str, n: usize) -> &str {
    s.char_indices().nth(n).map_or(s, |(i, _)| &s[..i])
}

/// Removes quoted history, signatures and disclaimers, and keeps at most `max_chars` characters.
/// Laya's default is [`MAX_CHARS`].
#[must_use]
pub fn clean_email_body(body: &str, max_chars: usize) -> String {
    let p = patterns();
    let text = body.replace("\r\n", "\n").replace('\r', "\n").replace("\\n", "\n");
    // Laya bounds the work before the expensive patterns, since only max_chars come out.
    let text = head(&text, max_chars.saturating_mul(4));
    let src: Vec<&str> = text.split('\n').collect();
    let mut lines: Vec<&str> = Vec::new();
    for (i, &line) in src.iter().enumerate() {
        let quote = p.quote.iter().any(|r| r.is_match(line))
            || p.quote_dated.iter().any(|r| r.is_match(line));
        if quote && !lines.is_empty() {
            break;
        }
        if !lines.is_empty()
            && p.header_from_name.is_match(line)
            && src.get(i + 1).is_some_and(|next| p.header_next.is_match(next))
        {
            break;
        }
        if p.attribution_tail.is_match(line) && !lines.is_empty() {
            if lines.last().is_some_and(|l| p.attribution_head.is_match(l)) {
                lines.pop();
            }
            break;
        }
        if line.trim_start_matches(space).starts_with('>') {
            continue;
        }
        lines.push(line.trim_end_matches(space));
    }
    let n = lines.len();
    let mut cut = n;
    let from = ((n as f64 * 0.6) as i64).min(n as i64 - 8).max(1);
    for (i, line) in lines.iter().enumerate().skip(from as usize) {
        let len = strip(line).chars().count();
        if (len <= 40 && p.signature.iter().any(|r| r.is_match(line)))
            || (len <= 60 && p.device_footer.is_match(line))
        {
            cut = i;
            break;
        }
    }
    let joined = lines[..cut].join("\n");
    let paragraphs: Vec<String> = p
        .paragraph
        .split(&joined)
        .map(strip_disclaimer)
        .filter(|s| !strip(s).is_empty())
        .map(|s| strip(&s).to_string())
        .collect();
    let text = p.blanks.replace_all(&paragraphs.join("\n\n"), " ").into_owned();
    head(&text, max_chars).to_string()
}

/// A state for email questions: the subject, the cleaned body, the sender when there is one, and
/// any other fields, which replace these when they share a name. Fields whose value is null are
/// left out.
#[must_use]
pub fn email_state(
    subject: &str,
    body: &str,
    sender: Option<&str>,
    clean: bool,
    extra: &[(&str, Value)],
) -> Map<String, Value> {
    let mut state = Map::new();
    state.insert("subject".into(), strip(subject).into());
    let body = if clean { clean_email_body(body, MAX_CHARS) } else { body.to_string() };
    state.insert("body".into(), body.into());
    if let Some(s) = sender.filter(|s| !s.is_empty()) {
        state.insert("from".into(), s.into());
    }
    for (k, v) in extra {
        if !v.is_null() {
            state.insert((*k).to_string(), v.clone());
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_whitespace() {
        assert_eq!(py(r"^\s*[\s,]\S"), r"^[\s\x1c-\x1f]*[\s\x1c-\x1f,][^\s\x1c-\x1f]");
        assert_eq!(strip("\x1c a \u{3000}"), "a");
    }

    #[test]
    fn sentence_split() {
        assert_eq!(sentences("A. B!  C? d e.\n\nF"), ["A.", "B!", "C?", "d e.", "F"]);
        assert_eq!(sentences("no stop "), ["no stop "]);
        assert_eq!(sentences("end. "), ["end.", ""]);
    }

    #[test]
    fn quoted_history_and_signature() {
        let body = "Hi,\n\nPlease refund invoice 4411.\n\nThanks,\nAnna\n\nOn Mon, 3 Mar 2025 at 10:00, Bob <bob@x.com> wrote:\n> old";
        assert_eq!(clean_email_body(body, MAX_CHARS), "Hi,\n\nPlease refund invoice 4411.");
    }
}
