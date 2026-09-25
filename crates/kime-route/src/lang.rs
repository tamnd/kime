//! A port of Laya's `laya/lang.py`: the script of a state, a best effort guess of the language of
//! Latin script text from function words and diacritics, and from those whether the English
//! checkpoint can read it. The rules, word lists, thresholds and tie breaks are Laya's, so
//! [`analyse`] gives the same answer as `laya.lang.analyse` for the same state.
//!
//! Python's string predicates are matched with the Unicode general category rather than Rust's
//! own: `str.isalpha` is the letter categories, where `char::is_alphabetic` also takes marks and
//! letter numbers.

use std::collections::HashSet;
use std::sync::OnceLock;

use kime_core::answer::py_round;
use regex::Regex;
use serde_json::{Map, Value};
use unicode_normalization::char::canonical_combining_class;
use unicode_properties::{GeneralCategory, UnicodeGeneralCategory};

/// Unicode blocks the English checkpoint cannot read, by script name.
const SCRIPT_RANGES: &[(&str, &[(u32, u32)])] = &[
    ("greek", &[(0x0370, 0x03FF), (0x1F00, 0x1FFF)]),
    ("cyrillic", &[(0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)]),
    ("armenian", &[(0x0530, 0x058F)]),
    ("hebrew", &[(0x0590, 0x05FF)]),
    (
        "arabic",
        &[(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    ),
    ("devanagari", &[(0x0900, 0x097F), (0xA8E0, 0xA8FF)]),
    ("bengali", &[(0x0980, 0x09FF)]),
    ("gurmukhi", &[(0x0A00, 0x0A7F)]),
    ("gujarati", &[(0x0A80, 0x0AFF)]),
    ("oriya", &[(0x0B00, 0x0B7F)]),
    ("tamil", &[(0x0B80, 0x0BFF)]),
    ("telugu", &[(0x0C00, 0x0C7F)]),
    ("kannada", &[(0x0C80, 0x0CFF)]),
    ("malayalam", &[(0x0D00, 0x0D7F)]),
    ("sinhala", &[(0x0D80, 0x0DFF)]),
    ("thai", &[(0x0E00, 0x0E7F)]),
    ("lao", &[(0x0E80, 0x0EFF)]),
    ("tibetan", &[(0x0F00, 0x0FFF)]),
    ("myanmar", &[(0x1000, 0x109F)]),
    ("georgian", &[(0x10A0, 0x10FF)]),
    ("ethiopic", &[(0x1200, 0x137F)]),
    ("khmer", &[(0x1780, 0x17FF)]),
    ("hangul", &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF)]),
    ("kana", &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)]),
    ("han", &[(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)]),
];

/// Function words by language, in Laya's order, which is the order ties are broken in. See
/// `_STOP` in `laya/lang.py` for why each list holds what it does.
const STOP: &[(&str, &[&str])] = &[
    (
        "en",
        &[
            "the", "and", "is", "are", "was", "were", "to", "of", "in", "for", "with", "that",
            "this", "it", "you", "have", "has", "not", "but", "on", "at", "be", "as", "from",
            "will", "can", "would", "there", "their", "what", "which", "please", "we", "i",
        ],
    ),
    (
        "fr",
        &[
            "le", "la", "les", "des", "une", "est", "pour", "dans", "que", "qui", "avec", "sur",
            "pas", "plus", "nous", "vous", "être", "cette", "mais", "sont", "ont", "aux", "ce",
            "et", "du", "au", "ou", "je", "tu", "il", "elle", "ils", "elles", "mon", "ton", "ma",
            "ta", "sa", "mes", "tes", "ses", "ces", "deux", "trois", "très", "bien", "tout",
            "tous", "toute", "fait", "veux", "veut", "peux", "peut", "dois", "doit", "merci",
            "bonjour", "jour", "jours", "mois", "fois", "quand", "comment", "pourquoi", "alors",
            "donc",
        ],
    ),
    (
        "de",
        &[
            "der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "nicht", "mit", "für",
            "auf", "von", "zu", "sich", "auch", "werden", "wurde", "haben", "sind", "oder", "aber",
            "ich", "wir", "mir", "mich", "dir", "dich", "uns", "mein", "meine", "meinen", "meinem",
            "meiner", "diese", "dieser", "diesen", "dieses", "einen", "einem", "einer", "wie",
            "wo", "wann", "welche", "im", "zum", "zur", "aus", "bei", "nach", "noch", "bitte",
            "heute", "jetzt", "kann", "kannst", "habe", "gibt", "wird", "in", "was",
        ],
    ),
    (
        "es",
        &[
            "el", "los", "las", "que", "por", "con", "para", "una", "es", "se", "del", "como",
            "pero", "son", "está", "este", "esta", "todo", "más", "muy", "hay", "sus", "la", "un",
            "y", "al", "lo", "le", "les", "su", "mi", "tu", "nos", "ni", "dos", "tres", "fue",
            "fueron", "ser", "tiene", "tienen", "tengo", "puede", "pueden", "quiero", "necesito",
            "hemos", "han", "sobre", "entre", "cuando", "donde", "porque", "aunque", "también",
            "ya", "eso", "esto", "esa", "ese", "nada", "algo", "aquí", "hoy", "gracias",
        ],
    ),
    (
        "pt",
        &[
            "os", "as", "que", "em", "um", "uma", "para", "com", "não", "é", "se", "do", "da",
            "dos", "das", "mas", "são", "está", "este", "esta", "muito", "pelo", "pela", "o", "e",
            "na", "nas", "nos", "ao", "aos", "por", "foi", "era", "ser", "sou", "tem", "tenho",
            "pode", "podem", "quero", "preciso", "eu", "meu", "minha", "seu", "sua", "isso",
            "isto", "aqui", "ali", "como", "quando", "onde", "porque", "mais", "já", "ainda",
            "agora", "hoje", "ontem", "dois", "três", "tudo", "nada", "obrigado", "olá", "você",
            "vocês", "voce", "voces", "vc", "vcs", "nao", "sao", "ja", "até", "tá", "pra",
            "gostaria", "obrigada", "também", "tambem", "estou", "estamos", "meus", "minhas",
            "nosso", "nossa", "consigo", "cadê", "boa", "tarde", "noite", "depois", "antes",
            "então", "entao", "ninguém", "ninguem", "alguém", "alguem", "nenhum", "nenhuma",
            "estava", "ficou", "fiz", "deu",
        ],
    ),
    (
        "it",
        &[
            "il", "lo", "gli", "che", "di", "per", "con", "non", "è", "si", "del", "della", "sono",
            "questo", "questa", "anche", "come", "più", "nella", "alla", "la", "le", "un", "uno",
            "una", "e", "ed", "o", "da", "su", "tra", "fra", "mi", "ci", "ne", "ho", "hai", "ha",
            "abbiamo", "avete", "hanno", "era", "stato", "stata", "devo", "deve", "devono",
            "voglio", "vorrei", "mio", "mia", "tuo", "sua", "quando", "dove", "perche", "molto",
            "poco", "sempre", "mai", "già", "ancora", "adesso", "oggi", "ieri", "grazie", "ciao",
            "scusa", "nel", "nell", "negli", "sul", "sulla", "sulle", "dal", "dalla", "dallo",
            "dagli", "dei", "delle", "dello", "degli", "agli", "alle", "col",
        ],
    ),
    (
        "nl",
        &[
            "het", "een", "van", "is", "op", "te", "dat", "niet", "met", "voor", "zijn", "aan",
            "door", "maar", "ook", "worden", "deze", "naar", "wordt",
        ],
    ),
    (
        "ro",
        &[
            "și", "să", "este", "sunt", "care", "pentru", "din", "dar", "după", "până", "fără",
            "ale", "lui", "în", "fost", "acum", "vreau", "trebuie", "foarte", "acest", "această",
            "acesta", "aceasta", "mi", "ți", "vă", "nu",
        ],
    ),
    (
        "bn",
        &[
            "ami",
            "amar",
            "amake",
            "amra",
            "amader",
            "apni",
            "apnar",
            "apnake",
            "apnara",
            "tumi",
            "tomar",
            "tomake",
            "tomra",
            "tader",
            "ota",
            "eita",
            "oita",
            "ekta",
            "ei",
            "oi",
            "ki",
            "keno",
            "kivabe",
            "kibhabe",
            "kothay",
            "kokhon",
            "kobe",
            "koto",
            "kintu",
            "jodi",
            "tahole",
            "ar",
            "theke",
            "jonno",
            "sathe",
            "shathe",
            "diye",
            "niye",
            "moddhe",
            "kore",
            "korte",
            "korchi",
            "korsi",
            "korbo",
            "korechi",
            "koreche",
            "korun",
            "koren",
            "korlam",
            "hobe",
            "hoyeche",
            "hoise",
            "hocche",
            "hoyni",
            "chai",
            "chaina",
            "lagbe",
            "parchi",
            "parbo",
            "parchina",
            "peyechi",
            "paini",
            "dite",
            "dilam",
            "diyechi",
            "nai",
            "khub",
            "onek",
            "ekhon",
            "akhon",
            "ekhono",
            "abar",
            "ekbar",
            "duibar",
            "ajke",
            "kalke",
            "taka",
            "bhalo",
            "valo",
            "kharap",
            "shomossa",
            "somossa",
            "dhonnobad",
            "bhai",
            "shob",
            "keu",
            "kichu",
            "bolte",
            "bolun",
            "parben",
            "asbe",
            "jabe",
            "pabo",
            "ferot",
            "dorkar",
            "hoye",
            "geche",
            "gese",
        ],
    ),
    (
        "az",
        &[
            "və", "ve", "bir", "bu", "üçün", "ucun", "ilə", "ile", "olan", "olub", "olmasa", "var",
            "yox", "yoxdur", "mən", "sən", "biz", "siz", "onlar", "daha", "çox", "cox", "hər",
            "nə", "kimi", "görə", "sonra", "əgər", "eger", "deyil", "lakin", "amma", "ancaq",
            "artıq", "artiq", "də", "isə", "həm", "yalnız", "yalniz",
        ],
    ),
];

/// Letters ordinary English does not use, after lowercasing.
const NON_EN_DIACRITICS: &str = concat!(
    "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœ",
    "ăâîșțşţ",
    "ąćęłńśźż",
    "čďěňřšťůž",
    "őű",
    "ğı",
    "āēģīķļņūž",
    "đ",
    "ə",
);

/// A diacritic rate at or above this is evidence the text is not English.
pub const NON_EN_DIACRITIC_RATE: f64 = 0.02;
/// The share of non-Latin letters that makes mostly Latin text count as the other script.
pub const NON_LATIN_FRACTION: f64 = 0.2;
/// The smaller share that counts when there are enough non-Latin letters.
pub const NON_LATIN_MIN_FRACTION: f64 = 0.1;
/// The non-Latin letters needed with the smaller share.
pub const NON_LATIN_MIN_LETTERS: usize = 10;

/// The most characters of a state [`analyse`] reads.
pub const MAX_CHARS: usize = 4000;

struct Tables {
    stop: Vec<(&'static str, HashSet<&'static str>)>,
    shared: HashSet<&'static str>,
    diacritics: HashSet<char>,
    identifier: Regex,
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let stop: Vec<_> =
            STOP.iter().map(|(lg, words)| (*lg, words.iter().copied().collect())).collect();
        let shared = STOP
            .iter()
            .flat_map(|(_, w)| w.iter().copied())
            .filter(|w| stop.iter().filter(|(_, s): &&(_, HashSet<_>)| s.contains(w)).count() > 1)
            .collect();
        Tables {
            stop,
            shared,
            diacritics: NON_EN_DIACRITICS.chars().collect(),
            // `[\w-]*(?:[.@][\w-]+)+` with Python's `\w`: letters, numbers and the underscore.
            identifier: Regex::new(r"[\p{L}\p{N}_-]*(?:[.@][\p{L}\p{N}_-]+)+")
                .unwrap_or_else(|e| unreachable!("{e}")),
        }
    })
}

/// Python's `str.isalpha` for one character.
pub(crate) fn is_alpha(c: char) -> bool {
    matches!(
        c.general_category(),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// A character of Python's `[^\W\d_]`: a letter, or a number that is not a decimal digit.
fn is_word(c: char) -> bool {
    is_alpha(c)
        || matches!(
            c.general_category(),
            GeneralCategory::LetterNumber | GeneralCategory::OtherNumber
        )
}

fn is_latin(cp: u32, below: u32) -> bool {
    cp < below
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
}

fn named_script(cp: u32) -> Option<&'static str> {
    SCRIPT_RANGES
        .iter()
        .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)))
        .map(|(name, _)| *name)
}

/// Whether a letter counts as Latin script.
pub(crate) fn is_latin_letter(c: char) -> bool {
    is_latin(c as u32, 0x02B0)
}

/// The script of one letter for [`detect_script`] and [`script_profile`].
fn script_of_letter(c: char) -> &'static str {
    let cp = c as u32;
    if is_latin(cp, 0x02B0) { "latin" } else { named_script(cp).unwrap_or("other") }
}

/// Letter counts by script, in the order Python's dict would hold them.
fn counts(text: &str, latin_first: bool) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> =
        if latin_first { vec![("latin", 0)] } else { Vec::new() };
    let mut latin = 0;
    for c in text.chars().filter(|&c| is_alpha(c)) {
        let s = script_of_letter(c);
        if s == "latin" && !latin_first {
            latin += 1;
        } else if let Some(e) = counts.iter_mut().find(|(k, _)| *k == s) {
            e.1 += 1;
        } else {
            counts.push((s, 1));
        }
    }
    if !latin_first {
        counts.push(("latin", latin));
    }
    counts
}

/// The first key with the largest value, as Python's `max` over a dict picks it.
fn first_max<'a, T: PartialOrd + Copy + 'a>(
    it: impl Iterator<Item = &'a (&'static str, T)>,
) -> Option<&'static str> {
    let mut best: Option<(&'static str, T)> = None;
    for &(k, v) in it {
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((k, v));
        }
    }
    best.map(|(k, _)| k)
}

/// The script most of the letters are in: `latin`, `han`, `devanagari` and so on, `other` for a
/// script Laya does not name, or `unknown` when there are no letters.
#[must_use]
pub fn detect_script(text: &str) -> &'static str {
    let c = counts(text, false);
    if c.iter().all(|(_, n)| *n == 0) {
        return "unknown";
    }
    first_max(c.iter()).unwrap_or("unknown")
}

/// The share of the letters in each script, leaving out scripts with none.
#[must_use]
pub fn script_profile(text: &str) -> Vec<(&'static str, f64)> {
    let c = counts(text, true);
    let total: usize = c.iter().map(|(_, n)| n).sum();
    if total == 0 {
        return Vec::new();
    }
    c.into_iter().filter(|(_, n)| *n > 0).map(|(k, n)| (k, n as f64 / total as f64)).collect()
}

/// Non-Latin runs that read as words rather than as a symbol, a name or a pronunciation inside
/// English prose.
fn has_non_latin_words(text: &str) -> bool {
    let mut run = 0usize;
    let mut first_upper = false;
    let mut script: Option<&str> = None;
    let word = |run: usize, upper: bool| run >= 2 && !upper;
    for c in text.chars() {
        if canonical_combining_class(c) != 0 {
            continue;
        }
        let cp = c as u32;
        let s = if is_latin(cp, 0x0250) { None } else { named_script(cp) };
        if s.is_some() && s == script {
            run += 1;
            continue;
        }
        if word(run, first_upper) {
            return true;
        }
        (run, first_upper, script) = match s {
            Some(_) => (1, c.is_uppercase(), s),
            None => (0, false, None),
        };
    }
    word(run, first_upper)
}

/// The text of a state that detection reads: its string leaves, joined by spaces, at most
/// `max_chars` characters. Keys are left out, since they are usually English.
#[must_use]
pub fn state_text(state: &Value, max_chars: usize) -> String {
    fn leaves<'a>(v: &'a Value, depth: usize, out: &mut Vec<&'a str>) {
        if depth > 6 {
            return;
        }
        match v {
            Value::String(s) => out.push(s),
            Value::Object(m) => m.values().for_each(|v| leaves(v, depth + 1, out)),
            Value::Array(a) => a.iter().for_each(|v| leaves(v, depth + 1, out)),
            _ => {}
        }
    }
    let mut all = Vec::new();
    leaves(state, 0, &mut all);
    let mut parts: Vec<&str> = Vec::new();
    let mut budget = max_chars as i64;
    for leaf in all {
        if budget <= 0 {
            break;
        }
        let n = leaf.chars().count() as i64;
        if n > budget {
            parts.push(head(leaf, budget as usize));
            break;
        }
        parts.push(leaf);
        budget -= n + 1;
    }
    let joined = parts.join(" ");
    head(&joined, max_chars).to_string()
}

fn head(s: &str, n: usize) -> &str {
    s.char_indices().nth(n).map_or(s, |(i, _)| &s[..i])
}

/// The evidence behind the language guess for Latin script text.
#[derive(Debug, Clone, PartialEq)]
pub struct LatinProfile {
    /// The language, or `None` when undecided.
    pub language: Option<&'static str>,
    /// English function words.
    pub english_hits: usize,
    /// Letters ordinary English does not use, over all characters.
    pub diacritic_rate: f64,
    /// Whether the diacritic rate alone says the text is not English.
    pub looks_non_english: bool,
}

/// Laya's `latin_profile`: function word hits per language and the diacritic rate.
#[must_use]
pub fn latin_profile(text: &str) -> LatinProfile {
    let t = tables();
    let cleaned = t.identifier.replace_all(text, " ").replace('İ', "i").to_lowercase();
    let words: Vec<&str> = cleaned.split(|c: char| !is_word(c)).filter(|w| !w.is_empty()).collect();
    let lowered = text.to_lowercase();
    let (mut n, mut diac) = (0usize, 0usize);
    for c in lowered.chars() {
        n += 1;
        diac += usize::from(t.diacritics.contains(&c));
    }
    let diacritic_rate = diac as f64 / n.max(1) as f64;
    let looks_non_english = diacritic_rate >= NON_EN_DIACRITIC_RATE;
    if words.len() < 4 {
        return LatinProfile { language: None, english_hits: 0, diacritic_rate, looks_non_english };
    }
    let scores: Vec<(&'static str, usize)> = t
        .stop
        .iter()
        .map(|(lg, sw)| (*lg, words.iter().filter(|w| sw.contains(*w)).count()))
        .collect();
    let en = scores[0].1;
    let evidenced: Vec<(&'static str, usize)> = scores
        .iter()
        .zip(&t.stop)
        .skip(1)
        .filter(|(_, (_, sw))| words.iter().any(|w| sw.contains(w) && !t.shared.contains(w)))
        .map(|(s, _)| *s)
        .collect();
    let best_lg = first_max(evidenced.iter());
    let best = best_lg.map_or(0, |lg| evidenced.iter().find(|(k, _)| *k == lg).map_or(0, |e| e.1));
    let language = match best_lg {
        Some(lg) if best >= 2.max(en + 2) => Some(lg),
        Some(lg) if looks_non_english && best >= 2.max(en) => Some(lg),
        _ if en > 0 && !looks_non_english => Some("en"),
        _ => None,
    };
    LatinProfile { language, english_hits: en, diacritic_rate, looks_non_english }
}

/// The function word lists by language, in Laya's order, as `_STOP` in `laya/lang.py`.
#[must_use]
pub fn stop_words() -> &'static [(&'static str, &'static [&'static str])] {
    STOP
}

/// A best effort language code for Latin script text, or `None` when undecided.
#[must_use]
pub fn guess_latin_language(text: &str) -> Option<&'static str> {
    latin_profile(text).language
}

/// What [`analyse`] found in a state.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// The script that decides: see [`detect_script`].
    pub script: &'static str,
    /// The share of the letters in each script.
    pub script_profile: Vec<(&'static str, f64)>,
    /// The language, for Latin script text where one was found.
    pub language: Option<&'static str>,
    /// Whether the English checkpoint can be expected to read the state.
    pub is_english: bool,
    /// Whether no language was named.
    pub language_undecided: bool,
    /// See [`LatinProfile::diacritic_rate`], rounded to 4 places.
    pub diacritic_rate: f64,
    /// The share of letters that are not Latin, rounded to 4 places.
    pub non_latin_fraction: f64,
}

impl Analysis {
    /// The analysis as the dict `laya.lang.analyse` returns.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("script".into(), self.script.into());
        let prof: Map<String, Value> =
            self.script_profile.iter().map(|(k, v)| ((*k).to_string(), (*v).into())).collect();
        m.insert("script_profile".into(), Value::Object(prof));
        m.insert("language".into(), self.language.into());
        m.insert("is_english".into(), self.is_english.into());
        m.insert("language_undecided".into(), self.language_undecided.into());
        m.insert("diacritic_rate".into(), self.diacritic_rate.into());
        m.insert("non_latin_fraction".into(), self.non_latin_fraction.into());
        Value::Object(m)
    }
}

/// Laya's `analyse`: the script, the language when it can be told, and whether the English
/// checkpoint can read the state. A string state is read as it is.
#[must_use]
pub fn analyse(state: &Value) -> Analysis {
    analyse_text(&state_text(state, MAX_CHARS))
}

/// [`analyse`] for text already flattened by [`state_text`].
#[must_use]
pub fn analyse_text(text: &str) -> Analysis {
    let prof = script_profile(text);
    let mut script = detect_script(text);
    let non_latin = if prof.is_empty() {
        0.0
    } else {
        py_round(1.0 - prof.iter().find(|(k, _)| *k == "latin").map_or(0.0, |p| p.1), 4)
    };
    if script == "latin" && (non_latin >= NON_LATIN_MIN_FRACTION) && has_non_latin_words(text) {
        let letters = text.chars().filter(|&c| is_alpha(c)).count();
        let n_non_latin = (non_latin * letters as f64).round_ties_even() as usize;
        if non_latin >= NON_LATIN_FRACTION || n_non_latin >= NON_LATIN_MIN_LETTERS {
            script = first_max(prof.iter().filter(|(k, _)| *k != "latin")).unwrap_or(script);
        }
    }
    let other = |script, english, non_latin_fraction| Analysis {
        script,
        script_profile: prof.clone(),
        language: None,
        is_english: english,
        language_undecided: true,
        diacritic_rate: 0.0,
        non_latin_fraction,
    };
    if script == "unknown" {
        return other("unknown", true, 0.0);
    }
    if script != "latin" {
        return other(script, false, non_latin);
    }
    let lat = latin_profile(text);
    let undecided = lat.language.is_none();
    Analysis {
        script: "latin",
        script_profile: prof,
        language: lat.language,
        is_english: lat.language == Some("en") || (undecided && !lat.looks_non_english),
        language_undecided: undecided,
        diacritic_rate: py_round(lat.diacritic_rate, 4),
        non_latin_fraction: non_latin,
    }
}

/// Whether the English checkpoint can be expected to read this state.
#[must_use]
pub fn is_english(state: &Value) -> bool {
    analyse(state).is_english
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scripts() {
        assert_eq!(detect_script(""), "unknown");
        assert_eq!(detect_script("hello"), "latin");
        assert_eq!(detect_script("Привет, как дела"), "cyrillic");
        assert_eq!(detect_script("ab 你好"), "han");
        assert_eq!(detect_script("a你"), "han");
        assert_eq!(detect_script("ㄅㄆ"), "other");
    }

    #[test]
    fn languages() {
        assert_eq!(guess_latin_language("the order is late and I want a refund"), Some("en"));
        assert_eq!(guess_latin_language("je veux un remboursement pour la commande"), Some("fr"));
        assert_eq!(guess_latin_language("hi"), None);
    }

    #[test]
    fn states() {
        let a =
            analyse(&json!({"subject": "Hola", "body": "necesito ayuda con mi pedido por favor"}));
        assert_eq!(a.language, Some("es"));
        assert!(!a.is_english);
        assert!(is_english(&json!("Set α to 0.05 and rerun the job")));
        assert!(!is_english(&json!("请帮我查一下订单 ORDER-12345 的状态")));
        assert_eq!(state_text(&json!(["ab", {"k": "cd"}, 3]), 4000), "ab cd");
    }
}
