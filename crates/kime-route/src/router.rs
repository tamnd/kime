//! The checkpoint choice of Laya's `Router._route` for a request that names no model, with the
//! reasons it gives, and the language identifier in [`crate::lid`] on top of its detection.
//! Precedence: an explicit `lang`, then `lang_guess`, then detection, then the default.

use serde_json::Value;

use crate::lang::{Analysis, MAX_CHARS, analyse_text, state_text};
use crate::lid::{MIN_WORDS, OVERRULE, model, words};

/// Which checkpoint a request goes to, and why, in the words of Laya's `RouteDecision`.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// `Some(true)` for the English checkpoint, `Some(false)` for the multilingual one and `None`
    /// for the server's default.
    pub english: Option<bool>,
    pub reason: String,
    /// Laya's `analyse` of the state, when detection ran.
    pub detection: Option<Analysis>,
}

/// Laya's `_english_from_code`: whether a language code names English, or `None` when it names
/// nothing. `en`, `EN`, `en-US`, `en_US` and `en_US.UTF-8` are all English.
#[must_use]
pub fn english_from_code(value: &Value) -> Option<bool> {
    let code = match value {
        Value::Null => return None,
        Value::String(s) => s.trim().to_lowercase(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        v => v.to_string(),
    };
    let code = code.split('.').next().unwrap_or_default();
    let primary = code.split(['-', '_']).next().unwrap_or_default();
    if primary.is_empty() {
        return None;
    }
    Some(matches!(primary, "en" | "eng" | "english"))
}

fn key(english: bool) -> &'static str {
    if english { "english" } else { "multilingual" }
}

/// Python's `repr` of a string, for the reasons that quote what the caller sent.
fn repr(v: &Value) -> String {
    match v {
        Value::String(s) if s.contains('\'') && !s.contains('"') => format!("\"{s}\""),
        Value::String(s) => format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'")),
        v => v.to_string(),
    }
}

/// The decision for a state, given the caller's `lang` and `lang_guess` hints and whether the
/// server's default is the English checkpoint.
#[must_use]
pub fn route(
    state: &Value,
    lang: Option<&Value>,
    lang_guess: Option<&Value>,
    default_english: bool,
) -> Decision {
    if let Some(l) = lang
        && let Some(en) = english_from_code(l)
    {
        return Decision {
            english: Some(en),
            reason: format!("explicit lang={}", repr(l)),
            detection: None,
        };
    }
    if let Some(en) = lang_guess.and_then(english_from_code) {
        let what = if en { "English" } else { "non-English" };
        return Decision {
            english: Some(en),
            reason: format!("lang_guess: the caller identified this as {what} text"),
            detection: None,
        };
    }
    detect(state, default_english)
}

/// Detection alone: Laya's rules, with the identifier deciding Latin script text of two words or
/// more the way [`crate::lid::english_model`] does.
#[must_use]
pub fn detect(state: &Value, default_english: bool) -> Decision {
    let text = state_text(state, MAX_CHARS);
    let det = analyse_text(&text);
    let default = key(default_english);
    let (english, reason) = if det.script == "unknown" {
        (None, format!("no letters detected in state; using default ({default})"))
    } else if det.script != "latin" {
        let pct = 100.0 * det.non_latin_fraction;
        let reason = format!(
            "non-Latin script ({}, {pct:.0}% of letters); the English checkpoint cannot read it",
            det.script
        );
        (Some(false), reason)
    } else if words(&text) < MIN_WORDS {
        laya_latin(&det, default)
    } else {
        let m = model();
        let p = m.p_english(&text);
        let pct = 100.0 * p;
        if det.is_english && p >= m.threshold {
            if det.language_undecided {
                let reason = format!(
                    "Latin script, language not identified by its word lists, but the language identifier puts English at {pct:.0}%"
                );
                (Some(true), reason)
            } else {
                (Some(true), "English Latin text".to_string())
            }
        } else if det.is_english {
            let reason = format!(
                "Latin script, but the language identifier puts English at {pct:.0}%; not safe for the English checkpoint"
            );
            (Some(false), reason)
        } else if p >= OVERRULE {
            let over = match det.language {
                Some(l) => format!("the word lists' guess of '{l}'"),
                None => format!("{:.0}% non-English letters", 100.0 * det.diacritic_rate),
            };
            (
                Some(true),
                format!(
                    "Latin script, the language identifier puts English at {pct:.0}%, over {over}"
                ),
            )
        } else {
            laya_latin(&det, default)
        }
    };
    Decision { english, reason, detection: Some(det) }
}

/// Laya's three branches for Latin script text.
fn laya_latin(det: &Analysis, default: &str) -> (Option<bool>, String) {
    if !det.is_english {
        let reason = match det.language {
            Some(l) => format!("Latin script but language looks like '{l}', not English"),
            None => format!(
                "Latin script, language not identified but {:.0}% non-English letters; not safe for the English checkpoint",
                100.0 * det.diacritic_rate
            ),
        };
        (Some(false), reason)
    } else if det.language_undecided {
        let reason = format!(
            "Latin script, language not identified and no non-English letters; using default ({default})"
        );
        (None, reason)
    } else {
        (Some(true), "English Latin text".to_string())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn codes() {
        for (v, want) in [
            (json!("en"), Some(true)),
            (json!("EN"), Some(true)),
            (json!("en-US"), Some(true)),
            (json!("en_US.UTF-8"), Some(true)),
            (json!("english"), Some(true)),
            (json!("de"), Some(false)),
            (json!("pt_BR"), Some(false)),
            (json!("  "), None),
            (json!(""), None),
            (json!(".UTF-8"), None),
            (Value::Null, None),
            (json!(7), Some(false)),
        ] {
            assert_eq!(english_from_code(&v), want, "{v}");
        }
    }

    #[test]
    fn precedence() {
        let de = json!("Ich möchte mein Abonnement kündigen, bitte helfen Sie mir.");
        let d = route(&de, Some(&json!("en")), Some(&json!("de")), true);
        assert_eq!((d.english, d.reason.as_str()), (Some(true), "explicit lang='en'"));
        let d = route(&de, Some(&json!(" ")), Some(&json!("en_US.UTF-8")), true);
        assert_eq!(d.english, Some(true));
        assert_eq!(d.reason, "lang_guess: the caller identified this as English text");
        let d = route(&de, None, None, true);
        assert_eq!(d.english, Some(false));
        assert!(d.detection.is_some());
    }

    #[test]
    fn detection() {
        let d = detect(&json!("12345 !!!"), true);
        assert_eq!(
            (d.english, d.reason.as_str()),
            (None, "no letters detected in state; using default (english)")
        );
        let d = detect(&json!("Мне нужно отменить подписку"), true);
        assert_eq!(d.english, Some(false));
        assert_eq!(
            d.reason,
            "non-Latin script (cyrillic, 100% of letters); the English checkpoint cannot read it"
        );
        let d = detect(&json!("saya mau pesan tiket ke jakarta besok pagi"), true);
        assert_eq!(d.english, Some(false));
        assert!(
            d.reason.starts_with("Latin script, but the language identifier puts English at "),
            "{}",
            d.reason
        );
        let d = detect(
            &json!(
                "Apple releases Mac OS X 10.3.7 Update As expected, Apple Computer today released Mac OS X 10.3.7 Update, a maintenance release for its Mac OS X 10.3 Panther operating system."
            ),
            true,
        );
        assert_eq!(d.english, Some(true));
        assert!(d.reason.ends_with("over the word lists' guess of 'pt'"), "{}", d.reason);
        let d = detect(&json!("I was charged twice for my subscription"), true);
        assert_eq!((d.english, d.reason.as_str()), (Some(true), "English Latin text"));
    }
}
