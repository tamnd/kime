//! Reading `tokenizer.json` and `tokenizer_config.json`.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::added::{Added, AddedToken};
use crate::bpe::Bpe;
use crate::{Decoder, Normalizer, PreTokenizer, Specials, Tokenizer};

/// Why a tokenizer failed to load.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read.
    Io(PathBuf, std::io::Error),
    /// A file is not valid JSON, or not the shape a tokenizer file has.
    Json(String),
    /// The file asks for something kime does not implement. The message names it.
    Unsupported(String),
    /// The file is self inconsistent, for example a merge of tokens that are not in the vocabulary.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(p, e) => write!(f, "reading {}: {e}", p.display()),
            Error::Json(m) => write!(f, "tokenizer file: {m}"),
            Error::Unsupported(m) => write!(f, "tokenizer file asks for something kime does not support: {m}"),
            Error::Invalid(m) => write!(f, "tokenizer file is invalid: {m}"),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Deserialize)]
struct File {
    added_tokens: Vec<AddedJson>,
    normalizer: Option<Value>,
    pre_tokenizer: Option<Value>,
    decoder: Option<Value>,
    model: ModelJson,
}

#[derive(Deserialize)]
struct AddedJson {
    id: u32,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default = "yes")]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

fn yes() -> bool {
    true
}

#[derive(Deserialize)]
struct ModelJson {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<Box<str>, u32>,
    merges: Merges,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    byte_fallback: bool,
    #[serde(default)]
    fuse_unk: bool,
    #[serde(default)]
    dropout: Option<f64>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    end_of_word_suffix: Option<String>,
    #[serde(default)]
    ignore_merges: bool,
}

/// Newer files write each merge as a pair and older ones as one string with a space in it.
#[derive(Deserialize)]
#[serde(untagged)]
enum Merges {
    Pairs(Vec<(String, String)>),
    Strings(Vec<String>),
}

fn kind(v: &Value) -> &str {
    v.get("type").and_then(Value::as_str).unwrap_or("")
}

fn normalizer(v: Option<&Value>) -> Result<Normalizer, Error> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(Normalizer::None);
    };
    match kind(v) {
        "NFC" => Ok(Normalizer::Nfc),
        "Replace" => {
            let from = v.pointer("/pattern/String").and_then(Value::as_str);
            let to = v.get("content").and_then(Value::as_str);
            match (from, to) {
                (Some(from), Some(to)) => Ok(Normalizer::Replace { from: from.to_string(), to: to.to_string() }),
                _ => Err(Error::Unsupported("a Replace normalizer with a regex pattern".into())),
            }
        }
        "Sequence" => {
            let list = v.get("normalizers").and_then(Value::as_array).cloned().unwrap_or_default();
            match list.as_slice() {
                [] => Ok(Normalizer::None),
                [one] => normalizer(Some(one)),
                _ => Err(Error::Unsupported("a Sequence of more than one normalizer".into())),
            }
        }
        other => Err(Error::Unsupported(format!("the {other:?} normalizer"))),
    }
}

fn pre_tokenizer(v: Option<&Value>) -> Result<PreTokenizer, Error> {
    let v = v.filter(|v| !v.is_null()).ok_or_else(|| Error::Unsupported("a tokenizer without a pre-tokenizer".into()))?;
    match kind(v) {
        "ByteLevel" => {
            if v.get("add_prefix_space").and_then(Value::as_bool) == Some(true) {
                return Err(Error::Unsupported("ByteLevel with add_prefix_space".into()));
            }
            if v.get("use_regex").and_then(Value::as_bool) == Some(false) {
                return Err(Error::Unsupported("ByteLevel without the split pattern".into()));
            }
            Ok(PreTokenizer::ByteLevel)
        }
        "Metaspace" => {
            let replacement = v.get("replacement").and_then(Value::as_str).and_then(|s| {
                let mut it = s.chars();
                match (it.next(), it.next()) {
                    (Some(c), None) => Some(c),
                    _ => None,
                }
            });
            let replacement = replacement.ok_or_else(|| Error::Invalid("Metaspace replacement is not one char".into()))?;
            let prepend = match v.get("prepend_scheme").and_then(Value::as_str) {
                Some("always") => true,
                Some("never") => false,
                Some(other) => return Err(Error::Unsupported(format!("Metaspace prepend_scheme {other:?}"))),
                None => v.get("add_prefix_space").and_then(Value::as_bool).unwrap_or(true),
            };
            let split = v.get("split").and_then(Value::as_bool).unwrap_or(true);
            Ok(PreTokenizer::Metaspace { replacement, prepend, split })
        }
        other => Err(Error::Unsupported(format!("the {other:?} pre-tokenizer"))),
    }
}

fn decoder(v: Option<&Value>, pre: &PreTokenizer) -> Result<Decoder, Error> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Err(Error::Unsupported("a tokenizer without a decoder".into()));
    };
    match (kind(v), pre) {
        ("ByteLevel", PreTokenizer::ByteLevel) => Ok(Decoder::ByteLevel),
        ("Metaspace", PreTokenizer::Metaspace { replacement, .. }) => Ok(Decoder::Metaspace { replacement: *replacement }),
        ("Sequence", PreTokenizer::Metaspace { replacement, .. }) => {
            let names: Vec<&str> = v.get("decoders").and_then(Value::as_array).map(|a| a.iter().map(kind).collect()).unwrap_or_default();
            if names == ["Replace", "ByteFallback", "Fuse"] {
                Ok(Decoder::Metaspace { replacement: *replacement })
            } else {
                Err(Error::Unsupported(format!("the decoder sequence {names:?}")))
            }
        }
        (other, _) => Err(Error::Unsupported(format!("the {other:?} decoder with this pre-tokenizer"))),
    }
}

fn config_token(config: &Value, key: &str) -> Option<String> {
    match config.get(key)? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

pub(crate) fn load(json: &[u8], config: Option<&[u8]>) -> Result<Tokenizer, Error> {
    let file: File = serde_json::from_slice(json).map_err(|e| Error::Json(e.to_string()))?;
    let m = file.model;
    if m.kind != "BPE" {
        return Err(Error::Unsupported(format!("the {:?} model", m.kind)));
    }
    if m.dropout.is_some_and(|d| d > 0.0) {
        return Err(Error::Unsupported("BPE dropout".into()));
    }
    if m.continuing_subword_prefix.is_some() || m.end_of_word_suffix.is_some() {
        return Err(Error::Unsupported("BPE subword prefixes or suffixes".into()));
    }
    if m.ignore_merges {
        return Err(Error::Unsupported("BPE ignore_merges".into()));
    }
    let merges: Vec<(String, String)> = match m.merges {
        Merges::Pairs(p) => p,
        Merges::Strings(s) => s
            .into_iter()
            .map(|line| match line.split_once(' ') {
                Some((a, b)) => Ok((a.to_string(), b.to_string())),
                None => Err(Error::Invalid(format!("merge {line:?} has no space"))),
            })
            .collect::<Result<_, _>>()?,
    };
    let bpe = Bpe::new(m.vocab, &merges, m.unk_token.as_deref(), m.byte_fallback, m.fuse_unk).map_err(Error::Invalid)?;

    let norm = normalizer(file.normalizer.as_ref())?;
    let pre = pre_tokenizer(file.pre_tokenizer.as_ref())?;
    let dec = decoder(file.decoder.as_ref(), &pre)?;

    let tokens = file
        .added_tokens
        .into_iter()
        .map(|a| AddedToken {
            id: a.id,
            content: a.content,
            single_word: a.single_word,
            lstrip: a.lstrip,
            rstrip: a.rstrip,
            normalized: a.normalized,
            special: a.special,
        })
        .collect();
    let added = Added::new(tokens, |s| norm.apply(s).into_owned());

    let config: Value = match config {
        Some(b) => serde_json::from_slice(b).map_err(|e| Error::Json(format!("tokenizer_config.json: {e}")))?,
        None => Value::Null,
    };
    let lookup = |key: &str, default: &str| -> Result<(u32, String), Error> {
        let name = config_token(&config, key).unwrap_or_else(|| default.to_string());
        let id = added.id_of(&name).or_else(|| bpe.id_of(&name)).ok_or_else(|| Error::Invalid(format!("{key} {name:?} is not a token")))?;
        Ok((id, name))
    };
    let (cls, _) = lookup("cls_token", "[CLS]")?;
    let (sep, _) = lookup("sep_token", "[SEP]")?;
    let (mask, mask_text) = lookup("mask_token", "[MASK]")?;
    let (pad, _) = lookup("pad_token", "[PAD]")?;

    Ok(Tokenizer { normalizer: norm, pre, decoder: dec, added, bpe, specials: Specials { cls, sep, mask, pad }, mask_text })
}
