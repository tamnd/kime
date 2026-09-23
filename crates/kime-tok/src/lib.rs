//! The two tokenizer families kime needs: byte level BPE for the ModernBERT vocabulary and SentencePiece style BPE with byte fallback for the Gemma 2 vocabulary that mmBERT uses. Ids must match Hugging Face tokenizers exactly. See spec/06-input.md.
//!
//! A [`Tokenizer`] is loaded from the `tokenizer.json` a checkpoint ships with. Only the pieces our
//! checkpoints use are supported, and anything else is a load error that names the piece, because a
//! tokenizer that quietly ignores a setting gives wrong ids and every number after it is wrong too.
//!
//! One of the six crates where `unsafe` is allowed. Every block carries a `// SAFETY:` comment
//! that names the invariant which makes it sound.

mod added;
mod bpe;
mod bytelevel;
mod hash;
mod load;
mod metaspace;

use std::borrow::Cow;
use std::fmt;
use std::path::Path;

use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};

pub use load::Error;

use added::Added;
use bpe::Bpe;

/// How text is normalized before the model sees it.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Normalizer {
    None,
    Nfc,
    /// Replace every occurrence of one string with another. mmBERT uses it to turn spaces into `▁`.
    Replace {
        from: String,
        to: String,
    },
}

impl Normalizer {
    fn apply<'a>(&self, s: &'a str) -> Cow<'a, str> {
        match self {
            Normalizer::None => Cow::Borrowed(s),
            Normalizer::Nfc => {
                // ASCII is always in NFC, and checking for it runs many bytes at a time.
                if s.is_ascii() || is_nfc_quick(s.chars()) == IsNormalized::Yes {
                    Cow::Borrowed(s)
                } else {
                    Cow::Owned(s.nfc().collect())
                }
            }
            Normalizer::Replace { from, to } => {
                if s.contains(from.as_str()) {
                    Cow::Owned(s.replace(from.as_str(), to))
                } else {
                    Cow::Borrowed(s)
                }
            }
        }
    }
}

/// How normalized text is cut into words before BPE.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PreTokenizer {
    /// The GPT-2 split pattern, then bytes mapped to the GPT-2 alphabet.
    ByteLevel,
    /// Spaces to `▁`, a `▁` put in front of every piece that does not start with one, then a split before every `▁`.
    Metaspace { replacement: char, prepend: bool, split: bool },
}

/// How ids go back to text.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Decoder {
    ByteLevel,
    /// `▁` back to a space, `<0xNN>` tokens back to bytes, then everything joined.
    Metaspace {
        replacement: char,
    },
}

/// The special tokens the input layouts need. Ids come from `tokenizer_config.json` when it is
/// given, and from the usual names otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Specials {
    /// The token a sequence starts with, `[CLS]` for ModernBERT and `<bos>` for mmBERT.
    pub cls: u32,
    /// The separator, `[SEP]` or `<eos>`.
    pub sep: u32,
    /// The mask token, which the compat layout puts in front of every option.
    pub mask: u32,
    /// The padding token.
    pub pad: u32,
}

/// A loaded tokenizer. Encoding is a pure function of the input text and never touches global state.
pub struct Tokenizer {
    normalizer: Normalizer,
    pre: PreTokenizer,
    decoder: Decoder,
    added: Added,
    bpe: Bpe,
    specials: Specials,
    mask_text: String,
}

impl fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tokenizer")
            .field("normalizer", &self.normalizer)
            .field("pre", &self.pre)
            .field("vocab", &self.bpe.vocab_len())
            .field("merges", &self.bpe.merges_len())
            .field("added", &self.added.len())
            .field("specials", &self.specials)
            .finish()
    }
}

impl Tokenizer {
    /// Load `tokenizer.json`, and `tokenizer_config.json` next to it when there is one.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
        let dir = dir.as_ref();
        let json = std::fs::read(dir.join("tokenizer.json"))
            .map_err(|e| Error::Io(dir.join("tokenizer.json"), e))?;
        let config = match std::fs::read(dir.join("tokenizer_config.json")) {
            Ok(b) => Some(b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(Error::Io(dir.join("tokenizer_config.json"), e)),
        };
        Tokenizer::from_bytes(&json, config.as_deref())
    }

    /// Load from the bytes of `tokenizer.json` and, optionally, `tokenizer_config.json`.
    pub fn from_bytes(json: &[u8], config: Option<&[u8]>) -> Result<Tokenizer, Error> {
        load::load(json, config)
    }

    /// The special token ids.
    pub fn specials(&self) -> Specials {
        self.specials
    }

    /// The literal text of the mask token, which the compat layout replaces with a space in user text.
    pub fn mask_text(&self) -> &str {
        &self.mask_text
    }

    /// The number of ids, counting added tokens.
    pub fn vocab_size(&self) -> usize {
        self.bpe.vocab_len().max(self.added.max_id().map_or(0, |m| m as usize + 1))
    }

    /// The id of a token, looked up among added tokens first and then the model vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added.id_of(token).or_else(|| self.bpe.id_of(token))
    }

    /// The text of one id, as it is stored in the vocabulary.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.added.content_of(id).or_else(|| self.bpe.token_of(id))
    }

    /// Encode `text` without adding special tokens, which is what Laya does for every piece it
    /// tokenizes. Ids are appended to `out`.
    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        // Added tokens that match the raw text come out first, then each gap between them is
        // normalized and searched again for the added tokens that match normalized text. This is
        // the order Hugging Face uses, and it matters: ModernBERT has added tokens for runs of
        // spaces, which only match after normalization.
        self.added.split_raw(text, &mut |piece| match piece {
            added::Piece::Token(id) => out.push(id),
            added::Piece::Text(gap) => {
                let normalized = self.normalizer.apply(gap);
                self.added.split_normalized(&normalized, &mut |piece| match piece {
                    added::Piece::Token(id) => out.push(id),
                    added::Piece::Text(t) => self.encode_words(t, out),
                });
            }
        });
    }

    /// Encode `text` without special tokens into a new vector.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        self.encode_into(text, &mut out);
        out
    }

    fn encode_words(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        match &self.pre {
            PreTokenizer::ByteLevel => {
                bytelevel::split(text, |word| self.bpe.encode_bytelevel(word, out))
            }
            PreTokenizer::Metaspace { replacement, prepend, split } => {
                metaspace::split(text, *replacement, *prepend, *split, |word| {
                    self.bpe.encode_chars(word, out)
                })
            }
        }
    }

    /// Decode ids back to text. Special tokens are skipped when `skip_special` is set, as Hugging
    /// Face's `decode(skip_special_tokens=True)` does.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let tokens = ids.iter().filter_map(|&id| {
            if skip_special && self.added.is_special(id) {
                return None;
            }
            self.id_to_token(id)
        });
        match &self.decoder {
            Decoder::ByteLevel => bytelevel::decode(tokens),
            Decoder::Metaspace { replacement } => metaspace::decode(tokens, *replacement),
        }
    }
}

#[cfg(test)]
mod tests;
