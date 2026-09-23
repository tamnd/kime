# Input pipeline

This covers everything between the parsed JSON request and the token id arrays the engine runs on: rendering the state and criteria to text, tokenizing, laying out sequences for each model family, enforcing budgets, and cleaning email. It lives in `kime-core::render` and `kime-tok`. It runs on the request thread, not the inference thread, and its target cost is under 30 microseconds for a typical 2 KB request.

## Rendering values to text

Jev lets `instructions`, criteria and the state be any JSON value. The renderer turns a value into text deterministically.

| Value | Rendering |
|---|---|
| string | as is, after NFC normalization |
| number, bool | JSON text (`3.5`, `true`) |
| null | empty string |
| array | compact JSON (`["a","b"]`) |
| object | compact JSON with keys in the order sent, `ensure_ascii=false` |

Compact JSON means no spaces after `:` or `,`. That matches Python's `json.dumps(separators=(",",":"))` for the native family. The compat family uses Laya's exact choice (`json.dumps(ensure_ascii=False)`, which has spaces after `:` and `,`) so its tokens match. Key order is the order in the request body. The JSON parser is sonic-rs, configured to preserve object order, and object values are kept as raw slices so rendering is a copy, not a re-serialization, whenever the input is already compact.

A literal special token string in user text (for example `[MASK]` or `<mask>`) is replaced by a space before tokenizing, as Laya does. Otherwise a user could forge a marker. In the native family we go further: kime's own markers are ids outside the base vocabulary and cannot be produced by the tokenizer at all, so the replacement is only kept for compat.

## Tokenizers

`kime-tok` implements the two tokenizer kinds the models use, with no dependency on the HF `tokenizers` crate at runtime. The HF crate is a dev dependency for parity tests only.

### Byte level BPE (ModernBERT, OLMo derived, vocab 50,368)

1. Pre-tokenize with the GPT-NeoX split pattern (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`). It is implemented as a hand written scanner over UTF-8 with a precomputed table of Unicode general categories (letters, numbers), not a regex engine. The scanner is about 150 lines, has a proptest that compares it to the `fancy-regex` form on random Unicode, and runs at about 1 GB/s.
2. Map bytes to the GPT-2 byte to unicode alphabet. This is a lookup table and never materializes a string.
3. Encode each pre-token with a linear time BPE. We use the approach of GitHub's `bpe` crate (rust-gems): an Aho-Corasick automaton over the vocabulary plus the "compatible pair" check gives the same result as merge rank BPE without a heap. We vendor the algorithm into `kime-tok` rather than depend on the crate, because we need to handle special tokens and our own id space.
4. A per-thread cache from pre-token bytes to ids (a 64k entry open addressing table with FxHash) covers repeated words. Most English words hit it.

### SentencePiece style BPE with byte fallback (mmBERT, Gemma 2, vocab 256,000)

1. Normalize: replace spaces with `▁`, with no prefix space added. mmBERT's tokenizer has no consistent prefix space, and we must match whatever `tokenizer.json` says, so the normalizer is read from the file and only the combinations actually used by our checkpoints are supported. Anything else fails to load.
2. There is no regex pre-split. Digits are split into single characters.
3. Merge by rank with a binary heap over a linked list of symbols. Characters not in the vocabulary fall back to `<0xNN>` byte tokens.
4. Because there is no pre-split, the cache is keyed on whitespace delimited chunks, which is safe given how `▁` works.

### Parity and speed

Both tokenizers must produce ids identical to HF `tokenizers` 0.23 on:

- the full text of every benchmark dataset in 13;
- 10 million lines sampled from multilingual web text;
- a fuzz corpus of random Unicode, control characters, combining marks and invalid surrogates (after lossy UTF-8 decoding).

Speed target: 2 KB of English text in 15 microseconds or less on one core of an M3 or Zen 4, and at least 3x faster than HF `tokenizers` single threaded on the same input. The tokenizer never allocates per token. It writes into a reusable `Vec<u32>` owned by the request.

## Sequence layout, compat family

The compat family reproduces Laya's `build_sequence` exactly:

```
[CLS] tok("<type> question: <instructions>")[:budget] [SEP]
[MASK] tok(" " + opt_0)[:48]  [MASK] tok(" " + opt_1)[:48] ... [SEP]
tok(state)[:room] [SEP]
```

- Options render as `label: description`, `level i: description`, or for noul `false: <c>` and `true: <c>` with Laya's default phrases.
- The option budget is `head_max_len` minus the total option length. If fewer than 16 tokens are left, each option is cut to `max(4, (head_max_len - 16) / k)`. Instructions get `max(8, remaining)` tokens.
- `room = max(0, max_len - len(ids) - 1)`. The state is cut from the tail by default, or from the head when `truncation = head`.
- The output is truncated to `max_len` and markers beyond it are dropped. A question that loses markers is an error, `question 'x' options exceed head_max_len=192`, returned as 422. The same message is used, because Laya users grep for it.

One sequence per question. The state tokens are tokenized once per request and copied into each sequence.

## Sequence layout, native family

The native family has two separate inputs.

### State tower input

```
[CLS_S] state tokens [SEP]
```

- A string state is tokenized as is. An object or array state is rendered to compact JSON.
- In segment mode, each top level key becomes a segment, `"key":value`. Arrays split at elements. Segments longer than 1,024 tokens are split at token boundaries. Each segment is tokenized separately, so it can be cached separately. The segment list and each segment's token count are part of the plan.
- If the state is longer than the model's `max_tokens` (32,768 for released models), it is truncated by the strategy in `kime.truncation`. `middle` keeps the first and last halves and inserts a `[CUT]` special token. Truncation is reported in the extension block and counted in a Prometheus counter. It is never silent.

### Question tower input

For every question (and every chunk of a large choice):

```
[CLS_Q] type_text instructions [SEP]  [OPT] option_0 [SEP]  [OPT] option_1 [SEP] ...
```

- `type_text` is `choose:`, `rate:` or `true or false:`. It is short, fixed, and the same for all questions of a type.
- The instructions are the rendered instruction value. Backticked state paths like `` `ticket.messages[0].text` `` are kept as text. The model learns to follow them through training examples that use them (see 12). We do not resolve paths in the runtime, because Jev does not either.
- Choice option text is `label` if the description is null or empty, else `label: description`. Score level text is the description only, with no index. Noul options are `[NO] <false criterion>` and `[YES] <true criterion>`, where the criterion text is empty if not given.
- Per option budget: 64 tokens by default, set by the checkpoint. Each option segment has its own budget. There is no shared head budget that forces cuts when k is large. Banking77's 77 labels each keep their full name and description.
- Header budget: 512 tokens. Instructions longer than that are cut at the tail and reported.
- Position ids: the header uses 0 to h-1. Every option segment uses h to h + len - 1, the same range for all options. See 05 for why.

The question tower batch for a request is all question chunks of all questions, packed without padding. The plan records, for each row, which state memory it reads (for the batch endpoint, rows read different states) and where its markers are.

## Budgets and limits

| Budget | Compat | Native |
|---|---|---|
| State tokens | `max_len` minus the head, 512 or 1,024 | up to 32,768 |
| Per option | 48 tokens, and cut further when crowded | 64 tokens, never crowded |
| Instructions | whatever the options leave, at least 8 | 512 |
| Options per pass | all, up to about 40 before they no longer fit | 32 per chunk, 255 per question, 4,096 with shortlist |
| Total per request | n/a | 65,536 tokens counted as in `usage.input_tokens` |

`usage.input_tokens` for the native family is: state tokens once, plus the tokens of every question tower row. For compat it is the sum of all sequence lengths, which is what Laya reports.

## Email cleaning

`kime-core::email` ports Laya's `clean_email_body` and extends it. `email_state(subject, body, sender, clean = true, extra)` returns `{"subject": ..., "body": ..., "from": ...}` with extra keys appended.

The cleaner applies these steps in order. Each step is a small hand written matcher, not a regex, so the whole pass is linear:

1. Normalize line endings and strip zero width characters.
2. Cut at the first quote header. Supported forms:
   - "On <date>, <name> wrote:" in English, Portuguese ("escreveu"), Spanish ("escribió"), German ("schrieb"), French ("a écrit"), Italian ("ha scritto"), Dutch ("schreef"), Japanese ("のメッセージ:" and "さんは書きました:"), Chinese ("写道:" and "寫道:"), Korean ("작성:"), Russian ("написал"), Polish ("napisał") and Turkish ("yazdı").
   - Outlook blocks starting with "From:", "De:", "Von:", "Da:", "Van:", "发件人:" or "差出人:", followed within 4 lines by a sent or date line.
   - "-----Original Message-----" and its translations.
3. Drop lines starting with `>`.
4. Cut signatures. A line of `-- ` or a sign-off line (a list of about 60 phrases across the 12 languages, such as "Best regards", "Atenciosamente", "Mit freundlichen Grüßen" or "よろしくお願いします") counts as a signature start if it falls in the last 40% of lines and the lines after it are each 40 characters or fewer.
5. Drop device footers ("Sent from my iPhone" and about 30 variants).
6. Drop disclaimer paragraphs. These are paragraphs with at least two of the words confidential, intended recipient, privileged, disclaimer or their translations.
7. Collapse runs of blank lines and trim. Cut at `max_chars` (default 3,000) on a char boundary.

Test fixtures cover each rule in each language, plus the Laya test fixtures, which must produce identical output for en, pt and es.
