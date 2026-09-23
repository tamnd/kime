# Model architecture

kime runs two model families. The `laya` compat family is the published Laya graph reproduced exactly, so existing checkpoints work and we have a baseline to beat on identical weights. The native `kime-v1` family is our own architecture. It encodes the state once and answers every question with a small tower that reads the cached state, and it is where the 10x comes from. Both are described as data (a `ModelSpec` in `kime-model`), so a backend implements a fixed set of ops and never has model specific code.

## Shared building blocks

Both families are built from ModernBERT style layers. All linear layers have no bias.

```
embed(x)       = LayerNorm(E[x])                                     # ModernBERT embedding norm, eps 1e-5, no bias
attn_block(h)  = h + Wo( Attention( RoPE(Wq n), RoPE(Wk n), Wv n, mask ) ),  n = LayerNorm(h)
mlp_block(h)   = h + Wout( GELU(a) * g ),  [a, g] = split(Win LayerNorm(h))   # GeGLU, Win has 2*I rows
layer(h)       = mlp_block(attn_block(h))
```

- In the first layer, `attn_norm` is the identity. ModernBERT does this and the MLX port reproduces it.
- Attention is global on every third layer (layer index % 3 == 0) and sliding window of 128 tokens (64 each side) on the others.
- RoPE theta is 160,000 on global layers. On local layers it is 10,000 for ModernBERT and 160,000 for mmBERT. Head dim is 64.
- GELU is the exact erf form, because ModernBERT uses `gelu` and not the tanh approximation. We keep erf so compat parity holds. Native checkpoints may switch to tanh if calibration and accuracy show no change, and this is recorded in the model config.
- The final norm is a LayerNorm without bias.

The local window follows the MLX port's working definition: key j is visible to query i when `|pos_i - pos_j| <= 64`. That is 129 positions, the HF `window_size // 2` convention.

## Family 1: laya compat

This is the exact Laya 0.3.7 `DecisionModel`. We need it for three reasons: drop-in use of the three published checkpoints, the M0 milestone parity test, and a same-weights speed comparison that isolates the engine gains from the model gains.

```
h  = encoder(ids)                                       # ModernBERT-large or mmBERT-base, all layers, final norm
h  = h + type_emb[qtype]                                # Embedding(3, d), broadcast over tokens
for l in 0..2:                                          # PyTorch TransformerEncoderLayer, norm_first, batch_first
    h = h + SelfAttn(LN1(h), key_padding_mask)          # nhead = d/64, WITH biases (PyTorch default)
    h = h + W2(ReLU(W1(LN2(h))))                        # 4d FFN, biases, LayerNorm with bias
m  = gather(h, marker_positions)                        # one vector per option
z  = scorer(m) = W_b(GELU(W_a(LN(m))))                  # LayerNorm(d), Linear(d,d), GELU, Linear(d,1), biases
act = act_head(concat(h[:,0], [top1, top1-top2, ent_norm, k/255]))   # Linear(d+4,256), GELU, Linear(256,2)
```

Notes for the implementer:

- The head layers use PyTorch conventions: biased linears, LayerNorm with bias and eps 1e-5, fused `in_proj_weight` of shape [3d, d], and `out_proj`. Their attention is full, with no RoPE and no window, and padding is masked by `src_key_padding_mask`.
- Weight name mapping from `model.safetensors`: `encoder.*` loads into the encoder (HF ModernBERT names: `embeddings.tok_embeddings`, `embeddings.norm`, `layers.N.attn.Wqkv`, `layers.N.attn.Wo`, `layers.N.attn_norm`, `layers.N.mlp.Wi`, `layers.N.mlp.Wo`, `layers.N.mlp_norm`, `final_norm`). `head.layers.N.self_attn.in_proj_weight`, `head.layers.N.self_attn.in_proj_bias`, `head.layers.N.linear1`, `head.layers.N.linear2`, `head.layers.N.norm1` and `head.layers.N.norm2` load into the head. `type_emb.weight`, `scorer.0` (LN), `scorer.1`, `scorer.3`, `act_head.0` and `act_head.2` load as named, and the buffer `temperature` holds the per type temperatures. The loader rejects missing or extra tensors and wrong shapes, naming each one, like Laya's `_verify_compatibility`.
- The act head output in compat mode is computed but, by default, replaced by the calibrated `max(p)` (see 04). `--laya-raw-act` restores the original value for parity testing.
- Compat mode keeps Laya's input format bit for bit (see 06), including the 48 token option cap and the head budget. That is the only way to get identical logits.

The engine speeds up compat mode through unpadded variable length batches, fused kernels and graphs, but the math is unchanged. The expected gain on a T4 is 3x to 4x for one question (about 33 to 40 ms down to about 9 to 12 ms). For 10 questions it is larger, about 5x, because padding and Python overhead grow with batch size. This is not 10x, and we say so in the README. The 10x targets are for the native family.

## Family 2: kime-v1 split encoder

### Why a split

In Laya every question costs a full pass over the state, because question tokens and state tokens attend to each other from layer 0. The laya-mlx research confirmed that nothing can be reused across questions: the first global layer already mixes them. The fix has to be in the architecture. kime-v1 lets the state be read by the questions through cross attention, but not the other way round. State encoding therefore depends only on the state and can be computed once per state and cached.

The cost is that state token representations cannot adapt to the question. Late interaction and split layer models (ColBERT, PreTTR) show that a few joint layers on top recover most of a cross encoder's quality. In kime-v1 the question tower's cross attention layers play that role. We protect accuracy by distilling from a full cross encoder teacher (see 12), and by making the question tower deep enough to do real reading (3 to 4 layers, each with self attention, cross attention and an MLP).

### Graph

```
# State tower, once per state
s0  = embed(state_ids)                         # [CLS_S] state tokens [SEP]
s   = layers_state(s0, mask = local/global, segment_mask optional)     # N_s ModernBERT layers
S   = LayerNorm(s)                             # final state representation
for j in 0..N_q:                               # state memory, once per state, cached
    K_j = RoPE_none(S Wk_j),  V_j = S Wv_j     # kv_heads = 2, head dim 64, so 128 dims each

# Question tower, once per question chunk, batched over all questions
q0  = embed(question_ids) + type_emb[qtype]    # header and option segments, see 06
for j in 0..N_q:
    q = q + SelfAttn(LN(q), RoPE(segment positions), mask = question mask)
    q = q + CrossAttn(LN(q), K_j, V_j)          # queries from q, 12 or 8 query heads grouped onto 2 kv heads
    q = q + GeGLU_MLP(LN(q))
Q   = LayerNorm(q)
m_i = Q[marker_i] + ord_emb(i, k) if score     # one vector per option
z_i = W_b(GELU(W_a(LN(m_i)))) + b_type         # scorer, per type bias
```

Design points:

- **Cross attention uses grouped query attention with 2 kv heads.** Each question tower layer has its own K and V projections of the final state representation. Two kv heads of dim 64 make the state memory 256 values per token per layer. With N_q = 4 in fp16 that is 2 KiB per state token, so a 5.3k token agent state is about 10.6 MiB. This keeps the state cache practical (see 11). Using the final state layer as the memory source, rather than matching layers, is the simplest choice that works with any N_s, and it is what the teacher distillation targets.
- **No RoPE on cross attention.** Keys carry position from the state tower already. The questions should find content, not positions.
- **The question mask makes options permutation equivariant.** Each option segment starts at the same position id, so RoPE gives no order signal between options. The mask is:
  - header tokens see the header and all option markers;
  - option tokens see the header, their own segment and all option markers;
  - markers see the header, their own segment and all markers.

  Options can therefore compare themselves to each other through the markers, which choice needs, but swapping two options just swaps their outputs. The only order dependent parts are the ordinal embedding for score, which is intended, and the floating point summation order. This is why kime can target a flip rate near 0 when Jev is at 0.13 and Laya at 0.15.
- **Ordinal embedding for score.** `ord_emb(i, k)` is a linear map of the features `[i/(k-1), (i/(k-1))^2, sin(pi i/(k-1)), cos(pi i/(k-1))]` to d. Mirrored scales in training teach the model that direction comes from the criteria text. The embedding only encodes adjacency.
- **Noul markers.** Two learned marker embeddings, `[NO]` and `[YES]`, replace the generic option marker for noul. No label words are inserted.
- **Type embedding.** Added to every question tower token. There are three rows, one per type.
- **Chunking.** A question with more than 32 options is split into chunks of at most 32 option segments. Every chunk carries the same header. Chunks of one question are independent rows in the batch. See 04 for how they are merged.
- **Pooled state embedding.** The mean of `S` over state tokens, L2 normalized. Used for shortlisting and returned by `return_embedding`.

### Segmented state encoding

Agent loops send nearly the same state every step: the same page with one field changed, plus one more history entry. Recomputing 5k tokens every step wastes most of the work. With `state_segments` on, an object state is split into segments at its top level keys (arrays are split at elements). Long segments are split every 1,024 tokens. The state tower then runs in two parts:

- Layers 0 to N_s - g - 1 use a block diagonal mask, so every token sees only its own segment, with positions restarting per segment. Each segment's output at layer N_s - g is a pure function of that segment's tokens, and it is cached under `blake3(model, segment tokens)`.
- The top g layers (g = 3 for the s tier, one global and two local) run over the concatenation of all segments with the normal mask, plus a learned segment index embedding added at layer N_s - g.

An agent step with one changed segment then costs the low layers on that segment, plus g layers on the full state, plus the question tower. For the Google Flights trace, that is about 20% of the full state tower cost (see the estimate in 13). The model is trained with segment mode on for half of the batches, so it works well either way, and `auto` turns it on for object states over 1,024 tokens.

Segment mode is also how the native model reaches Jev's 32k state context. Lower layers cost O(L times segment length). Only the top g layers pay for full sequence attention, and on those FlashAttention keeps memory linear.

### Tiers

| Model id | Tokenizer | d | Heads | I (GeGLU) | N_s | N_q | g | Total params | Non-embedding | Init |
|---|---|---|---|---|---|---|---|---|---|---|
| `kime-v1-s-en` | ModernBERT BPE, 50,368 | 512 | 8 | 1,344 | 12 | 4 | 3 | about 82M | about 56M | Ettin 68M layers where shapes match, else distilled |
| `kime-v1-m-en` | ModernBERT BPE, 50,368 | 768 | 12 | 1,152 | 19 | 3 | 3 | about 160M | about 121M | ModernBERT-base: layers 0 to 18 into the state tower, 19 to 21 into the question tower's self attention and MLP |
| `kime-v1-s-x` | Gemma 2 BPE, 256,000, trimmable | 384 | 6 | 1,152 | 18 | 4 | 3 | about 145M | about 47M | mmBERT-small |
| `kime-v1-m-x` | Gemma 2 BPE, 256,000, trimmable | 768 | 12 | 1,152 | 19 | 3 | 3 | about 318M | about 121M | mmBERT-base |
| `kime-t-en` (teacher) | ModernBERT BPE | 1,024 | 16 | 2,624 | full cross encoder, 28 layers + 2 head layers | | | about 425M | | ModernBERT-large, fixed Laya format |
| `kime-t-x` (teacher) | Gemma 2 BPE | 768 | 12 | 1,152 | full cross encoder, 22 + 2 | | | about 325M | | mmBERT-base |

The cross attention projections and the scorer are new weights in every tier. Parameter counts are estimates, and the exact values go in each release's model card. `kime-latest` routes between `s-en` and `s-x` by language. The m tiers are for callers who want more accuracy at about 2x the cost.

Vocabulary trimming for the x tiers: most of mmBERT's 256k Gemma vocabulary is used by few languages. `kime convert --trim-vocab <corpus>` keeps tokens seen in a sample of the target languages plus all byte fallback tokens, and remaps ids. It only reduces memory, not compute, so it is optional and off for released checkpoints.

### Compute budget

Per token, a state layer costs about `2 * (4 d^2 + 3 d I)` FLOPs plus attention. For `kime-v1-s-en` that is about 6.2 MFLOP per token per layer, or about 75 MFLOP per token for the whole state tower.

| Workload | State tokens | Question tokens | s-en FLOPs | Laya (large) FLOPs | Ratio |
|---|---|---|---|---|---|
| W1: 1 question, short email | 96 | 24 | about 9 GFLOP | about 100 GFLOP | 11x |
| W3: 10 questions, 512 token ticket | 512 | 10 x 40 | about 50 GFLOP | about 4 TFLOP | 80x |
| W5: agent step, 5.3k state, 3 heads of up to 250 options | 5,300 | about 3,000 in chunks | about 480 GFLOP cold, about 110 GFLOP with segment reuse | not runnable (512 max) | n/a |

At batch 1 on a GPU, FLOPs are not the bottleneck. Kernel launches and memory traffic are, and 07 and 08 deal with those. The table shows that the model change alone is worth 10x to 80x in arithmetic, and that the engine must not waste it.

## ModelSpec format

Each checkpoint directory (or single `.kime` file, see 07) has a `kime.json`:

```json
{
  "format": "kime/1",
  "family": "kime-v1",
  "id": "kime-v1-s-en",
  "version": "1.0.0",
  "tokenizer": {"kind": "bpe-bytelevel", "file": "tokenizer.json", "cls": 50281, "sep": 50282, "pad": 50283, "specials": {"opt": 50368, "no": 50369, "yes": 50370, "cls_s": 50371, "cls_q": 50372}},
  "dims": {"d": 512, "heads": 8, "head_dim": 64, "inter": 1344, "vocab": 50373},
  "state_tower": {"layers": 12, "global_every": 3, "window": 128, "rope_theta_global": 160000, "rope_theta_local": 10000, "segment_top_layers": 3, "max_tokens": 32768},
  "question_tower": {"layers": 4, "kv_heads": 2, "max_options_per_chunk": 32, "max_tokens": 4096},
  "gelu": "erf",
  "norm_eps": 1e-5,
  "calibration": "calibration.json",
  "correctness_head": true,
  "languages": ["en"],
  "trained_context": 8192,
  "provenance": {"base": "jhu-clsp/ettin-encoder-68m", "teacher": "kime-t-en-1.0.0", "data": "data-manifest-1.0.0.json"}
}
```

Special tokens added by kime take ids after the base vocabulary. Their embeddings are learned during fine tuning. The compat family uses Laya's own `rl_agent_config.json` and is described by a generated `kime.json` with `"family": "laya"`.
