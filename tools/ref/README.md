# Reference dumps

These scripts produce the numbers kime is checked against. They need PyTorch and the real `laya` package, so they are run by hand and their output is committed, and CI never installs either.

## What is here

`make_cases.py` writes the 200 parity cases to `crates/kime-eval/fixtures/parity/cases.jsonl` from a fixed seed. The cases cover 22 languages across 11 scripts, empty and whitespace states, conversations, the `[MASK]` and `<mask>` literals inside text, criteria that are numbers, booleans, objects and lists, one to 40 options, score questions with 2 to 32 levels, JSON states with Unicode keys, states long enough that both models truncate them, and requests with 10 and 50 questions.

`laya_ref.py` runs every case through Laya's own `Agent.system_one` and writes one line per case to `crates/kime-eval/fixtures/parity/<model>.jsonl`. Each question records the strings Laya tokenizes and their ids before truncation, the final ids and marker positions, the option logits and act probabilities taken from a hook on the model's forward, and the whole answer Laya returned.

`laya_tensors.py` builds Laya's `DecisionModel` from each checkpoint's own configs, the way `laya.Agent` does, and writes every parameter name and shape to `crates/kime-model/tests/fixtures/laya-tensors.json`, along with each tensor's dtype, sum and first four values as the safetensors library reads them. The loader test in `kime-model` checks its graph and its f16 reading against that file. The script also times `laya.Agent` loading on the CPU, best of three: 0.27 seconds for English and 1.57 seconds for multilingual on the i9-13900K.

## Regenerating

```
uv venv -p 3.12 venv
VIRTUAL_ENV=venv uv pip install laya==0.3.7 torch transformers safetensors tokenizers numpy
python tools/ref/make_cases.py
python tools/ref/laya_ref.py --model <dir>/laya --name laya --device cpu
python tools/ref/laya_ref.py --model <dir>/laya/multilingual --name laya-multilingual --device cpu
python tools/ref/laya_tensors.py <dir>/laya crates/kime-model/tests/fixtures/laya-tensors.json
```

`<dir>/laya` is a download of `convaiinnovations/laya` from Hugging Face. The English model is the repository root and the multilingual one is its `multilingual` folder.

The committed dumps were made with laya 0.3.7, torch 2.14.0, transformers 5.17.0 and tokenizers 0.23.2 on the CPU in FP32, on an i9-13900K. The English run takes 91 seconds for 625 questions with 32 threads, and the multilingual one 44 seconds. The weight files hash to `891102d372688fc2` (English) and `9d628fd971b70038` (multilingual), the first 16 hex digits of their SHA-256.

`--dump-hidden DIR` also writes the output of the embeddings, every encoder layer, the final norm and both head layers for every question, one `.npz` per question. They are too big to commit and are only for finding where a mismatch starts.

## How far Laya's own GPU path is from these numbers

Running the same cases with `--device cuda` on an RTX 4090 uses Laya's bf16 autocast. Against the CPU FP32 dumps:

| Model | Questions | Argmax agreement | Max probability error | Max logit error |
|---|---|---|---|---|
| laya | 625 | 622 (99.5%) | 0.084 | 1.52 |
| laya-multilingual | 625 | 623 (99.7%) | 0.068 | 0.41 |

So Laya on a GPU does not agree with Laya on a CPU for 5 of 1,250 questions. The tolerance in `spec/15-testing.md` for kime's FP16 backends is a probability error of 6e-3 with 100% argmax agreement, which is more than ten times tighter than that.

## Laya's speed and energy on a GPU

`laya_gpu.py <model dir> <fixture> [batch] [on|off]` runs every question of a parity fixture through Laya's own model on the first NVIDIA GPU, once one at a time and once in batches, the way `Agent.system_one` does, and prints the latency, the throughput and the energy per decision from NVML's total energy counter. `cargo run --release -p kime-cuda --example cuda_bench -- <dir>/laya <fixture> f16 16` prints the same numbers for kime, so the two can run back to back on one machine. The idle line is the board's draw in the five seconds after the timed runs, with the clocks still up, which is higher than a board that has been idle for a while (about 20 W on the 4090 against 56 to 64 W right after a run).

## Tokenizer corpus

`tok_corpus.py` writes the MASSIVE train utterances in all 51 languages plus 200,000 fuzz strings to a JSON lines file, with the ids Hugging Face tokenizers gives for each line under both Laya tokenizers, and prints how fast Hugging Face was on one thread. The parquet files are cached in `--cache`, and a cache that already has all 51 is used without touching the network. `cargo run --release -p kime-tok --example corpus -- <models>/laya tok-corpus.jsonl` then checks kime against every line and times it the same way.
