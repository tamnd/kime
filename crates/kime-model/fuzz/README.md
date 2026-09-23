# kime-model fuzz targets

Three targets over the checkpoint parsers, which read untrusted files (SECURITY.md):

- `safetensors`: raw bytes as a safetensors file.
- `kime_file`: raw bytes as a `.kime` file, which mostly reaches the fixed header block.
- `kime_index`: a well formed header block with a matching hash around fuzzer chosen index JSON and data, so every input reaches the index parser.

Each target reads every byte of every tensor the parser accepts, so an offset that slipped past the checks is a crash and not a silent pass.

```
cargo +nightly fuzz run safetensors -- -max_total_time=3600
```

Seeds come from `cargo run -p kime-model --example fuzz_seeds -- fuzz/corpus`, which writes small valid files of both formats.
