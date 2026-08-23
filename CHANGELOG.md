# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — found while validating against the real 350M checkpoint
- **`codegen download` reported success without downloading anything.**
  `huggingface-cli` has been renamed to `hf`; the old name now prints a deprecation
  notice, downloads nothing, and exits 0. The command believed the exit code and
  printed "✓ Download complete!" over an empty directory. It now tries `hf` first and
  verifies the file exists rather than trusting the status.
- **`complete` printed nothing in its default streaming mode.**
  `CodeGenGenerator::with_tokenizer` existed but was never called, so `decode_token`
  returned an empty string for every token.
- **`chat` sliced off the wrong tokens** — the same generated-only slicing bug fixed in
  `complete`, `repl` and the server last pass, missed in this fourth call site.
- **`-t` was bound to both `--temperature` and `--template`**, and `--stream` was a flag
  with `default_value = "true"`, so it could never be false and the non-streaming branch
  was unreachable. `--template` loses its short flag; `--stream` becomes `--no-stream`.
- **`info` printed hardcoded model values** that had drifted from the checkpoint
  (`max_seq_len` 1024 against the real 2048), while parsing the real config a few lines
  later, and only looked for `pytorch_model.bin` so it missed a converted safetensors file.

### Fixed — training
- **`glm-train` did not train.** Both cross-entropy paths accumulated into an `f32` and
  returned `Tensor::new(total, device)` — a fresh leaf with no autograd history — so
  `optimizer.backward_step(&loss)` received an empty gradient store and every step was a
  no-op. Rebuilt from tensor operations around `candle_nn::loss::cross_entropy`. A real
  80-step run now goes from loss 10.80 to 4.99; before the fix, 60 optimizer steps left
  the loss at 4.1934047 both before and after.
- **Every checkpoint save failed.** `save_safetensors` called `tensor.to_vec1::<u8>()`,
  which errors on anything of rank > 1, so training ended in
  "unexpected rank, expected: 1, got: 2" as soon as it reached a weight matrix.
- **`configs/train.yaml` did not deserialize.** `eval_steps` sat under a separate
  `evaluation:` section and `tokenizer_path` was absent. Nothing had ever loaded the file
  because `glm-train` had no `--config` flag. Config structs now use `#[serde(default)]`,
  so a partial YAML falls back to defaults field by field.
- `gradient_accumulation_steps` and `max_grad_norm` were declared in `TrainingConfig` and
  claimed here as delivered, but read nowhere. Both are now implemented:
  `gradient_accumulation_steps` micro-batch losses are averaged into one backward, and
  `clip_grad_norm` scales gradients to the global-norm bound.
- The learning rate was set *after* `optimizer.step`, so every step ran on the previous
  step's rate.
- Masking used a hand-rolled LCG re-seeded from the step counter, so consecutive steps drew
  near-identical corruption patterns. Replaced with one `StdRng` seeded from
  `TrainingConfig::seed`.

### Added — training
- `--config <path>` on `glm-train`, using the existing `TrainConfig::from_file`. It was
  marked done in the project notes but had never been wired to the CLI.
- `src/training/loss.rs` with tests that fail if gradients stop reaching parameters, if a
  tiny model stops memorising a batch, or if an optimizer step leaves weights unchanged
- Gradient-clipping tests, and a test that `configs/train.yaml` actually parses

### Removed — training
- `src/training/checkpoint.rs` (192 lines): never declared in `mod.rs`, so it had never
  been compiled, and it duplicated the checkpoint code in `train.rs`
- Per-tensor `println!` debug output from `TrainableGLMModel::save_safetensors`
- Three copies of the sequence-corruption logic and two of the tensor-to-safetensors
  conversion, collapsed into one each

### Performance
- **KV cache no longer rewrites itself on every token.** `KVCache::append` used
  `Tensor::slice_assign`, which is not a targeted write: candle zero-pads the source out
  to the full destination shape, allocates a full-size mask, pads that too, and runs
  `where_cond` over every element — the entire `[1, heads, max_seq_len, head_dim]` buffer,
  for `k` and `v`, in every layer, for every token. Replaced with `Tensor::slice_set`,
  which copies only the new tokens in place. On the benchmark model: prefill
  88.5 ms → 36.2 ms, and decode 58.4 ms → 5.6 ms per token, a 10.4× improvement.
- **`lm_head` no longer runs over the whole prompt.** Generation reads only the final
  position but paid the `hidden_dim × vocab_size` projection for every prefill token —
  6.0 ms of a 35.8 ms prefill. `forward_hidden` and `project_logits` are now separate, and
  generation narrows before projecting.
- `Embedding::forward` uses one `index_select` instead of a `get` plus `stack` per token.
  No measurable difference at benchmark sizes; it is simply less code.
- Removed the per-token `eprintln!` timing lines from `generate_stream` — a stderr flush
  per token inside the generation loop.

### Fixed
- **FP16 models built by `CodeGenModel::new_blank` panicked on the first forward.**
  `MultiHeadAttention::new_blank`, the FFN constructors and `Embedding::zeros` hardcoded
  F32 regardless of `config.dtype`, giving `dtype mismatch in mul, lhs: F32, rhs: F16`.
  Loading real weights masked this, because every tensor was replaced on the way in.
- **`cargo clippy --all-targets --all-features -- -D warnings` failed on every commit.**
  `main.rs` re-declared the whole module tree instead of using the library crate, so the
  binary compiled a second copy and every `pub` item the CLI did not call was reported as
  dead code — 34 warnings from one cause. The CI lint gate now passes.
- `KVCache::append` reports a clear error when a prompt exceeds `max_seq_len` instead of
  failing inside candle.
- **CodeGen produced incorrect output with real weights.** The fused `qkv_proj`
  tensor was split as `[all q | all v | all k]`, but CodeGen stores it as four
  interleaved model-parallel groups of `[q | v | k]` (`mp_num = 4` upstream).
  Every attention head was reading the wrong slice of the projection.
- **Sampling was always greedy.** The hand-rolled LCG in `sampling::sample`
  divided a `u64` state by `u32::MAX`, producing values around 6e9, so the
  selection loop always fell through to argmax — `temperature`, `top_k` and
  `top_p` had no effect. Replaced with a `StdRng` seeded once per generation.
- **`complete`, `repl` and `/generate` sliced off the wrong tokens.**
  `CodeGenGenerator::generate` returns generated tokens only, but all three
  callers stripped `prompt.len()` tokens from the front, dropping output or
  panicking when the prompt was longer than the completion.
- **The HTTP server ignored `config.json` and `--f16`**, loading with
  `CodeGenConfig::default()` (`rotary_dim` 64 rather than the checkpoint's 32).
  It now shares `ModelContext` with the CLI.
- **Missing checkpoint tensors were silently ignored**, leaving zeroed layers
  that produce nonsense instead of an error. `WeightLoader` now names the first
  missing tensor and fails.
- `TrainingConfig::to_glm_config()` now uses `self` instead of hardcoded defaults
- Hardcoded `0..6` layer loops replaced with `self.config.num_layers` / `model.num_layers`
- `GLMTrainer::from_config` now accepts `TrainConfig` (outer struct) so `to_glm_config()` resolves correctly
- Hardcoded tokenizer path now configurable via `TrainingConfig::tokenizer_path`
- Removed unused `_dtype_str` variable in `model.rs`

### Added
- `CodeGenTokenizer::vocab_size`, reported by `info`. CodeGen pads `vocab_size` to 51200
  while the tokenizer stops at 50294; the 905 untrained rows never win a sample, verified
  at temperature 2.0 with no top-k or nucleus cut, and an integration test keeps that true
- Integration tests that run against the real checkpoint: sampling stays inside the
  tokenizer's vocabulary, and a fixed seed reproduces while a different seed diverges
- `WeightLoader::load_from_safetensors` and `WeightLoader::load` (format picked by
  extension). `examples/convert_codegen_to_safetensors.rs` wrote a file nothing in the repo
  could read; `ModelContext` now prefers `model.safetensors` over `pytorch_model.bin`
- Round-trip test covering the converter's output
- `KVCache` test asserting token-by-token appends match a single block append
- `--seed` global flag for reproducible sampling
- Numerical parity tests (`tests/codegen_parity.rs`) against the HuggingFace
  CodeGen reference, using committed tiny-model fixtures generated by
  `scripts/gen_parity_fixture.py` — no weight download needed
- MIT License file
- Crate-level rustdoc for `lib.rs`
- `rust-version = "1.75.0"` MSRV in `Cargo.toml`
- `tokenizer_path` field in `TrainingConfig` (configurable, defaults to `codegen_weights/tokenizer.json`)
- `num_layers` field stored in `TrainableGLMModel` for param enumeration
- `Default` impl for `TrainConfig` (outer config struct)
- Unit tests covering config, data, layers, GLM, codegen, sampling, training, and generation
- 2 integration tests for CodeGen weight loading and tokenizer roundtrip
- CI workflows for benchmarks, security audit, and code coverage
- Branch triggers for `codegen` on all CI workflows

### Changed
- **`benches/transformer.rs` now benchmarks this crate.** Every benchmark was a hand-inlined
  copy of the model, including a QKV split carrying the bug the real model no longer has, so
  the suite could report neither a speedup nor a regression. Replaced with benchmarks that
  call the public API: prefill, prefill without the vocabulary projection, forward passes,
  the full generator, weight loading, and F32 against F16.
- Blank constructors (`Embedding::zeros`, `MultiHeadAttention::new_blank`, the FFN
  variants) take an explicit `DType`
- `sampling::sample` takes an RNG instead of a `u64` seed, so one stream spans a
  whole generation
- `server::start_server` takes the weights directory and `--f16` flag rather than
  a `.bin` path
- Removed blanket `#![allow(dead_code)]` from `lib.rs` and `main.rs`
- `cli` and `model` modules now re-exported from `lib.rs`
- README updated with full CLI reference, deps table, mermaid project structure
- Docs updated: `run-codegen-on-raspberry-pi.md`, `train-code-infill.md` (correct CLI commands)

### Removed
- **`src/codegen/quantized.rs`.** Never constructed outside its own tests, and
  `QuantizedLinear::forward` dequantized the whole weight matrix on every call, making it
  strictly slower than the F32 path it would have replaced. A real speedup needs int8
  matmul kernels, which candle does not expose. The README and docs no longer claim it.
- `CodeGenBlock::norm2`, unused: CodeGen blocks are parallel and have one norm

## [0.1.0] - 2024-01-01

### Added
- Initial release
- GLM (General Language Model) architecture for blank-infilling
- CodeGen-350M architecture for code generation
- INT8 dynamic quantization
- HTTP server (feature-gated)
- Training pipeline with YAML config
- CLI with chat, complete, repl, info, serve, download, glm-train commands
