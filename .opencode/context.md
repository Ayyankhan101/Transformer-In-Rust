# Project Context

## Environment
- Language: Rust 2021
- Runtime: Candle 0.8 (CPU-only, no GPU)
- Build: cargo build --release
- Test: cargo test (20 tests pass, 2 integration tests skip without weights)
- Bench: cargo bench (Criterion 0.5)
- Package Manager: Cargo

## Project Type
- [x] Application (CLI)
- [x] Library/Package (lib+bin crate since this session)
- Training playground for transformer architectures

## Infrastructure
- Container: None
- CI: .github/workflows/bench.yml
- Benchmarks: Criterion benches in benches/transformer.rs
- Server: Feature-gated HTTP inference server (cargo build --features server)

## Structure
- `src/lib.rs` — Library crate root (exports all modules as pub)
- `src/main.rs` — Binary crate (CLI entry point)
- `src/layers/` — Shared layer library (attention, ffn, norm, embedding, block)
- `src/glm/` — GLM blank-infilling model (config, model, trainable, positions, attention_mask)
- `src/codegen/` — CodeGen-350M model (config, model, kv_cache, rotary, weights, quantized)
- `src/generation/` — Generation pipeline (codegen_generate, glm_generate, sampling)
- `src/training/` — Training pipeline (config, data, lr_scheduler, train, checkpoint)
- `src/server.rs` — HTTP inference server (feature-gated: `server`)
- `tests/` — Integration tests (codegen_integration.rs)
- `examples/` — convert_codegen_to_safetensors.rs
- `benches/` — transformer.rs (criterion benchmarks)
- `docs/` — Tutorials and architecture documentation
- `configs/` — train.yaml example config

## Conventions
- Naming: snake_case for modules/functions, CamelCase for types
- Error handling: candle_core::Result / anyhow::Result
- Testing: unit tests in each module (#[cfg(test)]), integration tests in tests/
- State: 55/55 tasks complete

## Completed Work
- Phase 1: FP16 dtype propagation + validation (23% speedup)
- Phase 2: YAML config, DataLoader, LR scheduler, safetensors checkpoints, gradient clipping, eval loop
- Phase 3: Criterion benchmarks (attention, FFN, block, e2e, F16 vs F32)
- Phase 4: GLM safetensors save/load, CodeGen .bin→safetensors converter
- Phase 5: Token streaming, prompt templates, INT8 quantization, batched generation
- Phase 6: Tutorials (train on laptop, run on RPi), architecture deep-dive, framework comparison
- CI: GitHub Actions bench workflow, README auto-update script
- This session: Warning cleanup (26→0), HTTP server (axum), lib+bin crate, --serve/--download-codegen flags, integration tests

## Key Architecture
- TrainableGLMModel: embedding, pos_1, pos_2, per-layer (norm1, attn.qkv, attn.out, norm2, mlp.up, mlp.gate, mlp.down), final_norm, lm_head
- CodeGenModel: embedding, blocks (norm1, attn, norm2, ffn), final_norm, lm_head, lm_head_bias, rotary
- Dtype propagation: causal_mask(..., dtype), KVCache::new(..., dtype), RotaryEmbedding::new(..., dtype)
- QuantizedLinear: per-channel INT8 with offset (u8 + 128 offset), ~4x compression for large weights

## Server (feature-gated)
- Build: `cargo build --features server`
- Run: `cargo run --features server -- --serve [port]`
- Endpoints: GET /health, POST /generate (JSON: prompt, max_tokens, temperature, top_k, top_p)
- Uses Arc<Mutex<CodeGenGenerator>> for shared model access

## CodeGenGenerator Setters
- set_temperature(&mut self, f64), set_top_k(&mut self, usize), set_top_p(&mut self, f64)
- set_max_new_tokens(&mut self, usize), set_repetition_penalty(&mut self, f64)
