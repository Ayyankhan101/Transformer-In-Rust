# CodeGen Branch Context

## Your Focus
- Text-to-code generation (CodeGen-350M)
- Inference pipeline, CLI, server
- Model training (GLM blank-infilling)

## Modules You Own
- `src/codegen/` — CodeGen model (config, model, kv_cache, rotary, weights, quantized)
- `src/generation/` — Inference pipeline (codegen_generate, glm_generate, sampling)
- `src/generation/chat.rs` — Conversational code generation
- `src/commands/` — CLI subcommands (chat, complete, repl, info, serve, download, glm-train)
- `src/server.rs` — HTTP inference server (feature-gated: `server`)
- `src/training/` — Training pipeline (config, data, lr_scheduler, train, checkpoint)

## Project Structure
- Language: Rust 2021, Candle 0.8 (CPU-only)
- Build: `cargo build --release`
- Test: `cargo test`
- Bench: `cargo bench`
- Server: `cargo build --features server && cargo run --features server -- --serve`

## Sync Rule
**Run this before starting work:**
```bash
git fetch origin
git merge origin/image-gen --no-edit
```

This ensures you have the latest SVG/image work from the image-gen branch.

## Do NOT Modify
- `src/svg/` — Partner's domain (SVG generation)
- `src/image/` — Partner's domain (image generation)

## Completed Work
- Phase 1-6: All features implemented (see todo.md)
- CI: All 4 workflows green on develop
- 0 warnings, clean clippy, all tests pass

## Key Architecture
- CodeGenModel: embedding, blocks (norm1, attn, norm2, ffn), final_norm, lm_head, lm_head_bias, rotary
- TrainableGLMModel: embedding, pos_1, pos_2, per-layer (norm1, attn, norm2, mlp), final_norm, lm_head
- QuantizedLinear: per-channel INT8 with offset (~4x compression)

## New Work (SVG Generation - NOT YOUR RESPONSIBILITY)
The partner will add:
- SVG model module (`src/svg/`)
- SVG validator and renderer
- SVG CLI subcommand
- Integration with Norod78/gpt-fluentui-flat-svg model (~195M params)
