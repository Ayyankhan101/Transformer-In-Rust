# Image-Gen Branch Context

## Your Focus
- Text-to-SVG generation (GPT-2 ~195M params)
- SVG validation and optional PNG rendering
- Image generation pipeline

## Model
- Base model: `Norod78/gpt-fluentui-flat-svg`
- Architecture: GPT-2 (GPT2LMHeadModel)
- Size: ~195M parameters
- License: MIT
- Format: Safetensors + tokenizer.json available
- HuggingFace: https://huggingface.co/Norod78/gpt-fluentui-flat-svg

## Modules You Own
- `src/svg/` (new) — SVG generation module
  - `mod.rs` — Module root
  - `config.rs` — SVG model configuration
  - `model.rs` — GPT-2 model wrapper for SVG inference
  - `generate.rs` — Text-to-SVG generation pipeline
  - `validate.rs` — XML validation, well-formedness check
  - `render.rs` (optional) — SVG-to-PNG via resvg crate
- `src/image/` (new) — Future image generation

## Project Structure
- Language: Rust 2021, Candle 0.8 (CPU-only)
- Build: `cargo build --release`
- Test: `cargo test`
- Existing CLI: `src/cli.rs` with clap 4 derive

## Sync Rule
**Run this before starting work:**
```bash
git fetch origin
git merge origin/codegen --no-edit
```

This ensures you have the latest CodeGen work from the codegen branch.

## Do NOT Modify
- `src/codegen/` — Partner's domain (CodeGen model)
- `src/generation/` — Partner's domain (inference pipeline)
- `src/commands/` — Partner's domain (CLI subcommands)
- `src/server.rs` — Partner's domain (HTTP server)
- `src/training/` — Partner's domain (training pipeline)

## Getting Started
1. Clone repo: `git clone <repo-url>`
2. Check out: `git checkout image-gen`
3. Open opencode — it reads this file automatically
4. Before starting work: run sync script to merge codegen changes

## Existing Code to Understand
- `src/layers/` — Shared layer library (attention, ffn, norm, embedding, block)
- `src/tokenizer.rs` — Tokenizer wrapper
- `src/lib.rs` — Library crate root (exports all modules as pub)
- `src/main.rs` — Binary crate (CLI entry point)

## Key Architecture
- Shared layers: `src/layers/` has attention, ffn, norm, embedding, block
- Use candle-core for tensor ops (CPU-only target)
- Tokenizers crate for BPE tokenization (GPT-2 BPE)
- Safetensors for weight loading

## Tasks
1. Download Norod78/gpt-flentui-flat-svg weights (safetensors)
2. Create `src/svg/` module with config, model, generate, validate
3. Implement SVG tokenizer (handle XML tags, attributes, coordinates)
4. Implement SVG model wrapper (GPT-2 inference for SVG generation)
5. Implement SVG validator (XML validation, well-formedness check)
6. Add resvg dependency for optional SVG-to-PNG rendering
7. Create SVG CLI subcommand (text-to-SVG)
8. Add integration tests
