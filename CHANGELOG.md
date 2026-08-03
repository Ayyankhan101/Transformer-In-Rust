# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT License file
- Crate-level rustdoc for `lib.rs`
- `rust-version = "1.75.0"` MSRV in `Cargo.toml`
- `tokenizer_path` field in `TrainingConfig` (configurable, defaults to `codegen_weights/tokenizer.json`)
- `num_layers` field stored in `TrainableGLMModel` for param enumeration
- `Default` impl for `TrainConfig` (outer config struct)
- 69 unit tests covering config, data, layers, GLM, codegen, sampling, training, and generation
- 2 integration tests for CodeGen weight loading and tokenizer roundtrip
- CI workflows for benchmarks, security audit, and code coverage
- Branch triggers for `codegen` on all CI workflows

### Fixed
- `TrainingConfig::to_glm_config()` now uses `self` instead of hardcoded defaults
- Hardcoded `0..6` layer loops replaced with `self.config.num_layers` / `model.num_layers`
- `GLMTrainer::from_config` now accepts `TrainConfig` (outer struct) so `to_glm_config()` resolves correctly
- Hardcoded tokenizer path now configurable via `TrainingConfig::tokenizer_path`
- Removed unused `_dtype_str` variable in `model.rs`

### Changed
- Removed blanket `#![allow(dead_code)]` from `lib.rs` and `main.rs`
- `cli` and `model` modules now re-exported from `lib.rs`
- README updated with 69 tests, full CLI reference, deps table, mermaid project structure
- Docs updated: `run-codegen-on-raspberry-pi.md`, `train-code-infill.md` (correct CLI commands)

## [0.1.0] - 2024-01-01

### Added
- Initial release
- GLM (General Language Model) architecture for blank-infilling
- CodeGen-350M architecture for code generation
- INT8 dynamic quantization
- HTTP server (feature-gated)
- Training pipeline with YAML config
- CLI with chat, complete, repl, info, serve, download, glm-train commands
