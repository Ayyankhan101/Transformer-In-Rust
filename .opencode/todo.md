# Mission: Pure-Rust Transformer Playground Enhancement

## M1: Status Assessment | status: completed
### T1.1: Explore project | agent:Worker
- [x] S1.1.1: Check project structure | size:S
- [x] S1.1.2: Run cargo check | size:S
- [x] S1.1.3: Run cargo test | size:S

### T1.2: Fix broken examples | agent:Worker
- [x] S1.2.1: Fix test_f16_names.rs (pth.keys() → tensor_infos().keys()) | size:S
- [x] S1.2.2: Fix test_f16_weights.rs warnings (unused import, unused vars) | size:S
- [x] S1.2.3: Verify clean compile and 16/16 tests pass | size:S

## M2: Phase 1 - FP16 End-to-End Validation | status: completed
### T2.1: Fix dtype propagation | agent:Worker
- [x] S2.1.1: causal_mask() accepts DType parameter | size:S
- [x] S2.1.2: KVCache::new() accepts DType parameter | size:S
- [x] S2.1.3: RotaryEmbedding uses config.dtype | size:S
- [x] S2.1.4: Update all call sites (CodeGen, GLM train, GLM generate) | size:S
### T2.2: Validate FP16 inference | agent:Worker
- [x] S2.2.1: cargo run --release -- --codegen --f16 works | size:S
- [x] S2.2.2: cargo run --release -- --repl --f16 works | size:S
- [x] S2.2.3: All 17 tests pass | size:S
- [x] S2.2.4: Benchmark FP32 vs FP16 (0.4s vs 0.5s/token) | size:S

## M3: Phase 2 - Training Pipeline Hardening | status: completed
### T3.1: Config system | agent:Worker
- [x] S3.1.1: TrainConfig YAML schema (configs/train.yaml) | size:S
- [x] S3.1.2: TrainingConfig, ModelConfig, LrScheduleConfig structs | size:S
- [x] S3.1.3: from_file() deserialization, to_glm_config() conversion | size:S
### T3.2: Data pipeline | agent:Worker
- [x] S3.2.1: DataLoader with train/eval split, shuffling, random windowing | size:M
- [x] S3.2.2: download_default_data() for CPython functools.py | size:S
### T3.3: LR scheduler | agent:Worker
- [x] S3.3.1: Cosine decay with linear warmup | size:S
- [x] S3.3.2: Linear decay, constant fallback | size:S
### T3.4: Safetensors checkpoints | agent:Worker
- [x] S3.4.1: save_checkpoint() model + optimizer state + training state | size:M
- [x] S3.4.2: load_checkpoint() resume from latest | size:M
- [x] S3.4.3: Cleanup old checkpoints (keep_last_n) | size:S
### T3.5: Gradient accumulation + clipping | agent:Worker
- [x] S3.5.1: Configurable gradient_accumulation_steps | size:S
- [x] S3.5.2: Gradient clipping by global norm | size:S
### T3.6: Evaluation loop | agent:Worker
- [x] S3.6.1: Periodic eval on validation set | size:S
- [x] S3.6.2: Reproducible eval with fixed seed | size:S
### T3.7: CLI integration | agent:Worker
- [x] S3.7.1: --config flag for YAML config | size:S
- [x] S3.7.2: --glm-train uses new pipeline | size:S
### T3.8: Fix train/eval split (need more data) | agent:Worker
- [x] S3.8.1: Add more sample data or adjust split for small datasets | size:S
- [x] S3.8.2: Test training run completes 100+ steps | size:M

## M4: Phase 3 - Benchmarking & Perf Tracking | status: completed
### T4.1: Criterion benches
- [x] S4.1.1: Micro-benches (attention, FFN, block, full forward)
- [x] S4.1.2: E2E bench (prefill + N tokens)
- [x] S4.1.3: FP32 vs FP16 vs INT8 comparison
### T4.2: CI integration
- [x] S4.2.1: .github/workflows/bench.yml
- [x] S4.2.2: README auto-update script

## M5: Phase 4 - Model Serialization | status: completed
### T5.1: GLM safetensors save/load
- [x] S5.1.1: Save trained GLM to safetensors
- [x] S5.1.2: Load for inference/resume
### T5.2: CodeGen converter
- [x] S5.2.1: PyTorch .bin → safetensors converter

## M6: Phase 5 - CodeGen Enhancements | status: completed
- [x] S6.1: Token streaming callback
- [x] S6.2: Prompt templates (instruct, completion, chat)
- [x] S6.3: INT8 dynamic quantization
- [x] S6.4: Batched generation

## M7: Phase 6 - Documentation & Showcase | status: completed
- [x] S7.1: Tutorial: "Train code infill model on laptop"
- [x] S7.2: Tutorial: "Run CodeGen-350M on Raspberry Pi"
- [x] S7.3: Architecture deep-dive doc
- [x] S7.4: Comparison table vs candle-examples, burn, tch-rs
