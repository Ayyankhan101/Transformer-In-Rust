//! Benchmarks for the CodeGen inference path.
//!
//! These call the library directly. An earlier version of this file re-implemented
//! the model inline, which meant it measured a copy of the code rather than the
//! code — including a QKV split that had a bug the real model no longer has.

use std::path::Path;

use candle_core::{DType, Device};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use rust_transformer::codegen::config::CodeGenConfig;
use rust_transformer::codegen::kv_cache::KVCache;
use rust_transformer::codegen::model::CodeGenModel;
use rust_transformer::codegen::weights::WeightLoader;
use rust_transformer::generation::codegen_generate::CodeGenGenerator;

/// A stand-in for CodeGen-350M: small enough to build quickly, but with the real
/// `max_seq_len`. KV-cache cost scales with the cache buffer, not with parameter
/// count, so shrinking `max_seq_len` would hide exactly what these measure.
fn bench_config() -> CodeGenConfig {
    CodeGenConfig {
        vocab_size: 16384,
        hidden_dim: 512,
        num_layers: 12,
        num_heads: 8,
        ffn_dim: 2048,
        max_seq_len: 2048,
        rotary_dim: 32,
        ..Default::default()
    }
}

const PROMPT_LEN: usize = 32;
const NEW_TOKENS: usize = 16;

fn prompt_tokens(config: &CodeGenConfig) -> Vec<u32> {
    (0..PROMPT_LEN)
        .map(|i| (i * 7 % config.vocab_size) as u32)
        .collect()
}

/// Prefill: one forward pass over the whole prompt, empty cache.
fn bench_prefill(c: &mut Criterion) {
    let device = Device::Cpu;
    let config = bench_config();
    let model = CodeGenModel::new_blank(config.clone(), &device).unwrap();
    let tokens = prompt_tokens(&config);
    let positions: Vec<usize> = (0..tokens.len()).collect();

    let mut group = c.benchmark_group("prefill");
    group.throughput(Throughput::Elements(PROMPT_LEN as u64));
    group.bench_function("32_tokens", |b| {
        b.iter(|| {
            let mut cache: Option<Vec<KVCache>> = None;
            model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap()
        });
    });
    // Same work minus the vocabulary projection. The gap is what generation
    // saves by projecting only the last position.
    group.bench_function("32_tokens_hidden_only", |b| {
        b.iter(|| {
            let mut cache: Option<Vec<KVCache>> = None;
            model
                .forward_hidden(&tokens, &positions, &mut cache)
                .unwrap()
        });
    });
    group.finish();
}

/// The path `codegen complete` actually takes: prefill, decode, sampling.
fn bench_generator(c: &mut Criterion) {
    let device = Device::Cpu;
    let config = bench_config();
    let model = CodeGenModel::new_blank(config.clone(), &device).unwrap();
    let tokens = prompt_tokens(&config);
    let generator = CodeGenGenerator::new(model, 0.8, 40, 0.9, 1.2, NEW_TOKENS);

    let mut group = c.benchmark_group("generator");
    group.throughput(Throughput::Elements(NEW_TOKENS as u64));
    group.bench_function("32_prompt_16_new", |b| {
        b.iter(|| generator.generate(&tokens).unwrap());
    });
    group.finish();
}

/// Prefill plus autoregressive decode — the number that matters for `complete`.
/// Subtract the `prefill` result to isolate per-token decode cost.
fn bench_generate(c: &mut Criterion) {
    let device = Device::Cpu;
    let config = bench_config();
    let model = CodeGenModel::new_blank(config.clone(), &device).unwrap();
    let tokens = prompt_tokens(&config);
    let positions: Vec<usize> = (0..tokens.len()).collect();

    let mut group = c.benchmark_group("generate");
    group.throughput(Throughput::Elements(NEW_TOKENS as u64));
    group.bench_function("32_prompt_16_new", |b| {
        b.iter(|| {
            let mut cache: Option<Vec<KVCache>> = None;
            model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap();
            for step in 0..NEW_TOKENS {
                let pos = tokens.len() + step;
                model
                    .forward_with_cache(&[1u32], &[pos], &mut cache)
                    .unwrap();
            }
        });
    });
    group.finish();
}

/// Weight loading, on the committed parity fixture.
fn bench_weight_load(c: &mut Criterion) {
    let device = Device::Cpu;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let path = dir.join("tiny_h4.pth");
    if !path.exists() {
        eprintln!("skipping weight_load bench: {} not found", path.display());
        return;
    }
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("tiny_h4_config.json")).unwrap())
            .unwrap();
    let config = CodeGenConfig::from_hf_config(&config_json);

    c.bench_function("weight_load/tiny_h4", |b| {
        b.iter(|| WeightLoader::load_from_pytorch(&path, &config, &device).unwrap());
    });
}

/// F32 against F16 on the same shapes, to keep the README's precision claim honest.
fn bench_dtype(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("prefill_dtype");

    for (name, dtype) in [("f32", DType::F32), ("f16", DType::F16)] {
        let config = CodeGenConfig {
            dtype,
            ..bench_config()
        };
        let model = CodeGenModel::new_blank(config.clone(), &device).unwrap();
        let tokens = prompt_tokens(&config);
        let positions: Vec<usize> = (0..tokens.len()).collect();

        group.bench_function(name, |b| {
            b.iter(|| {
                let mut cache: Option<Vec<KVCache>> = None;
                model
                    .forward_with_cache(&tokens, &positions, &mut cache)
                    .unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_generate,
    bench_generator,
    bench_weight_load,
    bench_dtype
);
criterion_main!(benches);
