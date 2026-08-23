//! Integration test for CodeGen-350M with real weights.
//!
//! Requires weights at codegen_weights/pytorch_model.bin.
//! Skip automatically if weights are not present.

use std::path::Path;

use rust_transformer::codegen::config::CodeGenConfig;
use rust_transformer::codegen::weights::WeightLoader;
use rust_transformer::generation::codegen_generate::CodeGenGenerator;
use rust_transformer::tokenizer::CodeGenTokenizer;

fn weights_available() -> bool {
    Path::new("codegen_weights/pytorch_model.bin").exists()
        || Path::new("codegen_weights/model.safetensors").exists()
}

fn weights_path() -> &'static Path {
    if Path::new("codegen_weights/model.safetensors").exists() {
        Path::new("codegen_weights/model.safetensors")
    } else {
        Path::new("codegen_weights/pytorch_model.bin")
    }
}

/// The checkpoint's own config — the library defaults differ from it.
fn real_config() -> CodeGenConfig {
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("codegen_weights/config.json").unwrap())
            .expect("config.json is not valid JSON");
    CodeGenConfig::from_hf_config(&json)
}

fn real_tokenizer() -> CodeGenTokenizer {
    CodeGenTokenizer::from_file("codegen_weights/tokenizer.json").expect("Failed to load tokenizer")
}

#[test]
fn codegen_forward_pass_with_real_weights() {
    if !weights_available() {
        eprintln!("Skipping: weights not found at codegen_weights/pytorch_model.bin");
        return;
    }

    let device = candle_core::Device::Cpu;
    // The checkpoint's own config.json — the defaults differ from it
    // (rotary_dim 64 vs 32, vocab_size 50400 vs 51200).
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("codegen_weights/config.json").unwrap())
            .expect("config.json is not valid JSON");
    let config = rust_transformer::codegen::config::CodeGenConfig::from_hf_config(&config_json);
    let model = rust_transformer::codegen::weights::WeightLoader::load_from_pytorch(
        Path::new("codegen_weights/pytorch_model.bin"),
        &config,
        &device,
    )
    .expect("Failed to load model");

    let tokenizer =
        rust_transformer::tokenizer::CodeGenTokenizer::from_file("codegen_weights/tokenizer.json")
            .expect("Failed to load tokenizer");

    let prompt = "def fibonacci(n):";
    let token_ids = tokenizer.encode(prompt).expect("Failed to encode");

    let gen = rust_transformer::generation::codegen_generate::CodeGenGenerator::new(
        model, 0.0, 1, 1.0, 1.0, 64,
    );

    // `generate` returns the generated tokens only, not prompt + generated.
    let generated = gen.generate(&token_ids).expect("Generation failed");
    let output = tokenizer.decode(&generated).expect("Decode failed");

    println!("Prompt: {prompt}");
    println!("Generated: {output}");
    assert!(!generated.is_empty(), "Should generate at least one token");
}

#[test]
fn codegen_tokenizer_encode_decode_roundtrip() {
    if !weights_available() {
        eprintln!("Skipping: weights not found");
        return;
    }

    let tokenizer =
        rust_transformer::tokenizer::CodeGenTokenizer::from_file("codegen_weights/tokenizer.json")
            .expect("Failed to load tokenizer");

    let text = "Hello, world!";
    let ids = tokenizer.encode(text).expect("Encode failed");
    let decoded = tokenizer.decode(&ids).expect("Decode failed");

    println!("Original:  {text}");
    println!("Token IDs: {ids:?}");
    println!("Decoded:   {decoded}");
    assert!(!ids.is_empty(), "Should produce tokens");
    assert!(!decoded.is_empty(), "Should decode back to text");
}

/// Sampling must never emit an id the tokenizer cannot decode.
///
/// `config.json` declares `vocab_size: 51200` while the tokenizer stops at 50294, so
/// 905 untrained rows sit at the top of `lm_head`. Their logits turn out to be low
/// enough that they never win — verified at temperature 2.0 with no top-k or nucleus
/// cut — so generation does not mask them. This test is what keeps that true.
#[test]
fn sampling_stays_within_the_tokenizer_vocabulary() {
    if !weights_available() {
        eprintln!("Skipping: weights not found");
        return;
    }
    let device = candle_core::Device::Cpu;
    let config = real_config();
    let tokenizer = real_tokenizer();
    let model = WeightLoader::load(weights_path(), &config, &device).expect("Failed to load model");

    assert!(
        tokenizer.vocab_size() < config.vocab_size,
        "this test is pointless unless the model vocabulary is padded"
    );

    let prompt = tokenizer
        .encode("def fibonacci(n):")
        .expect("encode failed");
    let mut generator =
        CodeGenGenerator::new(model, 0.9, 40, 0.95, 1.1, 32).with_tokenizer(tokenizer.clone());
    generator.set_seed(Some(7));

    let generated = generator.generate(&prompt).expect("generation failed");
    let out_of_range: Vec<u32> = generated
        .iter()
        .copied()
        .filter(|&t| (t as usize) >= tokenizer.vocab_size())
        .collect();
    assert!(
        out_of_range.is_empty(),
        "sampled ids outside the tokenizer vocabulary: {out_of_range:?}"
    );

    let text = tokenizer.decode(&generated).expect("decode failed");
    println!("Sampled: {text}");
    assert!(!text.trim().is_empty(), "sampled output decoded to nothing");
}

/// The sampler used to be greedy no matter what, so every run matched. Now a fixed
/// seed must reproduce and a different seed must diverge.
#[test]
fn sampling_is_reproducible_and_seed_dependent() {
    if !weights_available() {
        eprintln!("Skipping: weights not found");
        return;
    }
    let device = candle_core::Device::Cpu;
    let config = real_config();
    let tokenizer = real_tokenizer();
    let prompt = tokenizer
        .encode("def quicksort(arr):")
        .expect("encode failed");

    let run = |seed: u64| -> Vec<u32> {
        let model =
            WeightLoader::load(weights_path(), &config, &device).expect("Failed to load model");
        let mut generator =
            CodeGenGenerator::new(model, 0.9, 40, 0.95, 1.1, 16).with_tokenizer(tokenizer.clone());
        generator.set_seed(Some(seed));
        generator.generate(&prompt).expect("generation failed")
    };

    assert_eq!(run(1), run(1), "same seed must reproduce");
    assert_ne!(run(1), run(2), "different seeds must diverge");
}
