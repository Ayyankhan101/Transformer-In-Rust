//! Integration test for CodeGen-350M with real weights.
//!
//! Requires weights at codegen_weights/pytorch_model.bin.
//! Skip automatically if weights are not present.

use std::path::Path;

fn weights_available() -> bool {
    Path::new("codegen_weights/pytorch_model.bin").exists()
}

#[test]
fn codegen_forward_pass_with_real_weights() {
    if !weights_available() {
        eprintln!("Skipping: weights not found at codegen_weights/pytorch_model.bin");
        return;
    }

    let device = candle_core::Device::Cpu;
    let config = rust_transformer::codegen::config::CodeGenConfig::default();
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

    let generated = gen.generate(&token_ids).expect("Generation failed");
    let output = tokenizer.decode(&generated).expect("Decode failed");

    println!("Prompt: {prompt}");
    println!("Generated: {output}");
    assert!(
        generated.len() > token_ids.len(),
        "Should generate at least one token"
    );
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
