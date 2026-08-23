//! Model loading utilities for CodeGen-350M.
//!
//! Provides [`ModelContext`] for loading weights, tokenizer, and building
//! the generator in a single call, eliminating boilerplate across commands.

use std::path::Path;

use anyhow::{bail, Result};
use candle_core::Device;

use crate::codegen::config::CodeGenConfig;
use crate::codegen::weights::WeightLoader;
use crate::generation::codegen_generate::CodeGenGenerator;
use crate::tokenizer::CodeGenTokenizer;

/// Shared context for model loading — eliminates boilerplate across commands.
pub struct ModelContext {
    pub generator: CodeGenGenerator,
    pub tokenizer: CodeGenTokenizer,
    pub config: CodeGenConfig,
}

impl ModelContext {
    /// Load CodeGen-350M weights, tokenizer, and build generator.
    pub fn load(weights_dir: &Path, use_f16: bool, temperature: f64) -> Result<Self> {
        let config_path = weights_dir.join("config.json");
        let tokenizer_path = weights_dir.join("tokenizer.json");

        // safetensors loads faster and needs no pickle, so prefer it when present.
        let safetensors_path = weights_dir.join("model.safetensors");
        let pytorch_path = weights_dir.join("pytorch_model.bin");
        let weights_path = if safetensors_path.exists() {
            safetensors_path
        } else if pytorch_path.exists() {
            pytorch_path
        } else {
            bail!(
                "Weights not found in {dir}: expected model.safetensors or pytorch_model.bin\n\
                 Run `codegen download` or:\n  \
                 huggingface-cli download Salesforce/codegen-350M-multi --local-dir {dir}",
                dir = weights_dir.display()
            );
        };

        let mut config = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
            CodeGenConfig::from_hf_config(&config_json)
        } else {
            CodeGenConfig::default()
        };

        if use_f16 {
            config.dtype = candle_core::DType::F16;
        }

        let device = Device::Cpu;

        let tokenizer = CodeGenTokenizer::from_file(tokenizer_path.to_str().unwrap())?;
        let model = WeightLoader::load(&weights_path, &config, &device)?;

        // Without this the generator has no tokenizer, so the streaming callback
        // receives an empty string for every token and `complete` prints nothing.
        let generator = CodeGenGenerator::new(model, temperature, 40, 0.9, 1.2, 256)
            .with_tokenizer(tokenizer.clone());

        Ok(Self {
            generator,
            tokenizer,
            config,
        })
    }

    /// Load with default temperature (0.6).
    pub fn load_default(weights_dir: &Path, use_f16: bool) -> Result<Self> {
        Self::load(weights_dir, use_f16, 0.6)
    }
}
