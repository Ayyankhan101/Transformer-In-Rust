use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Device;

use crate::cli::Cli;
use crate::training::{GLMTrainer, TrainingConfig};

pub fn run(cli: &Cli, data_path: &str, steps: usize) -> Result<()> {
    let device = Device::Cpu;
    let dtype_str = if cli.f16 { "FP16" } else { "FP32" };

    println!("\x1b[1mGLM Training Demo ({dtype_str})\x1b[0m\n");

    let mut config = TrainingConfig::default();
    config.data_dir = PathBuf::from(data_path);
    config.max_steps = steps;
    config.dtype = if cli.f16 { "f16".to_string() } else { "f32".to_string() };

    println!("GLM Config:");
    println!("  vocab_size:   50257");
    println!("  hidden_dim:   1024");
    println!("  num_layers:   4");
    println!("  num_heads:    8");
    println!("  ffn_dim:      4096");

    // Check if data exists
    let data_dir = Path::new(data_path);
    if !data_dir.exists() {
        println!("\nData directory not found: {data_path}");
        println!("Creating sample data...");
        std::fs::create_dir_all(data_dir)?;
        std::fs::write(
            data_dir.join("sample.py"),
            include_str!("../../data/sample.py"),
        )?;
    }

    let mut trainer = GLMTrainer::from_config(&config, &device)?;

    println!("Starting training for {steps} steps...\n");
    trainer.train(data_dir, &device)?;

    println!("\nTraining complete!");
    Ok(())
}
