use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Device;

use crate::cli::Cli;
use crate::training::{GLMTrainer, TrainConfig};

pub fn run(
    cli: &Cli,
    data_path: &str,
    steps: usize,
    config_path: Option<&Path>,
    resume: bool,
) -> Result<()> {
    let device = Device::Cpu;
    let dtype_str = if cli.f16 { "FP16" } else { "FP32" };

    println!("\x1b[1mGLM Training Demo ({dtype_str})\x1b[0m\n");

    let mut config = match config_path {
        Some(path) => {
            println!("Config: {}", path.display());
            TrainConfig::from_file(path).map_err(|e| anyhow::anyhow!("{e}"))?
        }
        None => TrainConfig::default(),
    };

    // CLI flags win over the file.
    config.training.data_dir = PathBuf::from(data_path);
    config.training.max_steps = steps;
    config.training.dtype = if cli.f16 { "f16" } else { "f32" }.to_string();

    let model = &config.model;
    println!("GLM Config:");
    println!("  vocab_size:   {}", model.vocab_size);
    println!("  hidden_dim:   {}", model.hidden_dim);
    println!("  num_layers:   {}", model.num_layers);
    println!("  num_heads:    {}", model.num_heads);
    println!("  ffn_dim:      {}", model.ffn_dim);
    println!(
        "  batch:        {} x {} (micro x accumulation)",
        config.training.micro_batch_size, config.training.gradient_accumulation_steps
    );

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
    trainer.train(data_dir, &device, resume)?;

    println!("\nTraining complete!");
    Ok(())
}
