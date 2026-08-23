use anyhow::Result;

use crate::cli::Cli;
use crate::codegen::config::CodeGenConfig;
use crate::glm::GLMConfig;
use crate::tokenizer::CodeGenTokenizer;

pub fn run(cli: &Cli) -> Result<()> {
    // GLM info
    let glm_config = GLMConfig::default();
    println!("\x1b[1mGLM Config (tiny):\x1b[0m");
    println!("  vocab_size:   {}", glm_config.vocab_size);
    println!("  hidden_dim:   {}", glm_config.hidden_dim);
    println!("  num_layers:   {}", glm_config.num_layers);
    println!("  num_heads:    {}", glm_config.num_heads);
    println!("  ffn_dim:      {}", glm_config.ffn_dim);

    let config_path = cli.weights_dir.join("config.json");
    let tokenizer_path = cli.weights_dir.join("tokenizer.json");

    // Report the checkpoint's own config rather than hardcoded numbers, which had
    // drifted from it (max_seq_len was printed as 1024; the checkpoint says 2048).
    let config = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        CodeGenConfig::from_hf_config(&config_json)
    } else {
        CodeGenConfig::default()
    };

    println!("\n\x1b[1mCodeGen-350M:\x1b[0m");
    println!("  vocab_size:   {}", config.vocab_size);
    println!("  hidden_dim:   {}", config.hidden_dim);
    println!("  num_layers:   {}", config.num_layers);
    println!("  num_heads:    {}", config.num_heads);
    println!("  ffn_dim:      {}", config.ffn_dim);
    println!("  max_seq_len:  {}", config.max_seq_len);
    println!("  rotary_dim:   {}", config.rotary_dim);
    println!("  dtype:        {:?}", if cli.f16 { "f16" } else { "f32" });

    println!("\n\x1b[1mWeight Status:\x1b[0m");

    // Same order of preference as `ModelContext::load`.
    let weights = ["model.safetensors", "pytorch_model.bin"]
        .into_iter()
        .map(|name| cli.weights_dir.join(name))
        .find(|path| path.exists());

    match &weights {
        Some(path) => {
            let size_mb = std::fs::metadata(path)?.len() as f64 / (1024.0 * 1024.0);
            println!(
                "  weights:    \x1b[32m✓\x1b[0m {} ({:.1} MB)",
                path.display(),
                size_mb
            );
        }
        None => println!(
            "  weights:    \x1b[31m✗\x1b[0m No model.safetensors or pytorch_model.bin in {}",
            cli.weights_dir.display()
        ),
    }

    if config_path.exists() {
        println!("  config:     \x1b[32m✓\x1b[0m {}", config_path.display());
    } else {
        println!("  config:     \x1b[33m-\x1b[0m Not found (using defaults)");
    }

    if tokenizer_path.exists() {
        let vocab = CodeGenTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .map(|t| t.vocab_size().to_string())
            .unwrap_or_else(|_| "unreadable".to_string());
        println!(
            "  tokenizer:  \x1b[32m✓\x1b[0m {} ({vocab} tokens)",
            tokenizer_path.display()
        );
    } else {
        println!(
            "  tokenizer:  \x1b[31m✗\x1b[0m Not found at {}",
            tokenizer_path.display()
        );
    }

    if weights.is_some() && tokenizer_path.exists() {
        println!(
            "\n  \x1b[32mReady to run: `codegen chat` or `codegen complete \"prompt\"`\x1b[0m"
        );
    } else {
        println!("\n  \x1b[33mRun `codegen download` to get weights.\x1b[0m");
    }

    Ok(())
}
