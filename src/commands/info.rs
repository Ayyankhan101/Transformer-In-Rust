use anyhow::Result;

use crate::cli::Cli;
use crate::codegen::config::CodeGenConfig;
use crate::glm::GLMConfig;

pub fn run(cli: &Cli) -> Result<()> {
    // GLM info
    let glm_config = GLMConfig::default();
    println!("\x1b[1mGLM Config (tiny):\x1b[0m");
    println!("  vocab_size:   {}", glm_config.vocab_size);
    println!("  hidden_dim:   {}", glm_config.hidden_dim);
    println!("  num_layers:   {}", glm_config.num_layers);
    println!("  num_heads:    {}", glm_config.num_heads);
    println!("  ffn_dim:      {}", glm_config.ffn_dim);

    // CodeGen info
    println!("\n\x1b[1mCodeGen-350M:\x1b[0m");
    println!("  vocab_size:   51200");
    println!("  hidden_dim:   1024");
    println!("  num_layers:   20");
    println!("  num_heads:    16");
    println!("  ffn_dim:      4096");
    println!("  max_seq_len:  1024");
    println!("  params:       ~350M");

    // Weight status
    let weights_path = cli.weights_dir.join("pytorch_model.bin");
    let config_path = cli.weights_dir.join("config.json");
    let tokenizer_path = cli.weights_dir.join("tokenizer.json");

    println!("\n\x1b[1mWeight Status:\x1b[0m");
    if weights_path.exists() {
        let meta = std::fs::metadata(&weights_path)?;
        let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
        println!("  weights:    \x1b[32m✓\x1b[0m {} ({:.1} MB)", weights_path.display(), size_mb);
    } else {
        println!("  weights:    \x1b[31m✗\x1b[0m Not found at {}", weights_path.display());
    }

    if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        let config = CodeGenConfig::from_hf_config(&config_json);
        println!("  config:     \x1b[32m✓\x1b[0m {} (dtype: {:?})", config_path.display(), config.dtype);
    } else {
        println!("  config:     \x1b[33m-\x1b[0m Not found (using defaults)");
    }

    if tokenizer_path.exists() {
        println!("  tokenizer:  \x1b[32m✓\x1b[0m {}", tokenizer_path.display());
    } else {
        println!("  tokenizer:  \x1b[31m✗\x1b[0m Not found at {}", tokenizer_path.display());
    }

    let all_present = weights_path.exists() && tokenizer_path.exists();
    if all_present {
        println!("\n  \x1b[32mReady to run: `codegen chat` or `codegen complete \"prompt\"`\x1b[0m");
    } else {
        println!("\n  \x1b[33mRun `codegen download` to get weights.\x1b[0m");
    }

    Ok(())
}
