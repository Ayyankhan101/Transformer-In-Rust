mod codegen;
mod generation;
mod glm;
mod layers;
mod tokenizer;
mod training;

use std::path::Path;

use anyhow::Result;
use candle_core::Device;

use crate::codegen::{CodeGenConfig, WeightLoader};
use crate::generation::{CodeGenGenerator, GLMGenerator};
use crate::glm::{GLMConfig, GLMModel};
use crate::glm::attention_mask::build_glm_mask;
use crate::tokenizer::CodeGenTokenizer;
use crate::training::GLMTrainer;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = Device::Cpu;

    let use_f16 = args.iter().any(|a| a == "--f16");

    match args.get(1).map(|s| s.as_str()) {
        Some("--glm-demo") => run_glm_demo(&device)?,
        Some("--glm-train") => run_glm_train(&args, &device)?,
        Some("--glm-generate") => run_glm_generate(&device)?,
        Some("--glm-fill-blanks") => run_glm_fill_blanks(&device)?,
        Some("--codegen") => run_codegen_inference(&device, use_f16)?,
        Some("--repl") => run_codegen_repl(&device, use_f16)?,
        Some("--info") => print_info()?,
        _ => print_usage(),
    }

    Ok(())
}

fn print_usage() {
    println!("Usage:");
    println!("  --glm-demo                    GLM forward pass demo");
    println!("  --glm-train [opts]            Train tiny GLM on code data");
    println!("    --data-path <dir>             Directory with .py files");
    println!("    --steps <N>                   Number of training steps (default: 1000)");
    println!("    --download-data [<url>]       Download sample data if empty");
    println!("  --glm-generate                GLM causal generation (random weights)");
    println!("  --glm-fill-blanks             GLM blank-infilling demo (random weights)");
    println!("  --codegen                     CodeGen-350M inference (needs weights)");
    println!("  --codegen --f16               CodeGen-350M inference in FP16");
    println!("  --repl                        CodeGen-350M interactive REPL");
    println!("  --repl --f16                  CodeGen-350M REPL in FP16");
    println!("  --info                        Print model info");
}

fn print_info() -> Result<()> {
    let config = GLMConfig::default();
    println!("GLM Config (tiny):");
    println!("  vocab_size:   {}", config.vocab_size);
    println!("  hidden_dim:   {}", config.hidden_dim);
    println!("  num_layers:   {}", config.num_layers);
    println!("  num_heads:    {}", config.num_heads);
    println!("  ffn_dim:      {}", config.ffn_dim);
    println!("  max_seq_len:  {}", config.max_seq_len);
    println!("  ~params:      {}", config.param_count_estimate());

    let codegen_config = CodeGenConfig::default();
    println!("\nCodeGen Config (350M):");
    println!("  vocab_size:   {}", codegen_config.vocab_size);
    println!("  hidden_dim:   {}", codegen_config.hidden_dim);
    println!("  num_layers:   {}", codegen_config.num_layers);
    println!("  num_heads:    {}", codegen_config.num_heads);
    println!("  ffn_dim:      {}", codegen_config.ffn_dim);
    println!("  max_seq_len:  {}", codegen_config.max_seq_len);
    println!("  rotary_dim:   {}", codegen_config.rotary_dim);

    Ok(())
}

fn run_glm_demo(device: &Device) -> Result<()> {
    println!("=== GLM Forward Pass Demo ===\n");

    let config = GLMConfig::default();
    println!("Config: {} layers, {} hidden, {} heads, ~{} params",
        config.num_layers, config.hidden_dim, config.num_heads, config.param_count_estimate());

    let model = GLMModel::new(config.clone(), device)?;
    println!("Model created successfully!\n");

    let context_tokens: Vec<u32> = vec![100, 101, 102, 103, 104];
    let blank_lens = vec![3, 1];

    let mask = build_glm_mask(context_tokens.len(), &blank_lens, device)?;
    println!("Attention mask shape: {:?}", mask.shape());

    let mask_id = config.vocab_size as u32 - 1;
    let mut all_tokens = context_tokens.clone();
    all_tokens.extend(std::iter::repeat(mask_id).take(4));

    let logits = model.forward(
        &all_tokens,
        context_tokens.len(),
        &blank_lens,
        &mask,
    )?;

    println!("Input tokens:  {:?}", all_tokens);
    println!("Logits shape:  {:?}", logits.shape());
    println!("\nGLM forward pass complete!");

    Ok(())
}

fn run_glm_train(args: &[String], device: &Device) -> Result<()> {
    println!("=== GLM Training ===\n");

    let config = GLMConfig::default();
    let mut trainer = GLMTrainer::new(config.clone(), device)?;

    let data_path = args.iter()
        .position(|a| a == "--data-path")
        .and_then(|i| args.get(i + 1))
        .map(|s| Path::new(s))
        .unwrap_or_else(|| Path::new("data"));

    let num_steps = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1000);

    let download = args.iter().any(|a| a == "--download-data");

    trainer.load_checkpoint()?;

    println!("Starting from step {}", trainer.step());
    println!("Training for {} steps, data from {:?}", num_steps, data_path);

    trainer.train(data_path, num_steps, download, device)?;

    Ok(())
}

fn run_glm_generate(device: &Device) -> Result<()> {
    println!("=== GLM Causal Generation ===\n");

    let config = GLMConfig::default();
    println!("Config: {} layers, {} hidden, {} heads, ~{} params",
        config.num_layers, config.hidden_dim, config.num_heads, config.param_count_estimate());

    let model = GLMModel::new(config.clone(), device)?;
    let mask_token_id = config.vocab_size as u32 - 1;
    println!("Model created!\n");

    let generator = GLMGenerator::new(model, mask_token_id, 0.8, 40, 0.9, 1.1, 30);

    let prompt: Vec<u32> = vec![100, 101, 102, 103, 104, 105, 106, 107];

    println!("Prompt: {:?}", prompt);
    println!("Generating...\n");

    let start = std::time::Instant::now();
    let output = generator.generate(&prompt)?;
    let elapsed = start.elapsed();

    println!("Generated {} tokens in {:.2}s ({:.2}s/token)",
        output.len(),
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / output.len() as f64);
    println!("Output tokens: {:?}", &output[prompt.len()..]);

    Ok(())
}

fn run_glm_fill_blanks(device: &Device) -> Result<()> {
    println!("=== GLM Blank-Infilling ===\n");

    let config = GLMConfig::default();
    println!("Config: {} layers, {} hidden, {} heads, ~{} params",
        config.num_layers, config.hidden_dim, config.num_heads, config.param_count_estimate());

    let model = GLMModel::new(config.clone(), device)?;
    let mask_token_id = config.vocab_size as u32 - 1;
    let generator = GLMGenerator::new(model, mask_token_id, 0.8, 40, 0.9, 1.1, 30);

    let context: Vec<u32> = vec![100, 101, 102, 103, 104, 105];
    let blank_lens = vec![3, 1];

    println!("Context tokens: {:?}", context);
    println!("Blank lengths: {:?}\n", blank_lens);

    let start = std::time::Instant::now();
    let output = generator.fill_blanks(&context, &blank_lens)?;
    let elapsed = start.elapsed();

    let total_blank: usize = blank_lens.iter().sum();
    let filled = &output[context.len()..];

    println!("Filled {} blank tokens in {:.2}s ({:.2}s/token)",
        total_blank,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() / total_blank as f64);
    println!("Full sequence: {:?}", output);
    println!("Context:       {:?}", &output[..context.len()]);
    println!("Blank 0:       {:?}", &filled[..blank_lens[0]]);
    if blank_lens.len() > 1 {
        let offset = blank_lens[0];
        println!("Blank 1:       {:?}", &filled[offset..offset + blank_lens[1]]);
    }

    Ok(())
}

fn run_codegen_inference(device: &Device, use_f16: bool) -> Result<()> {
    let dtype_str = if use_f16 { "FP16" } else { "FP32" };
    println!("=== CodeGen-350M Inference ({dtype_str}) ===\n");

    let weights_path = Path::new("codegen_weights/pytorch_model.bin");
    let config_path = Path::new("codegen_weights/config.json");
    let tokenizer_path = "codegen_weights/tokenizer.json";

    if !weights_path.exists() {
        println!("Weights not found at: {}", weights_path.display());
        println!("\nTo download CodeGen-350M-multi weights:");
        println!("  hf download Salesforce/codegen-350M-multi --local-dir codegen_weights");
        return Ok(());
    }

    let mut config = if config_path.exists() {
        let config_str = std::fs::read_to_string(config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        CodeGenConfig::from_hf_config(&config_json)
    } else {
        println!("No config.json found, using defaults");
        CodeGenConfig::default()
    };

    if use_f16 {
        config.dtype = candle_core::DType::F16;
    }

    let tokenizer = CodeGenTokenizer::from_file(tokenizer_path)?;
    println!("Tokenizer loaded (vocab_size: {})", config.vocab_size);

    println!("Loading CodeGen-350M weights ({dtype_str})...");
    let model = WeightLoader::load_from_pytorch(weights_path, &config, device)?;
    println!("Weights loaded!");

    let generator = CodeGenGenerator::new(model, 0.6, 40, 0.9, 1.2, 30);

    let prompts = vec![
        "def fibonacci(n):",
    ];

    for prompt in &prompts {
        println!("\nPrompt: {}", prompt);
        let token_ids = tokenizer.encode(prompt)?;
        println!("  {} input tokens: {:?}", token_ids.len(), token_ids);
        println!("  decoded: |{}|", tokenizer.decode(&token_ids)?);
        let start = std::time::Instant::now();
        let generated = generator.generate(&token_ids)?;
        let elapsed = start.elapsed();
        let output = tokenizer.decode(&generated)?;
        println!("  {} total tokens in {:.1}s", generated.len(), elapsed.as_secs_f64());
        println!("Output: {}", output);
    }

    Ok(())
}

fn run_codegen_repl(device: &Device, use_f16: bool) -> Result<()> {
    let dtype_str = if use_f16 { "FP16" } else { "FP32" };
    let weights_path = Path::new("codegen_weights/pytorch_model.bin");
    let config_path = Path::new("codegen_weights/config.json");
    let tokenizer_path = "codegen_weights/tokenizer.json";

    if !weights_path.exists() {
        println!("Weights not found at: {}", weights_path.display());
        println!("\nTo download CodeGen-350M-multi weights:");
        println!("  hf download Salesforce/codegen-350M-multi --local-dir codegen_weights");
        return Ok(());
    }

    let mut config = if config_path.exists() {
        let config_str = std::fs::read_to_string(config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)?;
        CodeGenConfig::from_hf_config(&config_json)
    } else {
        println!("No config.json found, using defaults");
        CodeGenConfig::default()
    };

    if use_f16 {
        config.dtype = candle_core::DType::F16;
    }

    let tokenizer = CodeGenTokenizer::from_file(tokenizer_path)?;
    println!("Tokenizer loaded (vocab_size: {})", config.vocab_size);

    println!("Loading CodeGen-350M weights ({dtype_str})...");
    let model = WeightLoader::load_from_pytorch(weights_path, &config, device)?;
    println!("Weights loaded!\n");

    let generator = CodeGenGenerator::new(model, 0.6, 40, 0.9, 1.2, 128);

    println!("CodeGen-350M REPL — type a prompt and get generated code.");
    println!("Type 'exit' or 'quit' to stop.\n");

    let stdin = std::io::stdin();
    let mut input = String::new();

    loop {
        print!(">>> ");
        use std::io::Write;
        std::io::stdout().flush()?;

        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }

        let prompt = input.trim();
        if prompt.is_empty() || prompt == "exit" || prompt == "quit" {
            break;
        }

        let token_ids = tokenizer.encode(prompt)?;
        let start = std::time::Instant::now();
        let generated = generator.generate(&token_ids)?;
        let elapsed = start.elapsed();

        let output = tokenizer.decode(&generated)?;
        println!("\n{}", output);
        println!("--- {} tokens in {:.1}s ---\n", generated.len(), elapsed.as_secs_f64());
    }

    println!("Bye!");
    Ok(())
}
