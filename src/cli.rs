use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Command-line interface for CodeGen-350M inference.
///
/// Supports conversational chat, single-shot completion, interactive REPL,
/// HTTP server, and GLM training demos.
///
/// # Global Flags
///
/// - `--f16` — Use FP16 precision (faster, less memory)
/// - `--weights-dir` — Path to weights directory (default: `codegen_weights`)
/// - `--seed` — Fixed sampling seed for reproducible output
#[derive(Parser)]
#[command(
    name = "codegen",
    about = "CodeGen-350M inference CLI — code generation in pure Rust",
    version
)]
pub struct Cli {
    /// Use FP16 precision (faster, less memory)
    #[arg(long, global = true)]
    pub f16: bool,

    /// Path to weights directory
    #[arg(long, global = true, default_value = "codegen_weights")]
    pub weights_dir: PathBuf,

    /// Fixed sampling seed for reproducible output (default: random each run)
    #[arg(long, global = true)]
    pub seed: Option<u64>,

    #[command(subcommand)]
    pub command: Commands,
}

/// Available subcommands for the CodeGen CLI.
#[derive(Subcommand)]
pub enum Commands {
    /// Conversational code generation (multi-turn)
    Chat {
        /// System prompt to set at start
        #[arg(short, long)]
        system: Option<String>,
    },

    /// Single-shot code generation from a prompt
    Complete {
        /// The prompt to complete
        prompt: String,

        /// Max tokens to generate
        #[arg(short, long, default_value = "128")]
        max_tokens: usize,

        /// Temperature (0.0 = deterministic, 2.0 = max creativity)
        #[arg(short, long, default_value = "0.6")]
        temperature: f64,

        /// Prompt template: completion, instruct, chat
        #[arg(short, long, default_value = "completion")]
        template: String,

        /// Stream tokens as they're generated
        #[arg(long, default_value = "true")]
        stream: bool,
    },

    /// Interactive REPL (single-turn)
    Repl,

    /// Print model info and weight status
    Info,

    /// Start HTTP inference server
    Serve {
        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },

    /// Download CodeGen-350M weights from HuggingFace
    Download,

    /// GLM training demo (trains from scratch on .py files)
    GlmTrain {
        /// Directory with .py files
        #[arg(short, long, default_value = "data")]
        data_path: String,

        /// Number of training steps
        #[arg(short, long, default_value = "1000")]
        steps: usize,

        /// YAML training config (see configs/train.yaml). Defaults are used if omitted.
        #[arg(short, long)]
        config: Option<PathBuf>,
    },
}
