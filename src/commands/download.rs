use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::cli::Cli;

/// The CLI names to try, in order.
///
/// `huggingface-cli` was renamed to `hf`. The old name still exists but now
/// prints a deprecation notice, downloads nothing, and **exits 0** — so trusting
/// the exit status reported a successful download of no files at all.
const DOWNLOAD_TOOLS: &[&str] = &["hf", "huggingface-cli"];

const REPO: &str = "Salesforce/codegen-350M-multi";

/// The checkpoint file, if one is already there. Mirrors the order
/// `ModelContext::load` prefers.
fn existing_weights(dir: &Path) -> Option<(PathBuf, f64)> {
    for name in ["model.safetensors", "pytorch_model.bin"] {
        let path = dir.join(name);
        if let Ok(meta) = std::fs::metadata(&path) {
            return Some((path, meta.len() as f64 / (1024.0 * 1024.0)));
        }
    }
    None
}

pub fn run(cli: &Cli) -> Result<()> {
    if let Some((path, size_mb)) = existing_weights(&cli.weights_dir) {
        println!(
            "Weights already present at: {} ({:.1} MB)",
            path.display(),
            size_mb
        );
        return Ok(());
    }

    println!("\x1b[1mDownloading CodeGen-350M-multi from HuggingFace...\x1b[0m\n");

    let dir = cli
        .weights_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("weights directory path is not valid UTF-8"))?;

    for tool in DOWNLOAD_TOOLS {
        let status = std::process::Command::new(tool)
            .args(["download", REPO, "--local-dir", dir])
            .status();

        let Ok(status) = status else {
            continue; // not installed, try the next name
        };

        // Check for the file rather than believing the exit code.
        if let Some((path, size_mb)) = existing_weights(&cli.weights_dir) {
            println!("\n\x1b[32m✓ Download complete!\x1b[0m");
            println!("  {} ({:.1} MB)", path.display(), size_mb);
            println!("  Run `codegen info` to verify.");
            return Ok(());
        }

        if status.success() {
            println!("\n\x1b[33m`{tool}` reported success but no weights appeared.\x1b[0m");
        } else {
            println!("\n\x1b[31m`{tool}` failed.\x1b[0m");
        }
    }

    print_manual_instructions(&cli.weights_dir);
    bail!("could not download weights automatically");
}

fn print_manual_instructions(weights_dir: &Path) {
    println!("\nInstall the HuggingFace CLI:");
    println!("  pip install -U huggingface_hub\n");
    println!("Or download manually:");
    println!("  1. Install git-lfs:");
    println!("     git lfs install\n");
    println!("  2. Clone the repo:");
    println!(
        "     git clone https://huggingface.co/{REPO} {}\n",
        weights_dir.display()
    );
    println!("  3. Or fetch the individual files:");
    for file in ["pytorch_model.bin", "tokenizer.json", "config.json"] {
        println!(
            "     wget -P {}/ https://huggingface.co/{REPO}/resolve/main/{file}",
            weights_dir.display()
        );
    }
}
