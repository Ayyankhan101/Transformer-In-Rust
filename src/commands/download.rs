use std::path::Path;

use anyhow::Result;

use crate::cli::Cli;

pub fn run(cli: &Cli) -> Result<()> {
    let weights_path = cli.weights_dir.join("pytorch_model.bin");

    if weights_path.exists() {
        let meta = std::fs::metadata(&weights_path)?;
        let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
        println!(
            "Weights already present at: {} ({:.1} MB)",
            weights_path.display(),
            size_mb
        );
        return Ok(());
    }

    println!("\x1b[1mDownloading CodeGen-350M-multi from HuggingFace...\x1b[0m\n");

    // Try huggingface-cli first
    let status = std::process::Command::new("huggingface-cli")
        .args([
            "download",
            "Salesforce/codegen-350M-multi",
            "--local-dir",
            cli.weights_dir.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("\n\x1b[32m✓ Download complete!\x1b[0m");
            println!("  Run `codegen info` to verify.");
        }
        Ok(_) => {
            println!("\n\x1b[31mhuggingface-cli failed.\x1b[0m");
            print_manual_instructions(&cli.weights_dir);
        }
        Err(_) => {
            println!("huggingface-cli not found. Install it:");
            println!("  pip install huggingface_hub\n");
            print_manual_instructions(&cli.weights_dir);
        }
    }

    Ok(())
}

fn print_manual_instructions(weights_dir: &Path) {
    println!("\nManual download:");
    println!("  1. Install git-lfs:");
    println!("     git lfs install\n");
    println!("  2. Clone the repo:");
    println!(
        "     git clone https://huggingface.co/Salesforce/codegen-350M-multi {}\n",
        weights_dir.display()
    );
    println!("  3. Or use wget for individual files:");
    println!("     wget -P {}/ https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/pytorch_model.bin", weights_dir.display());
    println!("     wget -P {}/ https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/tokenizer.json", weights_dir.display());
    println!("     wget -P {}/ https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/config.json", weights_dir.display());
}
