//! CLI entry point. Everything it drives lives in the library crate, so the
//! binary does not compile a second copy of the module tree.

use clap::Parser;

use rust_transformer::cli::{Cli, Commands};
use rust_transformer::commands;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Chat { system } => commands::chat::run(&cli, system.clone()),
        Commands::Complete {
            prompt,
            max_tokens,
            temperature,
            template,
            no_stream,
        } => commands::complete::run(
            &cli,
            prompt,
            *max_tokens,
            *temperature,
            template,
            !*no_stream,
        ),
        Commands::Repl => commands::repl::run(&cli),
        Commands::Info => commands::info::run(&cli),
        Commands::Serve { port } => commands::serve::run(&cli, *port),
        Commands::Download => commands::download::run(&cli),
        Commands::GlmTrain {
            data_path,
            steps,
            config,
        } => commands::glm_train::run(&cli, data_path, *steps, config.as_deref()),
    }
}
