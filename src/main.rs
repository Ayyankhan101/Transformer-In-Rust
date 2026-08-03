mod cli;
mod commands;
mod model;

mod codegen;
mod generation;
mod glm;
mod layers;
#[cfg(feature = "server")]
mod server;
mod tokenizer;
mod training;

use clap::Parser;

use cli::{Cli, Commands};

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Chat { system } => commands::chat::run(&cli, system.clone()),
        Commands::Complete {
            prompt,
            max_tokens,
            temperature,
            template,
            stream,
        } => commands::complete::run(&cli, prompt, *max_tokens, *temperature, template, *stream),
        Commands::Repl => commands::repl::run(&cli),
        Commands::Info => commands::info::run(&cli),
        Commands::Serve { port } => commands::serve::run(&cli, *port),
        Commands::Download => commands::download::run(&cli),
        Commands::GlmTrain { data_path, steps } => {
            commands::glm_train::run(&cli, data_path, *steps)
        }
    }
}
