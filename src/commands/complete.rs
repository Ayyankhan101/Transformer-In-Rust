use std::io::Write;

use anyhow::Result;

use crate::cli::Cli;
use crate::generation::codegen_generate::{CollectStream, PrintStream};
use crate::model::ModelContext;

pub fn run(
    cli: &Cli,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
    template: &str,
    stream: bool,
) -> Result<()> {
    let mut ctx = ModelContext::load(&cli.weights_dir, cli.f16, temperature)?;
    ctx.generator.set_max_new_tokens(max_tokens);

    // Apply prompt template
    let formatted = match template {
        "instruct" => {
            use crate::generation::codegen_generate::PromptTemplate;
            PromptTemplate::Instruct.apply(prompt)
        }
        "chat" => {
            use crate::generation::codegen_generate::PromptTemplate;
            PromptTemplate::Chat.apply(prompt)
        }
        _ => prompt.to_string(), // completion (raw)
    };

    let token_ids = ctx.tokenizer.encode(&formatted)?;

    if stream {
        print!("\x1b[1;32m{prompt}\x1b[0m");
        std::io::stdout().flush()?;

        let start = std::time::Instant::now();
        let mut print_stream = PrintStream;
        ctx.generator
            .generate_stream(&token_ids, &mut print_stream)?;
        let elapsed = start.elapsed();

        println!("\n\x1b[90m--- {:.1}s ---\x1b[0m", elapsed.as_secs_f64());
    } else {
        let start = std::time::Instant::now();
        let mut collector = CollectStream::new();
        ctx.generator.generate_stream(&token_ids, &mut collector)?;
        let elapsed = start.elapsed();

        let generated_ids = &collector.tokens[token_ids.len()..];
        let output = ctx.tokenizer.decode(generated_ids)?;
        println!("{output}");
        println!(
            "\n\x1b[90m--- {} tokens in {:.1}s ---\x1b[0m",
            generated_ids.len(),
            elapsed.as_secs_f64()
        );
    }

    Ok(())
}
