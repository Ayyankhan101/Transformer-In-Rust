use std::io::Write;

use anyhow::Result;

use crate::cli::Cli;
use crate::generation::codegen_generate::CollectStream;
use crate::model::ModelContext;

pub fn run(cli: &Cli) -> Result<()> {
    let ctx = ModelContext::load_default(&cli.weights_dir, cli.f16)?;

    println!("CodeGen-350M REPL — type a prompt and get generated code.");
    println!("Type 'exit' or 'quit' to stop.\n");

    let stdin = std::io::stdin();
    let mut input = String::new();

    loop {
        print!(">>> ");
        std::io::stdout().flush()?;

        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }

        let prompt = input.trim();
        if prompt.is_empty() || prompt == "exit" || prompt == "quit" {
            break;
        }

        let token_ids = ctx.tokenizer.encode(prompt)?;
        let start = std::time::Instant::now();
        let mut collector = CollectStream::new();
        ctx.generator.generate_stream(&token_ids, &mut collector)?;
        let elapsed = start.elapsed();

        let generated_ids = &collector.tokens[token_ids.len()..];
        let output = ctx.tokenizer.decode(generated_ids)?;
        println!("\n{output}");
        println!(
            "--- {} tokens in {:.1}s ---\n",
            generated_ids.len(),
            elapsed.as_secs_f64()
        );
    }

    println!("Bye!");
    Ok(())
}
