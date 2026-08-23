use std::io::Write;

use anyhow::Result;

use crate::cli::Cli;
use crate::generation::chat::ChatSession;
use crate::generation::codegen_generate::CollectStream;
use crate::model::ModelContext;

pub fn run(cli: &Cli, system: Option<String>) -> Result<()> {
    let mut ctx = ModelContext::load_default(&cli.weights_dir, cli.f16)?;
    ctx.generator.set_seed(cli.seed);

    let mut session = ChatSession::new(1024);
    if let Some(sys) = system {
        println!("\x1b[33m[System: {sys}]\x1b[0m\n");
        session.set_system_prompt(sys);
    }

    println!("\x1b[1;36m╔══════════════════════════════════════════════════╗");
    println!("║  CodeGen Chat — Conversational Code Generation  ║");
    println!("╚══════════════════════════════════════════════════╝\x1b[0m\n");
    println!("  Type your request in natural language.");
    println!("  Commands: /help, /clear, /history, /system, /temp, /tokens, /save\n");

    let stdin = std::io::stdin();
    let mut input = String::new();

    loop {
        print!("\x1b[1;34mYou:\x1b[0m ");
        std::io::stdout().flush()?;

        input.clear();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }

        let line = input.trim();
        if line.is_empty() {
            continue;
        }

        match line {
            "exit" | "quit" => {
                println!("\nGoodbye!");
                break;
            }
            "/help" => {
                println!("\n\x1b[1mCommands:\x1b[0m");
                println!("  /help           Show this help");
                println!("  /clear          Clear conversation history");
                println!("  /history        Show conversation so far");
                println!("  /system <msg>   Set system prompt");
                println!(
                    "  /temp <n>       Set temperature (current: {:.1})",
                    ctx.generator.temperature()
                );
                println!(
                    "  /tokens <n>     Set max tokens (current: {})",
                    ctx.generator.max_new_tokens()
                );
                println!("  /save <file>    Save last generated code to file");
                println!("  exit / quit     Exit\n");
                continue;
            }
            "/clear" => {
                session.clear();
                println!("\x1b[33m[History cleared]\x1b[0m\n");
                continue;
            }
            "/history" => {
                println!("\n{}", session.format_history());
                continue;
            }
            line if line.starts_with("/system ") => {
                let sys_msg = line.strip_prefix("/system ").unwrap().trim();
                session.set_system_prompt(sys_msg.to_string());
                println!("\x1b[33m[System prompt set: {}]\x1b[0m\n", sys_msg);
                continue;
            }
            line if line.starts_with("/temp ") => {
                if let Ok(t) = line.strip_prefix("/temp ").unwrap().trim().parse::<f64>() {
                    let t = t.clamp(0.0, 2.0);
                    ctx.generator.set_temperature(t);
                    println!("\x1b[33m[Temperature: {:.1}]\x1b[0m\n", t);
                } else {
                    println!("\x1b[31mInvalid temperature.\x1b[0m\n");
                }
                continue;
            }
            line if line.starts_with("/tokens ") => {
                if let Ok(n) = line
                    .strip_prefix("/tokens ")
                    .unwrap()
                    .trim()
                    .parse::<usize>()
                {
                    let n = n.min(1024);
                    ctx.generator.set_max_new_tokens(n);
                    println!("\x1b[33m[Max tokens: {}]\x1b[0m\n", n);
                } else {
                    println!("\x1b[31mInvalid number.\x1b[0m\n");
                }
                continue;
            }
            line if line.starts_with("/save ") => {
                let path = line.strip_prefix("/save ").unwrap().trim();
                if let Some(last) = session
                    .history()
                    .iter()
                    .rev()
                    .find(|m| m.role == crate::generation::chat::Role::Assistant)
                {
                    std::fs::write(path, &last.content)?;
                    println!("\x1b[33m[Saved to {}]\x1b[0m\n", path);
                } else {
                    println!("\x1b[31mNo generated code to save.\x1b[0m\n");
                }
                continue;
            }
            _ => {}
        }

        session.trim_to_fit(&ctx.tokenizer);

        let prompt = session.assemble_prompt(line);
        let prompt_ids = ctx.tokenizer.encode(&prompt)?;

        print!("\x1b[1;32mAssistant:\x1b[0m ");
        std::io::stdout().flush()?;

        let start = std::time::Instant::now();
        let mut collector = CollectStream::new();
        ctx.generator.generate_stream(&prompt_ids, &mut collector)?;
        let elapsed = start.elapsed();

        let all_tokens = collector.tokens;
        let generated_ids = &all_tokens[prompt_ids.len()..];
        let response = ctx.tokenizer.decode(generated_ids)?;

        if !response.is_empty() {
            println!("{response}");
        }

        println!(
            "\x1b[90m--- {} tokens in {:.1}s ---\x1b[0m\n",
            generated_ids.len(),
            elapsed.as_secs_f64()
        );

        session.add_user(line.to_string());
        session.add_assistant(response);
    }

    Ok(())
}
