#![allow(unsafe_code)]

use std::io::{self, Write};

use clap::Parser;

use llama_cli::args::Args;
use llama_cli::run_generate;

fn main() {
    let mut args = Args::parse();
    if let Some(arquivo) = &args.file {
        match std::fs::read_to_string(arquivo) {
            Ok(texto) => args.prompt = texto,
            Err(e) => {
                eprintln!("Erro: {}: {e}", arquivo.display());
                std::process::exit(1);
            }
        }
    }

    if !args.no_display_prompt {
        print!("{}", args.prompt);
        let _ = io::stdout().flush();
    }

    match run_generate(&args, &mut |piece| {
        print!("{piece}");
        let _ = io::stdout().flush();
    }) {
        Ok(timing) => {
            println!();
            if args.timings {
                eprintln!(
                    "{} tokens, {:.2} tok/s",
                    timing.n_tokens, timing.tokens_per_sec
                );
            }
        }
        Err(e) => {
            eprintln!("Erro: {e}");
            std::process::exit(1);
        }
    }
}
