extern crate clap;
use clap::{App, Arg};

use std::fs;

mod bytecode;
mod bytecode_interp;
mod compiler;
mod expr;
mod interpreter;
mod native_functions;
mod parser;
mod scanner;

static INPUT_STR: &str = "INPUT";
static SHOW_TOKENS_STR: &str = "tokens";
static SHOW_AST_STR: &str = "ast";
static SHOW_BYTECODE_STR: &str = "show-bytecode";
static BYTECODE_STR: &str = "bytecode";

fn main() {
    let matches = App::new("loxi")
        .arg(
            Arg::with_name(INPUT_STR)
                .help("input file to use")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name(SHOW_TOKENS_STR)
                .long("--show-tokens")
                .takes_value(false)
                .help("shows tokens"),
        )
        .arg(
            Arg::with_name(SHOW_AST_STR)
                .long("--show-ast")
                .takes_value(false)
                .help("shows AST"),
        )
        .arg(
            Arg::with_name(SHOW_BYTECODE_STR)
                .long("--show-bytecode")
                .takes_value(false)
                .help("shows full bytecode"),
        )
        .arg(
            Arg::with_name(BYTECODE_STR)
                .long("--bytecode")
                .takes_value(false)
                .help("shows short bytecode"),
        )
        .get_matches();
    if let Some(input_file) = matches.value_of(INPUT_STR) {
        let maybe_input = fs::read_to_string(input_file);
        match maybe_input {
            Ok(input) => {
                if matches.is_present(SHOW_BYTECODE_STR) || matches.is_present(BYTECODE_STR) {
                    let func_or_err = compiler::Compiler::compile(input);
                    match func_or_err {
                        Ok(func) => {
                            if matches.is_present(SHOW_BYTECODE_STR) {
                                bytecode_interp::disassemble_chunk(&func.chunk, input_file);
                            }
                            let res = bytecode_interp::Interpreter::default().interpret(func);
                            match res {
                                Ok(()) => {
                                    std::process::exit(0);
                                }
                                Err(err) => {
                                    println!("{:?}", err);
                                    std::process::exit(1);
                                }
                            }
                        }
                        Err(err) => {
                            println!("{}", err);
                            std::process::exit(1);
                        }
                    }
                }
                match scanner::scan_tokens(input) {
                    Ok(tokens) => {
                        if matches.is_present(SHOW_TOKENS_STR) {
                            println!("tokens: {:#?}", tokens);
                            std::process::exit(0);
                        }
                        let stmts_maybe = parser::parse(tokens);
                        match stmts_maybe {
                            Ok(stmts) => {
                                if matches.is_present(SHOW_AST_STR) {
                                    println!("AST: {:#?}", stmts);
                                    std::process::exit(0);
                                }
                                let interpret_result = interpreter::interpret(&stmts);
                                match interpret_result {
                                    Ok(output) => {
                                        println!("{}", output);
                                    }
                                    Err(err) => {
                                        println!("Interpreter Error: {}", err);
                                        std::process::exit(-1);
                                    }
                                }
                            }
                            Err(err) => {
                                println!("parse error: {}", err);
                                std::process::exit(-1)
                            }
                        }
                    }
                    Err(err) => {
                        println!("lexical error: {}", err);
                        std::process::exit(-1);
                    }
                }
            }
            Err(err) => {
                println!("Error reading {}: {}", input_file, err);
                std::process::exit(-1);
            }
        }
    }
}
