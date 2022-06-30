use std::env;
use std::fs;

mod expr;
mod interpreter;
mod parser;
mod scanner;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return println!("Expected file input arg");
    }

    match scanner::scan_tokens(fs::read_to_string(&args[1]).unwrap()) {
        Ok(tokens) => {
            let stmts_maybe = parser::parse(tokens);

            match stmts_maybe {
                Ok(stmts) => {
                    let interpret_res = interpreter::interpret(&stmts);

                    match interpret_res {
                        Ok(_) => {}

                        Err(err) => println!("Interpreter error:\n{}", err),
                    }
                }
                Err(err) => println!("Parse error: {}", err),
            }
        }
        Err(err) => println!("lexical error: {}", err),
    }
}
