use std::env;
use std::fs;

mod scanner;
mod expr;
mod parser;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return println!("Expected file input arg");
    }

        
        match scanner::scan_tokens(fs::read_to_string(&args[1]).unwrap()){
            Ok(tokens) => for t in tokens {println!("{:?}", t)}
            Err(err) => println!("lexical error : {}", err)
        }
}
    

