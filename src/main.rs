use std::env;
use std::fs;

mod lexer;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return println!("Expected file input arg");
    }

        let text = lexer::tokenize(fs::read_to_string(&args[1]).unwrap());
        match text{
            Ok(tokens) => for t in tokens {println!("{:?}", t)}
            Err(err) => println!("lexical error : {}", err)
        }
}
    

