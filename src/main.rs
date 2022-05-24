use std::env;
use std::fs;

mod scanner;
mod expr;
mod parser;
mod interpreter;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return println!("Expected file input arg");
    }

        
        match scanner::scan_tokens(fs::read_to_string(&args[1]).unwrap()){
            Ok(tokens) => {
                println!("Tokens");
                for t in &tokens {
                    println!("{:?}", t)
                }
                println!();

                let expr_maybe = parser::parse(tokens);

                match expr_maybe{
                    Ok(expr) => {
                        println!("ast:\n{:#?}", expr);
                        let interpret_res = interpreter::interpret(&expr);
                        println()!;
                        match interpret_res{
                            Ok(val) => println!("Result:\n{}", val),
                            Err(err) => println!("Intreprter error:\n{}", err),
                        }    
                    
                    }
                    Err(err) => println!("Parse error: {}", err),
                }
            }
            Err(err) => println!("lexical error: {}", err),
    }
}
    
 
          
    
        
