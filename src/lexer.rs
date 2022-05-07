#[derive(Debug)]
pub enum TokenType{
    LeftParen, RightParen, LeftBrace, RightBrace, 
    Comma, Dot, Minus, Plus, Semicolon, Slash, Star,

    Bang, BangEqual,
    Equal, EqualEqual,
    Greater, GreaterEqual,
    Less, LessEqual,

    Identifier, String, Number,

    And, Class, Else, False, Fun, For, If, Nil, Or, 
    Print, Return, Super, This, True, Var, While,

    Eof
}

#[derive(Debug)]
pub enum Literal{
    Identifier(String),
    Str(String),
    Number(f64)
}

#[derive(Debug)]
pub struct Token{
    ty: TokenType,
    lexeme: String,
    literal: Option<Literal>,
    line: i64
}

pub fn tokenize(_input: String) -> Result<Vec<Token>, String>{
    unimplemented!();
}
