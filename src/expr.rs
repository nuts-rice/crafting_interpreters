//Syntax, grammar tree

#[allow(dead_code)]
pub enum Expr{
    Literal(Literal),
    Unary(UnaryOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Grouping(Box<Expr>),
}

//single operand
#[allow(dead_code)]

pub enum UnaryOp{
    Minus,
    Bang,
}

//two operands
#[allow(dead_code)]
pub enum BinaryOp{
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Plus,
    Minus,
    Star,
    Slash,
}

#[allow(dead_code)]
pub enum Literal{
    Number,
    String,
    True,
    False,
    Nil,
}
