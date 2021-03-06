//Syntax, grammar tree
#[derive(Debug, Clone)]
pub enum Expr {
    Literal(Literal),
    This,
    Unary(UnaryOp, Box<Expr>),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Grouping(Box<Expr>),
    Variable(Symbol),
    Call(Box<Expr>, SourceLocation, Vec<Expr>),
    Get(Box<Expr>, Symbol),
    Assign(Symbol, Box<Expr>),
    Logical(Box<Expr>, LogicalOp, Box<Expr>),
    Set(Box<Expr>, Symbol, Box<Expr>),
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub line: usize,
    pub col: i64,
}

#[derive(Debug, Clone)]
pub enum LogicalOp {
    Or,
    And,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Symbol {
    pub name: String,
    pub line: usize,
    pub col: i64,
}

#[derive(Debug, Clone)]
pub struct FuncDecl {
    pub name: Symbol,
    pub params: Vec<Symbol>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Expr(Expr),
    Print(Expr),
    VarDecl(Symbol, Option<Expr>),
    Block(Vec<Stmt>),
    If(Expr, Box<Stmt>, Option<Box<Stmt>>),
    While(Expr, Box<Stmt>),
    FuncDecl(FuncDecl),
    ClassDecl(Symbol, Option<Symbol>, Vec<FuncDecl>), //superclass
    Return(SourceLocation, Option<Expr>),
}

//single operand

#[derive(Debug, Copy, Clone)]
pub enum UnaryOpType {
    Minus,
    Bang,
}

#[derive(Debug, Copy, Clone)]
pub struct UnaryOp {
    pub ty: UnaryOpType,
    pub line: usize,
    pub col: i64,
}

//two operands
#[derive(Debug, Copy, Clone)]
pub enum BinaryOpType {
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

#[derive(Debug, Copy, Clone)]
pub struct BinaryOp {
    pub ty: BinaryOpType,
    pub line: usize,
    pub col: i64,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Number(f64),
    String(String),
    True,
    False,
    Nil,
}
