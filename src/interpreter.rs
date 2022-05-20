use crate::expr;

#[derive(Debug)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Nil,
}

pub fn interpret(expr: &expr::Expr) -> Result<Value, String> {
    match expr {
        expr::Expr::Literal(lit) => Ok(interpret_literal(lit)),
        expr::Expr::Unary(op, e) => interpret_unary(*op, *&e),
        expr::Expr::Binary(lhs, op, rhs) => interpret_binary(*&lhs, *op, *&rhs),
        expr::Expr::Grouping(e) => interpret(&*e),
    }
}
