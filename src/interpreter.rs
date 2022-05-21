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

fn interpret_binary(lhs_expr: &expr::Expr, op: expr::BinaryOp, rhs_expr: &expr::Expr) -> Result<Value, String>{
    let lhs = interpret(lhs_expr)?;
    let rhs = interpret(rhs_expr)?;

    match (&lhs, op, &rhs){
        (Value::Number(n1), expr::BinaryOp::EqualEqual, Value::Number(n2)) => {
            Ok(Value::Bool(n1 == n2))
        }
        (Value::Number(n1), expr::BinaryOp::NotEqual, Value::Number(n2)) => {
            Ok(Value::Bool(n1 != n2))
        }
        (Value::Number(n1), expr::BinaryOp::Less, Value::Number(n2)) => Ok(Value::Bool(n1 < n2)),
        (Value::Number(n1), expr::BinaryOp::LessEqual, Value::Number(n2)) => {
            Ok(Value::Bool(n1 <= n2))
        }
        (Value::Number(n1), expr::BinaryOp::Greater, Value::Number(n2)) => Ok(Value::Bool(n1 > n2)),
        (Value::Number(n1), expr::BinaryOp::GreaterEqual, Value::Number(n2)) => {
                 Ok(Value::Bool(n1 >= n2))
        }
        (Value::Number(n1), expr::BinaryOp::Plus, Value::Number(n2)) => Ok(Value::Number(n1 + n2)),
        (Value::Number(n1), expr::BinaryOp::Minus, Value::Number(n2)) => Ok(Value::Number(n1 - n2)),
        (Value::Number(n1), expr::BinaryOp::Star, Value::Number(n2)) => Ok(Value::Number(n1 * n2)),
        (Value::Number(n1), expr::BinaryOp::Slash, Value::Number(n2)) => Ok(Value::Number(n1 / n2)),
        (Value::String(s1), expr::BinaryOp::EqualEqual, Value::String(s2)) => {
            Ok(Value::Bool(s1 == s2))
        }
        (Value::String(s1), expr::BinaryOp::NotEqual, Value::String(s2)) => {
            Ok(Value::Bool(s1 != s2))
        }
        (Value::String(s1), expr::BinaryOp::Less, Value::String(s2)) => Ok(Value::Bool(s1 < s2)),
        (Value::String(s1), expr::BinaryOp::LessEqual, Value::String(s2)) => {
            Ok(Value::Bool(s1 <= s2))
        }
        (Value::String(s1), expr::BinaryOp::Greater, Value::String(s2)) => Ok(Value::Bool(s1 > s2)),
        (Value::String(s1), expr::BinaryOp::GreaterEqual, Value::String(s2)) => {
            Ok(Value::Bool(s1 >= s2))
        }
        (Value::String(s1), expr::BinaryOp::Plus, Value::String(s2)) => {
            Ok(Value::String(format!("{}{}", s1, s2)))
        }
        (Value::Nil, exp::BinaryOp::EqualEqual, Value::Nil) => Ok(Value::Bool(true)),
        (Value::Nil, exp::BinaryOp::NotEqual, Value::Nil) => Ok(Value::Bool(false)),
        (Value::Nil, exp::BinaryOp::EqualEqual, _) => Ok(Value::Bool(false)),
        (_, expr::BinaryOp::EqualEqual, Value::Nil) => Ok(Value::Bool(false)),
        (Value::Nil, expr::BinaryOp::NotEqual, _) => Ok(Value::Bool(true)),
        (_, expr::BinaryOp::NotEqual, Value::Nil) => Ok(Value::Bool(true)),
        (Value::Bool(b1), expr::BinaryOp::EqualEqual, Value::Bool(b2)) => Ok(Value::Bool(b1 == b2)),
        (Value::Bool(b1), expr::BinaryOp::NotEqual, Value::Bool(b2)) => Ok(Value::Bool(b1 != b2)),
        (_, _, _) => Err(format!("Invalid operands in expression ({:?}, {:?}, {:?})")),
            lhs, op, rhs
    }
                ))
        }
}
