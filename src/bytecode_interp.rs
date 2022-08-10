use crate::bytecode;
use crate::value;

#[allow(dead_code)]
pub fn dissassemble_chunk(chunk: &bytecode::Chunk, name: &str) {
    println!("== {} ==", name);
    for (idx, (op, lineno)) in chunk.code.iter().enumerate() {
        print!("{:04} ", idx);
        match op {
            bytecode::Op::Return => print!("OP_RETURN"),
            bytecode::Op::Constant(const_idx) => {
                print!("OP_CONSTANT {:?}", chunk.constants[*const_idx])
            }
            bytecode::Op::Nil => print!("OP_NIL"),
            bytecode::Op::True => print!("OP_TRUE"),
            bytecode::Op::False => print!("OP_FALSE"),
            bytecode::Op::Negate => print!("OP_NEGATE"),
            bytecode::Op::Add => print!("OP_ADD"),
            bytecode::Op::Subtract => print!("OP_SUBTRACT"),
            bytecode::Op::Multiply => print!("OP_MULTIPLY"),
            bytecode::Op::Divide => print!("OP_DIVIDE"),
            bytecode::Op::Not => print!("OP_NOT"),
            bytecode::Op::Equal => print!("OP_EQUAL"),
            bytecode::Op::Greater => print!("OP_GREATER"),
            bytecode::Op::Less => print!("OP_LESS"),
        }
        println!("\t\tline {}", lineno.value);
    }
}

pub struct Interpreter {
    chunk: bytecode::Chunk,
    ip: usize,
    stack: Vec<value::Value>,
}

impl Default for Interpreter {
    fn default() -> Interpreter {
        let mut res = Interpreter {
            chunk: Default::default(),
            ip: 0,
            stack: Vec::new(),
        };
        res.stack.reserve(256);
        res
    }
}

enum Binop {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Eq, PartialEq, Debug, Clone)]
#[allow(dead_code)]
pub enum InterpreterError {
    Compile(String),
    Runtime(String),
}

impl Interpreter {
    pub fn interpret(&mut self, chunk: bytecode::Chunk) -> Result<(), InterpreterError> {
        self.chunk = chunk;
        self.run()
    }

    fn run(&mut self) -> Result<(), InterpreterError> {
        loop {
            match self.next_op() {
                (bytecode::Op::Return, _) => {
                    return Ok(());
                }
                (bytecode::Op::Constant(idx), _) => {
                    let constant = self.read_constant(idx);
                    self.stack.push(constant);
                }
                (bytecode::Op::Nil, _) => {
                    self.stack.push(value::Value::Nil);
                }
                (bytecode::Op::True, _) => {
                    self.stack.push(value::Value::Bool(true));
                }
                (bytecode::Op::False, _) => {
                    self.stack.push(value::Value::Bool(false));
                }
                (bytecode::Op::Negate, lineno) => {
                    let top_stack = self.peek();
                    let maybe_number = Interpreter::as_number(top_stack);

                    match maybe_number {
                        Some(to_negate) => {
                            self.pop_stack();
                            self.stack.push(value::Value::Number(-to_negate));
                        }
                        None => {
                            return Err(InterpreterError::Runtime(format!(
                                        "Invalid operand to unary op negate. Expected number, found {:?} at line {}",
                                        value::type_of(top_stack), lineno.value
                                        )))

                        }
                    }
                }
                (bytecode::Op::Add, lineno) => match self.numeric_binop(Binop::Add, lineno) {
                    Ok(()) => {}
                    Err(err) => return Err(err),
                },
                (bytecode::Op::Subtract, lineno) => match self.numeric_binop(Binop::Sub, lineno) {
                    Ok(()) => {}
                    Err(err) => return Err(err),
                },
                (bytecode::Op::Multiply, lineno) => match self.numeric_binop(Binop::Mul, lineno) {
                    Ok(()) => {}
                    Err(err) => return Err(err),
                },
                (bytecode::Op::Divide, lineno) => match self.numeric_binop(Binop::Div, lineno) {
                    Ok(()) => {}
                    Err(err) => return Err(err),
                },
                (bytecode::Op::Not, lineno) => {
                    let top_stack = self.peek();
                    let maybe_bool = Interpreter::as_bool(top_stack);
                    match maybe_bool {
                        Some(b) => {
                            self.pop_stack();
                            self.stack.push(value::Value::Bool(!b));
                        }
                        None => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected operand boolean, found {:?} at line {}",
                                value::type_of(top_stack),
                                lineno.value
                            )))
                        }
                    }
                }
                (bytecode::Op::Equal, _) => {
                    let val1 = self.pop_stack();
                    let val2 = self.pop_stack();
                    self.stack
                        .push(value::Value::Bool(Interpreter::values_equal(val1, val2)));
                }
                (bytecode::Op::Greater, lineno) => {
                    let val1 = self.peek_by(0);
                    let val2 = self.peek_by(1);
                    match (val1, val2) {
                        (value::Value::Number(n1), value::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack.push(value::Value::Bool(n1 > n2));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected numbers, found {:?} and {:?} at line {}",
                                value::type_of(val1),
                                value::type_of(val2),
                                lineno.value
                            )))
                        }
                    }
                }
                (bytecode::Op::Less, lineno) => {
                    let val1 = self.peek_by(0);
                    let val2 = self.peek_by(1);
                    match (val1, val2) {
                        (value::Value::Number(n1), value::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack.push(value::Value::Bool(n1 < n2));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected numbers, found {:?} and {:?} at line {}",
                                value::type_of(val1),
                                value::type_of(val2),
                                lineno.value
                            )))
                        }
                    }
                }
            }
        }
    }

    fn values_equal(val1: value::Value, val2: value::Value) -> bool {
        match (val1, val2) {
            (value::Value::Number(n1), value::Value::Number(n2)) => (n1 - n2).abs() < f64::EPSILON,
            (value::Value::Bool(b1), value::Value::Bool(b2)) => b1 == b2,
            (value::Value::Nil, value::Value::Nil) => true,
            (_, _) => false,
        }
    }

    fn numeric_binop(
        &mut self,
        binop: Binop,
        lineno: bytecode::Lineno,
    ) -> Result<(), InterpreterError> {
        let top_stack = self.peek();
        let maybe_left = Interpreter::as_number(top_stack);
        match maybe_left {
            Some(left) => {
                let top_stack = self.peek();
                let maybe_right = Interpreter::as_number(top_stack);
                match maybe_right {
                    Some(right) => {
                        self.pop_stack();
                        self.stack
                            .push(value::Value::Number(Interpreter::apply_numeric_binop(
                                left, right, binop,
                            )));
                        Ok(())
                    }
                    None => Err(InterpreterError::Runtime(format!(
                        "invalid operand of type {:?} at line {}",
                        value::type_of(top_stack),
                        lineno.value
                    ))),
                }
            }
            None => Err(InterpreterError::Runtime(format!(
                "Invalid operand of type {:?} at line {}",
                value::type_of(top_stack),
                lineno.value
            ))),
        }
    }

    fn apply_numeric_binop(left: f64, right: f64, binop: Binop) -> f64 {
        match binop {
            Binop::Add => left + right,
            Binop::Sub => left - right,
            Binop::Mul => left * right,
            Binop::Div => left / right,
        }
    }

    fn pop_stack(&mut self) -> value::Value {
        match self.stack.pop() {
            Some(val) => val,
            None => panic!("can't pop empty stack"),
        }
    }

    fn next_op(&mut self) -> (bytecode::Op, bytecode::Lineno) {
        let res = self.chunk.code[self.ip];
        self.ip += 1;
        res
    }

    fn peek(&self) -> value::Value {
        self.peek_by(0)
    }

    fn peek_by(&self, n: usize) -> value::Value {
        self.stack[self.stack.len() - n - 1]
    }

    fn read_constant(&self, idx: usize) -> value::Value {
        self.chunk.constants[idx]
    }

    fn as_bool(val: value::Value) -> Option<bool> {
        match val {
            value::Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    fn as_number(val: value::Value) -> Option<f64> {
        match val {
            value::Value::Number(f) => Some(f),
            _ => None,
        }
    }
}

//walk through these with rust gdb
#[cfg(test)]
mod tests {

    use crate::compiler::Compiler;

    #[test]
    fn compiler_test_1() {
        let code_or_err = Compiler::default().compile(String::from("(-1 + 2) + 3 - -4"));
        assert!(code_or_err.is_ok())
    }
}
