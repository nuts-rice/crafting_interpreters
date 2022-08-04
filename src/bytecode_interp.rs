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
            bytecode::Op::Negate => print!("OP_NEGATE"),
            bytecode::Op::Add => print!("OP_ADD"),
            bytecode::Op::Subtract => print!("OP_SUBTRACT"),
            bytecode::Op::Multiply => print!("OP_MULTIPLY"),
            bytecode::Op::Divide => print!("OP_DIVIDE"),
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
                (bytecode::Op::Negate, _) => {
                    let to_negate = self.pop_stack();
                    self.stack.push(Interpreter::negate(to_negate)?)
                }
                (bytecode::Op::Add, _) => {
                    let left = self.pop_stack();
                    let right = self.pop_stack();
                    self.stack.push(Interpreter::add(left, right)?)
                }
                (bytecode::Op::Subtract, _) => {
                    let left = self.pop_stack();
                    let right = self.pop_stack();
                    self.stack.push(Interpreter::subtract(left, right)?)
                }
                (bytecode::Op::Multiply, _) => {
                    let left = self.pop_stack();
                    let right = self.pop_stack();
                    self.stack.push(Interpreter::multiply(left, right)?)
                }
                (bytecode::Op::Divide, _) => {
                    let left = self.pop_stack();
                    let right = self.pop_stack();
                    self.stack.push(Interpreter::divide(left, right)?)
                }
            }
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

    fn read_constant(&self, idx: usize) -> value::Value {
        self.chunk.constants[idx]
    }

    fn negate(val: value::Value) -> Result<value::Value, InterpreterError> {
        match val {
            value::Value::Number(num) => Ok(value::Value::Number(-num)),
        }
    }

    fn add(val1: value::Value, val2: value::Value) -> Result<value::Value, InterpreterError> {
        match (val1, val2) {
            (value::Value::Number(num1), value::Value::Number(num2)) => {
                Ok(value::Value::Number(num1 + num2))
            }
        }
    }

    fn subtract(val1: value::Value, val2: value::Value) -> Result<value::Value, InterpreterError> {
        match (val1, val2) {
            (value::Value::Number(num1), value::Value::Number(num2)) => {
                Ok(value::Value::Number(num1 - num2))
            }
        }
    }

    fn multiply(val1: value::Value, val2: value::Value) -> Result<value::Value, InterpreterError> {
        match (val1, val2) {
            (value::Value::Number(num1), value::Value::Number(num2)) => {
                Ok(value::Value::Number(num1 * num2))
            }
        }
    }

    fn divide(val1: value::Value, val2: value::Value) -> Result<value::Value, InterpreterError> {
        match (val1, val2) {
            (value::Value::Number(num1), value::Value::Number(num2)) => {
                Ok(value::Value::Number(num1 / num2))
            }
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
