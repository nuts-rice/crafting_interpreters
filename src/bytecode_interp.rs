use crate::bytecode;
use std::collections::HashMap;
use std::fmt;

#[allow(dead_code)]
pub fn dissassemble_chunk(chunk: &bytecode::Chunk, name: &str) {
    println!("==== {} ====", name);
    println!("==== constants ====");
    for (idx, constant) in chunk.constants.iter().enumerate() {
        println!("{:<4} {:?}", idx, constant);
    }
    println!("==== code ====");
    for (idx, (op, lineno)) in chunk.code.iter().enumerate() {
        let formatted_op = match op {
            bytecode::Op::Return => format!("OP_RETURN"),
            bytecode::Op::Constant(const_idx) => {
                format!(
                    "OP_CONSTANT {:?} (idx={})",
                    chunk.constants[*const_idx], *const_idx
                )
            }
            bytecode::Op::Nil => format!("OP_NIL"),
            bytecode::Op::True => format!("OP_TRUE"),
            bytecode::Op::False => format!("OP_FALSE"),
            bytecode::Op::Negate => format!("OP_NEGATE"),
            bytecode::Op::Add => format!("OP_ADD"),
            bytecode::Op::Subtract => format!("OP_SUBTRACT"),
            bytecode::Op::Multiply => format!("OP_MULTIPLY"),
            bytecode::Op::Divide => format!("OP_DIVIDE"),
            bytecode::Op::Not => format!("OP_NOT"),
            bytecode::Op::Equal => format!("OP_EQUAL"),
            bytecode::Op::Greater => format!("OP_GREATER"),
            bytecode::Op::Less => format!("OP_LESS"),
            bytecode::Op::Print => format!("OP_PRINT"),
            bytecode::Op::Pop => format!("OP_POP"),
            bytecode::Op::DefineGlobal(global_idx) => format!(
                "OP_DEFINE_GLOBAL {:?} (idx={})",
                chunk.constants[*global_idx], *global_idx
            ),
            bytecode::Op::GetGlobal(global_idx) => format!(
                "OP_GET_GLOBAL {:?} (idx={})",
                chunk.constants[*global_idx], *global_idx
            ),
            bytecode::Op::SetGlobal(global_idx) => format!(
                "OP_SET_GLOBAL {:?} (idx={})",
                chunk.constants[*global_idx], *global_idx
            ),
            bytecode::Op::GetLocal(idx) => format!("OP_GET_LOCAL (idx={})", *idx),
            bytecode::Op::JumpIfFalse(loc) => format!("OP_JUMP_IF_FALSE {}", *loc),
            bytecode::Op::SetLocal(idx) => format!("OP_SET_LOCAL (idx={})", *idx),
            bytecode::Op::Jump(offset) => format!("OP_JUMP {}", *offset),
            bytecode::Op::Loop(offset) => format!("OP_LOOP {}", *offset),
        };
        println!(
            "{0: <04}  {1: <30} {2: <30}",
            idx,
            formatted_op,
            format!("line: {}", lineno.value)
        );
    }
}

pub struct Interpreter {
    chunk: bytecode::Chunk,
    ip: usize,
    stack: Vec<bytecode::Value>,
    output: Vec<String>,
    globals: HashMap<String, bytecode::Value>,
}

impl Default for Interpreter {
    fn default() -> Interpreter {
        let mut res = Interpreter {
            chunk: Default::default(),
            ip: 0,
            stack: Vec::new(),
            output: Vec::new(),
            globals: HashMap::new(),
        };
        res.stack.reserve(256);
        res
    }
}

#[allow(dead_code)]
#[derive(Debug)]
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
            if self.ip >= self.chunk.code.len() {
                return Ok(());
            }
            let op = self.next_op();
            match op {
                (bytecode::Op::Return, _) => {
                    return Ok(());
                }
                (bytecode::Op::Constant(idx), _) => {
                    let constant = self.read_constant(idx).clone();
                    self.stack.push(constant);
                }
                (bytecode::Op::Nil, _) => {
                    self.stack.push(bytecode::Value::Nil);
                }
                (bytecode::Op::True, _) => {
                    self.stack.push(bytecode::Value::Bool(true));
                }
                (bytecode::Op::False, _) => {
                    self.stack.push(bytecode::Value::Bool(false));
                }
                (bytecode::Op::Negate, lineno) => {
                    let top_stack = self.peek();
                    let maybe_number = Interpreter::extract_number(top_stack);

                    match maybe_number {
                        Some(to_negate) => {
                            self.pop_stack();
                            self.stack.push(bytecode::Value::Number(-to_negate));
                        }
                        None => {
                            return Err(InterpreterError::Runtime(format!(
                                        "Invalid operand to unary op negate. Expected number, found {:?} at line {}",
                                        bytecode::type_of(top_stack), lineno.value
                                        )))

                        }
                    }
                }
                (bytecode::Op::Add, lineno) => {
                    let val1 = self.peek_by(0).clone();
                    let val2 = self.peek_by(1).clone();
                    match (&val1, &val2) {
                        (bytecode::Value::Number(n1), bytecode::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack.push(bytecode::Value::Number(n1 + n2));
                        }
                        (bytecode::Value::String(s1), bytecode::Value::String(s2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack
                                .push(bytecode::Value::String(format!("{}{}", s2, s1)));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Invalid operands of type {:?} and {:?} in add expression \
                                        both operands must be number or string (line={})",
                                bytecode::type_of(&val1),
                                bytecode::type_of(&val2),
                                lineno.value
                            )))
                        }
                    }
                }
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
                (bytecode::Op::Print, _) => {
                    let to_print = self.peek().clone();
                    self.print_val(&to_print);
                }
                (bytecode::Op::Pop, _) => {
                    self.pop_stack();
                }
                (bytecode::Op::DefineGlobal(idx), _) => {
                    if let bytecode::Value::String(name) = self.read_constant(idx).clone() {
                        let val = self.pop_stack();
                        self.globals.insert(name, val);
                    } else {
                        panic!(
                            "expected string when defining global, found {:?}",
                            bytecode::type_of(self.read_constant(idx))
                        );
                    }
                }
                (bytecode::Op::GetGlobal(idx), lineno) => {
                    if let bytecode::Value::String(name) = self.read_constant(idx) {
                        match self.globals.get(name) {
                            Some(val) => {
                                self.stack.push(val.clone());
                            }
                            None => {
                                return Err(InterpreterError::Runtime(format!(
                                    "undefined variable '{}' at line {}",
                                    name, lineno.value
                                )));
                            }
                        }
                    } else {
                        panic!(
                            "expected string when defining global, found {:?}",
                            bytecode::type_of(self.read_constant(idx))
                        );
                    }
                }
                (bytecode::Op::SetGlobal(idx), lineno) => {
                    if let bytecode::Value::String(name) = self.read_constant(idx).clone() {
                        if self.globals.contains_key(&name) {
                            let val = self.peek().clone();
                            self.globals.insert(name, val);
                        } else {
                            return Err(InterpreterError::Runtime(format!(
                                "Use of undefined var {} in setitem expression at line {}.",
                                name, lineno.value
                            )));
                        }
                    } else {
                        panic!(
                            "expected string when stting globale, found {:?}",
                            bytecode::type_of(self.read_constant(idx))
                        );
                    }
                }
                (bytecode::Op::GetLocal(idx), _) => {
                    let val = self.stack[idx].clone();
                    self.stack.push(val);
                }
                (bytecode::Op::JumpIfFalse(offset), _) => {
                    if Interpreter::is_falsey(&self.peek()) {
                        self.ip += offset;
                    }
                }
                (bytecode::Op::Jump(offset), _) => {
                    self.ip += offset;
                }
                (bytecode::Op::Loop(offset), _) => {
                    self.ip -= offset;
                }
                (bytecode::Op::SetLocal(idx), _) => {
                    let val = self.peek();
                    self.stack[idx] = val.clone();
                }

                (bytecode::Op::Not, lineno) => {
                    let top_stack = self.peek();
                    let maybe_bool = Interpreter::extract_bool(top_stack);
                    match maybe_bool {
                        Some(b) => {
                            self.pop_stack();
                            self.stack.push(bytecode::Value::Bool(!b));
                        }
                        None => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected operand boolean, found {:?} at line {}",
                                bytecode::type_of(top_stack),
                                lineno.value
                            )))
                        }
                    }
                }
                (bytecode::Op::Equal, _) => {
                    let val1 = self.pop_stack();
                    let val2 = self.pop_stack();
                    self.stack
                        .push(bytecode::Value::Bool(Interpreter::values_equal(
                            &val1, &val2,
                        )));
                }
                (bytecode::Op::Greater, lineno) => {
                    let val1 = self.peek_by(0).clone();
                    let val2 = self.peek_by(1).clone();
                    match (&val1, &val2) {
                        (bytecode::Value::Number(n1), bytecode::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();

                            self.stack.push(bytecode::Value::Bool(n2 > n1));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected numbers, found {:?} and {:?} at line {}",
                                bytecode::type_of(&val1),
                                bytecode::type_of(&val2),
                                lineno.value
                            )))
                        }
                    }
                }
                (bytecode::Op::Less, lineno) => {
                    let val1 = self.peek_by(0).clone();
                    let val2 = self.peek_by(1).clone();
                    match (&val1, &val2) {
                        (bytecode::Value::Number(n1), bytecode::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack.push(bytecode::Value::Bool(n2 < n1));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected numbers, found {:?} and {:?} at line {}",
                                bytecode::type_of(&val1),
                                bytecode::type_of(&val2),
                                lineno.value
                            )))
                        }
                    }
                }
            }
        }
    }

    fn print_val(&mut self, val: &bytecode::Value) {
        let output = match val {
            bytecode::Value::Number(n) => format!("{}", n),
            bytecode::Value::Bool(b) => format!("{}", b),
            bytecode::Value::String(s) => s.to_string(),
            bytecode::Value::Nil => "nil".to_string(),
        };

        println!("{}", output);
        self.output.push(output);
    }

    fn values_equal(val1: &bytecode::Value, val2: &bytecode::Value) -> bool {
        match (val1, val2) {
            (bytecode::Value::Number(n1), bytecode::Value::Number(n2)) => {
                (n1 - n2).abs() < f64::EPSILON
            }
            (bytecode::Value::Bool(b1), bytecode::Value::Bool(b2)) => b1 == b2,
            (bytecode::Value::String(s1), bytecode::Value::String(s2)) => s1 == s2,
            (bytecode::Value::Nil, bytecode::Value::Nil) => true,
            (_, _) => false,
        }
    }

    fn numeric_binop(
        &mut self,
        binop: Binop,
        lineno: bytecode::Lineno,
    ) -> Result<(), InterpreterError> {
        let val1 = self.peek_by(0).clone();
        let val2 = self.peek_by(1).clone();
        match (&val1, &val2) {
            (bytecode::Value::Number(n1), bytecode::Value::Number(n2)) => {
                self.pop_stack();
                self.pop_stack();
                self.stack
                    .push(bytecode::Value::Number(Interpreter::apply_numeric_binop(
                        *n2, *n1, binop,
                    )));
                Ok(())
            }

            _ => Err(InterpreterError::Runtime(format!(
                "Expected numbers in {:?} expression. Found {:?} and {:?} (line={})",
                binop,
                bytecode::type_of(&val1),
                bytecode::type_of(&val2),
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

    fn pop_stack(&mut self) -> bytecode::Value {
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

    fn is_falsey(val: &bytecode::Value) -> bool {
        match val {
            bytecode::Value::Nil => false,
            bytecode::Value::Bool(b) => !*b,
            bytecode::Value::Number(f) => *f == 0.0,
            bytecode::Value::String(s) => s.is_empty(),
        }
    }

    fn peek(&self) -> &bytecode::Value {
        self.peek_by(0)
    }

    fn peek_by(&self, n: usize) -> &bytecode::Value {
        &self.stack[self.stack.len() - n - 1]
    }

    fn read_constant(&self, idx: usize) -> &bytecode::Value {
        &self.chunk.constants[idx]
    }

    fn extract_bool(val: &bytecode::Value) -> Option<bool> {
        match val {
            bytecode::Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn extract_number(val: &bytecode::Value) -> Option<f64> {
        match val {
            bytecode::Value::Number(f) => Some(*f),
            _ => None,
        }
    }
}

//walk through these with rust gdb
#[cfg(test)]
mod tests {

    use crate::bytecode_interp::*;
    use crate::compiler::Compiler;

    #[test]
    fn compiler_test_1() {
        let code_or_err = Compiler::default().compile(String::from("print 42 * 12;"));
        match code_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn compiler_test_2() {
        let code_or_err = Compiler::default().compile(String::from("print -2 * 3 + (-4 / 2);"));
        match code_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn var_decl_test() {
        let code_or_err = Compiler::default().compile(String::from("var x = 2;"));
        match code_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn var_decl_nil() {
        let code_or_err = Compiler::default().compile(String::from("var x;"));
        match code_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }
    #[test]
    fn test_var_reading() {
        let code_or_err = Compiler::default().compile(String::from("var x = 2; print x;"));
        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(code);
                match result {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["2"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn setting_locals_test() {
        let code_or_err = Compiler::default().compile(String::from("{\n\
                                                                   var breakfast = \"beignets\";\n\
                                                                   var beverage = \"cafe au lait\";\n\
                                                                   breakfast = \"beignets with \" + beverage;\n\
                                                                   print breakfast;\n\
                                                                   }\n",
                                                                   ));
        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(code);
                match result {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["beignets with cafe au lait"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn getting_locals_test() {
        let code_or_err = Compiler::default().compile(String::from("{var x = 2; print x;}"));
        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(code);
                match result {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["2"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn if_stmt_test() {
        let code_or_err = Compiler::default().compile(String::from(
            "var x = 0;\n\
                var y = 1;\n\
                if (x) {\n\
                    print x;\n\
                }\n\
                if (y) {\n\
                    print y;\n\
                }",
        ));
        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(code);
                match result {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["1"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }
    #[test]
    fn if_then_else_test_1() {
        let code_or_err = Compiler::default().compile(String::from(
            "var x = 0;\n\
             if (x) {\n\
               print \"hello\";\n\
             } else {\n\
               print \"goodbye\";\n\
             }",
        ));
        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(code);
                match result {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["goodbye"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn if_then_else_test_2() {
        let code_or_err = Compiler::default().compile(String::from(
            "var x = 1;\n\
             if (x) {\n\
               print \"hello\";\n\
             } else {\n\
               print \"goodbye\";\n\
             }",
        ));

        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(code);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["hello"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn while_test() {
        let code_or_err = Compiler::default().compile(String::from(
            "{var x = 0;\n\
             var sum = 0;\n\
             while (x < 100) {\n\
               x = x + 1;\n\
               sum = sum + x;\n\
             }\n\
             print sum;}",
        ));

        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(code);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["5151"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn and_test_1() {
        let code_or_err = Compiler::default().compile(String::from(
            "var x = false;\n\
             var y = true;\n\
             if (y and x) {\n\
               print \"cat\";\n\
             } else {\n\
               print \"dog\";\n\
             }\n",
        ));

        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(code);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["dog"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }
    #[test]
    fn for_test_1() {
        let code_or_err = Compiler::default().compile(String::from(
            "{\n\
                    var fact = 1;\n\
                    for (var i = 1; i <= 10; i = i + 1) {\n\
                        fact = fact * i;\n\
                    }\n\
                    print fact;\n\
                    }",
        ));

        match code_or_err {
            Ok(code) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(code);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["3628800"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{}", err),
        }
    }
}
