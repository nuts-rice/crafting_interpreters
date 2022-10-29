use crate::bytecode;
use crate::native_functions;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fmt;

#[allow(dead_code)]
pub fn disassemble_chunk(chunk: &bytecode::Chunk, name: &str) {
    if name.len() > 0 {
        println!("==== {} ====", name);
    }
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
            bytecode::Op::GetUpVal(idx) => format!("OP_GET_UPVAL idx={}", *idx),
            bytecode::Op::SetUpVal(idx) => format!("OP_SET_UPVAL idx={}", *idx),
            bytecode::Op::Jump(offset) => format!("OP_JUMP {}", *offset),
            bytecode::Op::Loop(offset) => format!("OP_LOOP {}", *offset),
            bytecode::Op::Call(arg_count) => format!("OP_CALL {}", *arg_count),
            bytecode::Op::Closure(idx, upvals) => {
                format!(
                    "OP_CLOSURE {:?} (idx={}, upvals={:?})",
                    chunk.constants[*idx], *idx, upvals
                )
            }
        };
        println!(
            "{0: <04}  {1: <30} {2: <30}",
            idx,
            formatted_op,
            format!("line: {}", lineno.value)
        );
    }
}

fn disassemble_builtin(args: Vec<bytecode::Value>) -> Result<bytecode::Value, String> {
    match &args[0] {
        bytecode::Value::Function(closure) => {
            disassemble_chunk(&closure.function.chunk, "");
            Ok(bytecode::Value::Nil)
        }
        _ => Err(format!(
            "Expected function, got {:?}.",
            bytecode::type_of(&args[0])
        )),
    }
}

pub struct Interpreter {
    frames: Vec<CallFrame>,
    stack: Vec<bytecode::Value>,
    output: Vec<String>,
    globals: HashMap<String, bytecode::Value>,
    upvalues: Vec<bytecode::Upvalue>,
}

impl Default for Interpreter {
    fn default() -> Interpreter {
        let mut res = Interpreter {
            frames: Default::default(),
            stack: Default::default(),
            output: Default::default(),
            globals: Default::default(),
        };
        res.stack.reserve(256);
        res.frames.reserve(64);
        res.globals.insert(
            String::from("disassemble"),
            bytecode::Value::NativeFunction(bytecode::NativeFunction {
                arity: 1,
                name: String::from("disassemble"),
                func: disassemble_builtin,
            }),
        );
        res.globals.insert(
            String::from("clock"),
            bytecode::Value::NativeFunction(bytecode::NativeFunction {
                arity: 0,
                name: String::from("clock"),
                func: native_functions::clock,
            }),
        );
        res.globals.insert(
            String::from("exponent"),
            bytecode::Value::NativeFunction(bytecode::NativeFunction {
                arity: 1,
                name: String::from("exponent"),
                func: native_functions::exponent,
            }),
        );
        res.globals.insert(
            String::from("sqrt"),
            bytecode::Value::NativeFunction(bytecode::NativeFunction {
                arity: 1,
                name: String::from("sqrt"),
                func: native_functions::sqrt,
            }),
        );

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

#[derive(Default)]
struct CallFrame {
    closure: bytecode::Closure,
    ip: usize,
    slots_offset: usize,
}

impl CallFrame {
    fn next_op(&mut self) -> (bytecode::Op, bytecode::Lineno) {
        let res = self.closure.function.chunk.code[self.ip].clone();
        self.ip += 1;
        res
    }

    fn read_constant(&self, idx: usize) -> &bytecode::Value {
        &self.closure.function.chunk.constants[idx]
    }
}

impl Interpreter {
    //Callframes basically a layer of abstraction yeah
    pub fn interpret(&mut self, func: bytecode::Function) -> Result<(), InterpreterError> {
        self.stack
            .push(bytecode::Value::Function(bytecode::Closure {
                function: func.clone(),
                upvalues: Vec::new(),
            }));
        self.frames.push(CallFrame {
            closure: bytecode::Closure { function: func },
            ip: 0,
            slots_offset: 1,
        });
        self.run()
    }

    fn frame_mut(&mut self) -> &mut CallFrame {
        let frames_len = self.frames.len();
        &mut self.frames[frames_len - 1]
    }

    fn frame(&self) -> &CallFrame {
        &self.frames[self.frames.len() - 1]
    }

    fn run(&mut self) -> Result<(), InterpreterError> {
        loop {
            if self.frames.len() == 0
                || self.frame().ip >= self.frame().closure.function.chunk.code.len()
            {
                return Ok(());
            }
            let op = self.next_op();
            println!("{:?}", op);
            match op {
                (bytecode::Op::Return, _) => {
                    let result = self.pop_stack();
                    let num_to_pop = self.stack.len() - self.frame().slots_offset
                        + usize::from(self.frame().closure.function.arity);
                    self.frames.pop();
                    self.pop_stack_n_times(num_to_pop);
                    if self.frames.is_empty() {
                        self.pop_stack();
                        return Ok(());
                    }
                    self.stack.push(result);
                }
                (bytecode::Op::Closure(idx, _), _) => {
                    let constant = self.read_constant(idx).clone();

                    if let bytecode::Value::Function(closure) = constant {
                        self.stack.push(bytecode::Value::Function(closure));
                    } else {
                        panic!(
                            "when interpereting closure, expected function, found {:?}",
                            bytecode::type_of(&constant)
                        );
                    }
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
                    let slots_offset = self.frame().slots_offset;
                    let val = self.stack[slots_offset + idx].clone();
                    self.stack.push(val);
                }
                (bytecode::Op::JumpIfFalse(offset), _) => {
                    if Interpreter::is_falsey(&self.peek()) {
                        self.frame_mut().ip += offset;
                    }
                }
                (bytecode::Op::Jump(offset), _) => {
                    self.frame_mut().ip += offset;
                }
                (bytecode::Op::Loop(offset), _) => {
                    self.frame_mut().ip -= offset;
                }
                (bytecode::Op::Call(arg_count), _) => {
                    self.call_value(self.peek_by(arg_count.into()).clone(), arg_count)?;
                }
                (bytecode::Op::SetLocal(idx), _) => {
                    let val = self.peek();
                    let slots_offset = self.frame().slots_offset;
                    self.stack[slots_offset + idx] = val.clone();
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
                (bytecode::Op::GetUpVal(_), _) => {
                    let upval = self.frame().closure.upvals[idx].clone();
                    let val = match &*upval.borrow() {
                        bytecode::Upvalue::Closed(value) => value.clone(),
                        bytecode::Upvalue::Open(stack_idx) => self.stack[*stack_idx].clone(),
                    };
                    self.stack.push(val);
                }
                (bytecode::Op::SetUpVal(_), _) => {
                    let new_value = self.peek().clone();
                    let upval = self.frame().closure.upvals[idx].clone();
                    match &mut *upval.borrow_mut() {
                        bytecode::Upvalue::Closed(value) => *value = new_value,
                        bytecode::Upvalue::Open(stack_idx) => self.stack[stack_idx] = new_value,
                    };
                }
            }
        }
    }

    fn call_value(
        &mut self,
        val_to_call: bytecode::Value,
        arg_count: u8,
    ) -> Result<(), InterpreterError> {
        match val_to_call {
            bytecode::Value::Function(func) => {
                self.call(func, arg_count)?;
                Ok(())
            }
            bytecode::Value::NativeFunction(native_func) => {
                self.native_call(native_func, arg_count)?;
                Ok(())
            }
            _ => Err(InterpreterError::Runtime(format!(
                "attempted to call non-callable value of type {:?}",
                bytecode::type_of(&val_to_call)
            ))),
        }
    }

    fn call(&mut self, closure: bytecode::Closure, arg_count: u8) -> Result<(), InterpreterError> {
        let func = &closure.function;
        if arg_count != func.arity {
            return Err(InterpreterError::Runtime(format!(
                "Expected {} arguments but found {}.",
                func.arity, arg_count
            )));
        }

        self.frames.push(CallFrame::default());
        let mut frame = self.frames.last_mut().unwrap();
        frame.closure = closure;
        frame.slots_offset = self.stack.len() - usize::from(arg_count);
        Ok(())
    }

    fn native_call(
        &mut self,
        native_func: bytecode::NativeFunction,
        arg_count: u8,
    ) -> Result<(), InterpreterError> {
        if arg_count != native_func.arity {
            return Err(InterpreterError::Runtime(format!(
                "Expected {} arguments but found {}",
                native_func.arity, arg_count
            )));
        }
        let mut args = Vec::new();
        for _ in 0..arg_count {
            args.push(self.pop_stack())
        }
        args.reverse();
        let args = args;
        self.pop_stack();

        match (native_func.func)(args) {
            Ok(result) => {
                self.stack.push(result);
                Ok(())
            }
            Err(err) => Err(InterpreterError::Runtime(format!(
                "{}: {}",
                native_func.name, err
            ))),
        }
    }

    fn print_val(&mut self, val: &bytecode::Value) {
        let output = format!("{:?}", val);
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

    fn pop_stack_n_times(&mut self, num_to_pop: usize) {
        for _ in 0..num_to_pop {
            self.pop_stack();
        }
    }

    fn next_op(&mut self) -> (bytecode::Op, bytecode::Lineno) {
        self.frame_mut().next_op()
    }

    fn is_falsey(val: &bytecode::Value) -> bool {
        match val {
            bytecode::Value::Nil => true,
            bytecode::Value::Bool(b) => !*b,
            bytecode::Value::Number(f) => *f == 0.0,
            bytecode::Value::Function(_) => false,
            bytecode::Value::NativeFunction(_) => false,
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
        &self.frame().read_constant(idx)
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
        let func_or_err = Compiler::compile(String::from("print 42 * 12;"));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn compiler_test_2() {
        let func_or_err = Compiler::compile(String::from("print -2 * 3 + (-4 / 2);"));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn var_decl_test() {
        let func_or_err = Compiler::compile(String::from("var x = 2;"));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn var_decl_nil() {
        let func_or_err = Compiler::compile(String::from("var x;"));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }
    #[test]
    fn test_var_reading() {
        let func_or_err = Compiler::compile(String::from("var x = 2; print x;"));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from("{\n\
                                                                   var breakfast = \"beignets\";\n\
                                                                   var beverage = \"cafe au lait\";\n\
                                                                   breakfast = \"beignets with \" + beverage;\n\
                                                                   print breakfast;\n\
                                                                   }\n",
                                                                   ));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from("{var x = 2; print x;}"));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from(
            "var x = 0;\n\
                var y = 1;\n\
                if (x) {\n\
                    print x;\n\
                }\n\
                if (y) {\n\
                    print y;\n\
                }",
        ));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from(
            "var x = 0;\n\
             if (x) {\n\
               print \"hello\";\n\
             } else {\n\
               print \"goodbye\";\n\
             }",
        ));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let result = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from(
            "var x = 1;\n\
             if (x) {\n\
               print \"hello\";\n\
             } else {\n\
               print \"goodbye\";\n\
             }",
        ));

        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from(
            "{var x = 0;\n\
             var sum = 0;\n\
             while (x < 100) {\n\
               x = x + 1;\n\
               sum = sum + x;\n\
             }\n\
             print sum;}",
        ));

        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
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
        let func_or_err = Compiler::compile(String::from(
            "var x = false;\n\
             var y = true;\n\
             if (y and x) {\n\
               print \"cat\";\n\
             } else {\n\
               print \"dog\";\n\
             }\n",
        ));

        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
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
    fn or_test_1() {
        let func_or_err = Compiler::compile(String::from(
            "var x = false;\n\
             var y = true;\n\
             if (y or x) {\n\
                 print \"cat\";\n\
             } else { \n\
                print \"dog\";\n\
             }\n",
        ));

        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["cat"]);
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
        let func_or_err = Compiler::compile(String::from(
            "{\n\
                    var fact = 1;\n\
                    for (var i = 1; i <= 10; i = i + 1) {\n\
                        fact = fact * i;\n\
                    }\n\
                    print fact;\n\
                    }",
        ));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
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

    #[test]
    fn functions_test_1() {
        let func_or_err = Compiler::compile(String::from(
            "fun function1() {\n\
                    print \"function test success\";\n\
                }\n\
                \n\
                print function1;\n",
        ));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn functions_test_2() {
        let func_or_err = Compiler::compile(String::from(
            "fun f(x, y) {\n\
                    print x + y;\n\
                }\n\
                \n\
                print f;\n",
        ));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }

    #[test]
    fn functions_test_3() {
        let func_or_err = Compiler::compile(String::from(
            "fun f() {\n\
                    return;\n\
                }\n\
                \n\
                print f();\n",
        ));
        match func_or_err {
            Ok(_) => {}
            Err(err) => panic!("{}", err),
        }
    }
    #[test]
    fn native_functions_test() {
        let func_or_err = Compiler::compile(String::from(
            "fun fib(n) {\n\
               if (n < 2) return n;\n\
               return fib(n - 2) + fib(n - 1);\n\
             }\n\
             \n\
             var start = clock();\n\
             print fib(5);\n\
             print clock() - start;",
        ));

        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output.len(), 2);
                        assert_eq!(interp.output[0], "5");
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
