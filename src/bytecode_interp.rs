use crate::bytecode;
use crate::garbage_collector;
use crate::value;

use crate::interpreter::Value;
use crate::native_functions;

use std::cell::RefCell;
use std::collections::HashMap;

use std::rc::Rc;

#[allow(dead_code)]
pub fn disassemble_chunk(chunk: &bytecode::Chunk, name: &str) {
    if name.len() > 0 {
        println!("==== {} ====", name);
    }
    println!("==== constants ====");
    for (idx, constant) in chunk.constants.iter().enumerate() {
        println!("{:<4} {:?}", idx, constant);
    }
    println!("\n==== code ====");
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
            bytecode::Op::CloseUpvalue => format!("OP_CLOSE_UPVALUE"),
        };
        println!(
            "{0: <04}  {1: <30} {2: <30}",
            idx,
            formatted_op,
            format!("line: {}", lineno.value)
        );
    }
}

fn disassemble_builtin(
    heap: &garbage_collector::Heap,
    args: Vec<value::Value>,
) -> Result<value::Value, String> {
    match &args[0] {
        value::Value::Function(closure_handle) => {
            let closure = heap.get_closure(*closure_handle);
            disassemble_chunk(&closure.function.chunk, "");
            Ok(value::Value::Nil)
        }
        _ => Err(format!(
            "Expected function, got {:?}.",
            value::type_of(&args[0])
        )),
    }
}

pub struct Interpreter {
    frames: Vec<CallFrame>,
    stack: Vec<value::Value>,
    output: Vec<String>,
    globals: HashMap<String, value::Value>,
    upvalues: Vec<Rc<RefCell<value::Upvalue>>>,
    heap: garbage_collector::Heap,
    gray_gc_stack: Vec<usize>,
}

impl Default for Interpreter {
    fn default() -> Interpreter {
        let mut res = Interpreter {
            frames: Default::default(),
            stack: Default::default(),
            output: Default::default(),
            globals: Default::default(),
            upvalues: Default::default(),
            heap: Default::default(),
            gray_gc_stack: Default::default(),
        };
        res.stack.reserve(256);
        res.frames.reserve(64);
        res.globals.insert(
            String::from("disassemble"),
            value::Value::NativeFunction(value::NativeFunction {
                arity: 1,
                name: String::from("disassemble"),
                func: disassemble_builtin,
            }),
        );
        res.globals.insert(
            String::from("clock"),
            value::Value::NativeFunction(value::NativeFunction {
                arity: 0,
                name: String::from("clock"),
                func: native_functions::clock,
            }),
        );
        res.globals.insert(
            String::from("exponent"),
            value::Value::NativeFunction(value::NativeFunction {
                arity: 1,
                name: String::from("exponent"),
                func: native_functions::exponent,
            }),
        );
        res.globals.insert(
            String::from("sqrt"),
            value::Value::NativeFunction(value::NativeFunction {
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
    fn next_op(&mut self) -> (&bytecode::Op, &bytecode::Lineno) {
        let res = self.closure.function.chunk.code[self.ip];
        self.ip += 1;
        &res
    }

    fn read_constant(&self, idx: usize) -> &bytecode::Constant {
        &self.closure.function.chunk.constants[idx]
    }
}

impl Interpreter {
    //Callframes basically a layer of abstraction yeah
    pub fn interpret(&mut self, func: bytecode::Function) -> Result<(), InterpreterError> {
        self.push_managed_closure(func.clone());
        self.push_call_frame(func);
        self.run()
    }

    fn push_managed_closure(&mut self, func: bytecode::Function) {
        let closure = bytecode::Closure {
            function: func,
            upvalues: Vec::new(),
        };
        let managed_closure = self.heap.manage_closure(closure);
        self.stack
            .push(bytecode::Constant::Function(managed_closure));
    }

    fn push_call_frame(&mut self, func: bytecode::Function) {
        let call_frame = CallFrame {
            closure: bytecode::Closure {
                upvalues: Vec::new(),
                function: func,
            },
            ip: 0,
            slots_offset: 1,
        };
        self.frames.push(call_frame);
    }

    fn frame_mut(&mut self) -> &mut CallFrame {
        let last_frame_index = self.frames.len() - 1;
        &mut self.frames[last_frame_index]
    }

    fn frame(&self) -> &CallFrame {
        let last_frame_index = self.frames.len() - 1;
        &self.frames[last_frame_index]
    }

    fn run(&mut self) -> Result<(), InterpreterError> {
        loop {
            if self.frames.is_empty()
                || self.frame().ip >= self.frame().closure.function.chunk.code.len()
            {
                return Ok(());
            }
            let op = self.next_op();
            println!("{:?}", op);
            match op {
                (bytecode::Op::Return, _) => {
                    let result = self.pop_stack();
                    for idx in self.frame().slots_offset..self.stack.len() {
                        self.close_upvals(idx);
                    }
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
                (bytecode::Op::Closure(idx, upvals), _) => {
                    let constant = self.read_constant(idx);

                    if let value::Value::Function(closure_handle) = constant {
                        let closure = self.get_closure(&closure_handle).clone();
                        let upvalues = upvals
                            .iter()
                            .map(|upval| match upval {
                                bytecode::UpvalueLoc::Upvalue(idx) => {
                                    self.frame().closure.upvalues[*idx].clone()
                                }
                                bytecode::UpvalueLoc::Local(idx) => {
                                    if let Some(upval) = self.find_open_upval(*idx) {
                                        upval
                                    } else {
                                        let index = self.frame().slots_offset + *idx;
                                        let upval =
                                            Rc::new(RefCell::new(value::Upvalue::Open(index)));
                                        self.upvalues.push(upval.clone());
                                        upval
                                    }
                                }
                            })
                            .collect();
                        self.stack
                            .push(value::Value::Function(self.heap.manage_closure(
                                bytecode::Closure {
                                    function: closure.function,
                                    upvalues,
                                },
                            )));
                    } else {
                        panic!(
                            "expected function for closure, found {:?}",
                            value::type_of(&constant)
                        );
                    }
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
                    let maybe_number = Interpreter::extract_number(top_stack);

                    match maybe_number {
                        Some(to_negate) => {
                            self.pop_stack();
                            self.stack.push(bytecode::Constant::Number(-to_negate));
                        }
                        None => {
                            return Err(InterpreterError::Runtime(format!(
                                        "Invalid operand to unary op negate. Expected number, found {:?} at line {}",
                                        value::type_of(top_stack), lineno.value
                                        )))

                        }
                    }
                }
                (bytecode::Op::Add, lineno) => {
                    let val1 = self.peek_by(0).clone();
                    let val2 = self.peek_by(1).clone();
                    match (&val1, &val2) {
                        (value::Value::Number(n1), value::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack.push(value::Value::Number(n1 + n2));
                        }
                        (value::Value::String(s1), value::Value::String(s2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack
                                .push(bytecode::Constant::String(format!("{}{}", s2, s1)));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Invalid operands of type {:?} and {:?} in add expression \
                                        both operands must be number or string (line={})",
                                value::type_of(&val1),
                                value::type_of(&val2),
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
                    if let value::Value::String(name) = self.read_constant(idx).clone() {
                        let val = self.pop_stack();
                        self.globals.insert(self.get_str(&name).clone(), val);
                    } else {
                        panic!(
                            "expected string when defining global, found {:?}",
                            value::type_of(&self.read_constant(idx))
                        );
                    }
                }
                (bytecode::Op::GetGlobal(idx), lineno) => {
                    if let value::Value::String(name_id) = self.read_constant(idx) {
                        match self.globals.get(self.get_str(name_id)) {
                            Some(val) => {
                                self.stack.push(val.clone());
                            }
                            None => {
                                return Err(InterpreterError::Runtime(format!(
                                    "undefined variable '{}' at line {}",
                                    self.get_str(name_id),
                                    lineno.value
                                )));
                            }
                        }
                    } else {
                        panic!(
                            "expected string when defining global, found {:?}",
                            value::type_of(&self.read_constant(idx))
                        );
                    }
                }
                (bytecode::Op::SetGlobal(idx), lineno) => {
                    if let value::Value::String(name) = self.read_constant(idx).clone() {
                        let name_str = self.get_str(&name).clone();
                        if self.globals.contains_key(&name_str) {
                            let val = self.peek().clone();
                            self.globals.insert(name_str, val);
                        } else {
                            return Err(InterpreterError::Runtime(format!(
                                "Use of undefined var {} in setitem expression at line {}.",
                                name_str, lineno.value
                            )));
                        }
                    } else {
                        panic!(
                            "expected string when stting globale, found {:?}",
                            value::type_of(&self.read_constant(idx))
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
                        .push(bytecode::Constant::Bool(Interpreter::values_equal(
                            &val1, &val2,
                        )));
                }
                (bytecode::Op::Greater, lineno) => {
                    let val1 = self.peek_by(0).clone();
                    let val2 = self.peek_by(1).clone();
                    match (&val1, &val2) {
                        (bytecode::Constant::Number(n1), value::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();

                            self.stack.push(bytecode::Constant::Bool(n2 > n1));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected numbers, found {:?} and {:?} at line {}",
                                value::type_of(&val1),
                                value::type_of(&val2),
                                lineno.value
                            )))
                        }
                    }
                }
                (bytecode::Op::Less, lineno) => {
                    let val1 = self.peek_by(0).clone();
                    let val2 = self.peek_by(1).clone();
                    match (&val1, &val2) {
                        (bytecode::Constant::Number(n1), value::Value::Number(n2)) => {
                            self.pop_stack();
                            self.pop_stack();
                            self.stack.push(bytecode::Constant::Bool(n2 < n1));
                        }
                        _ => {
                            return Err(InterpreterError::Runtime(format!(
                                "Expected numbers, found {:?} and {:?} at line {}",
                                value::type_of(&val1),
                                value::type_of(&val2),
                                lineno.value
                            )))
                        }
                    }
                }
                (bytecode::Op::GetUpVal(idx), _) => {
                    let upval = self.frame().closure.upvalues[idx].clone();
                    let val = match &*upval.borrow() {
                        value::Upvalue::Closed(value) => value.clone(),
                        value::Upvalue::Open(stack_idx) => self.stack[*stack_idx].clone(),
                    };
                    self.stack.push(val);
                }
                (bytecode::Op::SetUpVal(_idx), _) => {
                    unimplemented!()
                }
                /*
                    let new_value = self.peek().clone();
                    let upvalue = self.frame().closure.upvalues[idx].clone();
                    match &*upvalue.borrow_mut() {
                        value::Upvalue::Closed(value) => *value = new_value,
                        value::Upvalue::Open(stack_index) => {
                            self.stack[*stack_index] = new_value
                        }
                    };
                }
                */
                (bytecode::Op::CloseUpvalue, _) => {
                    let idx = self.stack.len() - 1;
                    self.close_upvals(idx);
                    self.stack.pop();
                }
            }
        }
    }

    fn call_value(
        &mut self,
        val_to_call: bytecode::Constant,
        arg_count: u8,
    ) -> Result<(), InterpreterError> {
        match val_to_call {
            bytecode::Constant::Function(func) => {
                self.call(func, arg_count)?;
                Ok(())
            }
            bytecode::Constant::NativeFunction(native_func) => {
                self.native_call(native_func, arg_count)?;
                Ok(())
            }
            _ => Err(InterpreterError::Runtime(format!(
                "attempted to call non-callable value of type {:?}",
                value::type_of(&val_to_call)
            ))),
        }
    }

    fn call(&mut self, closure_handle: usize, arg_count: u8) -> Result<(), InterpreterError> {
        let closure = self.get_closure(&closure_handle);
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
        native_func: value::NativeFunction,
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
        let res = (native_func.func)(&self.heap, args);
        match res {
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

    fn print_val(&mut self, val: &bytecode::Constant) {
        let output = format!("{:?}", val);
        println!("{}", output);
        self.output.push(output);
    }

    fn close_upvals(&mut self, index: usize) {
        let val = &self.stack[index];
        for upval in &self.upvalues {
            if upval.borrow().is_open_with_idx(index) {
                upval.replace(value::Upvalue::Closed(val.clone()));
            }
        }
        self.upvalues.retain(|u| u.borrow().is_open());
    }

    fn values_equal(val1: &bytecode::Constant, val2: &value::Value) -> bool {
        match (val1, val2) {
            (bytecode::Constant::Number(n1), value::Value::Number(n2)) => {
                (n1 - n2).abs() < f64::EPSILON
            }
            (bytecode::Constant::Bool(b1), value::Value::Bool(b2)) => b1 == b2,
            (bytecode::Constant::String(s1), value::Value::String(s2)) => s1 == s2,
            (bytecode::Constant::Nil, value::Value::Nil) => true,
            (_, _) => false,
        }
    }

    fn find_open_upval(&self, index: usize) -> Option<Rc<RefCell<value::Upvalue>>> {
        for upval in self.upvalues.iter().rev() {
            if upval.borrow().is_open_with_idx(index) {
                return Some(upval.clone());
            }
        }
        None
    }

    fn numeric_binop(
        &mut self,
        binop: Binop,
        lineno: bytecode::Lineno,
    ) -> Result<(), InterpreterError> {
        let val1 = self.peek_by(0).clone();
        let val2 = self.peek_by(1).clone();
        match (&val1, &val2) {
            (value::Value::Number(n1), value::Value::Number(n2)) => {
                self.pop_stack();
                self.pop_stack();
                self.stack
                    .push(value::Value::Number(Interpreter::apply_numeric_binop(
                        *n2, *n1, binop,
                    )));
                Ok(())
            }

            _ => Err(InterpreterError::Runtime(format!(
                "Expected numbers in {:?} expression. Found {:?} and {:?} (line={})",
                binop,
                value::type_of(&val1),
                value::type_of(&val2),
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

    fn pop_stack(&mut self) -> bytecode::Constant {
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

    fn is_falsey(val: &bytecode::Constant) -> bool {
        match val {
            bytecode::Constant::Nil => true,
            bytecode::Constant::Bool(b) => !*b,
            bytecode::Constant::Number(f) => *f == 0.0,
            bytecode::Constant::Function(_) => false,
            bytecode::Constant::NativeFunction(_) => false,
            bytecode::Constant::String(s) => s.is_empty(),
        }
    }

    fn peek(&self) -> &bytecode::Constant {
        self.peek_by(0)
    }

    fn peek_by(&self, n: usize) -> &bytecode::Constant {
        &self.stack[self.stack.len() - n - 1]
    }

    fn read_constant(&mut self, idx: usize) -> &value::Value {
        let constant = self.frame().read_constant(idx);
        match constant {
            bytecode::Constant::Number(num) => value::Value::Number(num),
            bytecode::Constant::String(s) => value::Value::String(self.heap.manage_closure(s)),
            bytecode::Constant::Function(f) => {
                value::Value::Function(self.heap.manage_closure(bytecode::Closure {
                    function: f.function,
                    upvalues: Vec::new(),
                }))
            }
        }
    }

    fn extract_bool(val: &bytecode::Constant) -> Option<bool> {
        match val {
            bytecode::Constant::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn extract_number(val: &value::Value) -> Option<f64> {
        match val {
            bytecode::Constant::Number(f) => Some(*f),
            _ => None,
        }
    }

    fn get_str(&self, str_handle: usize) -> &String {
        self.heap.get_str(str_handle)
    }

    fn get_closure(&self, closure_handle: usize) -> &bytecode::Closure {
        self.heap.get_closure(closure_handle)
    }

    fn collect_garbage(&mut self) {
        self.heap.unmark();
        self.mark_roots();
        self.heap.sweep();
    }

    fn trace_references(&mut self) {
        loop {
            let maybe_val = self.gray_gc_stack.pop();
            match maybe_val {
                Some(val) => self.blacken_object(val),
                None => break,
            }
        }
    }

    //need specific code statement for each type

    fn blacken_object(&mut self, val: usize) {
        todo!()
    }

    //strings and native functions contain no outgoing refs so dont traverse them
    //closed upvals when closed contain refrence to closed over val. Trace this ref here
    //Functions has ref to Objstring constaing fns name and constant table with refs to other objects
    fn mark_roots(&mut self) {
        todo!()
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
    #[test]
    fn get_upvals_test() {
        let func_or_err = Compiler::compile(String::from(
            "fun outer() {\n\
               var x = \"outside\";\n\
               fun inner() {\n\
                 print x;\n\
               }\n\
               inner();\n\
             }\n\
             outer();",
        ));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["outside"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{:?}", err),
        }
    }

    #[test]
    fn closing_upvals_test() {
        let func_or_err = Compiler::compile(String::from(
            "fun outer() {\n\
               var x = \"outside\";\n\
               fun inner() {\n\
                 print x;\n\
               }\n\
               \n\
               return inner;\n\
            }\n\
            \n\
            var closure = outer();\n\
            closure();",
        ));
        match func_or_err {
            Ok(func) => {
                let mut interp = Interpreter::default();
                let res = interp.interpret(func);
                match res {
                    Ok(()) => {
                        assert_eq!(interp.output, vec!["outside"]);
                    }
                    Err(err) => {
                        panic!("{:?}", err);
                    }
                }
            }
            Err(err) => panic!("{:?}", err),
        }
    }
}
