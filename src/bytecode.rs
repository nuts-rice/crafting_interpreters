use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
#[serde(untagged)]
pub enum Op {
    Return,
    Constant(usize),
    Closure(usize),
    Nil,
    True,
    False,
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
    Not,
    Equal,
    Greater,
    Less,
    Print,
    Pop,
    DefineGlobal(usize),
    GetGlobal(usize),
    SetGlobal(usize),
    GetLocal(usize),
    SetLocal(usize),
    GetUpVal(usize),
    SetUpVal(usize),
    JumpIfFalse(usize),
    Jump(usize),
    Loop(usize),
    Call(u8),
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum Value {
    Number(f64),
    Bool(bool),
    Nil,
    String(String),
    Function(Closure),
    NativeFunction(NativeFunction),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Type {
    Number,
    Bool,
    String,
    Nil,
    Function,
    NativeFunction,
}

#[derive(Default, Clone)]
pub struct Closure {
    pub function: Function,
}

#[derive(Default, Clone)]
pub struct Function {
    pub arity: u8,
    pub chunk: Chunk,
    pub name: String,
}

#[derive(Clone)]
pub struct NativeFunction {
    pub arity: u8,
    pub name: String,
    pub func: fn(Vec<Value>) -> Result<Value, String>,
}

pub fn type_of(value: &Value) -> Type {
    match value {
        Value::Number(_) => Type::Number,
        Value::Bool(_) => Type::Bool,
        Value::String(_) => Type::String,
        Value::Function(_) => Type::Function,
        Value::NativeFunction(_) => Type::NativeFunction,
        Value::Nil => Type::Nil,
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Number(num) => write!(f, "{}", num),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Function(closure) => write!(f, "<fn {}>", closure.function.name),
            Value::NativeFunction(func) => write!(f, "<native func {}>", func.name),
            Value::Nil => write!(f, "nil"),
        }
    }
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Lineno {
    pub value: usize,
}

pub fn Lineno(value: usize) -> Lineno {
    Lineno { value }
}

#[derive(Default, Clone)]
pub struct Chunk {
    pub code: Vec<(Op, Lineno)>,
    pub constants: Vec<Value>,
}

impl Chunk {
    pub fn add_constant_number(&mut self, c: f64) -> usize {
        self.add_constant(Value::Number(c))
    }

    pub fn add_constant_string(&mut self, s: String) -> usize {
        self.add_constant(Value::String(s))
    }

    pub fn add_constant(&mut self, val: Value) -> usize {
        let const_idx = self.constants.len();
        self.constants.push(val);
        const_idx
    }
}
