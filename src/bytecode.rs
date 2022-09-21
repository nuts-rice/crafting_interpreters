use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
#[serde(untagged)]
pub enum Op {
    Return,
    Constant(usize),
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
    JumpIfFalse(usize),
    Jump(usize),
    Loop(usize),
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum Value {
    Number(f64),
    Bool(bool),
    Nil,
    String(String),
    Function(Function),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Type {
    Number,
    Bool,
    String,
    Nil,
    Function,
}

#[derive(Default, Clone)]
#[allow(dead_code)]
pub struct Function {
    pub arity: u8,
    pub chunk: Chunk,
    pub name: String,
}

pub fn type_of(value: &Value) -> Type {
    match value {
        Value::Number(_) => Type::Number,
        Value::Bool(_) => Type::Bool,
        Value::String(_) => Type::String,
        Value::Function(_) => Type::Function,
        Value::Nil => Type::Nil,
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Number(num) => write!(f, "{}", num),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Function(func) => write!(f, "<fn {}>", func.name),
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
