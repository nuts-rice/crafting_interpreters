use std::fmt;

use crate::garbage_collector;




#[derive(Clone)]
#[allow(dead_code)]
pub enum Upvalue {
    Open(usize),
    Closed(Value),
}

impl Upvalue {
    pub fn is_open(&self) -> bool {
        match self {
            Upvalue::Open(_) => true,
            Upvalue::Closed(_) => false,
        }
    }

    pub fn is_open_with_index(&self, index: usize) -> bool {
        match self {
            Upvalue::Open(idx) => index == *idx,
            Upvalue::Closed(_) => false,
        }
    }
}

#[derive(Clone)]
pub struct NativeFunction {
    pub arity: u8,
    pub name: String,
    pub func: fn(&garbage_collector::Heap, Vec<Value>) -> Result<Value, String>,
}

#[derive(Clone)]
pub enum Value {
    Number(f64),
    Bool(bool),
    Class(usize),
    Nil,
    String(String),
    Function(usize),
    NativeFunction(NativeFunction)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Type {
    Number,
    Bool,
    String,
    Function,
    NativeFunction,
    Class,
    Nil,
}

#[derive(Clone)]
pub struct Class {
    pub name: String,
}

pub fn type_of(value: &Value) -> Type {
    match value {
        Value::Number(_) => Type::Number,
        Value::Bool(_) => Type::Bool,
        Value::Class(_) => todo!(),
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
            Value::Class(_) | Value::Function(_) | Value::NativeFunction(_) => todo!(),
            Value::Nil => write!(f, "nil"),
        }
    }
}
