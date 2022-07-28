use serde::{Deserialize, Serialize};
use std::fmt;

fn negate(val: value::Value) -> Value {
    match val {
        Value::Number(num) => Value::Number(-num),
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
#[serde(untagged)]
pub enum Op {
    Return,
    Constant(usize),
    Negate,
    Add,
    Subtract,
    Multiply,
    Divide,
}

#[derive(Default, Copy, Clone)]
pub struct Lineno(usize);

#[derive(Default)]
pub struct Chunk {
    code: Vec<(Op, Lineno)>,
    constants: Vec<Value>,
}

#[allow(dead_code)]
pub fn dissassemble_chunk(chunk: &Chunk, name: &str) {
    unimplemented!()
}
