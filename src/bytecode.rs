use crate::value;
use serde::{Deserialize, Serialize};

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
pub struct Lineno {
    pub value: usize,
}

pub fn Lineno(value: usize) -> Lineno {
    Lineno { value }
}

#[derive(Default)]
pub struct Chunk {
    code: Vec<(Op, Lineno)>,
    constants: Vec<value::Value>,
}

impl Chunk {
    pub fn add_constant(&mut self, c: f64) -> usize {
        let const_idx = self.constants.len();
        self.constants.push(value::Value::Number(c));
        const_idx
    }
}