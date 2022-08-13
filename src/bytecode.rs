use crate::value;
use serde::{Deserialize, Serialize};

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
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Lineno {
    pub value: usize,
}

pub fn Lineno(value: usize) -> Lineno {
    Lineno { value }
}

#[derive(Default)]
pub struct Chunk {
    pub code: Vec<(Op, Lineno)>,
    pub constants: Vec<value::Value>,
}

impl Chunk {
    pub fn add_constant_number(&mut self, c: f64) -> usize {
        self.add_constant(value::Value::Number(c))
    }

    pub fn add_constant_string(&mut self, s: String) -> usize {
        self.add_constant(value::Value::String(s))
    }

    pub fn add_constant(&mut self, val: value::Value) -> usize {
        let const_idx = self.constants.len();
        self.constants.push(val);
        const_idx
    }
}
