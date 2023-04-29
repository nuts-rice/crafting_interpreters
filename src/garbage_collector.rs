use std::collections::HashMap;

use crate::{bytecode};

enum GCData {
    String(String),
    Closure(bytecode::Closure),
}

impl GCData {
    fn as_str(&self) -> Option<&String> {
        match self {
            GCData::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_closure(&self) -> Option<&bytecode::Closure> {
        match &self {
            GCData::Closure(c) => Some(c),
            _ => None,
        }
    }
}

struct GCval {
    #[allow(dead_code)]
    //basically marked signifies is reachable in graph traversal
    is_marked: bool,
    data: GCData,
}

impl GCval {
    fn from(data: GCData) -> GCval {
        GCval {
            is_marked: false,
            data,
        }
    }
}

pub struct Heap {
    bytes_allocated: usize,
    id_counter: usize,
    next_gc: usize,
    values: HashMap<usize, GCval>,
}

impl Default for Heap {
    fn default() -> Heap {
        let next_gc = std::env::var("LOX_GC_TRIGGER_SIZE")
            .ok()
            .and_then(|env_str| env_str.parse::<usize>().ok())
            .unwrap_or(1024 * 1024);
        Heap {
            bytes_allocated: 0,
            id_counter: 0,
            next_gc,
            values: Default::default(),
        }
    }
}

impl Heap {
    fn generate_id(&mut self) -> usize {
        self.id_counter += 1;
        loop {
            if !self.values.contains_key(&self.id_counter) {
                return self.id_counter;
            }
            self.id_counter += 1;
        }
    }
    pub fn manage_str(&mut self, s: String) -> usize {
        self.bytes_allocated += s.len();
        let id = self.generate_id();
        self.values.insert(id, GCval::from(GCData::String(s)));
        id
    }

    pub fn manage_closure(&mut self, c: bytecode::Closure) -> usize {
        self.bytes_allocated += c.function.chunk.code.len();
        self.bytes_allocated += c.function.chunk.constants.len();
        let id = self.generate_id();
        self.values.insert(id, GCval::from(GCData::Closure(c)));
        id
    }

    pub fn get_str(&self, id: usize) -> &String {
        self.values.get(&id).unwrap().data.as_str().unwrap()
    }

    pub fn get_closure(&self, id: usize) -> &bytecode::Closure {
        self.values.get(&id).unwrap().data.as_closure().unwrap()
    }

    pub fn unmark(&mut self) {
        for val in self.values.values_mut() {
            val.is_marked = false;
        }
    }

    pub fn mark(&mut self, id: usize) {
        self.values.get_mut(&id).unwrap().is_marked = true;
    }

    pub fn is_marked(&self, id: usize) -> bool {
        self.values.get(&id).unwrap().is_marked
    }

    pub fn children(&self, _id:usize) -> Vec<usize> {
        todo!()
    }

    pub fn closure_children(&self, _closure: &bytecode::Closure) -> Vec<usize> {
        todo!()
    }
    pub fn sweep(&mut self) {
        todo!()
    }

    pub fn collect(&self) -> bool {
        todo!()
    }
}
