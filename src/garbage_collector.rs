use crate::bytecode;

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
#[derive(Default)]
pub struct Heap {
    bytes_allocated: usize,
    next_gc: usize,
    values: Vec<GCval>,
}

impl Heap {
    #[allow(dead_code)]
    pub fn manage_str(&mut self, s: String) -> usize {
        self.values.push(GCval::from(GCData::String(s)));
        self.values.len() - 1
    }

    #[allow(dead_code)]
    pub fn manage_closure(&mut self, c: bytecode::Closure) -> usize {
        self.values.push(GCval::from(GCData::Closure(c)));
        self.values.len() - 1
    }

    #[allow(dead_code)]
    pub fn get_str(&self, id: usize) -> &String {
        self.values[id].data.as_str().unwrap()
    }

    #[allow(dead_code)]
    pub fn get_closure(&self, id: usize) -> &bytecode::Closure {
        self.values[id].data.as_closure().unwrap()
    }

    pub fn mark(&mut self, id: usize) {
        self.values[id].is_marked = true;
    }

    pub fn mark_closure(&mut self, id: usize) {
        self.values[id].is_marked = true;
    }
}
