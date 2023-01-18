use crate::bytecode;
use crate::garbage_collector_vals;

enum GCData {
    String(String),
    #[allow(dead_code)]
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
    id_counter: garbage_collector_vals::Id,
    values: Vec<GCval>,
}

impl Heap {
    #[allow(dead_code)]
    pub fn manage_str(&mut self, s: String) -> garbage_collector_vals::GcString {
        self.values.push(GCval::from(GCData::String(s)));
        garbage_collector_vals::GcString(self.values.len() - 1)
    }

    #[allow(dead_code)]
    pub fn manage_closure(&mut self, c: bytecode::Closure) -> garbage_collector_vals::GcClosure {
        self.values.push(GCval::from(GCData::Closure(c)));
        garbage_collector_vals::GcClosure(self.values.len() - 1)
    }

    #[allow(dead_code)]
    pub fn get_str(&self, s: garbage_collector_vals::GcString) -> &String {
        self.values[s.0].data.as_str().unwrap()
    }

    #[allow(dead_code)]
    pub fn get_closure(&self, c: garbage_collector_vals::GcClosure) -> &bytecode::Closure {
        self.values[c.0].data.as_closure().unwrap()
    }
}
