use crate::bytecode;
use crate::garbage_collector_vals;

use std::collections::HashMap;

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

    fn as_str_mut(&mut self) -> Option<&mut String> {
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

    fn as_closure_mut(&mut self) -> Option<&mut bytecode::Closure> {
        match self {
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
    values: HashMap<garbage_collector_vals::Id, GCval>,
}

impl Heap {
    #[allow(dead_code)]
    pub fn manage_str(&mut self, s: String) -> garbage_collector_vals::GcString {
        let id = self.alloc_id();
        self.values.insert(id, GCval::from(GCData::String(s)));
        garbage_collector_vals::GcString(id)
    }

    #[allow(dead_code)]
    pub fn manage_closure(&mut self, c: bytecode::Closure) -> garbage_collector_vals::GcClosure {
        let id = self.alloc_id();
        self.values.insert(id, GCval::from(GCData::Closure(c)));
        garbage_collector_vals::GcClosure(id)
    }

    #[allow(dead_code)]
    fn alloc_id(&mut self) -> garbage_collector_vals::Id {
        while self.values.contains_key(&self.id_counter) {
            self.id_counter += 1
        }
        self.id_counter
    }

    #[allow(dead_code)]
    pub fn get_str(&self, s: garbage_collector_vals::GcString) -> &String {
        self.values.get(&s.0).unwrap().data.as_str().unwrap()
    }

    #[allow(dead_code)]
    pub fn get_str_mut(&mut self, s: garbage_collector_vals::GcString) -> &mut String {
        self.values
            .get_mut(&s.0)
            .unwrap()
            .data
            .as_str_mut()
            .unwrap()
    }
    #[allow(dead_code)]
    pub fn get_closure(&self, c: garbage_collector_vals::GcClosure) -> &bytecode::Closure {
        self.values.get(&c.0).unwrap().data.as_closure().unwrap()
    }
    #[allow(dead_code)]
    pub fn get_closure_mut(
        &mut self,
        c: garbage_collector_vals::GcClosure,
    ) -> &mut bytecode::Closure {
        self.values
            .get_mut(&c.0)
            .unwrap()
            .data
            .as_closure_mut()
            .unwrap()
    }
}
