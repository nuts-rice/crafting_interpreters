use crate::bytecode;

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
