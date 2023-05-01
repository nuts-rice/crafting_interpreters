use crate::garbage_collector;
use crate::value;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn exponent(
    _heap: &garbage_collector::Heap,
    args: Vec<value::Value>,
) -> Result<value::Value, String> {
    match args[0] {
        value::Value::Number(num) => Ok(value::Value::Number(num.exp())),
        _ => Err(format!(
            "Expected number, got {:?}",
            value::type_of(&args[0])
        )),
    }
}

pub fn sqrt(
    _heap: &garbage_collector::Heap,
    args: Vec<value::Value>,
) -> Result<value::Value, String> {
    match args[0] {
        value::Value::Number(num) => Ok(value::Value::Number(num.sqrt())),
        _ => Err(format!(
            "expected number, got {:?}",
            value::type_of(&args[0])
        )),
    }
}

pub fn clock(_args: Vec<value::Value>) -> Result<value::Value, String> {
    let start = SystemTime::now();
    let since_epoch = start.duration_since(UNIX_EPOCH).unwrap();
    Ok(value::Value::Number(since_epoch.as_millis() as f64))
}
