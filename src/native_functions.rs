use crate::bytecode;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn exponent(args: Vec<bytecode::Value>) -> Result<bytecode::Value, String> {
    match args[0] {
        bytecode::Value::Number(num) => Ok(bytecode::Value::Number(num.exp())),
        _ => Err(format!(
            "Expected number, got {:?}",
            bytecode::type_of(&args[0])
        )),
    }
}

pub fn sqrt(args: Vec<bytecode::Value>) -> Result<bytecode::Value, String> {
    match args[0] {
        bytecode::Value::Number(num) => Ok(bytecode::Value::Number(num.sqrt())),
        _ => Err(format!(
            "expected number, got {:?}",
            bytecode::type_of(&args[0])
        )),
    }
}

pub fn clock(args: Vec<bytecode::Value>) -> Result<bytecode::Value, String> {
    let start = SystemTime::now();
    let since_epoch = start.duration_since(UNIX_EPOCH).unwrap();
    Ok(bytecode::Value::Number(since_epoch.as_millis() as f64))
}
