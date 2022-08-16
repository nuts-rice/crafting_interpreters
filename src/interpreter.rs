use std::collections::HashMap;
use std::convert::TryInto;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::expr;
use std::fmt;

static INIT: &str = "init";

trait Callable {
    fn arity(&self, interpreter: &Interpreter) -> u8;
    fn call(&self, interpreter: &mut Interpreter, args: &[Value]) -> Result<Value, String>;
}

#[derive(Clone)]
pub struct NativeFunction {
    pub name: String,
    pub arity: u8,
    pub callable: fn(&[Value]) -> Result<Value, String>,
}
impl fmt::Debug for NativeFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NativeFunction({})", self.name)
    }
}

impl Callable for NativeFunction {
    fn arity(&self, _interpreter: &Interpreter) -> u8 {
        self.arity
    }

    fn call(&self, _interpreter: &mut Interpreter, args: &[Value]) -> Result<Value, String> {
        (self.callable)(args)
    }
}

#[derive(Debug, Clone)]
pub struct LoxFunction {
    pub name: expr::Symbol,
    pub parameters: Vec<expr::Symbol>,
    pub body: Vec<expr::Stmt>,
    pub closure: Enviroment,
    pub this_binding: Option<Box<Value>>,
    pub is_initializer: bool,
}

impl Callable for LoxFunction {
    fn arity(&self, _interpreter: &Interpreter) -> u8 {
        self.parameters.len().try_into().unwrap()
    }

    fn call(&self, interpreter: &mut Interpreter, args: &[Value]) -> Result<Value, String> {
        let args_env: HashMap<_, _> = self
            .parameters
            .iter()
            .zip(args.iter())
            .map(|(param, arg)| {
                (
                    param.name.clone(),
                    (
                        Some(arg.clone()),
                        SourceLocation {
                            line: param.line,
                            col: param.col,
                        },
                    ),
                )
            })
            .collect();
        let saved_env = interpreter.env.clone();
        let saved_retval = interpreter.retval.clone();

        let mut env = self.closure.clone();
        env.venv.extend(args_env);
        if let Some(this_val) = &self.this_binding {
            let this_symbol = Interpreter::this_symbol();
            env.venv.insert(
                this_symbol.name,
                (
                    Some(*this_val.clone()),
                    SourceLocation {
                        line: this_symbol.line,
                        col: this_symbol.col,
                    },
                ),
            );
        }

        let env = env;

        interpreter.env = env;
        interpreter.interpret(&self.body)?;

        let retval = interpreter.retval.clone();
        interpreter.env = saved_env;
        interpreter.retval = saved_retval;
        match retval {
            Some(val) => {
                let val_type = type_of(&val);
                if self.is_initializer && val_type != Type::Nil {
                    Err(format!(
                        "TypeError: init should only return nil, not {:?}",
                        val_type
                    ))
                } else {
                    Ok(val)
                }
            }
            None => {
                if self.is_initializer {
                    match &self.this_binding {
                        Some(this_val) => Ok(*this_val.clone()),
                        None => {
                            panic!("Internal interpreter error: COuld not find binding for this.")
                        }
                    }
                } else {
                    Ok(Value::Nil)
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoxClass {
    pub name: expr::Symbol,
    pub superclass: Option<u64>,
    pub id: u64,
    pub methods: HashMap<String, u64>,
}

impl Callable for LoxClass {
    fn arity(&self, interpreter: &Interpreter) -> u8 {
        match self.init(interpreter) {
            Some(initializer) => initializer.parameters.len().try_into().unwrap(),
            None => 0,
        }
    }
    fn call(&self, interpreter: &mut Interpreter, args: &[Value]) -> Result<Value, String> {
        let instance = interpreter.create_instance(&self.name, self.id);

        if let Some(mut initializer) = self.init(interpreter) {
            initializer.this_binding = Some(Box::new(instance.clone()));
            initializer.call(interpreter, args)?;
        }

        Ok(instance)
    }
}

impl LoxClass {
    fn init(&self, interpreter: &Interpreter) -> Option<LoxFunction> {
        match self.methods.get(&String::from("init")) {
            Some(initializer_id) => match interpreter.lox_functions.get(initializer_id) {
                Some(initializer) => Some(initializer.clone()),
                None => panic!(
                    "Internal interpreter error! Couldn't find an initializer method with id {}.",
                    initializer_id
                ),
            },
            None => None,
        }
    }

    fn find_method(
        &self,
        method_name: &str,
        interpreter: &Interpreter,
    ) -> Option<(expr::Symbol, u64)> {
        if let Some(method_id) = self.methods.get(method_name) {
            if let Some(lox_fn) = interpreter.lox_functions.get(method_id) {
                return Some((lox_fn.name.clone(), *method_id));
            }
            panic!(
                "Internal interpreter error: Could not find lox fn with id {}.",
                method_id
            );
        } else if let Some(superclass_id) = self.superclass {
            if let Some(superclass) = interpreter.lox_classes.get(&superclass_id) {
                return superclass.find_method(method_name, interpreter);
            }
            panic!(
                "Internal interpreter error: Could not find lox fn with id {}.",
                superclass_id
            )
        }
        None
    }
}

#[derive(Clone, Debug)]
pub struct LoxInstance {
    pub class_name: expr::Symbol,
    pub class_id: u64,
    pub fields: HashMap<String, Value>,
    pub id: u64,
}

impl LoxInstance {
    fn getattr(&self, attr: &str, interpreter: &Interpreter) -> Result<Value, String> {
        match self.fields.get(attr) {
            Some(val) => Ok(val.clone()),
            None => {
                if let Some(cls) = interpreter.lox_classes.get(&self.class_id) {
                    if let Some((func_name, method_id)) = cls.find_method(attr, interpreter) {
                        return Ok(Value::LoxFunction(
                            func_name,
                            method_id,
                            Some(Box::new(Value::LoxInstance(
                                self.class_name.clone(),
                                self.id,
                            ))),
                        ));
                    }
                    return Err(format!(
                        "Attribute error: '{}' instance has no '{}' attribute.",
                        self.class_name.name, attr
                    ));
                } else {
                    panic!(
                        "Internal interpreter error! Could not find class with id {}",
                        self.class_id
                    );
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    Number(f64),
    String(String),
    Bool(bool),
    Nil,
    NativeFunction(NativeFunction),
    LoxFunction(expr::Symbol, u64, Option<Box<Value>>),
    LoxClass(expr::Symbol, u64),
    LoxInstance(expr::Symbol, u64),
}

fn as_callable(interpreter: &Interpreter, value: &Value) -> Option<Box<dyn Callable>> {
    match value {
        Value::NativeFunction(f) => Some(Box::new(f.clone())),
        Value::LoxFunction(_, id, this_binding) => match interpreter.lox_functions.get(id) {
            Some(f) => {
                let mut f_copy = f.clone();
                f_copy.this_binding = this_binding.clone();
                Some(Box::new(f_copy))
            }
            None => panic!(
                "Internal interpreter error! Could not find loxFunction with id {}.",
                id
            ),
        },
        Value::LoxClass(_cls, id) => match interpreter.lox_classes.get(id) {
            Some(cls) => Some(Box::new(cls.clone())),
            None => panic!(
                "Internal interpreter error! Could not find loxclass with id {}.",
                id
            ),
        },
        _ => None,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    Number,
    String,
    Bool,
    Nil,
    NativeFunction,
    LoxFunction,
    LoxClass,
    LoxInstance,
}

pub fn type_of(val: &Value) -> Type {
    match val {
        Value::Number(_) => Type::Number,
        Value::String(_) => Type::String,
        Value::Bool(_) => Type::Bool,
        Value::Nil => Type::Nil,
        Value::NativeFunction(_) => Type::NativeFunction,
        Value::LoxFunction(_, _, _) => Type::LoxFunction,
        Value::LoxClass(_, _) => Type::LoxClass,
        Value::LoxInstance(_, _) => Type::LoxInstance,
    }
}

pub fn interpret(stmts: &[expr::Stmt]) -> Result<String, String> {
    let mut interpreter = Interpreter {
        ..Default::default()
    };
    interpreter.interpret(stmts)?;
    Ok(interpreter.output.join("\n"))
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    line: usize,
    col: i64,
}

#[derive(Debug, Default, Clone)]
pub struct Enviroment {
    enclosing: Option<Box<Enviroment>>,

    venv: HashMap<String, (Option<Value>, SourceLocation)>,
}

pub enum LookupResult<'a> {
    Ok(&'a Value),
    UndefButDeclared(SourceLocation),
    UndefAndNotDeclared,
}

impl Enviroment {
    pub fn with_enclosing(enclosing: Enviroment) -> Enviroment {
        Enviroment {
            enclosing: Some(Box::new(enclosing)),
            venv: HashMap::new(),
        }
    }
    pub fn define(&mut self, sym: expr::Symbol, maybe_val: Option<Value>) {
        self.venv.insert(
            sym.name,
            (
                maybe_val,
                SourceLocation {
                    line: sym.line,
                    col: sym.col,
                },
            ),
        );
    }

    pub fn lookup(&self, sym: &expr::Symbol) -> LookupResult {
        match self.venv.get(&sym.name) {
            Some((maybe_val, defn_source_location)) => match maybe_val {
                Some(val) => LookupResult::Ok(val),
                None => LookupResult::UndefButDeclared(SourceLocation {
                    line: defn_source_location.line,
                    col: defn_source_location.col,
                }),
            },
            None => LookupResult::UndefAndNotDeclared,
        }
    }

    pub fn get(&self, sym: &expr::Symbol) -> Result<&Value, String> {
        match self.lookup(sym) {
            LookupResult::Ok(val) => Ok(val),
            LookupResult::UndefButDeclared(source_location) => Err(format!(
                "Use of undefined variable {} at line={}, col={}.\
                    {} was previously declared at line={}, col={}, \
                    but was never defined",
                &sym.name, sym.line, sym.col, &sym.name, source_location.line, source_location.col
            )),
            LookupResult::UndefAndNotDeclared => match &self.enclosing {
                Some(enclosing) => enclosing.get(sym),
                None => Err(format!(
                    "Use of undefined variable {} at line={}, col={}. {} was never declared",
                    &sym.name, sym.line, sym.col, &sym.name
                )),
            },
        }
    }

    pub fn assign(&mut self, sym: expr::Symbol, val: &Value) -> Result<(), String> {
        if self.venv.contains_key(&sym.name) {
            self.define(sym, Some(val.clone()));
            return Ok(());
        }

        Err(format!(
            "attempting to assign to undeclared variable at line={},col={}",
            sym.line, sym.col
        ))
    }
}

pub struct Interpreter {
    pub counter: u64,
    pub lox_functions: HashMap<u64, LoxFunction>,
    pub lox_instances: HashMap<u64, LoxInstance>,
    pub lox_classes: HashMap<u64, LoxClass>,
    pub env: Enviroment,
    pub globals: Enviroment,
    pub retval: Option<Value>,
    pub output: Vec<String>,
}

impl Default for Interpreter {
    fn default() -> Interpreter {
        let mut globals_venv = HashMap::new();
        globals_venv.insert(
            String::from("clock"),
            (
                Some(Value::NativeFunction(NativeFunction {
                    name: String::from("clock"),
                    arity: 0,
                    callable: |_| {
                        let start = SystemTime::now();
                        let since_epoch = start.duration_since(UNIX_EPOCH).unwrap();

                        Ok(Value::Number(since_epoch.as_millis() as f64))
                    },
                })),
                SourceLocation {
                    line: 420,
                    col: 420,
                },
            ),
        );
        let globals = Enviroment {
            enclosing: None,
            venv: globals_venv,
        };

        Interpreter {
            counter: 0,
            lox_functions: HashMap::new(),
            lox_instances: HashMap::new(),
            lox_classes: HashMap::new(),
            env: Default::default(),
            globals,
            retval: None,
            output: Vec::new(),
        }
    }
}

impl Interpreter {
    pub fn interpret(&mut self, stmts: &[expr::Stmt]) -> Result<(), String> {
        for stmt in stmts.iter() {
            self.execute(stmt)?;
        }
        Ok(())
    }

    fn alloc_id(&mut self) -> u64 {
        let res = self.counter;
        self.counter += 1;
        res
    }

    fn create_instance(&mut self, class_name: &expr::Symbol, class_id: u64) -> Value {
        let inst_id = self.alloc_id();
        let inst = LoxInstance {
            class_name: class_name.clone(),
            class_id,
            id: inst_id,
            fields: HashMap::new(),
        };
        self.lox_instances.insert(inst_id, inst);
        Value::LoxInstance(class_name.clone(), inst_id)
    }

    pub fn execute(&mut self, stmt: &expr::Stmt) -> Result<(), String> {
        if self.retval.is_some() {
            return Ok(());
        }
        match stmt {
            expr::Stmt::Expr(e) => match self.interpret_expr(e) {
                Ok(_) => Ok(()),
                Err(err) => Err(err),
            },
            expr::Stmt::ClassDecl(sym, maybe_superclass, stmt_methods) => {
                let class_id = self.alloc_id();
                self.env
                    .define(sym.clone(), Some(Value::LoxClass(sym.clone(), class_id)));
                let mut methods = HashMap::new();
                for method in stmt_methods.iter() {
                    let func_id = self.alloc_id();
                    methods.insert(method.name.name.clone(), func_id);
                    let is_initializer = method.name.name == INIT;
                    let lox_function = LoxFunction {
                        name: method.name.clone(),
                        parameters: method.params.clone(),
                        body: method.body.clone(),
                        closure: self.env.clone(),
                        this_binding: None,
                        is_initializer,
                    };

                    self.lox_functions.insert(func_id, lox_function);
                }

                let superclass_id = if let Some(superclass_var) = maybe_superclass {
                    if superclass_var.name == sym.name {
                        return Err(format!(
                            "Class cannot inherit from itself (line={}, col={})",
                            sym.line, sym.col
                        ));
                    }

                    let superclass_val =
                        self.interpret_expr(&expr::Expr::Variable(superclass_var.clone()))?;
                    if let Value::LoxClass(_, id) = superclass_val {
                        Some(id)
                    } else {
                        return Err(format!(
                            "Only classes should appear as superclasses. Found {:?}.",
                            type_of(&superclass_val)
                        ));
                    }
                } else {
                    None
                };

                let cls = LoxClass {
                    name: sym.clone(),
                    superclass: superclass_id,
                    id: class_id,
                    methods,
                };

                self.lox_classes.insert(class_id, cls);
                Ok(())
            }
            expr::Stmt::FuncDecl(expr::FuncDecl {
                name,
                params: parameters,
                body,
            }) => {
                let func_id = self.alloc_id();
                self.env.define(
                    name.clone(),
                    Some(Value::LoxFunction(name.clone(), func_id, None)),
                );

                let lox_function = LoxFunction {
                    name: name.clone(),
                    parameters: parameters.clone(),
                    body: body.clone(),
                    closure: self.env.clone(),
                    this_binding: None,
                    is_initializer: false,
                };
                self.lox_functions.insert(func_id, lox_function);
                Ok(())
            }
            expr::Stmt::If(cond, if_true, maybe_if_false) => {
                if Interpreter::is_truthy(&self.interpret_expr(cond)?) {
                    return self.execute(if_true);
                }
                if let Some(if_false) = maybe_if_false {
                    return self.execute(if_false);
                }
                Ok(())
            }
            expr::Stmt::Print(e) => match self.interpret_expr(e) {
                Ok(val) => {
                    self.output.push(format!("{}", val));
                    Ok(())
                }
                Err(err) => Err(err),
            },
            expr::Stmt::VarDecl(sym, maybe_expr) => {
                let maybe_val = match maybe_expr {
                    Some(expr) => Some(self.interpret_expr(expr)?),
                    None => None,
                };
                self.env.define(sym.clone(), maybe_val);
                Ok(())
            }
            expr::Stmt::Block(stmts) => {
                self.env = Enviroment::with_enclosing(self.env.clone());

                for stmt in stmts.iter() {
                    self.execute(stmt)?;
                }

                if let Some(enclosing) = self.env.enclosing.clone() {
                    self.env = *enclosing
                } else {
                    panic!("nope")
                }

                Ok(())
            }

            expr::Stmt::While(cond, body) => {
                while Interpreter::is_truthy(&self.interpret_expr(cond)?) {
                    self.execute(body)?;
                }
                Ok(())
            }
            expr::Stmt::Return(_, maybe_res) => {
                self.retval = Some(if let Some(res) = maybe_res {
                    self.interpret_expr(res)?
                } else {
                    Value::Nil
                });
                Ok(())
            }
        }
    }

    pub fn lookup(&self, sym: &expr::Symbol) -> Result<&Value, String> {
        match self.env.get(sym) {
            Ok(val) => Ok(val),
            _ => self.globals.get(sym),
        }
    }

    pub fn interpret_expr(&mut self, expr: &expr::Expr) -> Result<Value, String> {
        match expr {
            expr::Expr::This => match self.lookup(&Interpreter::this_symbol()) {
                Ok(val) => Ok(val.clone()),
                Err(err) => Err(err),
            },
            expr::Expr::Literal(lit) => Ok(Interpreter::interpret_literal(lit)),
            expr::Expr::Unary(op, e) => self.interpret_unary(*op, e),
            expr::Expr::Binary(lhs, op, rhs) => self.interpret_binary(lhs, *op, rhs),
            expr::Expr::Call(callee, loc, args) => self.call(callee, loc, args),
            expr::Expr::Get(lhs, attr) => self.getattr(lhs, &attr.name),
            expr::Expr::Set(lhs, attr, rhs) => self.setattr(lhs, attr, rhs),
            expr::Expr::Grouping(e) => self.interpret_expr(e),
            expr::Expr::Variable(sym) => match self.env.get(sym) {
                Ok(val) => Ok(val.clone()),
                Err(err) => Err(err),
            },
            expr::Expr::Assign(sym, val_expr) => {
                let val = self.interpret_expr(val_expr)?;

                if let Err(err) = self.env.assign(sym.clone(), &val) {
                    return Err(err);
                }
                Ok(val)
            }
            expr::Expr::Logical(left_expr, expr::LogicalOp::Or, right_expr) => {
                let left = self.interpret_expr(left_expr)?;
                if Interpreter::is_truthy(&left) {
                    Ok(left)
                } else {
                    Ok(self.interpret_expr(right_expr)?)
                }
            }
            expr::Expr::Logical(left_expr, expr::LogicalOp::And, right_expr) => {
                let left = self.interpret_expr(left_expr)?;
                if !Interpreter::is_truthy(&left) {
                    Ok(left)
                } else {
                    Ok(self.interpret_expr(right_expr)?)
                }
            }
        }
    }

    fn getattr(&mut self, lhs: &expr::Expr, attr: &str) -> Result<Value, String> {
        let val = self.interpret_expr(lhs)?;
        match val {
            Value::LoxInstance(_, id) => match self.lox_instances.get(&id) {
                Some(inst) => inst.getattr(attr, self),
                None => panic!(
                    "Internal interpreter error: could not find an instance with id {}.",
                    id
                ),
            },
            _ => Err(format!(
                "Only LoxInstance Values have attributes. Found {:?}.",
                type_of(&val)
            )),
        }
    }

    fn setattr(
        &mut self,
        lhs_expr: &expr::Expr,
        attr: &expr::Symbol,
        rhs_expr: &expr::Expr,
    ) -> Result<Value, String> {
        let lhs = self.interpret_expr(lhs_expr)?;
        let rhs = self.interpret_expr(rhs_expr)?;
        match lhs {
            Value::LoxInstance(_, id) => match self.lox_instances.get_mut(&id) {
                Some(inst) => {
                    inst.fields.insert(attr.name.clone(), rhs.clone());
                    Ok(rhs)
                }
                None => panic!(
                    "Internal interpreter error: could not find instance with id {}",
                    id
                ),
            },
            _ => Err(format!(
                "Only LoxInstance values have attributes. Found {:?}.",
                type_of(&lhs)
            )),
        }
    }

    fn this_symbol() -> expr::Symbol {
        expr::Symbol {
            name: String::from("this"),
            line: 0,
            col: -1,
        }
    }

    fn call(
        &mut self,
        callee_expr: &expr::Expr,
        loc: &expr::SourceLocation,
        arg_exprs: &[expr::Expr],
    ) -> Result<Value, String> {
        let callee = self.interpret_expr(callee_expr)?;
        match as_callable(self, &callee) {
            Some(callable) => {
                let maybe_args: Result<Vec<_>, _> = arg_exprs
                    .iter()
                    .map(|arg| self.interpret_expr(arg))
                    .collect();

                match maybe_args {
                    Ok(args) => {
                        if args.len() != callable.arity(self).into() {
                            Err(format!("Invalid call at line={}, col={}: callee has arity {}, was called with {} arguments",
                                    loc.line,
                                    loc.col,
                                    callable.arity(self),
                                    args.len()
                                ))
                        } else {
                            callable.call(self, &args)
                        }
                    }
                    Err(err) => Err(err),
                }
            }
            None => Err(format!(
                "Value {:?} is not callable at line = {}, col = {}",
                callee, loc.line, loc.col
            )),
        }
    }

    fn execute_call(
        callee: Value,
        loc: &expr::SourceLocation,
        _args: Vec<Value>,
    ) -> Result<Value, String> {
        Err(format!(
            "Value {:?} not callable at line={}, col={}",
            callee, loc.line, loc.col
        ))
    }

    fn interpret_binary(
        &mut self,
        lhs_expr: &expr::Expr,
        op: expr::BinaryOp,
        rhs_expr: &expr::Expr,
    ) -> Result<Value, String> {
        let lhs = self.interpret_expr(lhs_expr)?;
        let rhs = self.interpret_expr(rhs_expr)?;

        match (&lhs, op.ty, &rhs) {
            (Value::Number(n1), expr::BinaryOpType::Less, Value::Number(n2)) => {
                Ok(Value::Bool(n1 < n2))
            }
            (Value::Number(n1), expr::BinaryOpType::LessEqual, Value::Number(n2)) => {
                Ok(Value::Bool(n1 <= n2))
            }

            (Value::Number(n1), expr::BinaryOpType::Greater, Value::Number(n2)) => {
                Ok(Value::Bool(n1 > n2))
            }
            (Value::Number(n1), expr::BinaryOpType::GreaterEqual, Value::Number(n2)) => {
                Ok(Value::Bool(n1 >= n2))
            }
            (Value::Number(n1), expr::BinaryOpType::Plus, Value::Number(n2)) => {
                Ok(Value::Number(n1 + n2))
            }
            (Value::Number(n1), expr::BinaryOpType::Minus, Value::Number(n2)) => {
                Ok(Value::Number(n1 - n2))
            }
            (Value::Number(n1), expr::BinaryOpType::Star, Value::Number(n2)) => {
                Ok(Value::Number(n1 * n2))
            }
            (Value::Number(n1), expr::BinaryOpType::Slash, Value::Number(n2)) => {
                if *n2 != 0.0 {
                    Ok(Value::Number(n1 / n2))
                } else {
                    Err(format!(
                        "division by zero at line={}, col={}",
                        op.line, op.col
                    ))
                }
            }
            (Value::String(s1), expr::BinaryOpType::Plus, Value::String(s2)) => {
                Ok(Value::String(format!("{}{}", s1, s2)))
            }
            (_, expr::BinaryOpType::EqualEqual, _) => {
                Ok(Value::Bool(Interpreter::equals(&lhs, &rhs)))
            }

            (_, expr::BinaryOpType::NotEqual, _) => {
                Ok(Value::Bool(!Interpreter::equals(&lhs, &rhs)))
            }
            _ => Err(format!(
                "invalid operands in binary opertor {:?} of type {:?} and {:?} at line={}, col={}",
                op.ty,
                type_of(&lhs),
                type_of(&rhs),
                op.line,
                op.col
            )),
        }
    }

    fn equals(lhs: &Value, rhs: &Value) -> bool {
        match (lhs, rhs) {
            (Value::Number(n1), Value::Number(n2)) => (n1 - n2).abs() < f64::EPSILON,
            (Value::String(s1), Value::String(s2)) => s1 == s2,
            (Value::Bool(b1), Value::Bool(b2)) => b1 == b2,
            (Value::Nil, Value::Nil) => true,
            (_, _) => false,
        }
    }

    //convert literal tree node of syntax tree into runtime value
    fn interpret_unary(&mut self, op: expr::UnaryOp, expr: &expr::Expr) -> Result<Value, String> {
        let val = self.interpret_expr(expr)?;

        match (op.ty, &val) {
            (expr::UnaryOpType::Minus, Value::Number(n)) => Ok(Value::Number(-n)),
            (expr::UnaryOpType::Bang, _) => Ok(Value::Bool(!Interpreter::is_truthy(&val))),
            (_, Value::String(_)) => Err(format!(
                "invalid application of unary op {:?} to object of type string at line = {}, col = {}",
                op.ty, op.line, op.col
            )),
            (_, Value::NativeFunction(_)) => Err(format!(
                    "invalid application of unary op {:?} to object of type NativeFunction at line= {}, col = {}",
                    op.ty, op.line, op.col
            )),
            (_, Value::LoxFunction(_, _, _)) => Err(format!(
                    "invalid application of unary op {:?} to obkect of type LoxFunction at line = {}, col = {}",
                    op.ty, op.line, op.col
            )),
            (_, Value::LoxClass(_, _)) => Err(format!(
                    "invalid application of unary op {:?} to object of type LoxClass at line = {}, col = {}",
                    op.ty, op.line, op.col
            )),
            (_, Value::LoxInstance(class_name, _inst)) => Err(format!(
                    "Invalid application of unary op {:?} to object of type {:?} at line={}, col={}",
                    class_name.name, op.ty, op.line, op.col
            )),
            (expr::UnaryOpType::Minus, Value::Bool(_)) => Err(format!(
                "invalid application of unary op {:?} to object of type bool at line = {}, col = {}",         
                       op.ty, op.line, op.col
            )),
            (_, Value::Nil) => Err(format!(
                "invalid application of unary op {:?} to object of type nil at line ={}, col = {}",
                op.ty, op.line, op.col
            )),
        }
    }

    fn is_truthy(val: &Value) -> bool {
        match val {
            Value::Nil => false,
            Value::Bool(b) => *b,
            _ => true,
        }
    }

    fn interpret_literal(literal: &expr::Literal) -> Value {
        match literal {
            expr::Literal::Number(n) => Value::Number(*n),
            expr::Literal::String(s) => Value::String(s.clone()),
            expr::Literal::True => Value::Bool(true),
            expr::Literal::False => Value::Bool(false),
            expr::Literal::Nil => Value::Nil,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self {
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "'{}'", s),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Nil => write!(f, "nil"),
            Value::NativeFunction(func) => write!(f, "NativeFunction({})", func.name),
            Value::LoxFunction(sym, _, _) => write!(f, "LoxFunction({})", sym.name),
            Value::LoxClass(cls, _) => write!(f, "LoxClass({})", cls.name),
            Value::LoxInstance(sym, _) => write!(f, "LoxInstance({})", sym.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::interpreter;
    use crate::parser;
    use crate::scanner;

    fn evaluate(code: &str) -> Result<String, String> {
        let tokens = scanner::scan_tokens(code.to_string()).unwrap();
        let stmts = parser::parse(tokens)?;
        interpreter::interpret(&stmts)
    }
    /*
        #[test]
        fn test_fact() {
            fn fact(n: i32) -> i32 {
                if n <= 1 {
                    return 1;
                }
                n * fact(n - 1)
            }

            let result = evaluate(
                "fun fact(n) { \n\
                    if (n <= 1) { \n\
                        return 1; \n\
                    }\n\
                    return n * fact(n - 1); \n\
                } \n\
                print fact(10); ",
            );
            match result {
                Ok(output) => assert_eq!(output, format!("{}", fact(10))),
                Err(err) => panic!("{}", err),
            }
    //    }
    */
    #[test]
    fn invalid_binary_operands_test() {
        let res = evaluate("1 + \"string\";");

        match res {
            Ok(output) => panic!("{}", output),
            Err(err) => assert!(err.starts_with("invalid operands in binary operator")),
        }
    }
}
