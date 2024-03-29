use crate::bytecode;
use crate::scanner;

pub struct Compiler {
    tokens: Vec<scanner::Token>,
    token_idx: usize,
    levels: Vec<Level>,
    level_idx: usize,
}

impl Default for Compiler {
    fn default() -> Compiler {
        Compiler {
            tokens: Default::default(),
            token_idx: 0,
            levels: vec![Default::default()],
            level_idx: 0,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
enum FunctionType {
    Function,
    Script,
}

#[derive(Debug)]
struct Local {
    name: scanner::Token,
    depth: i64,
    is_captured: bool,
}

pub struct Level {
    function: bytecode::Function,
    function_type: FunctionType,
    locals: Vec<Local>,
    scope_depth: i64,
    upvals: Vec<bytecode::UpvalueLoc>,
}

impl Default for Level {
    fn default() -> Level {
        Level {
            function: Default::default(),
            function_type: FunctionType::Script,
            locals: vec![Local {
                name: scanner::Token {
                    ty: scanner::TokenType::Identifier,
                    lexeme: Default::default(),
                    literal: Some(scanner::Literal::Identifier("".to_string())),
                    line: 0,
                    col: -1,
                },
                depth: 0,
                is_captured: false,
            }],
            scope_depth: 0,
            upvals: Default::default(),
        }
    }
}

enum Resolution {
    Local(usize),
    Global,
    UpValue(usize),
}

#[derive(Eq, PartialEq, PartialOrd, Copy, Clone, Debug)]
//Precedence levels from lowest to highest
enum Precedence {
    None,
    Assignment,
    Or,
    And,
    Equality,
    Comparison,
    Term,
    Factor,
    Unary,
    Call,
    Primary,
}

#[derive(Debug, Copy, Clone)]
enum ParseFn {
    Grouping,
    Unary,
    Binary,
    Number,
    Literal,
    String,
    Variable,
    And,
    Or,
    Call,
}

//Given a token type gives
//fn to compile Prefix  expr
//fn to compile infix expr whose left operand is folled by token of that type
//precende of infix expr that uses token as operator
#[derive(Debug, Copy, Clone)]
struct ParseRule {
    prefix: Option<ParseFn>,
    infix: Option<ParseFn>,
    precendence: Precedence,
}

impl Compiler {
    pub fn compile(input: String) -> Result<bytecode::Function, String> {
        let mut compiler = Compiler::default();
        match scanner::scan_tokens(input) {
            Ok(tokens) => {
                compiler.tokens = tokens;
                while !compiler.is_at_end() {
                    compiler.declaration()?;
                }
                compiler.emit_return();
                Ok(std::mem::take(&mut compiler.current_level_mut().function))
            }
            Err(err) => Err(err),
        }
    }

    fn declaration(&mut self) -> Result<(), String> {
        if self.matches(scanner::TokenType::Fun) {
            self.fun_decl()
        } else if self.matches(scanner::TokenType::Var) {
            self.var_decl()
        } else {
            self.statement()
        }
    }

    fn fun_decl(&mut self) -> Result<(), String> {
        let global_idx = self.parse_variable("expected function name.")?;
        self.mark_initialized();
        self.function(FunctionType::Function)?;
        self.define_var(global_idx);
        Ok(())
    }

    fn function(&mut self, _function_type: FunctionType) -> Result<(), String> {
        let mut level = Level::default();
        level.function_type = _function_type;
        //Call stack stuff
        //Return addresses and callframes etc
        level.function = bytecode::Function::default();
        //defining
        level.function.name =
            if let Some(scanner::Literal::Identifier(fun_name)) = &self.previous().literal {
                fun_name.clone()
            } else {
                panic!("expected identifier");
            };
        self.push_level(level);
        self.begin_scope();
        self.consume(
            scanner::TokenType::LeftParen,
            "Expected '(' after function name.",
        )?;

        if !self.check(scanner::TokenType::RightParen) {
            loop {
                self.current_function_mut().arity += 1;
                let param_const_idx = self.parse_variable("expected parameter name")?;
                self.define_var(param_const_idx);

                if !self.matches(scanner::TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(
            scanner::TokenType::RightParen,
            "Expected ')' after param list",
        )?;

        self.consume(
            scanner::TokenType::LeftBrace,
            "Expected '{' before function body",
        )?;
        self.block()?;
        self.emit_return();

        let function = std::mem::take(&mut self.current_level_mut().function);
        let upvals = std::mem::take(&mut self.current_level_mut().upvals);
        self.pop_level();
        let const_idx = self
            .current_chunk()
            .add_constant(bytecode::Constant::Function(bytecode::Closure {
                function,
                upvalues: Vec::new(),
            }));
        self.emit_op(
            bytecode::Op::Closure(const_idx, upvals),
            self.previous().line,
        );
        Ok(())
    }

    fn var_decl(&mut self) -> Result<(), String> {
        let global_idx = self.parse_variable("expected var name")?;
        if self.matches(scanner::TokenType::Equal) {
            self.expression()?;
        } else {
            let line = self.previous().line;
            self.emit_op(bytecode::Op::Nil, line)
        }

        self.consume(
            scanner::TokenType::Semicolon,
            "expected ';' after var declaration",
        )?;
        self.define_var(global_idx);
        Ok(())
    }

    fn mark_initialized(&mut self) -> bool {
        let scope_depth = self.scope_depth();
        if scope_depth > 0 {
            if let Some(last) = self.locals_mut().last_mut() {
                last.depth = scope_depth;
            } else {
                panic!("expected nonempty locals");
            }
            true
        } else {
            false
        }
    }

    fn identifier_equal(id1: &Option<scanner::Literal>, name2: &String) -> bool {
        match id1 {
            Some(scanner::Literal::Identifier(name1)) => name1 == name2,
            _ => {
                panic!(
                    "expected identifier in 'identifiers_equal' but found {:?}.",
                    id1
                );
            }
        }
    }

    fn define_var(&mut self, global_idx: usize) {
        if self.mark_initialized() {
            return;
        }
        let line = self.previous().line;
        self.emit_op(bytecode::Op::DefineGlobal(global_idx), line);
    }

    fn declare_variable(&mut self) -> Result<(), String> {
        if self.scope_depth() == 0 {
            return Ok(());
        }

        let name = self.previous().clone();

        let has_redeclaration = self.locals().iter().rev().any(|local| {
            local.depth != -1
                && local.depth == self.scope_depth()
                && Compiler::identifiers_equal(&local.name.literal, &name.literal)
        });
        if has_redeclaration {
            return Err(format!(
                "Redeclartion of var {} in same scope.",
                String::from_utf8(name.lexeme).unwrap()
            ));
        }
        self.add_local(name);
        Ok(())
    }

    fn identifiers_equal(id1: &Option<scanner::Literal>, id2: &Option<scanner::Literal>) -> bool {
        match (id1, id2) {
            (
                Some(scanner::Literal::Identifier(name1)),
                Some(scanner::Literal::Identifier(name2)),
            ) => name1 == name2,
            _ => {
                panic!(
                    "Expected identifiers in 'identifiers_equal' but found {:?} and {:?}",
                    id1, id2
                );
            }
        }
    }

    fn add_local(&mut self, name: scanner::Token) {
        self.locals_mut().push(Local {
            name,
            depth: -1, //not yet defined (refer to edge case)
            is_captured: false,
        });
    }

    fn parse_variable(&mut self, error_msg: &str) -> Result<usize, String> {
        self.consume(scanner::TokenType::Identifier, error_msg)?;
        self.declare_variable()?;

        if self.scope_depth() > 0 {
            return Ok(0);
        }

        if let Some(scanner::Literal::Identifier(name)) = &self.previous().literal.clone() {
            Ok(self.identifier_constant(name.clone()))
        } else {
            panic!(
                "expected identifier when parsing variable, found {:?}",
                self.previous()
            );
        }
    }

    fn identifier_constant(&mut self, name: String) -> usize {
        self.current_chunk().add_constant_string(name)
    }

    fn statement(&mut self) -> Result<(), String> {
        if self.matches(scanner::TokenType::Print) {
            self.print_statement()?;
        } else if self.matches(scanner::TokenType::For) {
            self.for_statement()?;
        } else if self.matches(scanner::TokenType::If) {
            self.if_statement()?;
        } else if self.matches(scanner::TokenType::Return) {
            self.return_statement()?;
        } else if self.matches(scanner::TokenType::While) {
            self.while_statement()?;
        } else if self.matches(scanner::TokenType::LeftBrace) {
            self.begin_scope();
            self.block()?;
            self.end_scope();
        } else {
            self.expression_statement()?;
        }
        Ok(())
    }

    fn for_statement(&mut self) -> Result<(), String> {
        self.begin_scope();
        self.consume(scanner::TokenType::LeftParen, "Expected '(' after 'for'.")?;
        if self.matches(scanner::TokenType::Semicolon) {
        } else if self.matches(scanner::TokenType::Var) {
            self.var_decl()?;
        } else {
            self.expression_statement()?;
        }

        let mut loop_start = self.current_chunk().code.len();
        let mut maybe_exit_jump = None;
        if !self.matches(scanner::TokenType::Semicolon) {
            self.expression()?;
            self.consume(
                scanner::TokenType::Semicolon,
                "expected ';' after loop condition",
            )?;
            maybe_exit_jump = Some(self.emit_jump(bytecode::Op::JumpIfFalse(0)));
            self.emit_op(bytecode::Op::Pop, self.previous().line);
        }
        let maybe_exit_jump = maybe_exit_jump;
        if !self.matches(scanner::TokenType::RightParen) {
            let body_jump = self.emit_jump(bytecode::Op::Jump(0));
            let increment_start = self.current_chunk().code.len() + 1;
            self.expression()?;
            self.emit_op(bytecode::Op::Pop, self.previous().line);
            self.consume(
                scanner::TokenType::RightParen,
                "Expected ')' after for clauses",
            )?;
            self.emit_loop(loop_start);
            loop_start = increment_start;
            self.patch_jump(body_jump);
        }
        self.statement()?;
        self.emit_loop(loop_start);
        if let Some(exit_jump) = maybe_exit_jump {
            self.patch_jump(exit_jump);
            self.emit_op(bytecode::Op::Pop, self.previous().line);
        }
        self.end_scope();

        Ok(())
    }

    fn if_statement(&mut self) -> Result<(), String> {
        self.consume(scanner::TokenType::LeftParen, "Expected '(' after 'if'.")?;
        self.expression()?;
        self.consume(
            scanner::TokenType::RightParen,
            "Expected ')' after condition.",
        )?;
        let then_jump = self.emit_jump(bytecode::Op::JumpIfFalse(0));
        self.emit_op(bytecode::Op::Pop, self.previous().line);
        self.statement()?;
        let else_jump = self.emit_jump(bytecode::Op::Jump(0));

        self.patch_jump(then_jump);
        self.emit_op(bytecode::Op::Pop, self.previous().line);
        if self.matches(scanner::TokenType::Else) {
            self.statement()?;
        }
        self.patch_jump(else_jump);
        Ok(())
    }

    fn return_statement(&mut self) -> Result<(), String> {
        if self.function_type() == FunctionType::Script {
            return Err("can't return from top level code".to_string());
        }
        if self.matches(scanner::TokenType::Semicolon) {
            self.emit_return();
        } else {
            self.expression()?;
            self.consume(
                scanner::TokenType::Semicolon,
                "Expected ';' after return value.",
            )?;
            self.emit_op(bytecode::Op::Return, self.previous().line);
        }
        Ok(())
    }

    fn patch_jump(&mut self, jump_loc: usize) {
        let true_jump = self.current_chunk().code.len() - jump_loc - 1;
        let (maybe_jump, lineno) = &self.current_chunk().code[jump_loc];
        if let bytecode::Op::JumpIfFalse(_) = maybe_jump {
            self.current_chunk().code[jump_loc] = (bytecode::Op::JumpIfFalse(true_jump), *lineno);
        } else if let bytecode::Op::Jump(_) = maybe_jump {
            self.current_chunk().code[jump_loc] = (bytecode::Op::Jump(true_jump), *lineno);
        } else {
            panic!(
                "patch jump attempted but couldnt find jump. Found {:?}.",
                maybe_jump
            );
        }
    }

    fn emit_jump(&mut self, op: bytecode::Op) -> usize {
        self.emit_op(op, self.previous().line);
        self.current_chunk().code.len() - 1
    }

    fn emit_loop(&mut self, loop_start: usize) {
        let offset = self.current_chunk().code.len() - loop_start + 2;
        self.emit_op(bytecode::Op::Loop(offset), self.previous().line);
    }

    fn while_statement(&mut self) -> Result<(), String> {
        let loop_start = self.current_chunk().code.len();
        self.consume(scanner::TokenType::LeftParen, "expected '(' after 'while'.")?;
        self.expression()?;
        self.consume(
            scanner::TokenType::RightParen,
            "Expected ')' after condition",
        )?;
        let exit_jump = self.emit_jump(bytecode::Op::JumpIfFalse(0));
        self.emit_op(bytecode::Op::Pop, self.previous().line);
        self.statement()?;
        self.emit_loop(loop_start);
        self.patch_jump(exit_jump);
        self.emit_op(bytecode::Op::Pop, self.previous().line);
        Ok(())
    }

    fn matches(&mut self, ty: scanner::TokenType) -> bool {
        if self.check(ty) {
            self.advance();
            return true;
        }
        false
    }

    fn block(&mut self) -> Result<(), String> {
        while !self.check(scanner::TokenType::RightBrace) && !self.check(scanner::TokenType::Eof) {
            self.declaration()?;
        }
        if let Err(err) = self.consume(scanner::TokenType::RightBrace, "expected '}' after block") {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn begin_scope(&mut self) {
        self.current_level_mut().scope_depth += 1;
    }

    fn end_scope(&mut self) {
        self.current_level_mut().scope_depth -= 1;
        println!("scope ended");
        let mut pop_count = 0;
        for local in self.locals().iter().rev() {
            if local.depth > self.scope_depth() {
                pop_count += 1
            } else {
                break;
            }
        }
        let pop_count = pop_count;

        let line = self.previous().line;
        for _ in 0..pop_count {
            let local = self.locals_mut().pop().unwrap();
            println!("local: {:?}", local);
            if local.is_captured {
                self.emit_op(bytecode::Op::CloseUpvalue, line);
            } else {
                self.emit_op(bytecode::Op::Pop, line);
            }
        }
    }

    fn expression_statement(&mut self) -> Result<(), String> {
        self.expression()?;
        self.consume(
            scanner::TokenType::Semicolon,
            "expected ';' after expression.",
        )?;
        let line = self.previous().line;
        self.emit_op(bytecode::Op::Pop, line);
        Ok(())
    }

    fn print_statement(&mut self) -> Result<(), String> {
        self.expression()?;
        self.consume(scanner::TokenType::Semicolon, "expected ';' after val")?;
        self.emit_op(bytecode::Op::Print, self.previous().clone().line);
        Ok(())
    }

    fn check(&self, ty: scanner::TokenType) -> bool {
        if self.is_at_end() {
            return false;
        }

        self.peek().ty == ty
    }

    fn expression(&mut self) -> Result<(), String> {
        self.parse_precendce(Precedence::Assignment)
    }

    fn grouping(&mut self, _can_assign: bool) -> Result<(), String> {
        self.expression()?;
        if let Err(err) = self.consume(
            scanner::TokenType::RightParen,
            "Expected ')' after expression.",
        ) {
            Err(err)
        } else {
            Ok(())
        }
    }

    fn number(&mut self, _can_assign: bool) -> Result<(), String> {
        let tok = self.previous().clone();
        match tok.literal {
            Some(scanner::Literal::Number(n)) => {
                self.emit_number(n, tok.line);
                Ok(())
            }
            _ => panic!(
                "Expected number at line ={}, col = {}. Current token {:?}",
                tok.line, tok.col, tok
            ),
        }
    }

    fn string(&mut self, _can_assign: bool) -> Result<(), String> {
        let tok = self.previous().clone();
        match tok.literal {
            Some(scanner::Literal::Str(s)) => {
                let const_idx = self.current_chunk().add_constant_string(s);
                self.emit_op(bytecode::Op::Constant(const_idx), tok.line);
                Ok(())
            }
            _ => panic!("expected literal when parsing string"),
        }
    }

    fn literal(&mut self, _can_assign: bool) -> Result<(), String> {
        let tok = self.previous().clone();
        match tok.ty {
            scanner::TokenType::Nil => {
                self.emit_op(bytecode::Op::Nil, tok.line);
                Ok(())
            }
            scanner::TokenType::True => {
                self.emit_op(bytecode::Op::True, tok.line);
                Ok(())
            }
            scanner::TokenType::False => {
                self.emit_op(bytecode::Op::False, tok.line);
                Ok(())
            }
            _ => {
                panic!("shouldn't get in literal with tok = {:?}.", tok);
            }
        }
    }

    fn variable(&mut self, _can_assign: bool) -> Result<(), String> {
        let tok = self.previous().clone();
        self.name_variable(tok, _can_assign)
    }

    fn name_variable(&mut self, tok: scanner::Token, _can_assign: bool) -> Result<(), String> {
        if tok.ty != scanner::TokenType::Identifier {
            return Err("expected identifier".to_string());
        }
        if let Some(scanner::Literal::Identifier(name)) = tok.literal.clone() {
            let get_op: bytecode::Op;
            let set_op: bytecode::Op;
            match self.resolve_local(&name) {
                Ok(Some(idx)) => {
                    get_op = bytecode::Op::GetLocal(idx);
                    set_op = bytecode::Op::SetLocal(idx);
                }
                Ok(None) => {
                    let idx = self.identifier_constant(name.clone());
                    get_op = bytecode::Op::GetGlobal(idx);
                    set_op = bytecode::Op::SetGlobal(idx);
                }
                Err(err) => {
                    return Err(err);
                }
            }
            if _can_assign && self.matches(scanner::TokenType::Equal) {
                self.expression()?;
                self.emit_op(set_op, tok.line);
            } else {
                self.emit_op(get_op, tok.line);
            }
            Ok(())
        } else {
            panic!("expected identifier when parsing var, found {:?}", tok);
        }
    }

    fn resolve_local(&self, name: &String) -> Result<Option<usize>, String> {
        Compiler::resolve_local_static(self.current_level(), name, self.previous())
    }

    fn resolve_local_static(
        level: &Level,
        name: &String,
        prev_tok: &scanner::Token,
    ) -> Result<Option<usize>, String> {
        for (idx, local) in level.locals.iter().rev().enumerate() {
            if Compiler::identifier_equal(&local.name.literal, name) {
                if local.depth == -1 {
                    return Err(Compiler::error_at_tok(
                        "Cannot read local var in its own initializor",
                        prev_tok,
                    ));
                }
                return Ok(Some(level.locals.len() - 2 - idx));
            }
        }
        Ok(None)
    }

    fn resolve_var(&mut self, name: &String) -> Result<Resolution, String> {
        if let Some(idx) = self.resolve_local(name)? {
            return Ok(Resolution::Local(idx));
        }
        if let Some(idx) = self.resolve_upval(name)? {
            return Ok(Resolution::UpValue(idx));
        }

        Ok(Resolution::Global)
    }

    fn resolve_upval(&mut self, name: &String) -> Result<Option<usize>, String> {
        if self.level_idx < 1 {
            return Ok(None);
        }
        if let Some(local_idx) =
            Compiler::resolve_local_static(&self.levels[self.level_idx - 1], name, self.previous())?
        {
            return Ok(Some(self.add_upval(bytecode::UpvalueLoc::Local(local_idx))));
        }

        self.level_idx -= 1;
        if let Some(upval_idx) = self.resolve_upval(name)? {
            self.current_level_mut().locals[upval_idx].is_captured = true;
            self.level_idx += 1;
            return Ok(Some(
                self.add_upval(bytecode::UpvalueLoc::Upvalue(upval_idx)),
            ));
        }
        self.level_idx += 1;

        Ok(None)
    }

    fn add_upval(&mut self, upvalue: bytecode::UpvalueLoc) -> usize {
        if let Some(res) = self
            .current_level()
            .upvals
            .iter()
            .position(|query_upval| *query_upval == upvalue)
        {
            return res;
        }
        self.current_level_mut().upvals.push(upvalue);
        self.current_level().upvals.len() - 1
    }

    //table colum for infix parse used here
    fn binary(&mut self, _can_assign: bool) -> Result<(), String> {
        let operator = self.previous().clone();
        let rule = Compiler::get_rule(operator.ty);
        self.parse_precendce(Compiler::next_precedence(rule.precendence))?;

        match operator.ty {
            scanner::TokenType::Plus => {
                self.emit_op(bytecode::Op::Add, operator.line);
                Ok(())
            }
            scanner::TokenType::Minus => {
                self.emit_op(bytecode::Op::Subtract, operator.line);
                Ok(())
            }
            scanner::TokenType::Star => {
                self.emit_op(bytecode::Op::Multiply, operator.line);
                Ok(())
            }
            scanner::TokenType::Slash => {
                self.emit_op(bytecode::Op::Divide, operator.line);
                Ok(())
            }
            scanner::TokenType::BangEqual => {
                self.emit_op(bytecode::Op::Equal, operator.line);
                self.emit_op(bytecode::Op::Not, operator.line);
                Ok(())
            }
            scanner::TokenType::EqualEqual => {
                self.emit_op(bytecode::Op::Equal, operator.line);
                Ok(())
            }
            scanner::TokenType::Greater => {
                self.emit_op(bytecode::Op::Greater, operator.line);
                Ok(())
            }
            scanner::TokenType::GreaterEqual => {
                self.emit_op(bytecode::Op::Less, operator.line);
                self.emit_op(bytecode::Op::Not, operator.line);
                Ok(())
            }
            scanner::TokenType::Less => {
                self.emit_op(bytecode::Op::Less, operator.line);
                Ok(())
            }
            scanner::TokenType::LessEqual => {
                self.emit_op(bytecode::Op::Greater, operator.line);
                self.emit_op(bytecode::Op::Not, operator.line);
                Ok(())
            }

            _ => Err(format!(
                "Invalid token {:?} in binary expression at line={}, col= {}.",
                operator.ty, operator.line, operator.col
            )),
        }
    }

    fn unary(&mut self, _can_assign: bool) -> Result<(), String> {
        let operator = self.previous().clone();

        self.parse_precendce(Precedence::Unary)?;
        match operator.ty {
            scanner::TokenType::Minus => {
                self.emit_op(bytecode::Op::Negate, operator.line);
                Ok(())
            }
            scanner::TokenType::Bang => {
                self.emit_op(bytecode::Op::Not, operator.line);
                Ok(())
            }
            _ => Err(format!(
                "Invalid token in unary op {:?} at line={}, col={}",
                operator.ty, operator.line, operator.col
            )),
        }
    }

    fn and_(&mut self, _can_assign: bool) -> Result<(), String> {
        let end_jump = self.emit_jump(bytecode::Op::JumpIfFalse(0));
        self.emit_op(bytecode::Op::Pop, self.previous().line);
        self.parse_precendce(Precedence::And)?;
        self.patch_jump(end_jump);
        Ok(())
    }

    fn or(&mut self, _can_assign: bool) -> Result<(), String> {
        let else_jump = self.emit_jump(bytecode::Op::JumpIfFalse(0));
        let end_jump = self.emit_jump(bytecode::Op::Jump(0));
        self.patch_jump(else_jump);
        self.emit_op(bytecode::Op::Pop, self.previous().line);
        self.parse_precendce(Precedence::Or)?;
        self.patch_jump(end_jump);
        Ok(())
    }

    fn call(&mut self, _can_assign: bool) -> Result<(), String> {
        let arg_count = self.argument_list()?;
        self.emit_op(bytecode::Op::Call(arg_count), self.previous().line);
        Ok(())
    }
    fn argument_list(&mut self) -> Result<u8, String> {
        let mut arg_count: u8 = 0;
        if !self.check(scanner::TokenType::RightParen) {
            loop {
                self.expression()?;
                arg_count += 1;
                if !self.matches(scanner::TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(
            scanner::TokenType::RightParen,
            "Expected ')' after argument list",
        )?;
        Ok(arg_count)
    }

    fn emit_number(&mut self, n: f64, lineno: usize) {
        let const_idx = self.current_chunk().add_constant_number(n);
        self.emit_op(bytecode::Op::Constant(const_idx), lineno);
    }

    fn emit_op(&mut self, op: bytecode::Op, lineno: usize) {
        self.current_chunk()
            .code
            .push((op, bytecode::Lineno(lineno)))
    }

    fn emit_return(&mut self) {
        self.emit_op(bytecode::Op::Nil, self.previous().line);
        self.emit_op(bytecode::Op::Return, self.previous().line);
    }

    fn consume(
        &mut self,
        tok: scanner::TokenType,
        on_err_str: &str,
    ) -> Result<&scanner::Token, String> {
        if self.check(tok) {
            return Ok(self.advance());
        }
        Err(format!(
            "Expected token {:?}, but found token {:?} at line={}, col={}: {} ",
            tok,
            self.peek().ty,
            self.peek().line,
            self.peek().col,
            on_err_str
        ))
    }

    fn current_chunk(&mut self) -> &mut bytecode::Chunk {
        &mut self.current_level_mut().function.chunk
    }

    fn current_level(&self) -> &Level {
        &self.levels[self.level_idx]
    }

    fn current_level_mut(&mut self) -> &mut Level {
        &mut self.levels[self.level_idx]
    }

    fn push_level(&mut self, level: Level) {
        self.levels.push(level);
        self.level_idx += 1;
    }

    fn pop_level(&mut self) {
        self.levels.pop();
        self.level_idx -= 1;
    }

    fn current_function_mut(&mut self) -> &mut bytecode::Function {
        &mut self.current_level_mut().function
    }

    fn function_type(&self) -> FunctionType {
        self.current_level().function_type
    }

    fn scope_depth(&self) -> i64 {
        self.current_level().scope_depth
    }

    fn locals(&self) -> &Vec<Local> {
        &self.current_level().locals
    }

    fn locals_mut(&mut self) -> &mut Vec<Local> {
        &mut self.current_level_mut().locals
    }

    fn parse_precendce(&mut self, precendence: Precedence) -> Result<(), String> {
        self.advance();
        let can_assign = precendence <= Precedence::Assignment;
        match Compiler::get_rule(self.previous().ty).prefix {
            Some(parse_fn) => self.apply_parse_fn(parse_fn, can_assign)?,
            None => {
                return Err(self.error("Expected expression."));
            }
        }
        while precendence <= Compiler::get_rule(self.peek().ty).precendence {
            self.advance();
            match Compiler::get_rule(self.previous().ty).infix {
                Some(parse_fn) => self.apply_parse_fn(parse_fn, can_assign)?,
                None => panic!("could not find infix rule to apply tok = {:?}", self.peek()),
            }
        }
        if can_assign && self.matches(scanner::TokenType::Equal) {
            return Err(self.error("invalid assignment target"));
        }
        Ok(())
    }

    fn error(&self, error_is: &str) -> String {
        Compiler::error_at_tok(error_is, self.previous())
    }

    fn error_at_tok(error_is: &str, prev_tok: &scanner::Token) -> String {
        format!(
            "{} at line={}, col={}.",
            error_is, prev_tok.line, prev_tok.col
        )
    }

    fn apply_parse_fn(&mut self, parse_fn: ParseFn, _can_assign: bool) -> Result<(), String> {
        match parse_fn {
            ParseFn::Grouping => self.grouping(_can_assign),
            ParseFn::Unary => self.unary(_can_assign),
            ParseFn::Binary => self.binary(_can_assign),
            ParseFn::Number => self.number(_can_assign),
            ParseFn::Literal => self.literal(_can_assign),
            ParseFn::String => self.string(_can_assign),
            ParseFn::Variable => self.variable(_can_assign),
            ParseFn::And => self.and_(_can_assign),
            ParseFn::Or => self.or(_can_assign),
            ParseFn::Call => self.call(_can_assign),
        }
    }

    //bunch of helper fns

    fn advance(&mut self) -> &scanner::Token {
        if !self.is_at_end() {
            self.token_idx += 1
        }

        self.previous()
    }

    fn previous(&self) -> &scanner::Token {
        &self.tokens[self.token_idx - 1]
    }

    fn is_at_end(&self) -> bool {
        self.peek().ty == scanner::TokenType::Eof
    }

    fn peek(&self) -> &scanner::Token {
        &self.tokens[self.token_idx]
    }

    fn next_precedence(precendence: Precedence) -> Precedence {
        match precendence {
            Precedence::None => Precedence::Assignment,
            Precedence::Assignment => Precedence::Or,
            Precedence::Or => Precedence::And,
            Precedence::And => Precedence::Equality,
            Precedence::Equality => Precedence::Comparison,
            Precedence::Comparison => Precedence::Term,
            Precedence::Term => Precedence::Factor,
            Precedence::Factor => Precedence::Unary,
            Precedence::Unary => Precedence::Call,
            Precedence::Call => Precedence::Primary,
            Precedence::Primary => panic!("primary has no next precendence"),
        }
    }
    //Pratts parse table
    fn get_rule(operator: scanner::TokenType) -> ParseRule {
        match operator {
            scanner::TokenType::LeftParen => ParseRule {
                prefix: Some(ParseFn::Grouping),
                infix: Some(ParseFn::Call),
                precendence: Precedence::Call,
            },
            scanner::TokenType::RightParen => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::LeftBrace => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::RightBrace => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Comma => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Dot => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            //Negation prefix, subtraction infix
            scanner::TokenType::Minus => ParseRule {
                prefix: Some(ParseFn::Unary),
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Term,
            },
            scanner::TokenType::Plus => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Term,
            },
            scanner::TokenType::Semicolon => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Slash => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Factor,
            },
            scanner::TokenType::Star => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Factor,
            },
            scanner::TokenType::Bang => ParseRule {
                prefix: Some(ParseFn::Unary),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::BangEqual => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Equal => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::EqualEqual => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Equality,
            },
            scanner::TokenType::Greater => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Comparison,
            },
            scanner::TokenType::GreaterEqual => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Comparison,
            },
            scanner::TokenType::Less => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Comparison,
            },
            scanner::TokenType::LessEqual => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Binary),
                precendence: Precedence::Comparison,
            },
            scanner::TokenType::Identifier => ParseRule {
                prefix: Some(ParseFn::Variable),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::String => ParseRule {
                prefix: Some(ParseFn::String),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Number => ParseRule {
                prefix: Some(ParseFn::Number),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::And => ParseRule {
                prefix: None,
                infix: Some(ParseFn::And),
                precendence: Precedence::And,
            },
            scanner::TokenType::Class => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Else => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::False => ParseRule {
                prefix: Some(ParseFn::Literal),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::For => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Fun => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::If => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Nil => ParseRule {
                prefix: Some(ParseFn::Literal),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Or => ParseRule {
                prefix: None,
                infix: Some(ParseFn::Or),
                precendence: Precedence::Or,
            },
            scanner::TokenType::Print => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Return => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },

            scanner::TokenType::Super => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::This => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::True => ParseRule {
                prefix: Some(ParseFn::Literal),
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Var => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::While => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
            scanner::TokenType::Eof => ParseRule {
                prefix: None,
                infix: None,
                precendence: Precedence::None,
            },
        }
    }
}
