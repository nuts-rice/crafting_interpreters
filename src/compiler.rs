use crate::bytecode;
use crate::scanner;

#[derive(Default)]
pub struct Compiler {
    tokens: Vec<scanner::Token>,
    current_chunk: bytecode::Chunk,
    current: usize,
    locals: Vec<Local>,
    scope_depth: i64,
}

struct Local {
    name: scanner::Token,
    depth: i64,
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
    pub fn compile(&mut self, input: String) -> Result<bytecode::Chunk, String> {
        match scanner::scan_tokens(input) {
            Ok(tokens) => {
                self.tokens = tokens;
                self.current_chunk = bytecode::Chunk::default();
                while !self.is_at_end() {
                    self.declaration()?;
                }
                Ok(std::mem::take(&mut self.current_chunk))
            }
            Err(err) => Err(err),
        }
    }

    fn declaration(&mut self) -> Result<(), String> {
        if self.matches(scanner::TokenType::Var) {
            self.var_decl()
        } else {
            self.statement()
        }
    }

    fn var_decl(&mut self) -> Result<(), String> {
        let global_idx = self.parse_variable("expected var name")?;
        if self.matches(scanner::TokenType::Equal) {
            self.expression()?;
        } else {
            let line = self.previous().line;
            self.emit_op(bytecode::Op::Nil, line)
        }

        if let Err(err) = self.consume(
            scanner::TokenType::Semicolon,
            "expected ';' after var declaration",
        ) {
            return Err(err);
        }

        self.define_global(global_idx);
        Ok(())
    }

    fn define_global(&mut self, global_idx: usize) {
        if self.scope_depth > 0 {
            return;
        }
        let line = self.previous().line;
        self.emit_op(bytecode::Op::DefineGlobal(global_idx), line);
    }

    fn declare_variable(&mut self) -> Result<(), String> {
        if self.scope_depth == 0 {
            return Ok(());
        }

        let name = self.previous().clone();

        if self.locals.iter().rev().any(|local| {
            local.depth != -1
                && local.depth == self.scope_depth
                && Compiler::identifiers_equal(&local.name.literal, &name.literal)
        }) {
            return Err(format!(
                "Redeclartion of var {} in same scope.",
                String::from_utf8(name.lexeme).unwrap()
            ));
        }
        self.add_local(name);
        Ok(())
    }

    fn identifiers_equal(id: &Option<scanner::Literal>, id2: &Option<scanner::Literal>) -> bool {
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
        self.locals.push(Local {
            name,
            depth: self.scope_depth,
        });
    }

    fn parse_variable(&mut self, error_msg: &str) -> Result<usize, String> {
        if let Err(err) = self.consume(scanner::TokenType::Identifier, error_msg) {
            return Err(err);
        }

        if let Err(err) = self.declare_variable() {
            return Err(err);
        }
        if self.scope_depth > 0 {
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
        self.current_chunk.add_constant_string(name)
    }

    fn statement(&mut self) -> Result<(), String> {
        if self.matches(scanner::TokenType::Print) {
            self.print_statement()?;
        } else if self.matchs(scanner::TokenType::LeftBrace) {
            self.begin_scope();
            self.block()?;
            self.end_scope();
        } else {
            self.expression_statement()?;
        }
        Ok(())
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
        self.scope_depth += 1;
    }

    fn end_scope(&mut self) {
        self.scope_depth -= 1;
    }

    fn expression_statement(&mut self) -> Result<(), String> {
        self.expression()?;
        if let Err(err) = self.consume(
            scanner::TokenType::Semicolon,
            "expected ';' after expression.",
        ) {
            return Err(err);
        }
        let line = self.previous().line;
        self.emit_op(bytecode::Op::Pop, line);
        Ok(())
    }

    fn print_statement(&mut self) -> Result<(), String> {
        self.expression()?;
        if let Err(err) = self.consume(scanner::TokenType::Semicolon, "expected ';' after val") {
            return Err(err);
        }
        self.emit_op(bytecode::Op::Print, self.previous().clone().line);
        Ok(())
    }

    fn matches(&mut self, ty: scanner::TokenType) -> bool {
        if self.check(ty) {
            self.advance();
            return true;
        }
        false
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
                let const_idx = self.current_chunk.add_constant_string(s);
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
            let idx = self.identifier_constant(name);
            if _can_assign && self.matches(scanner::TokenType::Equal) {
                self.expression()?;
                self.emit_op(bytecode::Op::SetGlobal(idx), tok.line);
            } else {
                self.emit_op(bytecode::Op::GetGlobal(idx), tok.line);
            }
            Ok(())
        } else {
            panic!("expected identifier when parsing var, found {:?}", tok);
        }
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

    fn emit_number(&mut self, n: f64, lineno: usize) {
        let const_idx = self.current_chunk.add_constant_number(n);
        self.emit_op(bytecode::Op::Constant(const_idx), lineno);
    }

    fn emit_op(&mut self, op: bytecode::Op, lineno: usize) {
        self.current_chunk.code.push((op, bytecode::Lineno(lineno)))
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

    fn parse_precendce(&mut self, precendence: Precedence) -> Result<(), String> {
        self.advance();
        let can_assign = precendence <= Precedence::Assignment;
        match Compiler::get_rule(self.previous().ty).prefix {
            Some(parse_fn) => self.apply_parse_fn(parse_fn, can_assign)?,
            None => {
                return Err(self.error("Expected expression."));
            }
        }

        while precendence <= Compiler::get_rule(self.current_tok().ty).precendence {
            println!("{:?} {:?}", self.current_tok(), precendence);
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
        let tok = self.previous();
        format!("{} at line={}, col ={}.", error_is, tok.line, tok.col)
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
        }
    }

    //bunch of helper fns

    fn advance(&mut self) -> &scanner::Token {
        if !self.is_at_end() {
            self.current += 1
        }

        self.previous()
    }

    fn current_tok(&self) -> &scanner::Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &scanner::Token {
        &self.tokens[self.current - 1]
    }

    fn is_at_end(&self) -> bool {
        self.peek().ty == scanner::TokenType::Eof
    }

    fn peek(&self) -> &scanner::Token {
        &self.tokens[self.current]
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
                infix: None,
                precendence: Precedence::None,
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
                infix: None,
                precendence: Precedence::None,
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
                infix: None,
                precendence: Precedence::None,
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
