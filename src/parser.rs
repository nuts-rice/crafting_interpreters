use crate::expr;
use crate::scanner;

#[allow(dead_code)]
#[derive(Default)]
struct Parser {
    tokens: Vec<scanner::Token>,
    current: usize,
}

pub fn parse(tokens: Vec<scanner::Token>) -> Result<Vec<expr::Stmt>, String> {
    let mut p = Parser {
        tokens,
        ..Default::default()
    };

    let stmts_or_err = p.parse();

    match stmts_or_err {
        Ok(stmts_or_err) => {
            if !p.is_at_end() {
                let tok = &p.tokens[p.current];
                Err(format!(
                    "Unexpected token of type {:?} at line={}, col={} ",
                    tok.ty, tok.line, tok.col
                ))
            } else {
                Ok(stmts_or_err)
            }
        }
        Err(err) => Err(err),
    }
}

//Grammar
//program   → declaration* EOF ;
//declartion → VarDecl
//              | stamement;
//statement  → exprStmt
//             | printstmt
//             | ifStmt
//             | whileStmt
//             | block ;
//whileStmt  -> "while" "(" expression ")" statement;
//ifStmt -> "if" "(" expression ")" statement
//block -> "{" declaration* "}";
//varDecl -> "var" IDENTIFIER ("=" expression)? ";" ;
//printStmt → "print" expression ";" ;
//expression     → assignment ;
//assignment -> IDENTIFIER "=" assignment
//                        |logic_or;
//logic_or -> logic_and ( "or" logic_and )* ;
//logic_and -> equality ( "and" equality )* ;
//equality       → comparison ( ( "!=" | "==" ) comparison )* ;
//comparison     → addition ( ( ">" | ">=" | "<" | "<=" ) addition )* ;
//addition       → multiplication ( ( "-" | "+" ) multiplication )* ;
//multiplication → unary ( ( "/" | "*" ) unary )* ;
//unary          → ( "!" | "-" ) unary
//                    |primary
//primary        → NUMBER | STRING | "false" | "true" | "nil"
//                            | "(" expression ")" ;
//                            | IDENTIFIER ;

impl Parser {
    pub fn parse(&mut self) -> Result<Vec<expr::Stmt>, String> {
        let mut statements = Vec::new();

        while !self.is_at_end() {
            let stmt = self.declaration()?;
            statements.push(stmt);
        }

        Ok(statements)
    }

    fn declaration(&mut self) -> Result<expr::Stmt, String> {
        if self.matches(scanner::TokenType::Var) {
            return self.var_decl();
        }
        self.statement()
    }

    fn var_decl(&mut self) -> Result<expr::Stmt, String> {
        let name_token = self
            .consume(scanner::TokenType::Identifier, "Expected variable name")?
            .clone();

        let maybe_initializer = if self.matches(scanner::TokenType::Equal) {
            Some(self.expression()?)
        } else {
            None
        };

        self.consume(
            scanner::TokenType::Semicolon,
            "Expected ; after variable declaration",
        )?;

        Ok(expr::Stmt::VarDecl(
            expr::Symbol {
                name: String::from_utf8(name_token.lexeme).unwrap(),
                line: name_token.line,
                col: name_token.col,
            },
            maybe_initializer,
        ))
    }

    fn statement(&mut self) -> Result<expr::Stmt, String> {
        if self.matches(scanner::TokenType::Print) {
            return self.print_statement();
        }
        if self.matches(scanner::TokenType::LeftBrace) {
            return Ok(expr::Stmt::Block(self.block()?));
        }

        if self.matches(scanner::TokenType::If) {
            return self.if_statement();
        }
        if self.matches(scanner::TokenType::While) {
            return self.while_statement();
        }

        self.expression_statement()
    }

    fn while_statement(&mut self) -> Result<expr::Stmt, String> {
        self.consume(scanner::TokenType::LeftParen, "Expected ( after while")?;
        let cond = self.expression()?;
        self.consume(
            scanner::TokenType::RightParen,
            "Expected ) after while condition",
        )?;
        let body = Box::new(self.statement()?);
        Ok(expr::Stmt::While(cond, body))
    }

    fn print_statement(&mut self) -> Result<expr::Stmt, String> {
        let expr = self.expression()?;
        self.consume(scanner::TokenType::Semicolon, "Expected ; after value ")?;
        Ok(expr::Stmt::Print(expr))
    }

    fn block(&mut self) -> Result<Vec<Box<expr::Stmt>>, String> {
        let mut stmts: Vec<Box<expr::Stmt>> = Vec::new();

        while !self.check(scanner::TokenType::RightBrace) && !self.is_at_end() {
            stmts.push(Box::new(self.declaration()?))
        }
        self.consume(scanner::TokenType::RightBrace, "Expected } after block")?;

        Ok(stmts)
    }

    fn if_statement(&mut self) -> Result<expr::Stmt, String> {
        self.consume(scanner::TokenType::LeftParen, "Expected ( after if")?;
        let cond = self.expression()?;
        self.consume(
            scanner::TokenType::RightParen,
            "Expected ) after if condition",
        )?;
        let then_branch = Box::new(self.statement()?);
        let maybe_else_branch = if self.matches(scanner::TokenType::Else) {
            Some(Box::new(self.statement()?))
        } else {
            None
        };

        Ok(expr::Stmt::If(cond, then_branch, maybe_else_branch))
    }

    fn expression_statement(&mut self) -> Result<expr::Stmt, String> {
        let expr = self.expression()?;
        self.consume(scanner::TokenType::Semicolon, "Expected ; after value")?;
        Ok(expr::Stmt::Expr(expr))
    }

    fn expression(&mut self) -> Result<expr::Expr, String> {
        self.assignment()
    }

    fn assignment(&mut self) -> Result<expr::Expr, String> {
        let expr = self.or()?;

        if self.matches(scanner::TokenType::Equal) {
            let equals = self.previous().clone();
            let value = self.assignment()?;

            if let expr::Expr::Variable(sym) = &value {
                return Ok(expr::Expr::Assign(sym.clone(), Box::new(value)));
            } else {
                return Err(format!(
                    "invalid assignment target at line={}, col={}",
                    equals.line, equals.col
                ));
            }
        }
        Ok(expr)
    }

    fn or(&mut self) -> Result<expr::Expr, String> {
        let mut expr = self.and()?;

        while self.matches(scanner::TokenType::Or) {
            let right = self.and()?;
            expr = expr::Expr::Logical(Box::new(expr), expr::LogicalOp::Or, Box::new(right));
        }
        Ok(expr)
    }

    fn and(&mut self) -> Result<expr::Expr, String> {
        let mut expr = self.equality()?;

        while self.matches(scanner::TokenType::And) {
            let right = self.equality()?;
            expr = expr::Expr::Logical(Box::new(expr), expr::LogicalOp::And, Box::new(right));
        }
        Ok(expr)
    }

    fn comparison(&mut self) -> Result<expr::Expr, String> {
        let mut expr = self.addition()?;

        while self.match_one_of(vec![
            scanner::TokenType::Greater,
            scanner::TokenType::GreaterEqual,
            scanner::TokenType::Less,
            scanner::TokenType::LessEqual,
        ]) {
            let operator_token = self.previous().clone();
            let right = Box::new(self.addition()?);

            let binop_maybe = Parser::op_token_to_binop(&operator_token);

            match binop_maybe {
                Ok(binop) => {
                    let left = Box::new(expr);
                    expr = expr::Expr::Binary(left, binop, right);
                }
                Err(err) => return Err(err),
            }
        }
        Ok(expr)
    }

    fn addition(&mut self) -> Result<expr::Expr, String> {
        let mut expr = self.multiplication()?;

        while self.match_one_of(vec![scanner::TokenType::Minus, scanner::TokenType::Plus]) {
            let operator_token = self.previous().clone();
            let right = Box::new(self.multiplication()?);

            let binop_maybe = Parser::op_token_to_binop(&operator_token);

            match binop_maybe {
                Ok(binop) => {
                    let left = Box::new(expr);
                    expr = expr::Expr::Binary(left, binop, right);
                }
                Err(err) => return Err(err),
            }
        }

        Ok(expr)
    }

    fn multiplication(&mut self) -> Result<expr::Expr, String> {
        let mut expr = self.unary()?;
        while self.match_one_of(vec![scanner::TokenType::Slash, scanner::TokenType::Star]) {
            let operator_token = self.previous().clone();
            let right = Box::new(self.unary()?);

            let binop_maybe = Parser::op_token_to_binop(&operator_token);
            match binop_maybe {
                Ok(binop) => {
                    let left = Box::new(expr);
                    expr = expr::Expr::Binary(left, binop, right);
                }
                Err(err) => return Err(err),
            }
        }
        Ok(expr)
    }

    fn unary(&mut self) -> Result<expr::Expr, String> {
        if self.match_one_of(vec![scanner::TokenType::Bang, scanner::TokenType::Minus]) {
            let operator_token = self.previous().clone();
            let right = Box::new(self.unary()?);
            let unary_op_maybe = Parser::op_token_to_unary_op(&operator_token);
            return match unary_op_maybe {
                Ok(unary_op) => Ok(expr::Expr::Unary(unary_op, right)),
                Err(err) => Err(err),
            };
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<expr::Expr, String> {
        if self.matches(scanner::TokenType::False) {
            return Ok(expr::Expr::Literal(expr::Literal::False));
        }
        if self.matches(scanner::TokenType::True) {
            return Ok(expr::Expr::Literal(expr::Literal::True));
        }
        if self.matches(scanner::TokenType::Nil) {
            return Ok(expr::Expr::Literal(expr::Literal::Nil));
        }
        if self.matches(scanner::TokenType::Number) {
            match &self.previous().literal {
                Some(scanner::Literal::Number(n)) => {
                    return Ok(expr::Expr::Literal(expr::Literal::Number(*n)))
                }
                Some(l) => panic!(
                    "internal error in parser: while parsing number, found literal {:?}",
                    l
                ),
                None => panic!("internal error in parser: when parsing number, found no literal"),
            }
        }
        if self.matches(scanner::TokenType::String) {
            match &self.previous().literal {
                Some(scanner::Literal::Str(s)) => {
                    return Ok(expr::Expr::Literal(expr::Literal::String(s.clone())))
                }
                Some(l) => panic!(
                    "internal error in parser: when parsing string, found literal {:?}",
                    l
                ),
                None => panic!("internal error in parser: when parsing string, found no literal"),
            }
        }
        if self.matches(scanner::TokenType::Identifier) {
            match &self.previous().literal {
                Some(scanner::Literal::Identifier(s)) => {
                    return Ok(expr::Expr::Variable(expr::Symbol {
                        name: s.clone(),
                        line: self.previous().line,
                        col: self.previous().col,
                    }))
                }
                Some(l) => panic!(
                    "internal error in parser: when parsing identifier, found literal {:?}",
                    l
                ),
                None => {
                    panic!("internal error in parser: when parsing identifier, found no literal")
                }
            }
        }
        if self.matches(scanner::TokenType::LeftParen) {
            let expr = Box::new(self.expression()?);
            if let Err(err) = self.consume(
                scanner::TokenType::RightParen,
                "Expected ')' after expression.",
            ) {
                return Err(err);
            }
            return Ok(expr::Expr::Grouping(expr));
        }
        Err(format!(
            "Expected expression, but found token {:?} at line={}, col={}",
            self.peek().ty,
            self.peek().line,
            self.peek().col
        ))
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
            "Expected token type {:?}, but found token {:?}: {}",
            tok,
            self.peek(),
            on_err_str
        ))
    }

    fn op_token_to_unary_op(tok: &scanner::Token) -> Result<expr::UnaryOp, String> {
        match tok.ty {
            scanner::TokenType::Minus => Ok(expr::UnaryOp {
                ty: expr::UnaryOpType::Minus,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Bang => Ok(expr::UnaryOp {
                ty: expr::UnaryOpType::Bang,
                line: tok.line,
                col: tok.col,
            }),
            _ => Err(format!(
                "invalid token in unary op {:?} at line={},col={}",
                tok.ty, tok.line, tok.col
            )),
        }
    }

    fn equality(&mut self) -> Result<expr::Expr, String> {
        let mut expr = self.comparison()?;

        while self.match_one_of(vec![
            scanner::TokenType::BangEqual,
            scanner::TokenType::EqualEqual,
        ]) {
            let operator_token = self.previous().clone();
            let right = Box::new(self.comparison()?);

            let binop_maybe = Parser::op_token_to_binop(&operator_token);

            match binop_maybe {
                Ok(binop) => {
                    let left = Box::new(expr);
                    expr = expr::Expr::Binary(left, binop, right);
                }
                Err(err) => return Err(err),
            }
        }
        Ok(expr)
    }

    fn op_token_to_binop(tok: &scanner::Token) -> Result<expr::BinaryOp, String> {
        match tok.ty {
            scanner::TokenType::EqualEqual => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::BangEqual => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Less => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::LessEqual => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Greater => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::GreaterEqual => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Plus => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Minus => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Star => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),
            scanner::TokenType::Slash => Ok(expr::BinaryOp {
                ty: expr::BinaryOpType::EqualEqual,
                line: tok.line,
                col: tok.col,
            }),

            _ => Err(format!(
                "invalid token in binary operation {:?} at line={}, col={}",
                tok.ty, tok.line, tok.col
            )),
        }
    }

    fn match_one_of(&mut self, types: Vec<scanner::TokenType>) -> bool {
        for ty in types.iter() {
            if self.matches(*ty) {
                return true;
            }
        }
        false
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

    fn advance(&mut self) -> &scanner::Token {
        if !self.is_at_end() {
            self.current += 1
        }

        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.peek().ty == scanner::TokenType::Eof
    }

    fn peek(&self) -> &scanner::Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &scanner::Token {
        &self.tokens[self.current - 1]
    }
}
