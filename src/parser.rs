use crate::ast::*;
use crate::lexer::*;
use crate::msg::*;
use crate::reader::Reader;
use crate::token::*;
use nom::combinator::value;
use nom::{
    branch::alt,
    bytes::complete::{escaped, tag},
    character::complete::{alpha1, anychar, none_of, one_of},
    combinator::{map, opt},
    multi::separated_list,
    number::complete::double,
    sequence::{pair, preceded, terminated, tuple},
    Err, IResult,
};
use std::mem;
//#[macro_use] extern crate nom;

pub struct Parser<'a> {
    lexer: Lexer,
    token: Token,
    ast: &'a mut Vec<Box<Expr>>,
}

macro_rules! expr {
    ($e:expr,$pos:expr) => {
        Box::new(Expr {
            pos: $pos,
            expr: $e,
        })
    };
}
macro_rules! exp {
    ($e:expr) => {
        Expr {
            pos: Position::default(),
            expr: $e,
        }
    };
}

type EResult<'a> = IResult<&'a str, Expr>; //Result<Box<Expr>, MsgWithPos>;

fn map_str<'b>(
    i: impl (Fn(&'b str) -> IResult<&'b str, &'_ str>),
) -> impl Fn(&'b str) -> IResult<&'b str, String> {
    map(i, |s| s.to_string())
}

fn expect_identifier<'b>(i: &'b str) -> IResult<&'b str, &str> {
    //re_match!(i, r"[a-zA-Z_][a-zA-Z0-9_]*")
    alpha1(i)
}

fn parse_nil<'b>(i: &'b str) -> EResult {
    value(exp!(ExprKind::Nil), tag("nil"))(i)
}

fn parse_bool_literal<'b>(i: &'b str) -> EResult {
    let parse_true = value(exp!(ExprKind::ConstBool(true)), tag("true"));
    let parse_false = value(exp!(ExprKind::ConstBool(false)), tag("false"));

    alt((parse_true, parse_false))(i)
}

fn lit_int<'b>(i: &'b str) -> EResult {
    /*let tok = self.advance_token()?;
    let pos = tok.position;
    if let TokenKind::LitInt(i, _, _) = tok.kind {
        Ok(expr!(ExprKind::ConstInt(i.parse().unwrap()), pos))
    } else {
        unreachable!()
    }*/
    unimplemented!()
}

fn lit_char<'b>(i: &'b str) -> EResult {
    map(preceded(tag("'"), terminated(anychar, tag("'"))), |c| {
        exp!(ExprKind::ConstChar(c))
    })(i)
}

fn lit_float<'b>(i: &'b str) -> EResult {
    map(double, |f| exp!(ExprKind::ConstFloat(f)))(i)
}

fn lit_str<'b>(i: &'b str) -> EResult {
    map(
        preceded(
            tag("\""),
            terminated(escaped(none_of("\\\""), '"', one_of("\\\"")), tag("\"")),
        ),
        |s: &'b str| exp!(ExprKind::ConstStr(s.to_string())),
    )(i)
}

pub fn parse<'b>(i: &'b str) -> Result<(), MsgWithPos> {
    /*    self.init()?;
    while !self.token.is_eof() {
        self.parse_top_level()?;
    }
    Ok(())*/
    unimplemented!()
}

fn expect_token<'b>(i: &'b str, kind: TokenKind) -> Result<Token, MsgWithPos> {
    /*if self.token.kind == kind {
        let token = self.advance_token()?;

        Ok(token)
    } else {
        Err(MsgWithPos::new(
            self.token.position,
            Msg::ExpectedToken(kind.name().into(), self.token.name()),
        ))
    }*/
    unimplemented!()
}

fn parse_top_level<'b>(i: &'b str) -> Result<(), MsgWithPos> {
    /*let expr = self.parse_expression()?;

    self.ast.push(expr);
    Ok(())*/
    unimplemented!()
}

fn parse_function_param<'b>(i: &'b str) -> Result<String, MsgWithPos> {
    /*let name = self.expect_identifier()?;
    Ok(name)*/
    unimplemented!()
}

fn parse_function<'b>(i: &'b str) -> EResult {
    let fn_arg_sep = tag(",");
    let fn_arg = expect_identifier;
    let tup = tuple((
        opt(map_str(expect_identifier)),
        tag("("),
        separated_list(fn_arg_sep, map_str(fn_arg)),
        tag(")"),
        parse_block,
    ));
    map(tup, |(name, _, params, _, block)| {
        exp!(ExprKind::Function(name, params, Box::new(block)))
    })(i)
    //Ok(expr!(ExprKind::Function(name, params, block), pos))
}

fn parse_let<'b>(i: &'b str) -> EResult {
    //let reassignable = alt((value(true, tag("var")), value(false, tag("let"))));

    let initialization = map(
        tuple((
            //reassignable,
            alt((value(true, tag("var")), value(false, tag("let")))),
            map_str(expect_identifier),
            tag("="),
            map(parse_expression, |expr| Some(Box::new(expr))),
        )),
        |(r, i, _, e)| exp!(ExprKind::Var(r, i, e)),
    );
    let declaration = map(
        tuple((
            //reassignable,
            alt((value(true, tag("var")), value(false, tag("let")))),
            map_str(expect_identifier),
        )),
        |(r, i)| exp!(ExprKind::Var(r, i, None)),
    );
    alt((initialization, declaration))(i)
}

fn parse_return<'b>(i: &'b str) -> EResult {
    map(pair(tag("return"), opt(parse_expression)), |(_, expr)| {
        exp!(ExprKind::Return(expr.map(Box::new)))
    })(i)
}

fn parse_expression<'b>(i: &'b str) -> EResult {
    let parse_new = preceded(tag("new"), parse_expression);
    alt((
        parse_new,
        parse_function,
        parse_match,
        parse_let,
        parse_block,
        parse_if,
        parse_while,
        parse_break,
        parse_continue,
        parse_return,
        parse_throw,
        |i| parse_binary(i, 0),
    ))(i)
}

fn parse_self<'b>(i: &'b str) -> EResult {
    value(exp!(ExprKind::This), tag("self"))(i)
}

fn parse_break<'b>(i: &'b str) -> EResult {
    value(exp!(ExprKind::Break), tag("break"))(i)
}

fn parse_continue<'b>(i: &'b str) -> EResult {
    value(exp!(ExprKind::Continue), tag("continue"))(i)
}

fn parse_throw<'b>(i: &'b str) -> EResult {
    map(pair(tag("throw"), parse_expression), |(_, expr)| {
        exp!(ExprKind::Throw(Box::new(expr)))
    })(i)
}

fn parse_while<'b>(i: &'b str) -> EResult {
    map(tuple((tag("while"), parse_expression, parse_block)), |(_, expr, block)| {
        exp!(ExprKind::While(Box::new(expr), Box::new(block)))
    })(i)
}

fn parse_match<'b>(i: &'b str) -> EResult {
    /*let pos = self.expect_token(TokenKind::Match)?.position;
    let value = self.parse_expression()?;
    self.expect_token(TokenKind::LBrace)?;
    let mut data = vec![];
    let mut or = None;
    while !self.token.is(TokenKind::RBrace) && !self.token.is_eof() {
        if self.token.is(TokenKind::Underscore) {
            self.expect_token(TokenKind::Underscore)?;
            self.expect_token(TokenKind::Arrow)?;
            let expr = self.parse_expression()?;
            or = Some(expr);
            continue;
        }
        let cond = self.parse_expression()?;
        self.expect_token(TokenKind::Arrow)?;
        let expr = self.parse_expression()?;
        data.push((cond, expr));
    }

    self.expect_token(TokenKind::RBrace)?;

    Ok(expr!(ExprKind::Match(value, data, or), pos))*/
    unimplemented!()
}

fn parse_if<'b>(i: &'b str) -> EResult {
    /*let pos = self.expect_token(TokenKind::If)?.position;
    let cond = self.parse_expression()?;
    let then_block = self.parse_expression()?;
    let else_block = if self.token.is(TokenKind::Else) {
        self.advance_token()?;

        if self.token.is(TokenKind::If) {
            let if_block = self.parse_if()?;
            let block = expr!(ExprKind::Block(vec![if_block]), if_block.pos);

            Some(block)
        } else {
            Some(self.parse_expression()?)
        }
    } else {
        None
    };

    Ok(expr!(ExprKind::If(cond, then_block, else_block), pos))*/
    unimplemented!()
}

fn parse_block<'b>(i: &'b str) -> EResult {
    /*let pos = self.expect_token(TokenKind::LBrace)?.position;
    let mut exprs = vec![];
    while !self.token.is(TokenKind::RBrace) && !self.token.is_eof() {
        let expr = self.parse_expression()?;
        exprs.push(expr);
    }
    self.expect_token(TokenKind::RBrace)?;
    Ok(expr!(ExprKind::Block(exprs), pos))*/
    unimplemented!()
}

fn create_binary(tok: Token, left: Box<Expr>, right: Box<Expr>) -> Box<Expr> {
    /*let op = match tok.kind {
        TokenKind::Eq => return expr!(ExprKind::Assign(left, right), tok.position),
        TokenKind::Or => "||",
        TokenKind::And => "&&",
        TokenKind::BitOr => "|",
        TokenKind::BitAnd => "&",
        TokenKind::EqEq => "==",
        TokenKind::Ne => "!=",
        TokenKind::Lt => "<",
        TokenKind::Gt => ">",
        TokenKind::Le => "<=",
        TokenKind::Ge => ">=",
        TokenKind::Caret => "^",
        TokenKind::Add => "+",
        TokenKind::Sub => "-",
        TokenKind::Mul => "*",
        TokenKind::Div => "/",
        TokenKind::LtLt => "<<",
        TokenKind::GtGt => ">>",
        TokenKind::Mod => "%",
        _ => unimplemented!(),
    };

    expr!(ExprKind::BinOp(left, op.to_owned(), right), tok.position)*/
    unimplemented!()
}

fn parse_binary<'b>(i: &'b str, precedence: u32) -> EResult {
    /*let mut left = self.parse_unary()?;
    loop {
        let right_precedence = match self.token.kind {
            TokenKind::Or => 1,
            TokenKind::And => 2,
            TokenKind::Eq => 3,
            TokenKind::EqEq
            | TokenKind::Ne
            | TokenKind::Lt
            | TokenKind::Le
            | TokenKind::Gt
            | TokenKind::Ge => 4,
            TokenKind::BitOr | TokenKind::BitAnd | TokenKind::Caret => 6,
            TokenKind::LtLt | TokenKind::GtGt | TokenKind::Add | TokenKind::Sub => 8,
            TokenKind::Mul | TokenKind::Div | TokenKind::Mod => 9,
            _ => {
                return Ok(left);
            }
        };
        if precedence >= right_precedence {
            return Ok(left);
        }

        let tok = self.advance_token()?;
        left = {
            let right = self.parse_binary(right_precedence)?;
            self.create_binary(tok, left, right)
        };
    }*/
    unimplemented!()
}

pub fn parse_unary<'b>(i: &'b str) -> EResult<'b> {
    map(pair(one_of("+-!"), parse_primary), |(op, expr)| {
        exp!(ExprKind::Unop(op.to_string(), Box::new(expr)))
    })(i)
}

/*pub fn parse_expression(&mut self) -> EResult {
    self.parse_binary(0)
}*/

fn parse_call<'b>(i: &'b str) -> EResult {
    /*let expr = self.parse_expression()?;

    self.expect_token(TokenKind::LParen)?;

    let args = self.parse_comma_list(TokenKind::RParen, |p| p.parse_expression())?;

    Ok(expr!(ExprKind::Call(expr, args), expr.pos))*/
    unimplemented!()
}

pub fn parse_primary<'b>(i: &'b str) -> EResult {
    /*let mut left = self.parse_factor()?;
    loop {
        left = match self.token.kind {
            TokenKind::Dot => {
                let tok = self.advance_token()?;
                let ident = self.expect_identifier()?;
                expr!(ExprKind::Access(left, ident), tok.position)
            }
            TokenKind::LBracket => {
                let tok = self.advance_token()?;
                let index = self.parse_expression()?;
                self.expect_token(TokenKind::RBracket)?;
                expr!(ExprKind::ArrayIndex(left, index), tok.position)
            }
            _ => {
                if self.token.is(TokenKind::LParen) {
                    let expr = left;

                    self.expect_token(TokenKind::LParen)?;

                    let args =
                        self.parse_comma_list(TokenKind::RParen, |p| p.parse_expression())?;

                    expr!(ExprKind::Call(expr, args), expr.pos)
                } else {
                    return Ok(left);
                }
            }
        }
    }*/
    unimplemented!()
}

fn parse_comma_list<F, R>(
    //    &mut self,
    stop: TokenKind,
    mut parse: F,
) -> Result<Vec<R>, MsgWithPos>
where
    F: FnMut(&mut Parser) -> Result<R, MsgWithPos>,
{
    /*let mut data = vec![];
    let mut comma = true;

    while !self.token.is(stop.clone()) && !self.token.is_eof() {
        if !comma {
            return Err(MsgWithPos::new(
                self.token.position,
                Msg::ExpectedToken(TokenKind::Comma.name().into(), self.token.name()),
            ));
        }

        let entry = parse(self)?;
        data.push(entry);

        comma = self.token.is(TokenKind::Comma);
        if comma {
            self.advance_token()?;
        }
    }

    self.expect_token(stop)?;

    Ok(data)*/
    unimplemented!()
}

fn advance_token(/*&mut self*/) -> Result<Token, MsgWithPos> {
    /*let tok = self.lexer.read_token()?;

    Ok(mem::replace(&mut self.token, tok))*/
    unimplemented!()
}

fn parse_lambda<'b>(i: &'b str) -> EResult {
    /*let tok = self.advance_token()?;
    let params = if tok.kind == TokenKind::Or {
        vec![]
    } else {
        self.parse_comma_list(TokenKind::BitOr, |f| f.parse_function_param())?
    };

    let block = self.parse_expression()?;
    Ok(expr!(ExprKind::Lambda(params, block), tok.position))*/
    unimplemented!()
}
pub fn parse_factor<'b>(i: &'b str) -> EResult {
    alt((
        preceded(tag("function"), parse_function),
        parse_parentheses,
        lit_char,
        lit_int,
        lit_float,
        lit_str,
        parse_self,
        ident,
        parse_lambda,
        parse_bool_literal,
        parse_bool_literal,
        parse_nil,
    ))(i)
}

fn parse_parentheses<'b>(i: &'b str) -> EResult {
    preceded(tag("("), terminated(parse_expression, tag(")")))(i)
}

fn ident<'b>(i: &'b str) -> EResult {
    /*let pos = self.token.position;
    let ident = self.expect_identifier()?;

    Ok(expr!(ExprKind::Ident(ident), pos))*/
    unimplemented!()
}

impl<'a> Parser<'a> {
    pub fn new(reader: Reader, ast: &'a mut Vec<Box<Expr>>) -> Parser<'a> {
        Self {
            lexer: Lexer::new(reader),
            token: Token::new(TokenKind::End, Position::new(1, 1)),
            ast,
        }
    }

    fn init(&mut self) -> Result<(), MsgWithPos> {
        //self.advance_token()?;

        Ok(())
    }
}
