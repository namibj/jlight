use crate::ast::Ident;
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
    multi::{many0, separated_list},
    number::complete::double,
    sequence::{pair, preceded, terminated, tuple},
    Err, IResult,
};
use nom_locate::LocatedSpanEx;
use nom_recursive::{recursive_parser, RecursiveInfo};
use nom_packrat::{init, packrat_parser, storage, HasExtraState};
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

#[derive(Ord, PartialOrd, Eq, PartialEq, Clone)]
pub enum OperatorPrecedence {
    None,
    Or,
    And,
    Assign,
    Cmp,
    Bit,
    AddShift,
    Mul,
}

type Span<'a> = LocatedSpanEx<&'a str, Option<OperatorPrecedence>>;

type EResult<'a> = IResult<Span<'a>, Expr>;

fn expect_identifier<'b>(i: Span<'b>) -> IResult<Span<'b>, String> {
    //re_match!(i, r"[a-zA-Z_][a-zA-Z0-9_]*")
    map(alpha1, |s: Span| s.to_string())(i)
}

fn parse_nil<'b>(i: Span) -> EResult {
    value(exp!(ExprKind::Nil), tag("nil"))(i)
}

fn parse_bool_literal<'b>(i: Span) -> EResult {
    let parse_true = value(exp!(ExprKind::ConstBool(true)), tag("true"));
    let parse_false = value(exp!(ExprKind::ConstBool(false)), tag("false"));

    alt((parse_true, parse_false))(i)
}

fn lit_int<'b>(i: Span) -> EResult {
    /*let tok = self.advance_token()?;
    let pos = tok.position;
    if let TokenKind::LitInt(i, _, _) = tok.kind {
        Ok(expr!(ExprKind::ConstInt(i.parse().unwrap()), pos))
    } else {
        unreachable!()
    }*/
    unimplemented!()
}

fn lit_char<'b>(i: Span) -> EResult {
    map(preceded(tag("'"), terminated(anychar, tag("'"))), |c| {
        exp!(ExprKind::ConstChar(c))
    })(i)
}

fn lit_float<'b>(i: Span) -> EResult {
    map(double, |f| exp!(ExprKind::ConstFloat(f)))(i)
}

fn lit_str<'b>(i: Span) -> EResult {
    map(
        preceded(
            tag("\""),
            terminated(escaped(none_of("\\\""), '"', one_of("\\\"")), tag("\"")),
        ),
        |s: Span| exp!(ExprKind::ConstStr(s.to_string())),
    )(i)
}

pub fn parse<'b>(i: Span) -> Result<(), MsgWithPos> {
    /*    self.init()?;
    while !self.token.is_eof() {
        self.parse_top_level()?;
    }
    Ok(())*/
    unimplemented!()
}

fn expect_token<'b>(i: Span, kind: TokenKind) -> Result<Token, MsgWithPos> {
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

fn parse_top_level<'b>(i: Span) -> Result<(), MsgWithPos> {
    /*let expr = self.parse_expression()?;

    self.ast.push(expr);
    Ok(())*/
    unimplemented!()
}

fn parse_function_param<'b>(i: Span) -> Result<String, MsgWithPos> {
    /*let name = self.expect_identifier()?;
    Ok(name)*/
    unimplemented!()
}

fn parse_function<'b>(i: Span) -> EResult {
    let fn_arg_sep = tag(",");
    let fn_arg = expect_identifier;
    let tup = tuple((
        opt(expect_identifier),
        tag("("),
        separated_list(fn_arg_sep, fn_arg),
        tag(")"),
        parse_block,
    ));
    map(tup, |(name, _, params, _, block)| {
        exp!(ExprKind::Function(name, params, Box::new(block)))
    })(i)
}

fn parse_let<'b>(i: Span) -> EResult {
    //let reassignable = alt((value(true, tag("var")), value(false, tag("let"))));

    let initialization = map(
        tuple((
            //reassignable,
            alt((value(true, tag("var")), value(false, tag("let")))),
            expect_identifier,
            tag("="),
            map(parse_expression, |expr| Some(Box::new(expr))),
        )),
        |(r, i, _, e)| exp!(ExprKind::Var(r, i, e)),
    );
    let declaration = map(
        tuple((
            //reassignable,
            alt((value(true, tag("var")), value(false, tag("let")))),
            expect_identifier,
        )),
        |(r, i)| exp!(ExprKind::Var(r, i, None)),
    );
    alt((initialization, declaration))(i)
}

fn parse_return<'b>(i: Span) -> EResult {
    map(pair(tag("return"), opt(parse_expression)), |(_, expr)| {
        exp!(ExprKind::Return(expr.map(Box::new)))
    })(i)
}

fn parse_expression<'b>(i: Span) -> EResult {
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
        map(parse_binary, |(expr, _)| expr),
    ))(i)
}

fn parse_self<'b>(i: Span) -> EResult {
    value(exp!(ExprKind::This), tag("self"))(i)
}

fn parse_break<'b>(i: Span) -> EResult {
    value(exp!(ExprKind::Break), tag("break"))(i)
}

fn parse_continue<'b>(i: Span) -> EResult {
    value(exp!(ExprKind::Continue), tag("continue"))(i)
}

fn parse_throw<'b>(i: Span) -> EResult {
    map(pair(tag("throw"), parse_expression), |(_, expr)| {
        exp!(ExprKind::Throw(Box::new(expr)))
    })(i)
}

fn parse_while<'b>(i: Span) -> EResult {
    map(
        tuple((tag("while"), parse_expression, parse_block)),
        |(_, expr, block)| exp!(ExprKind::While(Box::new(expr), Box::new(block))),
    )(i)
}

fn parse_match<'b>(i: Span) -> EResult {
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

fn parse_if<'b>(i: Span) -> EResult {
    map(
        tuple((
            preceded(tag("if"), parse_expression),
            preceded(tag("then"), parse_expression),
            opt(preceded(tag("else"), parse_expression)),
        )),
        |(cond, then_block, else_block)| {
            exp!(ExprKind::If(
                Box::new(cond),
                Box::new(then_block),
                else_block.map(Box::new)
            ))
        },
    )(i)
}

fn parse_block<'b>(i: Span) -> EResult {
    map(
        preceded(
            tag("{"),
            terminated(
                many0(map(parse_expression, |expr| Box::new(expr))),
                tag("}"),
            ),
        ),
        |exprs| exp!(ExprKind::Block(exprs)),
    )(i)
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

//#[recursive_parser]
fn parse_binary<'b>(s: Span<'b>) -> IResult<Span<'b>, (Expr, OperatorPrecedence)> {
    let parse_operator = alt((
        map(tag("||"), |t| (t, OperatorPrecedence::Or)),
        map(tag("&&"), |t| (t, OperatorPrecedence::And)),
        map(tag("="), |t| (t, OperatorPrecedence::Assign)),
        map(
            alt((
                tag("=="),
                tag("!="),
                tag("<"),
                tag("<="),
                tag(">"),
                tag(">="),
            )),
            |t| (t, OperatorPrecedence::Cmp),
        ),
        map(alt((tag("|"), tag("&"), tag("^"))), |t| {
            (t, OperatorPrecedence::Bit)
        }),
        map(alt((tag("<<"), tag(">>"), tag("+"), tag("-"))), |t| {
            (t, OperatorPrecedence::AddShift)
        }),
        map(alt((tag("*"), tag("/"), tag("%"))), |t| {
            (t, OperatorPrecedence::Mul)
        }),
    ));
    let subexpr = alt((
        map(parse_unary, |expr| (expr, OperatorPrecedence::None)),
        parse_binary,
    ));
    let (s, (left, left_precedence)) = subexpr(s)?;
    let (s, (op, precedence)) = parse_operator(s)?;
    let (s, (right, right_precedence)) = subexpr(s)?;
    Ok((
        s,
        (
            exp!(ExprKind::BinOp(
                Box::new(left),
                op.fragment.to_string(),
                Box::new(right)
            )),
            precedence,
        ),
    ))
}

pub fn parse_unary<'b>(i: Span<'b>) -> EResult<'b> {
    map(pair(one_of("+-!"), parse_primary), |(op, expr)| {
        exp!(ExprKind::Unop(op.to_string(), Box::new(expr)))
    })(i)
}

/*pub fn parse_expression(&mut self) -> EResult {
    self.parse_binary(0)
}*/

fn parse_call<'b>(
    expr_parser: impl Fn(Span<'b>) -> EResult<'b>,
) -> impl Fn(Span<'b>) -> EResult<'b> {
    let fn_arg_sep = tag(",");
    let fn_arg = parse_expression;
    let tup = tuple((
        expr_parser,
        tag("("),
        separated_list(fn_arg_sep, map(fn_arg, Box::new)),
        tag(")"),
    ));
    map(tup, |(expr, _, params, _)| {
        exp!(ExprKind::Call(Box::new(expr), params))
    })
}

fn parse_primary<'b>(s: Span<'b>) -> EResult<'b> {
    alt((
        map(
            tuple((parse_factor, preceded(tag("."), ident))),
            |(left, ident)| exp!(ExprKind::Access(Box::new(left), ident)),
        ),
        map(
            tuple((
                parse_factor,
                preceded(tag("["), terminated(parse_expression, tag("]"))),
            )),
            |(left, index)| exp!(ExprKind::ArrayIndex(Box::new(left), Box::new(index))),
        ),
        parse_call(parse_factor),
        parse_factor,
    ))(s)
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

fn parse_lambda<'b>(i: Span) -> EResult {
    let fn_arg_sep = tag(",");
    let fn_arg = expect_identifier;
    let tup = tuple((
        tag("|"),
        separated_list(fn_arg_sep, fn_arg),
        tag("|"),
        parse_expression,
    ));
    map(tup, |(_, params, _, block)| {
        exp!(ExprKind::Lambda(params, Box::new(block)))
    })(i)
}

pub fn parse_factor<'b>(i: Span) -> EResult {
    alt((
        preceded(tag("function"), parse_function),
        parse_parentheses,
        lit_char,
        lit_int,
        lit_float,
        lit_str,
        parse_self,
        map(ident, |ident| exp!(ExprKind::Ident(ident))),
        parse_lambda,
        parse_bool_literal,
        parse_bool_literal,
        parse_nil,
    ))(i)
}

fn parse_parentheses<'b>(i: Span) -> EResult {
    preceded(tag("("), terminated(parse_expression, tag(")")))(i)
}

fn ident<'b>(i: Span) -> IResult<Span<'b>, Ident> {
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
