use crate::token::Position;
use core::ops::Deref;
use std::borrow::Borrow;

#[derive(Clone, PartialEq, Default)]
pub struct Expr {
    pub pos: Position,
    pub expr: ExprKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ident(pub String);

impl From<Ident> for String {
    fn from(ident: Ident) -> Self {
        ident.0
    }
}

impl Deref for Ident {
    type Target = String;
    fn deref(&self) -> &String {
        &self.0
    }
}

impl Borrow<String> for Ident {
    fn borrow(&self) -> &String {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Assign(Box<Expr>, Box<Expr>),
    BinOp(Box<Expr>, String, Box<Expr>),
    Unop(String, Box<Expr>),
    Access(Box<Expr>, Ident),
    Ident(Ident),
    Function(Option<Ident>, Vec<Ident>, Box<Expr>),
    Class(String, Box<Expr>, Option<Box<Expr>>),
    Lambda(Vec<Ident>, Box<Expr>),
    Match(Box<Expr>, Vec<(Box<Expr>, Box<Expr>)>, Option<Box<Expr>>),
    If(Box<Expr>, Box<Expr>, Option<Box<Expr>>),
    ConstInt(i64),
    ConstChar(char),
    ConstStr(String),
    New(Box<Expr>),
    ConstFloat(f64),
    Object(Vec<(Box<Expr>, Box<Expr>)>),
    Var(bool, Ident, Option<Box<Expr>>),
    While(Box<Expr>, Box<Expr>),
    Block(Vec<Box<Expr>>),
    Return(Option<Box<Expr>>),
    Call(Box<Expr>, Vec<Box<Expr>>),
    Nil,
    Break,
    Continue,
    Throw(Box<Expr>),
    ConstBool(bool),
    Array(Vec<Box<Expr>>),
    ArrayIndex(Box<Expr>, Box<Expr>),
    This,
}

impl Default for ExprKind {
    fn default() -> Self {
        ExprKind::Nil
    }
}

use std::fmt;

impl Expr {
    pub fn is_access(&self) -> bool {
        if let ExprKind::Access(_, _) = self.expr {
            return true;
        };
        false
    }

    pub fn is_binop(&self) -> bool {
        if let ExprKind::BinOp(_, _, _) = self.expr {
            return true;
        };
        false
    }

    pub fn is_binop_cmp(&self) -> bool {
        if let ExprKind::BinOp(_, ref op, _) = self.expr {
            let op: &str = op;
            match op {
                ">" | "<" | ">=" | "<=" | "==" | "!=" => return true,
                _ => return false,
            }
        }
        return false;
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#?}", self.expr)
    }
}
