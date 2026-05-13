from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from optimization_methods.results import as_float, as_vector

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

SYMPY_NAMES = {
    "Abs": sp.Abs,
    "E": sp.E,
    "acos": sp.acos,
    "asin": sp.asin,
    "atan": sp.atan,
    "cos": sp.cos,
    "cosh": sp.cosh,
    "e": sp.E,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "pi": sp.pi,
    "sin": sp.sin,
    "sinh": sp.sinh,
    "sqrt": sp.sqrt,
    "tan": sp.tan,
    "tanh": sp.tanh,
}


@dataclass(slots=True)
class ParsedScalarFunction:
    expression: sp.Expr
    variable: sp.Symbol
    f: Callable[[float], float]
    derivative: Callable[[float], float]
    second_derivative: Callable[[float], float]


@dataclass(slots=True)
class ParsedMultivariateFunction:
    expression: sp.Expr
    variables: tuple[sp.Symbol, ...]
    f: Callable[[np.ndarray], float]
    gradient: Callable[[np.ndarray], np.ndarray]
    hessian: Callable[[np.ndarray], np.ndarray]


def parse_math_expression(expression: str, variables: list[str] | tuple[str, ...] = ()) -> sp.Expr:
    symbols = {name.strip(): sp.Symbol(name.strip()) for name in variables if name.strip()}
    local_dict = {**SYMPY_NAMES, **symbols}
    return parse_expr(expression, local_dict=local_dict, transformations=TRANSFORMATIONS)


def infer_variable_names(expression: str) -> list[str]:
    expr = parse_math_expression(expression)
    return [str(symbol) for symbol in sorted(expr.free_symbols, key=lambda item: item.name)]


def build_scalar_function(expression: str, variable: str = "x") -> ParsedScalarFunction:
    symbol = sp.Symbol(variable)
    expr = parse_math_expression(expression, [variable])
    raw_f = sp.lambdify(symbol, expr, modules=["numpy"])
    raw_d1 = sp.lambdify(symbol, sp.diff(expr, symbol), modules=["numpy"])
    raw_d2 = sp.lambdify(symbol, sp.diff(expr, symbol, 2), modules=["numpy"])

    def f(x: float) -> float:
        return as_float(raw_f(x))

    def derivative(x: float) -> float:
        return as_float(raw_d1(x))

    def second_derivative(x: float) -> float:
        return as_float(raw_d2(x))

    return ParsedScalarFunction(expr, symbol, f, derivative, second_derivative)


def build_multivariate_function(
    expression: str,
    variables: list[str] | tuple[str, ...],
) -> ParsedMultivariateFunction:
    names = [name.strip() for name in variables if name.strip()]
    if not names:
        names = infer_variable_names(expression)
    symbols = tuple(sp.Symbol(name) for name in names)
    expr = parse_math_expression(expression, names)
    gradient_expr = [sp.diff(expr, symbol) for symbol in symbols]
    hessian_expr = sp.Matrix(gradient_expr).jacobian(symbols)

    raw_f = sp.lambdify(symbols, expr, modules=["numpy"])
    raw_gradient = sp.lambdify(symbols, gradient_expr, modules=["numpy"])
    raw_hessian = sp.lambdify(symbols, hessian_expr, modules=["numpy"])

    def f(x) -> float:
        vector = as_vector(x)
        return as_float(raw_f(*vector))

    def gradient(x) -> np.ndarray:
        vector = as_vector(x)
        return np.asarray(raw_gradient(*vector), dtype=float).reshape(-1)

    def hessian(x) -> np.ndarray:
        vector = as_vector(x)
        return np.asarray(raw_hessian(*vector), dtype=float)

    return ParsedMultivariateFunction(expr, symbols, f, gradient, hessian)
