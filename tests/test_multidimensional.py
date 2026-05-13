from __future__ import annotations

import numpy as np

from optimization_methods.multidimensional import (
    conjugate_gradient,
    coordinate_descent,
    gradient_descent,
    newton_method,
)


def f(x) -> float:
    vector = np.asarray(x, dtype=float)
    return float((vector[0] - 1.0) ** 2 + 3.0 * (vector[1] + 2.0) ** 2)


def grad(x) -> np.ndarray:
    vector = np.asarray(x, dtype=float)
    return np.array([2.0 * (vector[0] - 1.0), 6.0 * (vector[1] + 2.0)])


def hess(x) -> np.ndarray:
    return np.array([[2.0, 0.0], [0.0, 6.0]])


def assert_near_minimum(result) -> None:
    assert np.linalg.norm(result.x_min - np.array([1.0, -2.0])) < 1e-2
    assert result.f_min < 1e-4


def test_coordinate_descent_finds_minimum() -> None:
    result = coordinate_descent(
        f,
        x0=[0.0, 0.0],
        gradient=grad,
        line_search_interval=(-5.0, 5.0),
        eps=1e-7,
    )
    assert_near_minimum(result)


def test_gradient_variants_find_minimum() -> None:
    results = [
        gradient_descent(f, grad, [0.0, 0.0], strategy="fixed", step=0.1, eps=1e-7),
        gradient_descent(f, grad, [0.0, 0.0], strategy="backtracking", eps=1e-7),
        gradient_descent(
            f,
            grad,
            [0.0, 0.0],
            strategy="steepest",
            line_search_interval=(0.0, 2.0),
            eps=1e-7,
        ),
    ]
    for result in results:
        assert_near_minimum(result)


def test_conjugate_gradient_and_newton_find_minimum() -> None:
    results = [
        conjugate_gradient(f, grad, [0.0, 0.0], eps=1e-7, line_search_interval=(0.0, 2.0)),
        newton_method(f, grad, hess, [0.0, 0.0], eps=1e-7),
    ]
    for result in results:
        assert_near_minimum(result)
