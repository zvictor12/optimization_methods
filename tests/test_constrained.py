from __future__ import annotations

import numpy as np

from optimization_methods.constrained import (
    conditional_gradient_method,
    external_penalty_method,
    internal_penalty_method,
)


def quadratic(x) -> float:
    vector = np.asarray(x, dtype=float)
    return float(vector[0] ** 2 + vector[1] ** 2)


def quadratic_gradient(x) -> np.ndarray:
    vector = np.asarray(x, dtype=float)
    return 2.0 * vector


def linear_constraint(x) -> float:
    vector = np.asarray(x, dtype=float)
    return float(2.0 * vector[0] + vector[1] + 4.0)


def test_external_penalty_approaches_boundary_solution() -> None:
    result = external_penalty_method(
        quadratic,
        [1.0, 1.0],
        inequality_constraints=[linear_constraint],
        eps=1e-4,
        inner_eps=1e-5,
        max_outer_iter=6,
    )
    assert np.linalg.norm(result.x_min - np.array([-1.6, -0.8])) < 0.08
    assert linear_constraint(result.x_min) <= 0.02


def test_internal_penalty_approaches_boundary_solution_from_inside() -> None:
    result = internal_penalty_method(
        quadratic,
        [-3.0, -3.0],
        inequality_constraints=[linear_constraint],
        eps=1e-4,
        inner_eps=1e-5,
        max_outer_iter=6,
    )
    assert np.linalg.norm(result.x_min - np.array([-1.6, -0.8])) < 0.12
    assert linear_constraint(result.x_min) < 0.0


def test_conditional_gradient_finds_box_solution() -> None:
    result = conditional_gradient_method(
        quadratic,
        quadratic_gradient,
        [1.0, -1.0],
        feasible_set=("box", [(-1.0, 2.0), (-2.0, 1.0)]),
        eps=1e-5,
    )
    assert np.linalg.norm(result.x_min) < 1e-3
