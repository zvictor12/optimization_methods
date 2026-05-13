from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from optimization_methods.multidimensional import (
    gradient_descent,
    line_search_along,
    vector_norm,
)
from optimization_methods.results import FunctionCounter, MultiDimResult, as_float, as_vector

FeasibleSetKind = Literal["box", "ball"]


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x,
    step: float = 1e-6,
) -> np.ndarray:
    x = as_vector(x)
    grad = np.zeros_like(x)
    for index in range(len(x)):
        h = step * max(1.0, abs(x[index]))
        left = x.copy()
        right = x.copy()
        left[index] -= h
        right[index] += h
        f_left = as_float(f(left))
        f_right = as_float(f(right))
        if np.isfinite(f_left) and np.isfinite(f_right):
            grad[index] = (f_right - f_left) / (2 * h)
            continue

        center = as_float(f(x))
        if np.isfinite(f_right):
            grad[index] = (f_right - center) / h
        elif np.isfinite(f_left):
            grad[index] = (center - f_left) / h
        else:
            grad[index] = 0.0
    return grad


def external_penalty_value(
    inequality_constraints: list[Callable[[np.ndarray], float]],
    equality_constraints: list[Callable[[np.ndarray], float]],
    x,
) -> float:
    vector = as_vector(x)
    value = 0.0
    for constraint in inequality_constraints:
        value += max(0.0, as_float(constraint(vector))) ** 2
    for constraint in equality_constraints:
        value += as_float(constraint(vector)) ** 2
    return float(value)


def internal_barrier_value(
    inequality_constraints: list[Callable[[np.ndarray], float]],
    x,
) -> float:
    vector = as_vector(x)
    value = 0.0
    for constraint in inequality_constraints:
        g_value = as_float(constraint(vector))
        if g_value >= 0:
            return float("inf")
        value -= 1.0 / g_value
    return float(value)


def external_penalty_method(
    f: Callable[[np.ndarray], float],
    x0,
    inequality_constraints: list[Callable[[np.ndarray], float]] | None = None,
    equality_constraints: list[Callable[[np.ndarray], float]] | None = None,
    eps: float = 1e-4,
    inner_eps: float = 1e-5,
    initial_r: float = 1.0,
    r_multiplier: float = 10.0,
    max_outer_iter: int = 8,
    max_inner_iter: int = 300,
) -> MultiDimResult:
    inequality_constraints = inequality_constraints or []
    equality_constraints = equality_constraints or []
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    path = [x.copy()]
    history = []
    r = initial_r

    for outer_iteration in range(1, max_outer_iter + 1):
        current_r = r

        def penalty(z) -> float:
            return external_penalty_value(inequality_constraints, equality_constraints, z)

        def phi(z, penalty_weight: float = current_r) -> float:
            return counted_f(z) + penalty_weight * penalty(z)

        def phi_gradient(z) -> np.ndarray:
            return numerical_gradient(phi, z)

        inner = gradient_descent(
            phi,
            phi_gradient,
            x,
            strategy="backtracking",
            eps=inner_eps,
            max_iter=max_inner_iter,
            initial_step=1.0,
            shrinkage=0.5,
            armijo=0.25,
        )
        x = inner.x_min
        path.extend(point.copy() for point in inner.path[1:])
        penalty_at_x = penalty(x)
        history.append(
            {
                "outer_iteration": outer_iteration,
                "r": current_r,
                "x": x.copy(),
                "f_x": counted_f(x),
                "penalty": penalty_at_x,
                "inner_iterations": inner.iterations,
            }
        )
        if penalty_at_x < eps:
            break
        r *= r_multiplier

    return MultiDimResult(
        method="External penalty",
        x_min=x,
        f_min=counted_f(x),
        iterations=len(history),
        function_calls=counted_f.calls,
        gradient_norm=None,
        path=path,
        history=history,
    )


def internal_penalty_method(
    f: Callable[[np.ndarray], float],
    x0,
    inequality_constraints: list[Callable[[np.ndarray], float]],
    eps: float = 1e-4,
    inner_eps: float = 1e-5,
    initial_t: float = 1.0,
    t_multiplier: float = 0.1,
    max_outer_iter: int = 8,
    max_inner_iter: int = 300,
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    if not np.isfinite(internal_barrier_value(inequality_constraints, x)):
        raise ValueError("Initial point must be strictly feasible: all g_i(x0) < 0.")

    path = [x.copy()]
    history = []
    t = initial_t

    for outer_iteration in range(1, max_outer_iter + 1):
        current_t = t

        def barrier(z) -> float:
            return internal_barrier_value(inequality_constraints, z)

        def psi(z, penalty_weight: float = current_t) -> float:
            barrier_value = barrier(z)
            if not np.isfinite(barrier_value):
                return float("inf")
            return counted_f(z) + penalty_weight * barrier_value

        def psi_gradient(z) -> np.ndarray:
            return numerical_gradient(psi, z)

        inner = gradient_descent(
            psi,
            psi_gradient,
            x,
            strategy="backtracking",
            eps=inner_eps,
            max_iter=max_inner_iter,
            initial_step=1.0,
            shrinkage=0.5,
            armijo=0.25,
        )
        x = inner.x_min
        path.extend(point.copy() for point in inner.path[1:])
        barrier_at_x = barrier(x)
        weighted_barrier = current_t * barrier_at_x
        history.append(
            {
                "outer_iteration": outer_iteration,
                "t": current_t,
                "x": x.copy(),
                "f_x": counted_f(x),
                "barrier": barrier_at_x,
                "t_barrier": weighted_barrier,
                "inner_iterations": inner.iterations,
            }
        )
        if weighted_barrier < eps:
            break
        t *= t_multiplier

    return MultiDimResult(
        method="Internal penalty",
        x_min=x,
        f_min=counted_f(x),
        iterations=len(history),
        function_calls=counted_f.calls,
        gradient_norm=None,
        path=path,
        history=history,
    )


def _linear_minimizer_box(gradient: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    if len(bounds) != len(gradient):
        raise ValueError("bounds length must match gradient length.")
    return np.array(
        [lower if gradient[index] >= 0 else upper for index, (lower, upper) in enumerate(bounds)],
        dtype=float,
    )


def _linear_minimizer_ball(gradient: np.ndarray, center, radius: float) -> np.ndarray:
    center = as_vector(center)
    grad_norm = vector_norm(gradient)
    if grad_norm == 0:
        return center.copy()
    return center - radius * gradient / grad_norm


def conditional_gradient_method(
    f: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0,
    feasible_set: tuple[FeasibleSetKind, object],
    eps: float = 1e-5,
    max_iter: int = 200,
    line_search_eps: float = 1e-5,
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    path = [x.copy()]
    history = []
    set_kind, set_data = feasible_set

    for iteration in range(1, max_iter + 1):
        grad = as_vector(gradient(x))
        if set_kind == "box":
            x_bar = _linear_minimizer_box(grad, set_data)
        elif set_kind == "ball":
            center, radius = set_data
            x_bar = _linear_minimizer_ball(grad, center, radius)
        else:
            raise ValueError(f"Unknown feasible set kind: {set_kind}")

        direction = x_bar - x
        frank_wolfe_gap = -float(np.dot(grad, direction))
        if vector_norm(direction) <= eps or frank_wolfe_gap <= eps:
            break

        alpha, _ = line_search_along(
            counted_f,
            x,
            direction,
            interval=(0.0, 1.0),
            eps=line_search_eps,
            method="golden",
        )
        x_next = x + alpha * direction
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "x": x_next.copy(),
                "linear_solution": x_bar.copy(),
                "f_x": counted_f(x_next),
                "gap": frank_wolfe_gap,
                "direction": direction.copy(),
            }
        )
        path.append(x_next.copy())

        if vector_norm(x_next - x) <= eps:
            x = x_next
            break
        x = x_next

    return MultiDimResult(
        method="Conditional gradient",
        x_min=x,
        f_min=counted_f(x),
        iterations=len(history),
        function_calls=counted_f.calls,
        gradient_norm=vector_norm(gradient(x)),
        path=path,
        history=history,
    )
