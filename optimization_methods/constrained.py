from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optimization_methods.multidimensional import gradient_descent
from optimization_methods.results import FunctionCounter, MultiDimResult, as_float, as_vector


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
    gradient: Callable[[np.ndarray], np.ndarray],
    x0,
    inequality_constraints: list[Callable[[np.ndarray], float]] | None = None,
    inequality_gradients: list[Callable[[np.ndarray], np.ndarray]] | None = None,
    equality_constraints: list[Callable[[np.ndarray], float]] | None = None,
    equality_gradients: list[Callable[[np.ndarray], np.ndarray]] | None = None,
    eps: float = 1e-4,
    inner_eps: float = 1e-5,
    initial_r: float = 1.0,
    r_multiplier: float = 10.0,
    max_outer_iter: int = 8,
    max_inner_iter: int = 300,
) -> MultiDimResult:
    inequality_constraints = inequality_constraints or []
    inequality_gradients = inequality_gradients or []
    equality_constraints = equality_constraints or []
    equality_gradients = equality_gradients or []
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

        def phi_gradient(z, penalty_weight: float = current_r) -> np.ndarray:
            vector = as_vector(z)
            result = as_vector(gradient(vector))
            for constraint, constraint_gradient in zip(
                inequality_constraints,
                inequality_gradients,
                strict=True,
            ):
                g_value = as_float(constraint(vector))
                result = result + 2 * penalty_weight * max(0.0, g_value) * as_vector(
                    constraint_gradient(vector)
                )
            for constraint, constraint_gradient in zip(
                equality_constraints,
                equality_gradients,
                strict=True,
            ):
                h_value = as_float(constraint(vector))
                result = result + 2 * penalty_weight * h_value * as_vector(
                    constraint_gradient(vector)
                )
            return result

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
    gradient: Callable[[np.ndarray], np.ndarray],
    x0,
    inequality_constraints: list[Callable[[np.ndarray], float]],
    inequality_gradients: list[Callable[[np.ndarray], np.ndarray]],
    eps: float = 1e-4,
    inner_eps: float = 1e-5,
    initial_t: float = 1.0,
    t_multiplier: float = 0.1,
    max_outer_iter: int = 8,
    max_inner_iter: int = 300,
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)

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

        def psi_gradient(z, penalty_weight: float = current_t) -> np.ndarray:
            vector = as_vector(z)
            result = as_vector(gradient(vector))
            for constraint, constraint_gradient in zip(
                inequality_constraints,
                inequality_gradients,
                strict=True,
            ):
                g_value = as_float(constraint(vector))
                result = result + penalty_weight * as_vector(constraint_gradient(vector)) / g_value**2
            return result

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
