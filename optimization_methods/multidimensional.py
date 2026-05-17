from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optimization_methods.one_dimensional import (
    dichotomy_search,
    fibonacci_search,
    golden_section_search,
    passive_search,
)
from optimization_methods.results import FunctionCounter, MultiDimResult, as_float, as_vector


def vector_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def line_search_along(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    interval: tuple[float, float] = (0.0, 2.0),
    eps: float = 1e-5,
    method: str = "golden",
) -> tuple[float, object]:
    start, end = interval
    direction = as_vector(direction)

    def phi(alpha: float) -> float:
        return as_float(f(x + alpha * direction))

    if method == "golden":
        result = golden_section_search(phi, start, end, eps=eps)
    elif method == "dichotomy":
        result = dichotomy_search(phi, start, end, eps=eps, delta=min(eps / 2, 1e-5))
    elif method == "fibonacci":
        result = fibonacci_search(phi, start, end, eps=eps)
    elif method == "passive":
        result = passive_search(phi, start, end, eps=eps)
    else:
        raise ValueError(f"Unknown line search method: {method}")
    return result.x_min, result


def coordinate_descent(
    f: Callable[[np.ndarray], float],
    x0,
    eps: float = 1e-5,
    max_iter: int = 200,
    line_search_interval: tuple[float, float] = (-2.0, 2.0),
    line_search_eps: float = 1e-5,
    line_search_method: str = "golden",
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    path = [x.copy()]
    history = []
    n = len(x)

    for cycle in range(1, max_iter + 1):
        previous = x.copy()
        previous_f = counted_f(previous)

        for coordinate in range(n):
            direction = np.zeros(n)
            direction[coordinate] = 1.0
            alpha, ls_result = line_search_along(
                counted_f,
                x,
                direction,
                interval=line_search_interval,
                eps=line_search_eps,
                method=line_search_method,
            )
            x = x + alpha * direction
            path.append(x.copy())
            history.append(
                {
                    "iteration": cycle,
                    "coordinate": coordinate,
                    "alpha": alpha,
                    "x": x.copy(),
                    "f_x": counted_f(x),
                    "line_search_calls": getattr(ls_result, "function_calls", 0),
                }
            )

        current_f = counted_f(x)
        if vector_norm(x - previous) <= eps or abs(current_f - previous_f) <= eps:
            break

    grad_norm = None if gradient is None else vector_norm(gradient(x))
    return MultiDimResult(
        method="Coordinate descent",
        x_min=x,
        f_min=counted_f(x),
        iterations=cycle,
        function_calls=counted_f.calls,
        gradient_norm=grad_norm,
        path=path,
        history=history,
    )


def gradient_descent(
    f: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0,
    strategy: str = "backtracking",
    eps: float = 1e-5,
    max_iter: int = 500,
    step: float = 0.01,
    initial_step: float = 1.0,
    shrinkage: float = 0.5,
    armijo: float = 0.25,
    line_search_interval: tuple[float, float] = (0.0, 2.0),
    line_search_eps: float = 1e-5,
    line_search_method: str = "golden",
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    path = [x.copy()]
    history = []

    for iteration in range(1, max_iter + 1):
        grad = as_vector(gradient(x))
        grad_norm = vector_norm(grad)
        f_x = counted_f(x)
        if grad_norm <= eps:
            break

        direction = -grad
        if strategy == "fixed":
            alpha = step
        elif strategy == "scheduled":
            alpha = initial_step / iteration
        elif strategy == "steepest":
            alpha, _ = line_search_along(
                counted_f,
                x,
                direction,
                interval=line_search_interval,
                eps=line_search_eps,
                method=line_search_method,
            )
        elif strategy == "backtracking":
            alpha = initial_step
            while alpha > 1e-14:
                candidate = x + alpha * direction
                if counted_f(candidate) - f_x <= -alpha * armijo * grad_norm**2:
                    break
                alpha *= shrinkage
        else:
            alpha = step

        x_next = x + alpha * direction
        f_next = counted_f(x_next)
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "x": x_next.copy(),
                "f_x": f_next,
                "gradient_norm": grad_norm,
                "direction": direction.copy(),
            }
        )
        path.append(x_next.copy())

        if vector_norm(x_next - x) <= eps or abs(f_next - f_x) <= eps:
            x = x_next
            break
        x = x_next

    final_grad_norm = vector_norm(gradient(x))
    return MultiDimResult(
        method=f"Gradient descent: {strategy}",
        x_min=x,
        f_min=counted_f(x),
        iterations=len(history),
        function_calls=counted_f.calls,
        gradient_norm=final_grad_norm,
        path=path,
        history=history,
    )


def conjugate_gradient(
    f: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    x0,
    eps: float = 1e-5,
    max_iter: int = 500,
    line_search_interval: tuple[float, float] = (0.0, 2.0),
    line_search_eps: float = 1e-5,
    line_search_method: str = "golden",
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    grad = as_vector(gradient(x))
    direction = -grad
    n = len(x)
    path = [x.copy()]
    history = []

    for iteration in range(1, max_iter + 1):
        grad_norm = vector_norm(grad)
        if grad_norm <= eps:
            break

        alpha, _ = line_search_along(
            counted_f,
            x,
            direction,
            interval=line_search_interval,
            eps=line_search_eps,
            method=line_search_method,
        )
        x_next = x + alpha * direction
        grad_next = as_vector(gradient(x_next))
        denominator = float(np.dot(grad, grad))
        if denominator <= 1e-30 or iteration % n == 0:
            beta = 0.0
        else:
            beta = float(np.dot(grad_next, grad_next) / denominator)

        direction_next = -grad_next + beta * direction
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "beta": beta,
                "x": x_next.copy(),
                "f_x": counted_f(x_next),
                "gradient_norm": grad_norm,
                "direction": direction.copy(),
            }
        )
        path.append(x_next.copy())

        if vector_norm(x_next - x) <= eps:
            x = x_next
            grad = grad_next
            break
        x, grad, direction = x_next, grad_next, direction_next

    return MultiDimResult(
        method="Fletcher-Reeves",
        x_min=x,
        f_min=counted_f(x),
        iterations=len(history),
        function_calls=counted_f.calls,
        gradient_norm=vector_norm(gradient(x)),
        path=path,
        history=history,
    )


def newton_method(
    f: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    x0,
    eps: float = 1e-5,
    max_iter: int = 100,
    modified: bool = True,
    line_search_interval: tuple[float, float] = (0.0, 1.0),
    line_search_eps: float = 1e-5,
    line_search_method: str = "golden",
) -> MultiDimResult:
    counted_f = FunctionCounter(lambda x: as_float(f(as_vector(x))))
    x = as_vector(x0)
    path = [x.copy()]
    history = []

    for iteration in range(1, max_iter + 1):
        grad = as_vector(gradient(x))
        grad_norm = vector_norm(grad)
        if grad_norm <= eps:
            break

        hess = np.asarray(hessian(x), dtype=float)
        direction = np.linalg.solve(hess, -grad)

        if modified:
            alpha, _ = line_search_along(
                counted_f,
                x,
                direction,
                interval=line_search_interval,
                eps=line_search_eps,
                method=line_search_method,
            )
        else:
            alpha = 1.0

        x_next = x + alpha * direction
        history.append(
            {
                "iteration": iteration,
                "alpha": alpha,
                "x": x_next.copy(),
                "f_x": counted_f(x_next),
                "gradient_norm": grad_norm,
                "direction": direction.copy(),
            }
        )
        path.append(x_next.copy())

        if vector_norm(x_next - x) <= eps:
            x = x_next
            break
        x = x_next

    return MultiDimResult(
        method="Modified Newton" if modified else "Newton",
        x_min=x,
        f_min=counted_f(x),
        iterations=len(history),
        function_calls=counted_f.calls,
        gradient_norm=vector_norm(gradient(x)),
        path=path,
        history=history,
    )
