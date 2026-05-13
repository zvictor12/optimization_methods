from __future__ import annotations

from collections.abc import Callable
from math import ceil, sqrt

from optimization_methods.results import FunctionCounter, OneDimResult, as_float


def passive_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float | None = None,
    segments: int | None = None,
) -> OneDimResult:
    if segments is None:
        if eps is None or eps <= 0:
            raise ValueError("Either positive eps or segments must be provided.")
        segments = max(1, ceil((b - a) / eps))
    if segments <= 0:
        raise ValueError("segments must be positive.")

    counted_f = FunctionCounter(lambda x: as_float(f(x)))
    step = (b - a) / segments
    best_x = a
    best_fx = counted_f(a)
    history = [
        {
            "iteration": 0,
            "x": a,
            "f_x": best_fx,
            "best_x": best_x,
            "best_f": best_fx,
        }
    ]

    for i in range(1, segments + 1):
        x = a + i * step
        f_x = counted_f(x)
        if f_x < best_fx:
            best_x = x
            best_fx = f_x
        history.append(
            {
                "iteration": i,
                "x": x,
                "f_x": f_x,
                "best_x": best_x,
                "best_f": best_fx,
            }
        )

    return OneDimResult(
        method="Passive search",
        x_min=best_x,
        f_min=best_fx,
        interval=(max(a, best_x - step), min(b, best_x + step)),
        iterations=segments,
        function_calls=counted_f.calls,
        history=history,
    )


def dichotomy_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-4,
    delta: float = 1e-5,
    max_iter: int = 10_000,
) -> OneDimResult:
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if delta >= 2 * eps:
        raise ValueError("delta must be smaller than 2 * eps.")

    counted_f = FunctionCounter(lambda x: as_float(f(x)))
    left, right = float(a), float(b)
    history = []

    for iteration in range(1, max_iter + 1):
        middle = (left + right) / 2
        c = middle - delta / 2
        d = middle + delta / 2
        f_c = counted_f(c)
        f_d = counted_f(d)
        old_left, old_right = left, right

        if f_c <= f_d:
            right = d
        else:
            left = c

        history.append(
            {
                "iteration": iteration,
                "a": old_left,
                "b": old_right,
                "c": c,
                "d": d,
                "f_c": f_c,
                "f_d": f_d,
                "next_a": left,
                "next_b": right,
                "error_bound": (right - left) / 2,
            }
        )

        if (right - left) / 2 <= eps:
            break

    x_min = (left + right) / 2
    return OneDimResult(
        method="Dichotomy",
        x_min=x_min,
        f_min=counted_f(x_min),
        interval=(left, right),
        iterations=len(history),
        function_calls=counted_f.calls,
        history=history,
    )


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-4,
    max_iter: int = 10_000,
) -> OneDimResult:
    if eps <= 0:
        raise ValueError("eps must be positive.")

    counted_f = FunctionCounter(lambda x: as_float(f(x)))
    left, right = float(a), float(b)
    tau = (sqrt(5) - 1) / 2
    c = right - tau * (right - left)
    d = left + tau * (right - left)
    f_c = counted_f(c)
    f_d = counted_f(d)
    history = []

    for iteration in range(1, max_iter + 1):
        old_left, old_right = left, right
        if f_c <= f_d:
            right = d
            d = c
            f_d = f_c
            c = right - tau * (right - left)
            f_c = counted_f(c)
        else:
            left = c
            c = d
            f_c = f_d
            d = left + tau * (right - left)
            f_d = counted_f(d)

        history.append(
            {
                "iteration": iteration,
                "a": old_left,
                "b": old_right,
                "c": c,
                "d": d,
                "f_c": f_c,
                "f_d": f_d,
                "next_a": left,
                "next_b": right,
                "error_bound": (right - left) / 2,
            }
        )

        if (right - left) / 2 <= eps:
            break

    x_min = (left + right) / 2
    return OneDimResult(
        method="Golden section",
        x_min=x_min,
        f_min=counted_f(x_min),
        interval=(left, right),
        iterations=len(history),
        function_calls=counted_f.calls,
        history=history,
    )


def fibonacci_numbers(limit: int) -> list[int]:
    numbers = [0, 1]
    while len(numbers) <= limit:
        numbers.append(numbers[-1] + numbers[-2])
    return numbers


def fibonacci_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    eps: float = 1e-4,
) -> OneDimResult:
    if eps <= 0:
        raise ValueError("eps must be positive.")

    counted_f = FunctionCounter(lambda x: as_float(f(x)))
    left, right = float(a), float(b)
    target = (right - left) / eps
    fib = [0, 1]
    while fib[-1] < target:
        fib.append(fib[-1] + fib[-2])

    n = len(fib) - 1
    if n < 3:
        n = 3
        fib = fibonacci_numbers(n)

    c = left + fib[n - 2] / fib[n] * (right - left)
    d = left + fib[n - 1] / fib[n] * (right - left)
    f_c = counted_f(c)
    f_d = counted_f(d)
    history = []

    for iteration in range(1, n - 1):
        old_left, old_right = left, right
        if f_c <= f_d:
            right = d
            d = c
            f_d = f_c
            c = left + fib[n - iteration - 2] / fib[n - iteration] * (right - left)
            f_c = counted_f(c)
        else:
            left = c
            c = d
            f_c = f_d
            d = left + fib[n - iteration - 1] / fib[n - iteration] * (right - left)
            f_d = counted_f(d)

        history.append(
            {
                "iteration": iteration,
                "a": old_left,
                "b": old_right,
                "c": c,
                "d": d,
                "f_c": f_c,
                "f_d": f_d,
                "next_a": left,
                "next_b": right,
                "error_bound": (right - left) / 2,
            }
        )

    x_min = (left + right) / 2
    return OneDimResult(
        method="Fibonacci",
        x_min=x_min,
        f_min=counted_f(x_min),
        interval=(left, right),
        iterations=len(history),
        function_calls=counted_f.calls,
        history=history,
    )
