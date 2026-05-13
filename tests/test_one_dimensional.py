from __future__ import annotations

from optimization_methods.one_dimensional import (
    dichotomy_search,
    fibonacci_search,
    golden_section_search,
    passive_search,
)


def quadratic(x: float) -> float:
    return (x - 2.0) ** 2 + 1.0


def test_zero_order_methods_find_quadratic_minimum() -> None:
    methods = [
        passive_search(quadratic, -5.0, 5.0, eps=0.01),
        dichotomy_search(quadratic, -5.0, 5.0, eps=1e-3, delta=1e-4),
        golden_section_search(quadratic, -5.0, 5.0, eps=1e-3),
        fibonacci_search(quadratic, -5.0, 5.0, eps=1e-3),
    ]
    for result in methods:
        assert abs(result.x_min - 2.0) < 0.02
        assert abs(result.f_min - 1.0) < 1e-3
        assert result.function_calls > 0
