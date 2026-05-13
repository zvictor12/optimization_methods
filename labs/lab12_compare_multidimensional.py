from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from optimization_methods.multidimensional import (
    conjugate_gradient,
    coordinate_descent,
    gradient_descent,
)
from optimization_methods.parsing import build_multivariate_function
from optimization_methods.reporting import print_multi_dim_result


def main() -> None:
    expression = "4*x**2 + y**2 + x*y + 0.25*sin(x + y)**2"
    parsed = build_multivariate_function(expression, ["x", "y"])
    x0 = [1.5, -1.0]
    results = [
        coordinate_descent(parsed.f, x0=x0, gradient=parsed.gradient),
        gradient_descent(parsed.f, parsed.gradient, x0=x0, strategy="fixed", step=0.04),
        gradient_descent(parsed.f, parsed.gradient, x0=x0, strategy="backtracking"),
        gradient_descent(parsed.f, parsed.gradient, x0=x0, strategy="scheduled", initial_step=0.2),
        gradient_descent(parsed.f, parsed.gradient, x0=x0, strategy="steepest", line_search_interval=(0.0, 1.0)),
        conjugate_gradient(parsed.f, parsed.gradient, x0=x0, formula="fletcher_reeves", line_search_interval=(0.0, 1.0)),
    ]
    for result in results:
        print("=" * 72)
        print_multi_dim_result(result)


if __name__ == "__main__":
    main()
