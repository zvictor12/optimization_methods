from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from optimization_methods.multidimensional import gradient_descent
from optimization_methods.parsing import build_multivariate_function
from optimization_methods.reporting import print_multi_dim_result, print_path
from optimization_methods.visualization import plot_2d_path, save_figure


def main() -> None:
    expression = "100*(y - x**2)**2 + (1 - x)**2"
    parsed = build_multivariate_function(expression, ["x", "y"])
    result = gradient_descent(
        parsed.f,
        parsed.gradient,
        x0=[-1.2, 1.0],
        strategy="backtracking",
        initial_step=1.0,
        shrinkage=0.5,
        armijo=0.25,
        eps=1e-8,
        max_iter=5000,
    )
    print_multi_dim_result(result)
    print_path(result)
    figure = plot_2d_path(parsed.f, parsed.gradient, result.path, title="Gradient descent: backtracking")
    print(f"Plot: {save_figure(figure, 'outputs/lab07_gradient_backtracking.png')}")


if __name__ == "__main__":
    main()
