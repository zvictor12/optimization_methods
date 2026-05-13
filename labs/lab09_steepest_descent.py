from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from optimization_methods.multidimensional import gradient_descent
from optimization_methods.parsing import build_multivariate_function
from optimization_methods.reporting import print_multi_dim_result, print_path
from optimization_methods.visualization import plot_2d_path, save_figure


def main() -> None:
    expression = "4*x**2 + y**2 + x*y + 0.25*sin(x + y)**2"
    parsed = build_multivariate_function(expression, ["x", "y"])
    result = gradient_descent(
        parsed.f,
        parsed.gradient,
        x0=[1.5, -1.0],
        strategy="steepest",
        line_search_interval=(0.0, 1.0),
        line_search_eps=1e-5,
        eps=1e-5,
        max_iter=120,
    )
    print_multi_dim_result(result)
    print_path(result)
    figure = plot_2d_path(parsed.f, parsed.gradient, result.path, title="Steepest gradient descent")
    print(f"Plot: {save_figure(figure, 'outputs/lab09_steepest_descent.png')}")


if __name__ == "__main__":
    main()
