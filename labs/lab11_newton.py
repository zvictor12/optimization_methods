from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from optimization_methods.multidimensional import newton_method
from optimization_methods.parsing import build_multivariate_function
from optimization_methods.reporting import print_multi_dim_result, print_path
from optimization_methods.visualization import plot_2d_path, save_figure


def main() -> None:
    expression = "100*(y - x**2)**2 + (1 - x)**2"
    parsed = build_multivariate_function(expression, ["x", "y"])
    result = newton_method(
        parsed.f,
        parsed.gradient,
        parsed.hessian,
        x0=[1.2, 1.2],
        modified=True,
        eps=1e-5,
        max_iter=50,
    )
    print_multi_dim_result(result)
    print_path(result)
    figure = plot_2d_path(parsed.f, parsed.gradient, result.path, title="Modified Newton")
    print(f"Plot: {save_figure(figure, 'outputs/lab11_newton.png')}")


if __name__ == "__main__":
    main()
