from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from optimization_methods.constrained import internal_penalty_method
from optimization_methods.parsing import build_multivariate_function
from optimization_methods.reporting import print_multi_dim_result, print_path
from optimization_methods.visualization import plot_2d_path, save_figure


def main() -> None:
    parsed = build_multivariate_function("x**2 + y**2", ["x", "y"])
    constraint = build_multivariate_function("2*x + y + 4", ["x", "y"])
    result = internal_penalty_method(
        parsed.f,
        x0=[-3.0, -3.0],
        inequality_constraints=[constraint.f],
        eps=1e-4,
        inner_eps=1e-5,
        initial_t=1.0,
        t_multiplier=0.1,
        max_outer_iter=6,
    )
    print_multi_dim_result(result)
    print_path(result)
    figure = plot_2d_path(parsed.f, parsed.gradient, result.path, title="Internal penalty")
    print(f"Plot: {save_figure(figure, 'outputs/lab15_internal_penalty.png')}")


if __name__ == "__main__":
    main()
