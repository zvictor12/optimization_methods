from __future__ import annotations

from _bootstrap import add_project_root

add_project_root()

from optimization_methods.one_dimensional import dichotomy_search
from optimization_methods.parsing import build_scalar_function
from optimization_methods.reporting import print_one_dim_result
from optimization_methods.visualization import plot_1d_function, save_figure


def main() -> None:
    parsed = build_scalar_function("x + 2/x")
    result = dichotomy_search(parsed.f, a=0.5, b=3.5, eps=0.001, delta=0.0001)
    print_one_dim_result(result)
    figure = plot_1d_function(parsed.f, 0.5, 3.5, result.x_min)
    print(f"Plot: {save_figure(figure, 'outputs/lab02_dichotomy.png')}")


if __name__ == "__main__":
    main()
