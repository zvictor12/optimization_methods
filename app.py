from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from optimization_methods.constrained import (
    conditional_gradient_method,
    external_penalty_method,
    internal_penalty_method,
)
from optimization_methods.multidimensional import (
    conjugate_gradient,
    coordinate_descent,
    gradient_descent,
    newton_method,
)
from optimization_methods.one_dimensional import (
    dichotomy_search,
    fibonacci_search,
    golden_section_search,
    passive_search,
)
from optimization_methods.parsing import (
    build_multivariate_function,
    build_scalar_function,
    infer_variable_names,
)
from optimization_methods.visualization import plot_1d_function, plot_2d_path


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_bounds(text: str) -> list[tuple[float, float]]:
    bounds = []
    for part in text.split(";"):
        if not part.strip():
            continue
        lower, upper = parse_float_list(part)
        bounds.append((lower, upper))
    return bounds


def build_constraints(text: str, variables: list[str]):
    constraints = []
    for line in text.splitlines():
        expression = line.strip()
        if expression:
            constraints.append(build_multivariate_function(expression, variables).f)
    return constraints


def readable_history(rows: list[dict]) -> pd.DataFrame:
    converted = []
    for row in rows:
        next_row = {}
        for key, value in row.items():
            if isinstance(value, np.ndarray):
                next_row[key] = np.array2string(value, precision=5)
            else:
                next_row[key] = value
        converted.append(next_row)
    return pd.DataFrame(converted)


def run_one_dimensional() -> None:
    st.subheader("One-dimensional minimization")
    expression = st.text_input("f(x)", "x + 2/x")
    method = st.selectbox(
        "Method",
        ["Passive search", "Dichotomy", "Golden section", "Fibonacci"],
    )

    left, right, eps = st.columns(3)
    a = left.number_input("a", value=0.5, step=0.1, format="%.6f")
    b = right.number_input("b", value=3.5, step=0.1, format="%.6f")
    tolerance = eps.number_input("eps", value=0.001, min_value=1e-12, format="%.8f")

    delta = 0.0001
    segments = None
    if method == "Dichotomy":
        delta = st.number_input("delta", value=0.0001, min_value=1e-12, format="%.8f")
    if method == "Passive search":
        segments = st.number_input("segments", value=0, min_value=0, step=1)

    parsed = build_scalar_function(expression)
    if method == "Passive search":
        result = passive_search(
            parsed.f,
            a,
            b,
            eps=tolerance,
            segments=None if segments == 0 else int(segments),
        )
    elif method == "Dichotomy":
        result = dichotomy_search(parsed.f, a, b, eps=tolerance, delta=delta)
    elif method == "Golden section":
        result = golden_section_search(parsed.f, a, b, eps=tolerance)
    else:
        result = fibonacci_search(parsed.f, a, b, eps=tolerance)

    metric_columns = st.columns(4)
    metric_columns[0].metric("x*", f"{result.x_min:.8f}")
    metric_columns[1].metric("f(x*)", f"{result.f_min:.8f}")
    metric_columns[2].metric("iterations", result.iterations)
    metric_columns[3].metric("calls", result.function_calls)

    st.pyplot(plot_1d_function(parsed.f, a, b, result.x_min), clear_figure=True)
    st.dataframe(readable_history(result.history), use_container_width=True)


def run_multidimensional() -> None:
    st.subheader("Multidimensional minimization")
    expression = st.text_input(
        "f",
        "4*x**2 + y**2 + x*y + 0.25*sin(x + y)**2",
    )
    inferred = ",".join(infer_variable_names(expression) or ["x", "y"])
    variable_text = st.text_input("variables", inferred)
    variables = [item.strip() for item in variable_text.split(",") if item.strip()]
    x0_text = st.text_input("x0", ",".join(["1.5", "-1.0"][: len(variables)]))

    method = st.selectbox(
        "Method",
        [
            "Coordinate descent",
            "Gradient: fixed step",
            "Gradient: backtracking",
            "Gradient: scheduled step",
            "Steepest gradient descent",
            "Fletcher-Reeves",
            "Polak-Ribiere",
            "Modified Newton",
            "External penalty",
            "Internal penalty",
            "Conditional gradient",
        ],
    )

    parsed = build_multivariate_function(expression, variables)
    x0 = parse_float_list(x0_text)
    if len(x0) != len(variables):
        st.error("x0 length must match variables length.")
        return

    params = st.columns(4)
    tolerance = params[0].number_input("eps", value=1e-5, min_value=1e-12, format="%.8f")
    max_iter = params[1].number_input("max_iter", value=200, min_value=1, step=1)
    alpha = params[2].number_input("step", value=0.04, min_value=1e-12, format="%.8f")
    line_max = params[3].number_input("line max", value=1.0, min_value=1e-12, format="%.8f")

    if method in {"External penalty", "Internal penalty"}:
        constraints_text = st.text_area("inequality constraints g_i(x) <= 0", "2*x + y + 4")
        constraints = build_constraints(constraints_text, variables)
    else:
        constraints = []

    if method == "Conditional gradient":
        feasible_kind = st.selectbox("Feasible set", ["box", "ball"])
        if feasible_kind == "box":
            bounds_text = st.text_input("bounds: lower,upper; lower,upper", "-1,2; -2,1")
            feasible_set = ("box", parse_bounds(bounds_text))
        else:
            center_text = st.text_input("ball center", "2,2")
            radius = st.number_input("ball radius", value=2.828427, min_value=1e-12)
            feasible_set = ("ball", (parse_float_list(center_text), radius))
    else:
        feasible_set = None

    if method == "Coordinate descent":
        result = coordinate_descent(
            parsed.f,
            x0,
            eps=tolerance,
            max_iter=int(max_iter),
            line_search_interval=(-line_max, line_max),
            gradient=parsed.gradient,
        )
    elif method == "Gradient: fixed step":
        result = gradient_descent(
            parsed.f,
            parsed.gradient,
            x0,
            strategy="fixed",
            step=alpha,
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "Gradient: backtracking":
        result = gradient_descent(
            parsed.f,
            parsed.gradient,
            x0,
            strategy="backtracking",
            initial_step=1.0,
            shrinkage=0.5,
            armijo=0.25,
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "Gradient: scheduled step":
        result = gradient_descent(
            parsed.f,
            parsed.gradient,
            x0,
            strategy="scheduled",
            initial_step=0.2,
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "Steepest gradient descent":
        result = gradient_descent(
            parsed.f,
            parsed.gradient,
            x0,
            strategy="steepest",
            line_search_interval=(0.0, line_max),
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "Fletcher-Reeves":
        result = conjugate_gradient(
            parsed.f,
            parsed.gradient,
            x0,
            formula="fletcher_reeves",
            line_search_interval=(0.0, line_max),
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "Polak-Ribiere":
        result = conjugate_gradient(
            parsed.f,
            parsed.gradient,
            x0,
            formula="polak_ribiere",
            line_search_interval=(0.0, line_max),
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "Modified Newton":
        result = newton_method(
            parsed.f,
            parsed.gradient,
            parsed.hessian,
            x0,
            modified=True,
            eps=tolerance,
            max_iter=int(max_iter),
        )
    elif method == "External penalty":
        result = external_penalty_method(
            parsed.f,
            x0,
            inequality_constraints=constraints,
            eps=tolerance,
            inner_eps=tolerance,
            max_outer_iter=int(max_iter),
        )
    elif method == "Internal penalty":
        result = internal_penalty_method(
            parsed.f,
            x0,
            inequality_constraints=constraints,
            eps=tolerance,
            inner_eps=tolerance,
            max_outer_iter=int(max_iter),
        )
    else:
        result = conditional_gradient_method(
            parsed.f,
            parsed.gradient,
            x0,
            feasible_set=feasible_set,
            eps=tolerance,
            max_iter=int(max_iter),
        )

    metric_columns = st.columns(4)
    metric_columns[0].metric("f(x*)", f"{result.f_min:.8f}")
    metric_columns[1].metric("iterations", result.iterations)
    metric_columns[2].metric("calls", result.function_calls)
    if result.gradient_norm is None:
        metric_columns[3].metric("||grad||", "n/a")
    else:
        metric_columns[3].metric("||grad||", f"{result.gradient_norm:.3e}")
    st.code(f"x* = {np.array2string(result.x_min, precision=8)}")

    if len(variables) == 2 and len(result.path) > 1:
        path = result.path_array()
        margin = 0.75
        x_range = (float(path[:, 0].min() - margin), float(path[:, 0].max() + margin))
        y_range = (float(path[:, 1].min() - margin), float(path[:, 1].max() + margin))
        st.pyplot(
            plot_2d_path(
                parsed.f,
                parsed.gradient,
                result.path,
                x_range=x_range,
                y_range=y_range,
                title=result.method,
            ),
            clear_figure=True,
        )

    st.dataframe(readable_history(result.history), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Optimization Methods", layout="wide")
    st.title("Optimization Methods")
    mode = st.sidebar.radio("Mode", ["One-dimensional", "Multidimensional"])
    try:
        if mode == "One-dimensional":
            run_one_dimensional()
        else:
            run_multidimensional()
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
