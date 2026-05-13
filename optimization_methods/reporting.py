from __future__ import annotations

import numpy as np

from optimization_methods.results import MultiDimResult, OneDimResult


def print_one_dim_result(result: OneDimResult) -> None:
    print(f"Method: {result.method}")
    print(f"x*: {result.x_min:.10f}")
    print(f"f(x*): {result.f_min:.10f}")
    print(f"Interval: [{result.interval[0]:.10f}, {result.interval[1]:.10f}]")
    print(f"Iterations: {result.iterations}")
    print(f"Function calls: {result.function_calls}")


def print_multi_dim_result(result: MultiDimResult) -> None:
    print(f"Method: {result.method}")
    print(f"x*: {np.array2string(result.x_min, precision=8)}")
    print(f"f(x*): {result.f_min:.10f}")
    print(f"Iterations: {result.iterations}")
    print(f"Function calls: {result.function_calls}")
    if result.gradient_norm is not None:
        print(f"Gradient norm: {result.gradient_norm:.10e}")


def print_path(result: MultiDimResult, limit: int = 15) -> None:
    print("Path:")
    for index, point in enumerate(result.path[:limit]):
        print(f"  {index:>3}: {np.array2string(point, precision=8)}")
    if len(result.path) > limit:
        print(f"  ... {len(result.path) - limit} more points")
