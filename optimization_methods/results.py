from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class OneDimResult:
    method: str
    x_min: float
    f_min: float
    interval: tuple[float, float]
    iterations: int
    function_calls: int
    history: list[dict[str, Any]] = field(default_factory=list)

    def rows(self) -> list[dict[str, Any]]:
        return self.history


@dataclass(slots=True)
class MultiDimResult:
    method: str
    x_min: np.ndarray
    f_min: float
    iterations: int
    function_calls: int
    gradient_norm: float | None
    path: list[np.ndarray] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def path_array(self) -> np.ndarray:
        if not self.path:
            return np.empty((0, 0))
        return np.vstack(self.path)

    def rows(self) -> list[dict[str, Any]]:
        return self.history


class FunctionCounter:
    def __init__(self, function):
        self.function = function
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.function(*args)


def as_float(value) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape == ():
        return float(array)
    return float(array.reshape(-1)[0])


def as_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)
