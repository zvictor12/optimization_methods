from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_1d_function(
    f: Callable[[float], float],
    a: float,
    b: float,
    x_min: float | None = None,
    points: int = 400,
):
    xs = np.linspace(a, b, points)
    ys = np.array([f(float(x)) for x in xs], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, color="#235789", linewidth=2)
    if x_min is not None:
        ax.axvline(x_min, color="#d00000", linestyle="--", linewidth=1.5)
        ax.scatter([x_min], [f(x_min)], color="#d00000", zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_2d_path(
    f: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray] | None,
    path: list[np.ndarray],
    x_range: tuple[float, float] = (-2.5, 2.5),
    y_range: tuple[float, float] = (-2.5, 2.5),
    title: str = "Optimization path",
    grid_size: int = 140,
):
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.empty_like(xx)
    for row in range(grid_size):
        for column in range(grid_size):
            zz[row, column] = f(np.array([xx[row, column], yy[row, column]]))

    path_array = np.vstack(path) if path else np.empty((0, 2))
    fig, ax = plt.subplots(figsize=(7, 6))
    filled = ax.contourf(xx, yy, zz, levels=32, cmap="viridis", alpha=0.88)
    ax.contour(xx, yy, zz, levels=16, colors="white", linewidths=0.35, alpha=0.65)
    fig.colorbar(filled, ax=ax, label="f(x, y)")

    if len(path_array):
        ax.plot(path_array[:, 0], path_array[:, 1], color="#ffdd57", linewidth=2.2, marker="o")
        ax.scatter(path_array[0, 0], path_array[0, 1], color="#ffffff", edgecolor="#111111", s=90, label="start")
        ax.scatter(path_array[-1, 0], path_array[-1, 1], color="#ef476f", edgecolor="#111111", s=90, label="finish")

    if gradient is not None and len(path_array):
        stride = max(1, len(path_array) // 14)
        arrow_points = path_array[::stride]
        directions = []
        for point in arrow_points:
            anti_grad = -np.asarray(gradient(point), dtype=float).reshape(-1)
            norm = np.linalg.norm(anti_grad)
            if norm > 0:
                anti_grad = anti_grad / norm
            directions.append(anti_grad)
        directions = np.asarray(directions)
        ax.quiver(
            arrow_points[:, 0],
            arrow_points[:, 1],
            directions[:, 0],
            directions[:, 1],
            angles="xy",
            scale_units="xy",
            scale=7,
            color="#f4f1de",
            width=0.004,
            label="-grad direction",
        )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
