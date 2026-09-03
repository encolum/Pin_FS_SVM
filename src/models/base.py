"""Shared validation and solver-result helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from src.utils.matrices import numeric_matrix


@dataclass
class SolverDiagnostics:
    status: str
    objective_value: float | None = None
    best_bound: float | None = None
    mip_gap: float | None = None
    node_count: int | None = None
    message: str | None = None
    backend: str = "scipy-highs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_training_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = numeric_matrix(X)
    y = np.asarray(y).reshape(-1)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y contain different numbers of observations")
    if y.dtype.kind not in "iuf" or not np.isfinite(y).all() or set(np.unique(y)) != {-1, 1}:
        raise ValueError("y must contain both binary labels {-1, +1}")
    return X, y.astype(int)


def validate_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_coefficient_bounds(lower: float, upper: float) -> tuple[float, float]:
    lower, upper = float(lower), float(upper)
    if not np.isfinite([lower, upper]).all() or not lower < 0 < upper:
        raise ValueError("coefficient bounds must satisfy lower < 0 < upper")
    return lower, upper


def scipy_status(result: Any, *, mixed_integer: bool) -> str:
    if result.status == 0:
        return "optimal"
    if result.status == 1:
        return "feasible_with_gap" if mixed_integer and result.x is not None else "time_limit"
    if result.status == 2:
        return "infeasible"
    if result.status == 3:
        return "unbounded"
    return "solver_error"
