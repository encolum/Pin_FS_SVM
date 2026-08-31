"""Shared validation, prediction, and solver-result helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
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


class BaseLinearClassifier:
    """Common interface for every corrected linear classifier."""

    explicit_feature_selection = True

    def __init__(self, *, feature_tolerance: float = 1e-3) -> None:
        self.feature_tolerance = float(feature_tolerance)
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.fit_time_: float | None = None
        self.diagnostics_: SolverDiagnostics | None = None
        self._fit_started: float | None = None

    @property
    def w(self) -> np.ndarray | None:  # legacy-compatible read alias
        return self.w_

    @property
    def b(self) -> float | None:  # legacy-compatible read alias
        return self.b_

    @property
    def train_time(self) -> float | None:  # legacy-compatible read alias
        return self.fit_time_

    def _start_fit(self) -> None:
        self._fit_started = perf_counter()

    def _finish_fit(
        self,
        w: np.ndarray,
        b: float,
        diagnostics: SolverDiagnostics,
    ) -> None:
        self.w_ = np.asarray(w, dtype=float)
        self.b_ = float(b)
        self.diagnostics_ = diagnostics
        self.fit_time_ = perf_counter() - (self._fit_started or perf_counter())

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None or self.b_ is None:
            raise ValueError("model is not fitted")
        X = numeric_matrix(X)
        if X.ndim != 2 or X.shape[1] != self.w_.shape[0]:
            raise ValueError("X has an incompatible feature dimension")
        return X @ self.w_ + self.b_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return only {-1, +1}; a zero decision score belongs to +1."""
        return np.where(self.decision_function(X) >= 0.0, 1, -1).astype(int)

    def get_selected_features(self) -> list[int]:
        if self.w_ is None:
            raise ValueError("model is not fitted")
        if not self.explicit_feature_selection:
            return list(range(self.w_.shape[0]))
        return np.flatnonzero(np.abs(self.w_) > self.feature_tolerance).astype(int).tolist()

    def get_num_selected_features(self) -> int:
        return len(self.get_selected_features())

    def solver_diagnostics(self) -> dict[str, Any]:
        return self.diagnostics_.to_dict() if self.diagnostics_ else {}
