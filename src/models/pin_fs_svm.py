"""Paper formulation (29)-(38): Pin-FS-SVM."""

from __future__ import annotations

import numpy as np

from .base import (
    BaseLinearClassifier,
    validate_coefficient_bounds,
    validate_positive,
    validate_training_data,
)
from .cplex_backend import validate_backend


class PinFSSVM(BaseLinearClassifier):
    def __init__(
        self,
        B: int,
        C: float = 1.0,
        tau: float = 0.5,
        *,
        lower_bound: float,
        upper_bound: float,
        time_limit: float | None = None,
        mip_gap: float | None = None,
        backend: str = "scipy",
        threads: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(B, bool) or int(B) != B or B < 1:
            raise ValueError("B must be a positive integer")
        self.B = int(B)
        self.C = validate_positive(C, "C")
        self.tau = validate_positive(tau, "tau")
        if self.tau > 1:
            raise ValueError("tau must satisfy 0 < tau <= 1")
        self.lower_bound, self.upper_bound = validate_coefficient_bounds(lower_bound, upper_bound)
        self.time_limit = time_limit
        self.requested_mip_gap = mip_gap
        self.backend = validate_backend(backend)
        self.threads = int(threads)
        self.z_: np.ndarray | None = None
        self.xi_: np.ndarray | None = None
        self.v_: np.ndarray | None = None
        self.progress_: list[object] = []
        self.mip_start_status_: str | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PinFSSVM":
        # Imported lazily so ``src.search`` can also be used as a top-level public API.
        from src.search.restricted_solver import solve_restricted_pin_fs

        X, y = validate_training_data(X, y)
        self._start_fit()
        _, n = X.shape
        if self.B > n:
            raise ValueError(f"B={self.B} exceeds the number of features ({n})")
        result = solve_restricted_pin_fs(
            X,
            y,
            kernel=set(range(n)),
            B=self.B,
            C=self.C,
            tau=self.tau,
            coefficient_bounds=(self.lower_bound, self.upper_bound),
            backend=self.backend,
            time_limit=self.time_limit,
            mip_gap=self.requested_mip_gap,
            threads=self.threads,
            collect_progress=False,
        )
        self.z_ = result.z
        self.xi_ = result.xi
        self.v_ = result.v
        self.progress_ = result.progress
        self.mip_start_status_ = result.mip_start_status
        self._finish_fit(
            result.coefficients,
            result.intercept,
            result.diagnostics,
        )
        return self

    def formulation_residuals(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Maximum violations of the manuscript constraints, useful for tests/QA."""
        if self.w_ is None or self.z_ is None or self.xi_ is None or self.v_ is None:
            raise ValueError("model is not fitted")
        margins = np.asarray(y) * self.decision_function(X)
        return {
            "pinball_lower": float(np.maximum(0.0, 1.0 - self.xi_ - margins).max(initial=0.0)),
            "pinball_upper": float(np.maximum(0.0, margins - 1.0 - self.xi_ / self.tau).max(initial=0.0)),
            "absolute_value": float(np.maximum(0.0, np.abs(self.w_) - self.z_).max(initial=0.0)),
            "lower_link": float(np.maximum(0.0, self.lower_bound * self.v_ - self.w_).max(initial=0.0)),
            "upper_link": float(np.maximum(0.0, self.w_ - self.upper_bound * self.v_).max(initial=0.0)),
            "budget": float(max(0, int(self.v_.sum()) - self.B)),
        }
