"""Paper formulation (29)-(38): Pin-FS-SVM."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .base import (
    BaseLinearClassifier,
    SolverDiagnostics,
    scipy_status,
    validate_coefficient_bounds,
    validate_positive,
    validate_training_data,
)
from .cplex_backend import solve_docplex, validate_backend


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PinFSSVM":
        X, y = validate_training_data(X, y)
        self._start_fit()
        m, n = X.shape
        if self.B > n:
            raise ValueError(f"B={self.B} exceeds the number of features ({n})")

        # Variables: [w(n), b, z(n), xi(m), v(n)]. w and b have infinite bounds.
        w0, b_idx = 0, n
        z0, xi0, v0 = n + 1, 2 * n + 1, 2 * n + m + 1
        total = 3 * n + m + 1
        row_count = 2 * m + 4 * n + 1
        A = lil_matrix((row_count, total), dtype=float)
        lb = np.full(row_count, -np.inf)
        ub = np.full(row_count, np.inf)
        row = 0
        for i in range(m):
            A[row, w0 : w0 + n] = y[i] * X[i]
            A[row, b_idx] = y[i]
            A[row, xi0 + i] = 1.0
            lb[row] = 1.0
            row += 1
            A[row, w0 : w0 + n] = y[i] * X[i]
            A[row, b_idx] = y[i]
            A[row, xi0 + i] = -1.0 / self.tau
            ub[row] = 1.0
            row += 1
        for j in range(n):
            A[row, w0 + j], A[row, z0 + j], ub[row] = 1.0, -1.0, 0.0
            row += 1
            A[row, w0 + j], A[row, z0 + j], ub[row] = -1.0, -1.0, 0.0
            row += 1
            A[row, w0 + j], A[row, v0 + j], ub[row] = 1.0, -self.upper_bound, 0.0
            row += 1
            A[row, w0 + j], A[row, v0 + j], ub[row] = -1.0, self.lower_bound, 0.0
            row += 1
        A[row, v0 : v0 + n] = 1.0
        ub[row] = self.B

        c = np.zeros(total)
        c[z0 : z0 + n] = 1.0
        c[xi0 : xi0 + m] = self.C
        lower = np.concatenate([np.full(n + 1, -np.inf), np.zeros(n + m + n)])
        upper = np.concatenate([np.full(n + 1 + n + m, np.inf), np.ones(n)])
        integrality = np.zeros(total, dtype=int)
        integrality[v0 : v0 + n] = 1
        options: dict[str, float] = {}
        if self.time_limit is not None:
            options["time_limit"] = float(self.time_limit)
        if self.requested_mip_gap is not None:
            options["mip_rel_gap"] = float(self.requested_mip_gap)
        if self.backend == "cplex":
            result = solve_docplex(
                c,
                lower_bounds=lower,
                upper_bounds=upper,
                constraint_matrix=A.tocsr(),
                constraint_lower=lb,
                constraint_upper=ub,
                integrality=integrality,
                time_limit=self.time_limit,
                mip_gap=self.requested_mip_gap,
                threads=self.threads,
                model_name="pin-fs-svm",
            )
            status = result.status
        else:
            result = milp(
                c,
                integrality=integrality,
                bounds=Bounds(lower, upper),
                constraints=LinearConstraint(A.tocsr(), lb, ub),
                options=options or None,
            )
            status = scipy_status(result, mixed_integer=True)
            if result.x is None or status not in {"optimal", "feasible_with_gap"}:
                raise RuntimeError(f"Pin-FS-SVM solve failed ({status}): {result.message}")
        self.z_ = result.x[z0 : z0 + n]
        self.xi_ = result.x[xi0 : xi0 + m]
        self.v_ = np.rint(result.x[v0 : v0 + n]).astype(int)
        self._finish_fit(
            result.x[w0 : w0 + n],
            result.x[b_idx],
            SolverDiagnostics(
                status=status,
                objective_value=float(result.fun),
                best_bound=_optional_float(result, "mip_dual_bound"),
                mip_gap=_optional_float(result, "mip_gap"),
                node_count=_optional_int(result, "mip_node_count"),
                message=str(result.message),
                backend=getattr(result, "backend", "scipy-highs"),
            ),
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


def _optional_float(result: object, name: str) -> float | None:
    value = getattr(result, name, None)
    return None if value is None else float(value)


def _optional_int(result: object, name: str) -> int | None:
    value = getattr(result, name, None)
    return None if value is None else int(value)
