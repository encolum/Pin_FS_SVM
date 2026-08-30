"""Paper formulation (11)-(17): hinge-loss SVM with a feature budget."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .base import (
    BaseLinearClassifier,
    SolverDiagnostics,
    scipy_status,
    validate_coefficient_bounds,
    validate_training_data,
)
from .cplex_backend import solve_docplex, validate_backend


class BudgetedMILPSVM(BaseLinearClassifier):
    """The objective is sum(xi); C and z variables are intentionally absent."""

    C = None
    tau = None

    def __init__(
        self,
        B: int,
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
        self.lower_bound, self.upper_bound = validate_coefficient_bounds(lower_bound, upper_bound)
        self.time_limit = time_limit
        self.requested_mip_gap = mip_gap
        self.backend = validate_backend(backend)
        self.threads = int(threads)
        self.xi_: np.ndarray | None = None
        self.v_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BudgetedMILPSVM":
        X, y = validate_training_data(X, y)
        self._start_fit()
        m, n = X.shape
        if self.B > n:
            raise ValueError(f"B={self.B} exceeds the number of features ({n})")
        # Variables: [w(n), b, xi(m), v(n)]. No z and no C.
        b_idx, xi0, v0 = n, n + 1, n + m + 1
        total = 2 * n + m + 1
        A = lil_matrix((m + 2 * n + 1, total), dtype=float)
        lb = np.full(A.shape[0], -np.inf)
        ub = np.full(A.shape[0], np.inf)
        row = 0
        for i in range(m):
            A[row, :n] = y[i] * X[i]
            A[row, b_idx] = y[i]
            A[row, xi0 + i] = 1.0
            lb[row] = 1.0
            row += 1
        for j in range(n):
            A[row, j], A[row, v0 + j], ub[row] = 1.0, -self.upper_bound, 0.0
            row += 1
            A[row, j], A[row, v0 + j], ub[row] = -1.0, self.lower_bound, 0.0
            row += 1
        A[row, v0 : v0 + n] = 1.0
        ub[row] = self.B
        c = np.zeros(total)
        c[xi0 : xi0 + m] = 1.0
        lower = np.concatenate([np.full(n + 1, -np.inf), np.zeros(m + n)])
        upper = np.concatenate([np.full(n + 1 + m, np.inf), np.ones(n)])
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
                model_name="budgeted-milp-svm",
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
                raise RuntimeError(f"Budgeted MILP-SVM solve failed ({status}): {result.message}")
        self.xi_ = result.x[xi0 : xi0 + m]
        self.v_ = np.rint(result.x[v0 : v0 + n]).astype(int)
        self._finish_fit(
            result.x[:n],
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


def _optional_float(result: object, name: str) -> float | None:
    value = getattr(result, name, None)
    return None if value is None else float(value)


def _optional_int(result: object, name: str) -> int | None:
    value = getattr(result, name, None)
    return None if value is None else int(value)
