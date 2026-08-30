"""Paper formulation (5)-(10): L1-regularized hinge-loss SVM."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .base import (
    BaseLinearClassifier,
    SolverDiagnostics,
    scipy_status,
    validate_positive,
    validate_training_data,
)
from .cplex_backend import solve_docplex, validate_backend


class L1SVM(BaseLinearClassifier):
    def __init__(
        self,
        C: float = 1.0,
        *,
        time_limit: float | None = None,
        backend: str = "scipy",
        threads: int = 1,
    ) -> None:
        super().__init__()
        self.C = validate_positive(C, "C")
        self.time_limit = time_limit
        self.backend = validate_backend(backend)
        self.threads = int(threads)
        self.z_: np.ndarray | None = None
        self.xi_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L1SVM":
        X, y = validate_training_data(X, y)
        self._start_fit()
        m, n = X.shape
        # Variables: [w(n), b, z(n), xi(m)]. w and b are explicitly free.
        total = 2 * n + m + 1
        c = np.zeros(total)
        c[n + 1 : n + 1 + n] = 1.0
        c[n + 1 + n :] = self.C

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for i in range(m):
            row = np.zeros(total)
            row[:n] = -y[i] * X[i]
            row[n] = -y[i]
            row[n + 1 + n + i] = -1.0
            rows.append(row)
            rhs.append(-1.0)
        for j in range(n):
            row = np.zeros(total)
            row[j] = 1.0
            row[n + 1 + j] = -1.0
            rows.append(row)
            rhs.append(0.0)
            rows.append(-row.copy())
            rows[-1][n + 1 + j] = -1.0
            rhs.append(0.0)

        matrix = np.vstack(rows)
        rhs_array = np.asarray(rhs)
        if self.backend == "cplex":
            result = solve_docplex(
                c,
                lower_bounds=np.concatenate([np.full(n + 1, -np.inf), np.zeros(n + m)]),
                upper_bounds=np.full(total, np.inf),
                constraint_matrix=matrix,
                constraint_lower=np.full(rhs_array.size, -np.inf),
                constraint_upper=rhs_array,
                time_limit=self.time_limit,
                threads=self.threads,
                model_name="l1-svm",
            )
            status = result.status
        else:
            bounds = [(None, None)] * (n + 1) + [(0.0, None)] * (n + m)
            options = {"time_limit": float(self.time_limit)} if self.time_limit else None
            result = linprog(c, A_ub=matrix, b_ub=rhs_array, bounds=bounds, method="highs", options=options)
            status = scipy_status(result, mixed_integer=False)
            if not result.success or result.x is None:
                raise RuntimeError(f"L1-SVM solve failed ({status}): {result.message}")
        self.z_ = result.x[n + 1 : n + 1 + n]
        self.xi_ = result.x[n + 1 + n :]
        self._finish_fit(
            result.x[:n],
            result.x[n],
            SolverDiagnostics(
                status=status,
                objective_value=float(result.fun),
                message=str(result.message),
                backend=getattr(result, "backend", "scipy-highs"),
            ),
        )
        return self
