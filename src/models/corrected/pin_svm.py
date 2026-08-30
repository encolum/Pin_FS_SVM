"""Paper formulation (23)-(27): L2-regularized Pin-SVM."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .base import BaseLinearClassifier, SolverDiagnostics, validate_positive, validate_training_data
from .cplex_backend import solve_docplex, validate_backend


class PinSVM(BaseLinearClassifier):
    explicit_feature_selection = False

    def __init__(
        self,
        C: float = 1.0,
        tau: float = 0.5,
        *,
        max_iter: int = 2000,
        time_limit: float | None = None,
        backend: str = "scipy",
        threads: int = 1,
    ) -> None:
        super().__init__()
        self.C = validate_positive(C, "C")
        self.tau = validate_positive(tau, "tau")
        if self.tau > 1:
            raise ValueError("tau must satisfy 0 < tau <= 1")
        self.max_iter = int(max_iter)
        self.time_limit = time_limit
        self.backend = validate_backend(backend)
        self.threads = int(threads)
        self.xi_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PinSVM":
        X, y = validate_training_data(X, y)
        self._start_fit()
        m, n = X.shape
        total = n + 1 + m
        lower = np.zeros((m, total))
        upper = np.zeros((m, total))
        for i in range(m):
            lower[i, :n] = y[i] * X[i]
            lower[i, n] = y[i]
            lower[i, n + 1 + i] = 1.0
            upper[i, :n] = y[i] * X[i]
            upper[i, n] = y[i]
            upper[i, n + 1 + i] = -1.0 / self.tau
        constraint = LinearConstraint(
            np.vstack([lower, upper]),
            np.concatenate([np.ones(m), np.full(m, -np.inf)]),
            np.concatenate([np.full(m, np.inf), np.ones(m)]),
        )
        bounds = Bounds(
            np.concatenate([np.full(n + 1, -np.inf), np.zeros(m)]),
            np.full(total, np.inf),
        )
        x0 = np.concatenate([np.zeros(n + 1), np.ones(m)])

        def objective(q: np.ndarray) -> float:
            return 0.5 * float(q[:n] @ q[:n]) + self.C * float(q[n + 1 :].sum())

        def jacobian(q: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(q)
            grad[:n] = q[:n]
            grad[n + 1 :] = self.C
            return grad

        if self.backend == "cplex":
            linear_objective = np.zeros(total)
            linear_objective[n + 1 :] = self.C
            result = solve_docplex(
                linear_objective,
                lower_bounds=np.concatenate([np.full(n + 1, -np.inf), np.zeros(m)]),
                upper_bounds=np.full(total, np.inf),
                constraint_matrix=np.vstack([lower, upper]),
                constraint_lower=np.concatenate([np.ones(m), np.full(m, -np.inf)]),
                constraint_upper=np.concatenate([np.full(m, np.inf), np.ones(m)]),
                quadratic_indices=range(n),
                time_limit=self.time_limit,
                threads=self.threads,
                model_name="pin-svm",
            )
        else:
            result = minimize(
                objective,
                x0,
                jac=jacobian,
                method="SLSQP",
                bounds=bounds,
                constraints=[constraint],
                options={"maxiter": self.max_iter, "ftol": 1e-9},
            )
            if not result.success or result.x is None:
                raise RuntimeError(f"Pin-SVM solve failed: {result.message}")
        self.xi_ = result.x[n + 1 :]
        margins = y * (X @ result.x[:n] + result.x[n])
        violation = max(
            float(np.maximum(0.0, 1.0 - self.xi_ - margins).max(initial=0.0)),
            float(np.maximum(0.0, margins - 1.0 - self.xi_ / self.tau).max(initial=0.0)),
            float(np.maximum(0.0, -self.xi_).max(initial=0.0)),
        )
        if violation > 1e-6:
            raise RuntimeError(f"Pin-SVM returned a constraint violation of {violation:.3g}")
        self._finish_fit(
            result.x[:n],
            result.x[n],
            SolverDiagnostics(
                status=getattr(result, "status", "optimal") if self.backend == "cplex" else "optimal",
                objective_value=float(result.fun),
                message=str(result.message),
                backend=getattr(result, "backend", "scipy-slsqp"),
            ),
        )
        return self
