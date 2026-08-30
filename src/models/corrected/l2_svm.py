"""Paper formulation (1)-(4): L2-regularized hinge-loss SVM."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from .base import BaseLinearClassifier, SolverDiagnostics, validate_positive, validate_training_data
from .cplex_backend import solve_docplex, validate_backend


class L2SVM(BaseLinearClassifier):
    explicit_feature_selection = False

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
        self.time_limit = time_limit  # retained for a uniform constructor; libsvm has no wall-clock option
        self.backend = validate_backend(backend)
        self.threads = int(threads)
        self.xi_: np.ndarray | None = None
        self._model: SVC | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L2SVM":
        X, y = validate_training_data(X, y)
        self._start_fit()
        if self.backend == "cplex":
            m, n = X.shape
            total = n + 1 + m
            matrix = np.zeros((m, total))
            for i in range(m):
                matrix[i, :n] = y[i] * X[i]
                matrix[i, n] = y[i]
                matrix[i, n + 1 + i] = 1.0
            linear_objective = np.zeros(total)
            linear_objective[n + 1 :] = self.C
            result = solve_docplex(
                linear_objective,
                lower_bounds=np.concatenate([np.full(n + 1, -np.inf), np.zeros(m)]),
                upper_bounds=np.full(total, np.inf),
                constraint_matrix=matrix,
                constraint_lower=np.ones(m),
                constraint_upper=np.full(m, np.inf),
                quadratic_indices=range(n),
                time_limit=self.time_limit,
                threads=self.threads,
                model_name="l2-svm",
            )
            w = result.x[:n]
            b = float(result.x[n])
            self.xi_ = result.x[n + 1 :]
            objective = float(result.fun)
            diagnostics = SolverDiagnostics(
                status=result.status,
                objective_value=objective,
                message=result.message,
                backend=result.backend,
            )
        else:
            model = SVC(C=self.C, kernel="linear")
            model.fit(X, y)
            w = model.coef_.reshape(-1)
            b = float(model.intercept_[0])
            margins = y * (X @ w + b)
            self.xi_ = np.maximum(0.0, 1.0 - margins)
            objective = 0.5 * float(w @ w) + self.C * float(self.xi_.sum())
            self._model = model
            diagnostics = SolverDiagnostics(status="optimal", objective_value=objective, backend="libsvm")
        self._finish_fit(
            w,
            b,
            diagnostics,
        )
        return self
