"""Recursive feature elimination using the corrected L1-SVM."""

from __future__ import annotations

import numpy as np

from .base import BaseLinearClassifier, validate_positive, validate_training_data
from .l1_svm import L1SVM


class L1SVMRFE(BaseLinearClassifier):
    def __init__(
        self,
        target_features: int,
        C: float = 1.0,
        *,
        time_limit: float | None = None,
        backend: str = "scipy",
        threads: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(target_features, bool) or int(target_features) != target_features or target_features < 1:
            raise ValueError("target_features must be a positive integer")
        self.target_features = int(target_features)
        self.C = validate_positive(C, "C")
        self.time_limit = time_limit
        self.backend = backend
        self.threads = int(threads)
        self.selected_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L1SVMRFE":
        X, y = validate_training_data(X, y)
        self._start_fit()
        if self.target_features > X.shape[1]:
            raise ValueError("target_features exceeds the number of input features")
        remaining = list(range(X.shape[1]))
        while len(remaining) > self.target_features:
            model = L1SVM(
                C=self.C,
                time_limit=self.time_limit,
                backend=self.backend,
                threads=self.threads,
            ).fit(X[:, remaining], y)
            remaining.pop(int(np.argmin(np.square(model.w_))))
        final_model = L1SVM(
            C=self.C,
            time_limit=self.time_limit,
            backend=self.backend,
            threads=self.threads,
        ).fit(X[:, remaining], y)
        full_w = np.zeros(X.shape[1])
        full_w[remaining] = final_model.w_
        self.selected_indices_ = np.asarray(remaining, dtype=int)
        self._finish_fit(full_w, final_model.b_, final_model.diagnostics_)
        return self
