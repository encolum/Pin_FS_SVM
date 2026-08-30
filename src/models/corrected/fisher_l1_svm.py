"""Fisher-score screening followed by the corrected L1-SVM.

The percentile is an ordinary hyperparameter.  It is selected by the outer
experiment pipeline's inner cross-validation, so Fisher scores are always
computed from the corresponding training partition only.
"""

from __future__ import annotations

import numpy as np

from .base import BaseLinearClassifier, validate_positive, validate_training_data
from .l1_svm import L1SVM


def fisher_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    positive, negative = X[y == 1], X[y == -1]
    overall = X.mean(axis=0)
    pos_mean, neg_mean = positive.mean(axis=0), negative.mean(axis=0)
    pos_var = positive.var(axis=0, ddof=1) if positive.shape[0] > 1 else np.zeros(X.shape[1])
    neg_var = negative.var(axis=0, ddof=1) if negative.shape[0] > 1 else np.zeros(X.shape[1])
    numerator = (pos_mean - overall) ** 2 + (neg_mean - overall) ** 2
    denominator = pos_var + neg_var
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


class FisherL1SVM(BaseLinearClassifier):
    def __init__(
        self,
        C: float = 1.0,
        threshold_percentile: int = 50,
        *,
        time_limit: float | None = None,
        backend: str = "scipy",
        threads: int = 1,
    ) -> None:
        super().__init__()
        self.C = validate_positive(C, "C")
        if isinstance(threshold_percentile, bool) or int(threshold_percentile) not in {25, 50, 75}:
            raise ValueError("threshold_percentile must be one of 25, 50, or 75")
        self.threshold_percentile = int(threshold_percentile)
        self.time_limit = time_limit
        self.backend = backend
        self.threads = int(threads)
        self.screened_indices_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FisherL1SVM":
        X, y = validate_training_data(X, y)
        self._start_fit()
        scores = fisher_scores(X, y)
        threshold = float(np.percentile(scores, self.threshold_percentile))
        self.screened_indices_ = np.flatnonzero(scores >= threshold)
        if self.screened_indices_.size == 0:
            raise RuntimeError("Fisher scoring produced no candidate feature set")
        final_model = L1SVM(
            C=self.C,
            time_limit=self.time_limit,
            backend=self.backend,
            threads=self.threads,
        ).fit(X[:, self.screened_indices_], y)
        full_w = np.zeros(X.shape[1])
        full_w[self.screened_indices_] = final_model.w_
        self._finish_fit(full_w, final_model.b_, final_model.diagnostics_)
        return self
