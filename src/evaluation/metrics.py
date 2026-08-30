"""The four classification metrics retained by the implementation plan."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


METRIC_NAMES = ("balanced_accuracy", "weighted_f1", "accuracy", "gmean")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty arrays with equal shape")
    if set(np.unique(y_true)) != {-1, 1}:
        raise ValueError("y_true must contain both classes {-1, +1}")
    if not set(np.unique(y_pred)).issubset({-1, 1}):
        raise ValueError("y_pred must contain only labels in {-1, +1}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[-1, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "gmean": float(np.sqrt(sensitivity * specificity)),
    }
