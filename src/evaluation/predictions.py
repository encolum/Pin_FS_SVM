"""Helpers for turning fold predictions into machine-readable rows."""

from __future__ import annotations

import numpy as np


def prediction_rows(
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
) -> list[dict[str, int | float]]:
    return [
        {
            "sample_index": int(index),
            "y_true": int(truth),
            "y_pred": int(prediction),
            "decision_score": float(score),
        }
        for index, truth, prediction, score in zip(indices, y_true, y_pred, decision_scores)
    ]
