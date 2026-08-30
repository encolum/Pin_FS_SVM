"""Train-only standardization helpers."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_transform_training(
    X_train: np.ndarray,
    *other_partitions: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], StandardScaler]:
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(np.asarray(X_train, dtype=float))
    transformed_others = [scaler.transform(np.asarray(partition, dtype=float)) for partition in other_partitions]
    return transformed_train, transformed_others, scaler
