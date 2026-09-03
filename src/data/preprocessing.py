"""Explicit, train-only preprocessing with sparse storage and memory guards."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

from src.utils.matrices import numeric_matrix, guarded_dense, matrix_metadata


PREPROCESSING_POLICIES = {"standard", "standard_sparse", "max_abs", "none", "passthrough_upstream_normalized"}


@dataclass
class FittedPreprocessor:
    name: str
    transformer: object | None
    input_storage: str
    output_storage: str
    metadata: dict


def fit_preprocessor(X_train, *, policy, allow_densify=False, max_dense_bytes=None):
    if policy not in PREPROCESSING_POLICIES:
        raise ValueError(f"unknown preprocessing policy: {policy}")
    X = numeric_matrix(X_train)
    input_storage = "csr" if sparse.issparse(X) else "dense"
    if policy == "standard" and sparse.issparse(X):
        X = guarded_dense(X, allow_densify=allow_densify, max_dense_bytes=max_dense_bytes)
    if policy == "standard_sparse" and not sparse.issparse(X):
        raise ValueError("standard_sparse requires sparse input; use standard for dense input")
    transformer = None
    if policy in {"standard", "standard_sparse"}:
        transformer = StandardScaler(with_mean=policy == "standard").fit(X)
    elif policy == "max_abs":
        transformer = MaxAbsScaler().fit(X)
    parameters = {name: np.asarray(getattr(transformer, name)).tolist()
                  for name in ("mean_", "scale_", "var_", "max_abs_")
                  if transformer is not None and getattr(transformer, name, None) is not None}
    output_storage = "csr" if sparse.issparse(X) else "dense"
    metadata = {"policy": policy, "fit_partition": "training_only", "fit_samples": int(X.shape[0]),
                "features": int(X.shape[1]), "parameters": parameters,
                "input_storage": input_storage, "output_storage": output_storage,
                "densified": input_storage == "csr" and output_storage == "dense",
                "allow_densify": allow_densify, "max_dense_bytes": max_dense_bytes,
                "input": matrix_metadata(X_train),
                "warnings": ["Input already normalized upstream; no additional scaling."]
                if policy == "passthrough_upstream_normalized" else []}
    return FittedPreprocessor(policy, transformer, input_storage, output_storage, metadata)


def transform_partition(fitted, X):
    X = numeric_matrix(X)
    if X.shape[1] != fitted.metadata["features"]:
        raise ValueError("partition has different feature dimensions from training")
    if ("csr" if sparse.issparse(X) else "dense") != fitted.input_storage:
        raise ValueError("partition storage differs from preprocessor training input")
    if fitted.metadata["densified"]:
        X = guarded_dense(X, allow_densify=fitted.metadata["allow_densify"],
                          max_dense_bytes=fitted.metadata["max_dense_bytes"])
    output = X.copy() if fitted.transformer is None else fitted.transformer.transform(X)
    return numeric_matrix(output)
