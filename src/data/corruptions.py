"""Deterministic, partition-local corruption generators with complete manifests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from src.models.corrected.l1_svm import L1SVM


@dataclass
class CorruptionResult:
    X: np.ndarray
    y: np.ndarray
    manifest: dict[str, Any]


def array_hash(X: np.ndarray, y: np.ndarray) -> str:
    digest = sha256()
    digest.update(np.ascontiguousarray(X).view(np.uint8))
    digest.update(np.ascontiguousarray(y).view(np.uint8))
    return digest.hexdigest()


def apply_corruption(
    X: np.ndarray,
    y: np.ndarray,
    condition: str,
    *,
    seed: int,
    config: dict[str, Any] | None = None,
) -> CorruptionResult:
    """Corrupt one already-preprocessed training partition.

    The nested-CV pipeline fits scaling on the clean training partition before
    calling this function.  Validation and test partitions are never passed in.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    condition = condition.lower()
    config = dict(config or {})
    input_digest = array_hash(X, y)
    if condition == "clean":
        return CorruptionResult(X.copy(), y.copy(), _manifest(condition, seed, config, input_digest, X, y))
    if condition == "mixed":
        return _mixed(X, y, seed=seed, config=config, input_digest=input_digest)
    if condition == "high_margin":
        return _high_margin(X, y, seed=seed, config=config, input_digest=input_digest)
    if condition == "combined":
        mixed_config = _required_mapping(config, "mixed")
        margin_config = _required_mapping(config, "high_margin")
        mixed = _mixed(X, y, seed=seed, config=mixed_config, input_digest=input_digest)
        margin = _high_margin(
            mixed.X,
            mixed.y,
            seed=seed + 1,
            config=margin_config,
            input_digest=array_hash(mixed.X, mixed.y),
        )
        manifest = _manifest(condition, seed, config, input_digest, margin.X, margin.y)
        manifest["stages"] = [mixed.manifest, margin.manifest]
        return CorruptionResult(margin.X, margin.y, manifest)
    raise ValueError(f"unknown condition: {condition}")


def _mixed(X: np.ndarray, y: np.ndarray, *, seed: int, config: dict[str, Any], input_digest: str) -> CorruptionResult:
    required = ("label_flip_rate", "additive_rate", "multiplicative_rate", "additive_std", "multiplicative_std")
    _require_numeric(config, required)
    rng = np.random.default_rng(seed)
    X_out, y_out = X.copy(), y.copy()
    label_count = _rate_count(config["label_flip_rate"], y.size)
    label_indices = np.sort(rng.choice(y.size, size=label_count, replace=False)) if label_count else np.array([], dtype=int)
    y_out[label_indices] *= -1
    total_cells = X.size
    additive_count = _rate_count(config["additive_rate"], total_cells)
    multiplicative_count = _rate_count(config["multiplicative_rate"], total_cells)
    additive_flat = np.sort(rng.choice(total_cells, size=additive_count, replace=False)) if additive_count else np.array([], dtype=int)
    multiplicative_flat = np.sort(rng.choice(total_cells, size=multiplicative_count, replace=False)) if multiplicative_count else np.array([], dtype=int)
    flat = X_out.reshape(-1)
    if additive_count:
        flat[additive_flat] += rng.normal(0.0, float(config["additive_std"]), size=additive_count)
    if multiplicative_count:
        flat[multiplicative_flat] *= rng.normal(1.0, float(config["multiplicative_std"]), size=multiplicative_count)
    manifest = _manifest("mixed", seed, config, input_digest, X_out, y_out)
    manifest.update({
        "flipped_label_indices": label_indices.tolist(),
        "additive_cells": _cell_pairs(additive_flat, X.shape[1]),
        "multiplicative_cells": _cell_pairs(multiplicative_flat, X.shape[1]),
        "modified_sample_indices": np.unique(
            np.concatenate([label_indices, additive_flat // X.shape[1], multiplicative_flat // X.shape[1]])
        ).astype(int).tolist(),
        "modified_feature_indices": np.unique(
            np.concatenate([additive_flat % X.shape[1], multiplicative_flat % X.shape[1]])
        ).astype(int).tolist(),
    })
    return CorruptionResult(X_out, y_out, manifest)


def _high_margin(X: np.ndarray, y: np.ndarray, *, seed: int, config: dict[str, Any], input_digest: str) -> CorruptionResult:
    _require_numeric(config, ("flip_rate", "reference_C"))
    count = _rate_count(config["flip_rate"], y.size)
    reference = L1SVM(
        C=float(config["reference_C"]),
        time_limit=config.get("reference_time_limit"),
        backend=str(config.get("reference_backend", "scipy")),
        threads=int(config.get("reference_threads", 1)),
    ).fit(X, y)
    signed_margins = y * reference.decision_function(X)
    # Stable tie-breaking by original observation index.
    ranked = np.lexsort((np.arange(y.size), -signed_margins))
    flipped = np.sort(ranked[:count])
    y_out = y.copy()
    y_out[flipped] *= -1
    manifest = _manifest("high_margin", seed, config, input_digest, X, y_out)
    manifest.update({
        "flipped_label_indices": flipped.astype(int).tolist(),
        "modified_sample_indices": flipped.astype(int).tolist(),
        "modified_feature_indices": [],
        "reference_solver": reference.solver_diagnostics(),
    })
    return CorruptionResult(X.copy(), y_out, manifest)


def _manifest(
    condition: str,
    seed: int,
    config: dict[str, Any],
    input_digest: str,
    X_out: np.ndarray,
    y_out: np.ndarray,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "random_seed": int(seed),
        "parameters": config,
        "input_hash": input_digest,
        "generated_output_hash": array_hash(X_out, y_out),
        "samples": int(X_out.shape[0]),
        "features": int(X_out.shape[1]),
    }


def _rate_count(rate: float, total: int) -> int:
    rate = float(rate)
    if not 0 <= rate <= 1:
        raise ValueError("corruption rates must lie in [0, 1]")
    return min(total, int(round(rate * total)))


def _require_numeric(config: dict[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if config.get(name) is None]
    if missing:
        raise ValueError(f"missing explicit corruption parameters: {missing}")
    for name in names:
        float(config[name])


def _required_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"combined corruption requires a '{name}' configuration mapping")
    return value


def _cell_pairs(flat_indices: np.ndarray, n_features: int) -> list[list[int]]:
    return [[int(index // n_features), int(index % n_features)] for index in flat_indices]
