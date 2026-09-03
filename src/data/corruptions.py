"""Deterministic, partition-local corruption generators with complete manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.utils.sparsefuncs import mean_variance_axis
from src.utils.matrices import data_hash, numeric_matrix


@dataclass
class CorruptionResult:
    X: np.ndarray
    y: np.ndarray
    manifest: dict[str, Any]


def validate_corruption_profile(condition, config):
    if condition == "clean":
        return
    if condition == "combined":
        validate_corruption_profile("mixed", _required_mapping(config, "mixed"))
        validate_corruption_profile(
            "feature_outlier", _required_mapping(config, "feature_outlier")
        )
        return
    fields = {"label_noise": ("label_flip_rate",),
              "mixed": ("label_flip_rate", "additive_rate", "multiplicative_rate", "additive_std", "multiplicative_std"),
              "feature_outlier": ("sample_rate", "feature_rate", "scale")}
    if condition not in fields:
        raise ValueError(f"unknown condition: {condition}")
    _require_numeric(config, fields[condition])
    for key in fields[condition]:
        if key.endswith("rate"):
            _rate_count(config[key], 1)
    if condition == "mixed" and float(config["additive_rate"]) + float(config["multiplicative_rate"]) > 1:
        raise ValueError("disjoint additive_rate + multiplicative_rate must be <= 1")


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
    X = numeric_matrix(X)
    y = np.asarray(y, dtype=int)
    condition = condition.lower()
    config = dict(config or {})
    validate_corruption_profile(condition, config)
    input_digest = data_hash(X, y)
    if condition == "clean":
        return CorruptionResult(X.copy(), y.copy(), _manifest(condition, seed, config, input_digest, X, y))
    if condition == "label_noise":
        profile = {**config, "additive_rate": 0., "multiplicative_rate": 0.,
                   "additive_std": 0., "multiplicative_std": 0.}
        result = _mixed(X, y, seed=seed, config=profile, input_digest=input_digest)
        result.manifest.update(condition=condition, parameters=config)
        return result
    if condition == "mixed":
        return _mixed(X, y, seed=seed, config=config, input_digest=input_digest)
    if condition == "feature_outlier":
        return _feature_outlier(X, y, seed=seed, config=config, input_digest=input_digest)
    if condition == "combined":
        # Freeze scale BEFORE either corruption stage, using clean training only.
        feature_scale = _training_feature_scale(X)
        mixed_config = _required_mapping(config, "mixed")
        mixed = _mixed(X, y, seed=seed, config=mixed_config, input_digest=input_digest)
        outlier = _feature_outlier(
            mixed.X,
            mixed.y,
            seed=seed + 1,
            config=_required_mapping(config, "feature_outlier"),
            input_digest=data_hash(mixed.X, mixed.y),
            feature_scale=feature_scale,
        )
        manifest = _manifest(condition, seed, config, input_digest, outlier.X, outlier.y)
        manifest["stages"] = [mixed.manifest, outlier.manifest]
        return CorruptionResult(outlier.X, outlier.y, manifest)
    raise ValueError(f"unknown condition: {condition}")


def _mixed(X: np.ndarray, y: np.ndarray, *, seed: int, config: dict[str, Any], input_digest: str) -> CorruptionResult:
    required = ("label_flip_rate", "additive_rate", "multiplicative_rate", "additive_std", "multiplicative_std")
    _require_numeric(config, required)
    rng = np.random.default_rng(seed)
    X_out, y_out = X.copy(), y.copy()
    label_count = _rate_count(config["label_flip_rate"], y.size)
    label_indices = np.sort(rng.choice(y.size, size=label_count, replace=False)) if label_count else np.array([], dtype=int)
    y_out[label_indices] *= -1
    eligible = _eligible_sparse_cells(X) if sparse.issparse(X) else None
    total_cells = len(eligible) if eligible is not None else int(X.shape[0] * X.shape[1])
    additive_count = _rate_count(config["additive_rate"], total_cells)
    # Rounding two disjoint proportions can exceed N by one; additive first.
    multiplicative_count = min(total_cells - additive_count, _rate_count(config["multiplicative_rate"], total_cells))
    _sparse_corruption_guard(X, additive_count + multiplicative_count, config)
    selected = rng.choice(total_cells, size=additive_count + multiplicative_count, replace=False)
    if eligible is not None:
        selected = eligible[selected]
    additive_flat = np.sort(selected[:additive_count])
    multiplicative_flat = np.sort(selected[additive_count:])
    if additive_count:
        X_out = _modify_cells(X_out, additive_flat, rng.normal(0.0, float(config["additive_std"]), size=additive_count))
    if multiplicative_count:
        X_out = _modify_cells(X_out, multiplicative_flat,
                             rng.normal(1.0, float(config["multiplicative_std"]), size=multiplicative_count), multiply=True)
    manifest = _manifest("mixed", seed, config, input_digest, X_out, y_out)
    changed = _feature_noise_audit(manifest, X, X_out, total_cells,
                                   {"additive": additive_flat, "multiplicative": multiplicative_flat})
    manifest.update({
        "flipped_label_indices": label_indices.tolist(),
        "label_changed_count": int(label_count),
        "label_effective_rate": label_count / y.size,
        "masks_disjoint": True,
        "rounding_policy": "round counts; cap multiplicative to remaining eligible cells",
        "additive_cells": _cell_pairs(additive_flat, X.shape[1]),
        "multiplicative_cells": _cell_pairs(multiplicative_flat, X.shape[1]),
        "modified_sample_indices": np.unique(
            np.concatenate([label_indices, changed // X.shape[1]])
        ).astype(int).tolist(),
        "modified_feature_indices": np.unique(
            changed % X.shape[1]
        ).astype(int).tolist(),
    })
    return CorruptionResult(X_out, y_out, manifest)


def _sparse_corruption_guard(X, count, config):
    if sparse.issparse(X) and count:
        limit = config.get("max_modified_cells")
        if type(limit) is not int or limit < count:
            raise ValueError("sparse feature corruption requires explicit max_modified_cells >= selected cells")


def _modify_cells(X, flat_indices, values, *, multiply=False):
    if not sparse.issparse(X):
        rows, columns = flat_indices // X.shape[1], flat_indices % X.shape[1]
        if multiply:
            X[rows, columns] *= values
        else:
            X[rows, columns] += values
        return X
    result = X.tocsr(copy=True)
    # CSR data updates cannot introduce new structural nonzeros or densify.
    stored = np.repeat(np.arange(X.shape[0]), np.diff(result.indptr)) * X.shape[1] + result.indices
    positions = np.searchsorted(stored, flat_indices)
    if multiply:
        result.data[positions] *= values
    else:
        result.data[positions] += values
    return result


def _feature_outlier(X, y, *, seed, config, input_digest, feature_scale=None):
    _require_numeric(config, ("sample_rate", "feature_rate", "scale"))
    rng = np.random.default_rng(seed)
    n_rows = _rate_count(config["sample_rate"], X.shape[0])
    n_columns = _rate_count(config["feature_rate"], X.shape[1])
    rows = np.sort(rng.choice(X.shape[0], n_rows, replace=False))
    columns = np.sort(rng.choice(X.shape[1], n_columns, replace=False))
    if sparse.issparse(X):
        eligible = _eligible_sparse_cells(X)
        total_cells = len(eligible)
        indices = eligible[np.isin(eligible // X.shape[1], rows) & np.isin(eligible % X.shape[1], columns)]
    else:
        total_cells = int(X.shape[0] * X.shape[1])
        indices = (rows[:, None] * X.shape[1] + columns).ravel()
    _sparse_corruption_guard(X, len(indices), config)
    if feature_scale is None:
        feature_scale = _training_feature_scale(X)
    deviations = float(config["scale"]) * feature_scale[indices % X.shape[1]]
    result = _modify_cells(X.copy(), indices, rng.normal(scale=deviations, size=len(indices)))
    manifest = _manifest("feature_outlier", seed, config, input_digest, result, y)
    changed = _feature_noise_audit(manifest, X, result, total_cells, {"outlier": indices})
    manifest.update(selected_sample_indices=rows.tolist(), selected_feature_indices=columns.tolist(),
                    modified_sample_indices=np.unique(changed // X.shape[1]).tolist(),
                    modified_feature_indices=np.unique(changed % X.shape[1]).tolist(),
                    outlier_cells=_cell_pairs(indices, X.shape[1]), flipped_label_indices=[],
                    feature_scale=feature_scale.tolist(),
                    feature_scale_source="clean_preprocessed_training_population_std",
                    zero_variance_policy="leave unchanged; report as selected but ineffective")
    return CorruptionResult(result, y.copy(), manifest)


def _eligible_sparse_cells(X):
    # nonzero() excludes explicitly stored zeros, unlike CSR.nnz.
    rows, columns = X.nonzero()
    return rows.astype(np.int64) * X.shape[1] + columns


def _training_feature_scale(X):
    if sparse.issparse(X):
        _, variance = mean_variance_axis(X, axis=0)
        scale = np.sqrt(np.maximum(variance, 0.))
    else:
        scale = np.std(X, axis=0, ddof=0)
    if not np.isfinite(scale).all():
        raise ValueError("training feature scale must be finite")
    return scale


def _feature_noise_audit(manifest, before, after, eligible_count, masks):
    """Distinguish sampled cells from effective changes (e.g. zero severity)."""
    manifest.update(feature_sampling="nonzero_entries" if sparse.issparse(before) else "all_entries",
                    eligible_feature_cells=int(eligible_count))
    changed_masks = []
    for name, indices in masks.items():
        rows, columns = indices // before.shape[1], indices % before.shape[1]
        # SciPy returns a sparse (1, 0) slice for empty paired indexing.
        old = np.asarray(before[rows, columns]).reshape(-1) if indices.size else np.empty(0)
        new = np.asarray(after[rows, columns]).reshape(-1) if indices.size else np.empty(0)
        if not np.isfinite(new).all():
            raise ValueError("feature corruption produced nonfinite values")
        changed = indices[old != new]
        changed_masks.append(changed)
        manifest.update({f"{name}_selected_count": int(len(indices)),
                         f"{name}_changed_count": int(len(changed)),
                         f"{name}_effective_rate": len(changed) / eligible_count if eligible_count else 0.,
                         f"effective_{name}_cells": _cell_pairs(changed, before.shape[1])})
    return np.unique(np.concatenate(changed_masks)).astype(np.int64)


def _manifest(
    condition: str,
    seed: int,
    config: dict[str, Any],
    input_digest: str,
    X_out: np.ndarray,
    y_out: np.ndarray,
) -> dict[str, Any]:
    return {
        "corruption_protocol_version": 2,
        "condition": condition,
        "random_seed": int(seed),
        "parameters": config,
        "input_hash": input_digest,
        "generated_output_hash": data_hash(X_out, y_out),
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
        value = float(config[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"corruption parameter {name} must be finite and non-negative")


def _required_mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"combined corruption requires a '{name}' configuration mapping")
    return value


def _cell_pairs(flat_indices: np.ndarray, n_features: int) -> list[list[int]]:
    return [[int(index // n_features), int(index % n_features)] for index in flat_indices]
