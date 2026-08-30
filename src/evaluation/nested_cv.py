"""Nested stratified cross-validation with partition-local corruption and scaling."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.corruptions import apply_corruption
from src.data.preprocessing import fit_transform_training
from src.experiments.registry import create_model
from src.experiments.search import parameter_candidates
from src.utils.seeds import derive_seed

from .metrics import classification_metrics
from .predictions import prediction_rows


def run_nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    dataset: str,
    condition: str,
    model_name: str,
    config: dict[str, Any],
    base_seed: int,
) -> dict[str, Any]:
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    candidates = parameter_candidates(model_name, X.shape[1], config)
    outer_cv = StratifiedKFold(
        n_splits=int(config["outer_folds"]),
        shuffle=True,
        random_state=int(base_seed),
    )
    folds: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_audit: list[dict[str, Any]] = []
    corruption_config = _condition_config(config, condition)

    for outer_fold, (outer_train, outer_test) in enumerate(outer_cv.split(X, y), start=1):
        fold_started = perf_counter()
        X_outer_train, y_outer_train = X[outer_train], y[outer_train]
        inner_seed = derive_seed(base_seed, dataset, condition, outer_fold, "inner")
        inner_cv = StratifiedKFold(
            n_splits=int(config["inner_folds"]),
            shuffle=True,
            random_state=inner_seed,
        )
        inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))
        split_audit.append({
            "outer_fold": outer_fold,
            "outer_train_indices": outer_train.astype(int).tolist(),
            "outer_test_indices": outer_test.astype(int).tolist(),
            "inner": [
                {
                    "train_indices": outer_train[train].astype(int).tolist(),
                    "validation_indices": outer_train[validation].astype(int).tolist(),
                }
                for train, validation in inner_splits
            ],
        })
        search_started = perf_counter()
        search_records: list[dict[str, Any]] = []
        best_parameters: dict[str, Any] | None = None
        best_score = -np.inf
        best_tie_key: tuple[Any, ...] | None = None
        selection_tolerance = float(config.get("selection_tolerance", 1e-12))

        prepared_inner: list[dict[str, Any]] = []
        inner_corruption_manifests: list[dict[str, Any]] = []
        for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits, start=1):
            corruption_seed = derive_seed(base_seed, dataset, condition, outer_fold, inner_fold, "corruption")
            X_train, transformed, _ = fit_transform_training(
                X_outer_train[inner_train],
                X_outer_train[inner_validation],
            )
            corrupted = apply_corruption(
                X_train,
                y_outer_train[inner_train],
                condition,
                seed=corruption_seed,
                config=corruption_config,
            )
            prepared_inner.append({
                "inner_fold": inner_fold,
                "seed": corruption_seed,
                "X_train": corrupted.X,
                "y_train": corrupted.y,
                "X_validation": transformed[0],
                "y_validation": y_outer_train[inner_validation],
            })
            inner_corruption_manifests.append({"inner_fold": inner_fold, **corrupted.manifest})

        max_workers = int(config.get("parallelism", {}).get("max_workers", 1))
        if max_workers > 1 and len(candidates) > 1:
            from joblib import Parallel, delayed

            evaluations = Parallel(n_jobs=min(max_workers, len(candidates)), backend="loky")(
                delayed(_evaluate_candidate)(model_name, parameters, config, prepared_inner)
                for parameters in candidates
            )
        else:
            evaluations = [
                _evaluate_candidate(model_name, parameters, config, prepared_inner)
                for parameters in candidates
            ]

        for record, candidate_failure in evaluations:
            parameters = record["parameters"]
            scores = record["balanced_accuracy_folds"]
            selected_feature_counts = record["selected_feature_count_folds"]
            candidate_failed = record["status"] == "failed"
            if candidate_failure:
                failures.append({
                    "dataset": dataset,
                    "condition": condition,
                    "model": model_name,
                    "outer_fold": outer_fold,
                    "parameters": parameters,
                    "stage": "inner_search",
                    **candidate_failure,
                })
            search_records.append(record)
            if not candidate_failed and len(scores) == len(inner_splits):
                mean_score = float(np.mean(scores))
                mean_selected = float(np.mean(selected_feature_counts))
                tie_key = _selection_tie_key(parameters, mean_selected)
                if (
                    mean_score > best_score + selection_tolerance
                    or (
                        abs(mean_score - best_score) <= selection_tolerance
                        and (best_tie_key is None or tie_key < best_tie_key)
                    )
                ):
                    best_score, best_parameters = mean_score, dict(parameters)
                    best_tie_key = tie_key

        search_time = perf_counter() - search_started
        if best_parameters is None:
            raise RuntimeError(
                f"all hyperparameter configurations failed for {dataset}/{condition}/{model_name}/outer-{outer_fold}"
            )
        final_seed = derive_seed(base_seed, dataset, condition, outer_fold, "outer_corruption")
        X_train, transformed, _ = fit_transform_training(X_outer_train, X[outer_test])
        corrupted = apply_corruption(
            X_train,
            y_outer_train,
            condition,
            seed=final_seed,
            config=corruption_config,
        )
        model = create_model(model_name, best_parameters, config, seed=final_seed)
        model.fit(corrupted.X, corrupted.y)
        scores = model.decision_function(transformed[0])
        predictions = np.where(scores >= 0.0, 1, -1).astype(int)
        metrics = classification_metrics(y[outer_test], predictions)
        selected = model.get_selected_features()
        folds.append({
            "outer_fold": outer_fold,
            "random_seed": int(final_seed),
            "best_parameters": best_parameters,
            "mean_inner_balanced_accuracy": best_score,
            "metrics": metrics,
            "selected_feature_indices": selected,
            "selected_feature_count": len(selected),
            "coefficients": model.w_.tolist(),
            "intercept": float(model.b_),
            "predictions": prediction_rows(outer_test, y[outer_test], predictions, scores),
            "model_fit_time": float(model.fit_time_),
            "hyperparameter_search_time": search_time,
            "total_outer_fold_time": perf_counter() - fold_started,
            "solver": model.solver_diagnostics(),
            "corruption_manifest": corrupted.manifest,
            "inner_corruption_manifests": inner_corruption_manifests,
            "search": search_records,
        })
    return {
        "dataset": dataset,
        "condition": condition,
        "model": model_name,
        "base_seed": int(base_seed),
        "candidate_count": len(candidates),
        "folds": folds,
        "failures": failures,
        "split_audit": split_audit,
    }


def _condition_config(config: dict[str, Any], condition: str) -> dict[str, Any]:
    if condition == "clean":
        return {}
    section = dict(config.get("corruption", {}).get(condition, {}))
    solver = config.get("solver", {})
    reference_settings = {
        "reference_backend": solver.get("backend", "scipy"),
        "reference_threads": int(solver.get("threads", 1)),
        "reference_time_limit": solver.get("time_limit"),
    }
    if condition == "high_margin":
        section = {**section, **reference_settings}
    elif condition == "combined" and isinstance(section.get("high_margin"), dict):
        section["high_margin"] = {**section["high_margin"], **reference_settings}
    return section


def _selection_tie_key(parameters: dict[str, Any], mean_selected_features: float) -> tuple[Any, ...]:
    """Paper-plan tie-break: sparsity, then B, then parameter order."""
    budget = int(parameters.get("B", 0))
    parameter_order = tuple((name, _sortable_parameter(parameters[name])) for name in sorted(parameters))
    return (mean_selected_features, budget, parameter_order)


def _sortable_parameter(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return (0, float(value))
    if isinstance(value, str):
        return (1, value)
    return (2, repr(value))


def _evaluate_candidate(
    model_name: str,
    parameters: dict[str, Any],
    config: dict[str, Any],
    prepared_inner: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    scores: list[float] = []
    selected_feature_counts: list[int] = []
    failure: dict[str, Any] | None = None
    for prepared in prepared_inner:
        try:
            model = create_model(model_name, parameters, config, seed=prepared["seed"])
            model.fit(prepared["X_train"], prepared["y_train"])
            metrics = classification_metrics(
                prepared["y_validation"],
                model.predict(prepared["X_validation"]),
            )
            scores.append(metrics["balanced_accuracy"])
            selected_feature_counts.append(model.get_num_selected_features())
        except Exception as exc:
            failure = {
                "inner_fold": prepared["inner_fold"],
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            break
    complete = failure is None and len(scores) == len(prepared_inner)
    record = {
        "parameters": parameters,
        "balanced_accuracy_folds": scores,
        "mean_balanced_accuracy": float(np.mean(scores)) if complete else None,
        "selected_feature_count_folds": selected_feature_counts,
        "mean_selected_feature_count": float(np.mean(selected_feature_counts)) if complete else None,
        "status": "complete" if complete else "failed",
    }
    return record, failure
