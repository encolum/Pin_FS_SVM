"""Deterministic, parity-preserving hyperparameter grid construction."""

from __future__ import annotations

from itertools import product
from typing import Any


def build_budget_grid(n_features: int, configured: str | list[int] | None = "auto", *, exhaustive_max: int = 30) -> list[int]:
    if configured not in (None, "auto"):
        values = sorted({int(value) for value in configured if 1 <= int(value) <= n_features})
        if not values:
            raise ValueError("the configured B grid contains no valid budgets")
        return values
    if n_features <= exhaustive_max:
        return list(range(1, n_features + 1))
    coarse = [1, 2, 3, 5, 10, 15, 25, 50, 100, 250, 500, n_features]
    return sorted({value for value in coarse if value <= n_features})


def target_feature_grid(n_features: int, configured: list[int] | None = None) -> list[int]:
    if configured:
        return sorted({max(1, min(n_features, int(value))) for value in configured})
    return sorted({max(1, round(n_features * fraction)) for fraction in (0.25, 0.5, 0.75)})


def parameter_candidates(model_name: str, n_features: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = config.get("model_grids", {}).get(model_name, {})
    C_values = overrides.get("C", config["C_grid"])
    tau_values = overrides.get("tau", config["tau_grid"])
    B_values = build_budget_grid(
        n_features,
        overrides.get("B", config.get("B_grid", "auto")),
        exhaustive_max=int(config.get("small_budget_exhaustive_max", 30)),
    )
    if model_name in {"l1_svm", "l2_svm"}:
        names, values = ("C",), (C_values,)
    elif model_name == "fisher_l1_svm":
        thresholds = overrides.get("threshold_percentile", [25, 50, 75])
        names, values = ("C", "threshold_percentile"), (C_values, thresholds)
    elif model_name == "pin_svm" or model_name == "pinball_l1_svm":
        names, values = ("C", "tau"), (C_values, tau_values)
    elif model_name == "pin_fs_svm":
        names, values = ("C", "tau", "B"), (C_values, tau_values, B_values)
    elif model_name == "budgeted_milp_svm":
        names, values = ("B",), (B_values,)
    elif model_name == "pinball_cardinality_svm":
        names, values = ("tau", "B"), (tau_values, B_values)
    elif model_name == "l1_svm_rfe":
        targets = target_feature_grid(n_features, overrides.get("target_features"))
        names, values = ("C", "target_features"), (C_values, targets)
    else:
        raise KeyError(f"unknown model: {model_name}")
    return [dict(zip(names, combination)) for combination in product(*values)]


def model_fit_cost(model_name: str, n_features: int, parameters: dict[str, Any], config: dict[str, Any]) -> int:
    """Number of underlying optimizer fits performed by one model.fit call."""
    if model_name == "l1_svm_rfe":
        return n_features - int(parameters["target_features"]) + 1
    return 1


def estimate_top_level_fits(config: dict[str, Any]) -> int:
    outer, inner = int(config["outer_folds"]), int(config["inner_folds"])
    seeds = len(config["seeds"])
    conditions = len(config["conditions"])
    total = 0
    from src.data.loaders import DATASET_SPECS

    for dataset in config["datasets"]:
        n_features = DATASET_SPECS[dataset].features
        for model_name in config["models"]:
            candidates = len(parameter_candidates(model_name, n_features, config))
            total += seeds * conditions * outer * (candidates * inner + 1)
    return total


def estimate_model_fits(config: dict[str, Any]) -> int:
    """Conservative count of underlying optimization fits, including Fisher/RFE internals."""
    outer, inner = int(config["outer_folds"]), int(config["inner_folds"])
    seeds = len(config["seeds"])
    conditions = len(config["conditions"])
    total = 0
    from src.data.loaders import DATASET_SPECS

    for dataset in config["datasets"]:
        n_features = DATASET_SPECS[dataset].features
        for model_name in config["models"]:
            candidates = parameter_candidates(model_name, n_features, config)
            costs = [model_fit_cost(model_name, n_features, parameters, config) for parameters in candidates]
            # The final selected RFE target is unknown before search, so use the largest possible cost.
            per_outer_fold = inner * sum(costs) + max(costs)
            total += seeds * conditions * outer * per_outer_fold
    return total
