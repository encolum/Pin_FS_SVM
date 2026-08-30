"""Configuration loading and validation.

The distributed .yaml files use JSON syntax, which is a strict subset of YAML. This
keeps the CLI usable with the standard library while still accepting PyYAML files
when PyYAML is installed.
"""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from src.data.loaders import DATASET_SPECS

from .registry import MODEL_NAMES, MILP_MODEL_NAMES


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("non-JSON YAML requires PyYAML") from exc
        config = yaml.safe_load(text)
    if not isinstance(config, dict):
        raise ValueError("the configuration root must be a mapping")
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict[str, Any], *, require_corruption_parameters: bool = True) -> None:
    for key in ("datasets", "models", "conditions", "outer_folds", "inner_folds", "seeds", "C_grid", "tau_grid"):
        if key not in config:
            raise ValueError(f"missing configuration key: {key}")
    unknown_datasets = sorted(set(config["datasets"]) - set(DATASET_SPECS))
    unknown_models = sorted(set(config["models"]) - MODEL_NAMES)
    unknown_conditions = sorted(set(config["conditions"]) - {"clean", "mixed", "high_margin", "combined"})
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}")
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {unknown_conditions}")
    if int(config["outer_folds"]) < 2 or int(config["inner_folds"]) < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2")
    if not config["seeds"]:
        raise ValueError("at least one random seed is required")
    selection_tolerance = float(config.get("selection_tolerance", 1e-12))
    if selection_tolerance < 0:
        raise ValueError("selection_tolerance must be nonnegative")
    solver = config.get("solver", {})
    if not isinstance(solver, dict):
        raise ValueError("solver must be a mapping")
    if solver.get("backend", "scipy") not in {"scipy", "cplex"}:
        raise ValueError("solver.backend must be scipy or cplex")
    if solver.get("backend") == "cplex" and (find_spec("docplex") is None or find_spec("cplex") is None):
        raise ValueError("solver.backend='cplex' requires requirements-cplex.txt")
    if int(solver.get("threads", 1)) < 1:
        raise ValueError("solver.threads must be at least 1")
    parallelism = config.get("parallelism", {"max_workers": 1, "allow_nested_parallelism": False})
    if not isinstance(parallelism, dict) or int(parallelism.get("max_workers", 1)) < 1:
        raise ValueError("parallelism.max_workers must be at least 1")
    if parallelism.get("allow_nested_parallelism", False):
        raise ValueError("nested parallelism is disabled by the corrected experiment protocol")
    metrics = config.get("metrics", ["balanced_accuracy", "weighted_f1", "accuracy", "gmean"])
    if metrics != ["balanced_accuracy", "weighted_f1", "accuracy", "gmean"]:
        raise ValueError("metrics must be exactly balanced_accuracy, weighted_f1, accuracy, and gmean")
    if set(config["models"]) & MILP_MODEL_NAMES:
        bounds = config.get("coefficient_bounds")
        if not isinstance(bounds, dict) or bounds.get("lower") is None or bounds.get("upper") is None:
            raise ValueError("MILP models require explicit coefficient_bounds.lower and coefficient_bounds.upper")
        if not float(bounds["lower"]) < 0 < float(bounds["upper"]):
            raise ValueError("coefficient bounds must satisfy lower < 0 < upper")
        if config.get("require_author_confirmation") and not bounds.get("author_confirmed", False):
            raise ValueError("the coefficient bounds require explicit author confirmation")
    if config.get("corrupt_outer_test", False):
        raise ValueError("test-time corruption is not part of the default main protocol")
    if require_corruption_parameters:
        corruption = config.get("corruption", {})
        for condition in set(config["conditions"]) - {"clean"}:
            section = corruption.get(condition)
            if not isinstance(section, dict):
                raise ValueError(f"condition '{condition}' requires an explicit corruption configuration")
            _validate_corruption_section(condition, section)
    _validate_statistics(config.get("statistics", {}))


def _validate_corruption_section(condition: str, section: dict[str, Any]) -> None:
    if condition == "mixed":
        required = {"label_flip_rate", "additive_rate", "multiplicative_rate", "additive_std", "multiplicative_std"}
    elif condition == "high_margin":
        required = {"flip_rate", "reference_C"}
    else:
        mixed, margin = section.get("mixed"), section.get("high_margin")
        if not isinstance(mixed, dict) or not isinstance(margin, dict):
            raise ValueError("combined corruption requires mixed and high_margin mappings")
        _validate_corruption_section("mixed", mixed)
        _validate_corruption_section("high_margin", margin)
        return
    missing = sorted(name for name in required if section.get(name) is None)
    if missing:
        raise ValueError(f"{condition} corruption has unresolved parameters: {missing}")


def _validate_statistics(section: Any) -> None:
    if not isinstance(section, dict):
        raise ValueError("statistics must be a mapping")
    supported = {
        "metric": {"balanced_accuracy", "weighted_f1", "accuracy", "gmean"},
        "alternative": {"greater", "less", "two-sided"},
        "zero_method": {"wilcox", "pratt", "zsplit"},
        "method": {"auto", "exact", "approx"},
        "correction": {"benjamini_hochberg"},
        "pairing_unit": {"outer_fold"},
    }
    for name, allowed in supported.items():
        value = section.get(name)
        if value is not None and value not in allowed:
            raise ValueError(f"unsupported statistics.{name}: {value}")
    alpha = float(section.get("alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError("statistics.alpha must lie strictly between 0 and 1")
