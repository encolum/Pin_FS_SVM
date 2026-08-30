"""Reproducibility manifest construction."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.data.loaders import file_hash


def build_manifest(config: dict[str, Any], *, run_id: str, planned_fits: int) -> dict[str, Any]:
    config_snapshot = {key: value for key, value in config.items() if not key.startswith("_")}
    return {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit(),
        "config_path": config.get("_config_path"),
        "config_sha256": sha256(
            json.dumps(config_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "python_version": sys.version,
        "package_versions": {name: _version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "docplex", "cplex"
        )},
        "machine_information": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
        },
        "dataset_hashes": {dataset: file_hash(dataset, "clean") for dataset in config["datasets"]},
        "generated_data_hashes": [],
        "random_seeds": [int(seed) for seed in config["seeds"]],
        "outer_cv_definition": {
            "class": "StratifiedKFold", "n_splits": int(config["outer_folds"]),
            "shuffle": True, "random_state": "base seed",
        },
        "inner_cv_definition": {
            "class": "StratifiedKFold", "n_splits": int(config["inner_folds"]),
            "shuffle": True, "random_state": "derived stable seed",
        },
        "model_grids": {
            "C_grid": config["C_grid"], "tau_grid": config["tau_grid"],
            "B_grid": config.get("B_grid", "auto"), "overrides": config.get("model_grids", {}),
        },
        "solver_settings": config.get("solver", {}),
        "parallelism": config.get("parallelism", {"max_workers": 1, "allow_nested_parallelism": False}),
        "preprocessing_protocol": "fit scaler on clean training partition, then corrupt standardized training only",
        "statistics_settings": config.get("statistics", {}),
        "coefficient_bounds": config.get("coefficient_bounds"),
        "number_of_planned_fits": int(planned_fits),
        "number_of_completed_fits": 0,
    }


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
