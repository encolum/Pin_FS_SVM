"""One-factor-at-a-time sensitivity execution with raw results saved first."""

from __future__ import annotations

import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.serialization import read_json, write_csv, write_json

from .runner import run_experiment


def run_sensitivity(
    config: dict[str, Any],
    *,
    resume_dir: str | Path | None = None,
) -> Path:
    parameter = config.get("sensitivity_parameter")
    values = _validate_and_values(config, parameter)
    if resume_dir:
        parent = Path(resume_dir).resolve()
        if not parent.is_dir():
            raise FileNotFoundError(parent)
    else:
        parent = (
            Path(config.get("output", {}).get("root", "results_v2"))
            / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-sensitivity-{parameter}"
        ).resolve()
        parent.mkdir(parents=True, exist_ok=False)
    (parent / "runs").mkdir(exist_ok=True)
    (parent / "figures").mkdir(exist_ok=True)
    state_path = parent / "value_runs.json"
    state = read_json(state_path) if state_path.exists() else {}

    for value in values:
        key = str(value)
        existing = state.get(key)
        if existing and Path(existing).is_dir():
            continue
        child = copy.deepcopy(config)
        child["output"] = {"root": str(parent / "runs")}
        if parameter == "C":
            child["C_grid"] = [value]
        elif parameter == "tau":
            child["tau_grid"] = [value]
        else:
            child["B_grid"] = [int(value)]
        run_dir = run_experiment(child, mode=f"sensitivity-{parameter}-{value}")
        state[key] = str(run_dir)
        write_json(state_path, state)

    raw_rows: list[dict[str, Any]] = []
    for value in values:
        run_dir = Path(state[str(value)])
        with (run_dir / "metrics" / "fold_metrics.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                raw_rows.append({"sensitivity_parameter": parameter, "sensitivity_value": value, **row})
    write_csv(parent / "raw_results.csv", raw_rows)
    write_json(
        parent / "manifest.json",
        {
            "mode": "sensitivity",
            "varied_parameter": parameter,
            "values": values,
            "held_fixed": _held_fixed(config, parameter),
            "child_runs": state,
            "primary_metric": "balanced_accuracy",
        },
    )
    _plot(parent, raw_rows, parameter, values)
    return parent


def _validate_and_values(config: dict[str, Any], parameter: str | None) -> list[float | int]:
    if parameter not in {"B", "C", "tau"}:
        raise ValueError("sensitivity_parameter must be exactly one of B, C, or tau")
    if len(config["models"]) != 1:
        raise ValueError("a sensitivity run must contain exactly one model")
    grids = {"C": config["C_grid"], "tau": config["tau_grid"], "B": config.get("B_grid")}
    for name, grid in grids.items():
        if not isinstance(grid, list):
            raise ValueError("sensitivity runs require explicit list grids for B, C, and tau")
        if name != parameter and len(grid) != 1:
            raise ValueError(f"{name} must contain exactly one held-fixed value")
    values = grids[parameter]
    if len(values) < 2:
        raise ValueError(f"{parameter} must contain at least two sensitivity values")
    return values


def _held_fixed(config: dict[str, Any], parameter: str) -> dict[str, Any]:
    return {
        name: grid[0]
        for name, grid in {"C": config["C_grid"], "tau": config["tau_grid"], "B": config["B_grid"]}.items()
        if name != parameter
    }


def _plot(parent: Path, rows: list[dict[str, Any]], parameter: str, values: list[float | int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means, standard_deviations = [], []
    for value in values:
        scores = np.asarray([
            float(row["balanced_accuracy"])
            for row in rows
            if str(row["sensitivity_value"]) == str(value)
        ])
        means.append(float(scores.mean()))
        standard_deviations.append(float(scores.std(ddof=1)) if scores.size > 1 else 0.0)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.errorbar(values, means, yerr=standard_deviations, marker="o", capsize=4)
    axis.set_xlabel(parameter)
    axis.set_ylabel("Balanced Accuracy")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(parent / "figures" / f"sensitivity_{parameter}.png", dpi=180)
    plt.close(figure)
