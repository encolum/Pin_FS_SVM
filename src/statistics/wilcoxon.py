"""Explicitly invoked one-sided Wilcoxon tests with BH correction."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from src.utils.serialization import read_json, write_csv, write_json


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return []
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate((ranked * values.size / np.arange(1, values.size + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted.tolist()


def run_wilcoxon_analysis(
    run_dir: str | Path,
    *,
    proposed_model: str = "pin_fs_svm",
    settings: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    run_config = read_json(config_path) if config_path.is_file() else {}
    options = {
        "metric": "balanced_accuracy",
        "alternative": "greater",
        "zero_method": "wilcox",
        "method": "auto",
        "correction": "benjamini_hochberg",
        "alpha": 0.05,
        "pairing_unit": "outer_fold",
        "correction_family": "all_declared_pairwise_comparisons",
        **run_config.get("statistics", {}),
        **(settings or {}),
    }
    if options["correction"] != "benjamini_hochberg":
        raise ValueError("only benjamini_hochberg correction is currently supported")
    if options["pairing_unit"] != "outer_fold":
        raise ValueError("only outer_fold pairing is currently supported")
    metric = str(options["metric"])
    metrics_path = run_dir / "metrics" / "fold_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indexed: dict[tuple[str, str, str, str, str], float] = {}
    models: set[str] = set()
    datasets: set[str] = set()
    for row in rows:
        key = (row["dataset"], row["condition"], row["outer_fold"], row["random_seed"], row["model"])
        if metric not in row:
            raise ValueError(f"metric column not found: {metric}")
        indexed[key] = float(row[metric])
        models.add(row["model"])
        datasets.add(row["dataset"])
    records: list[dict[str, Any]] = []
    for dataset in sorted(datasets):
        for baseline in sorted(models - {proposed_model}):
            differences: list[float] = []
            for (row_dataset, condition, fold, seed, model), proposed in indexed.items():
                if row_dataset != dataset or model != proposed_model:
                    continue
                baseline_key = (dataset, condition, fold, seed, baseline)
                if baseline_key in indexed:
                    differences.append(proposed - indexed[baseline_key])
            if not differences:
                continue
            array = np.asarray(differences)
            raw_p = 1.0 if np.allclose(array, 0.0) else float(
                wilcoxon(
                    array,
                    alternative=str(options["alternative"]),
                    zero_method=str(options["zero_method"]),
                    method=str(options["method"]),
                ).pvalue
            )
            records.append({
                "baseline": baseline,
                "dataset": dataset,
                "raw_p": raw_p,
                "mean_difference": float(array.mean()),
                "median_difference": float(np.median(array)),
                "n_pairs": int(array.size),
            })
    adjusted = benjamini_hochberg([record["raw_p"] for record in records])
    for record, adjusted_p in zip(records, adjusted):
        record["adjusted_p"] = adjusted_p
    output = run_dir / "aggregate" / "wilcoxon.csv"
    write_csv(
        output,
        records,
        fieldnames=["baseline", "dataset", "raw_p", "adjusted_p", "mean_difference", "median_difference", "n_pairs"],
    )
    write_json(
        run_dir / "aggregate" / "wilcoxon_metadata.json",
        {
            "proposed_model": proposed_model,
            "settings": options,
            "correction_family": [
                {"dataset": record["dataset"], "baseline": record["baseline"]}
                for record in records
            ],
            "number_of_tests": len(records),
        },
    )
    return output
