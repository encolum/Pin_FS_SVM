"""Dataset-first aggregation; across-dataset means give each dataset equal weight."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


MEASURES = (
    "balanced_accuracy", "weighted_f1", "accuracy", "gmean", "selected_feature_count",
    "model_fit_time", "hyperparameter_search_time", "total_outer_fold_time",
)


def aggregate_fold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["condition"]), str(row["model"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, condition, model), values in sorted(grouped.items()):
        record: dict[str, Any] = {
            "dataset": dataset, "condition": condition, "model": model, "outer_fold_count": len(values),
        }
        for measure in MEASURES:
            numbers = np.asarray([float(value[measure]) for value in values], dtype=float)
            record[f"mean_{measure}"] = float(numbers.mean())
            record[f"std_{measure}"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
        output.append(record)
    return output


def across_dataset_summary(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        grouped[(str(row["condition"]), str(row["model"]))].append(row)
    output: list[dict[str, Any]] = []
    for (condition, model), values in sorted(grouped.items()):
        record: dict[str, Any] = {"condition": condition, "model": model, "dataset_count": len(values)}
        for measure in MEASURES:
            numbers = np.asarray([float(value[f"mean_{measure}"]) for value in values])
            record[f"mean_{measure}"] = float(numbers.mean())
            record[f"std_across_datasets_{measure}"] = float(numbers.std(ddof=1)) if numbers.size > 1 else 0.0
        output.append(record)
    return output
