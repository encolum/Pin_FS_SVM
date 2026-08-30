"""Strict loaders for the six manuscript datasets and archived variants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Dataset" / "Dataset"


@dataclass(frozen=True)
class DatasetSpec:
    samples: int
    features: int
    positive: int
    negative: int
    files: dict[str, str]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "diabetes": DatasetSpec(768, 8, 268, 500, {
        "clean": "diabetes.csv", "mixed": "diabetes_noise_label_feature.csv",
        "high_margin": "diabetes_outlier.csv", "combined": "diabetes_both_noise_outlier.csv",
    }),
    "cleveland": DatasetSpec(303, 13, 139, 164, {
        "clean": "Heart_disease_cleveland_new.csv", "mixed": "clevaland_noise_label_feature.csv",
        "high_margin": "clevaland_outlier.csv", "combined": "cleveland_both_noise_outlier.csv",
    }),
    "wdbc": DatasetSpec(569, 30, 212, 357, {
        "clean": "wdbc.data.txt", "mixed": "wdbc_noisy_label_feature.txt",
        "high_margin": "wdbc_noisy_label_outlier.txt", "combined": "wdbc_both_noise_outlier.txt",
    }),
    "ionosphere": DatasetSpec(351, 34, 225, 126, {
        "clean": "ionosphere.data", "mixed": "ionosphere_noise_label_feature.txt",
        "high_margin": "ionosphere_outlier.txt", "combined": "ionosphere_both_noise_outlier.txt",
    }),
    "sonar": DatasetSpec(208, 60, 111, 97, {
        "clean": "sonar.txt", "mixed": "sonar_noise_label_feature.txt",
        "high_margin": "sonar_outlier.txt", "combined": "sonar_both_noise_outlier.txt",
    }),
    "colon": DatasetSpec(62, 2000, 22, 40, {
        "clean": "colon.csv", "mixed": "colon_noise_label_feature.csv",
        "high_margin": "colon_outlier.csv", "combined": "colon_both_noise_outlier.csv",
    }),
}


def _validate_expected_manifest() -> None:
    path = PROJECT_ROOT / "configs" / "datasets.yaml"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if set(manifest) != set(DATASET_SPECS):
        raise ValueError("configs/datasets.yaml and DATASET_SPECS list different datasets")
    for dataset, spec in DATASET_SPECS.items():
        expected = manifest[dataset]
        actual = {
            "samples": spec.samples,
            "features": spec.features,
            "positive": spec.positive,
            "negative": spec.negative,
        }
        if expected != actual:
            raise ValueError(f"configs/datasets.yaml disagrees with loader definition for {dataset}")


_validate_expected_manifest()

CONDITION_ALIASES = {
    "original": "clean",
    "noise": "mixed",
    "outlier": "high_margin",
    "both": "combined",
}


def _first_row_mode(path: Path) -> tuple[int | None, int | None]:
    """Return (header, skiprows), retaining numeric first observations."""
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        first = stream.readline()
    cells = [cell.strip() for cell in first.split(",")]
    if not any(cells):
        return None, 1
    try:
        float(cells[0])
        return None, None
    except ValueError:
        return 0, None


def _map_labels(dataset: str, raw: pd.Series) -> np.ndarray:
    values = raw.astype(str).str.strip()
    if dataset == "wdbc":
        mapped = values.map({"M": 1, "B": -1})
    elif dataset == "ionosphere":
        mapped = values.str.lower().map({"g": 1, "b": -1})
    elif dataset == "sonar":
        mapped = values.str.upper().map({"M": 1, "R": -1})
    elif dataset == "colon":
        numeric = pd.to_numeric(values, errors="raise")
        mapped = pd.Series(np.where(numeric == 2, -1, 1), index=raw.index)
    else:
        numeric = pd.to_numeric(values, errors="raise")
        mapped = pd.Series(np.where(numeric == 0, -1, 1), index=raw.index)
    if mapped.isna().any():
        unknown = sorted(values[mapped.isna()].unique().tolist())
        raise ValueError(f"unrecognized labels for {dataset}: {unknown}")
    return mapped.to_numpy(dtype=int)


def load_dataset(
    dataset: str,
    condition: str = "clean",
    *,
    data_root: str | Path | None = None,
    validate_classes: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = dataset.lower()
    condition = CONDITION_ALIASES.get(condition.lower(), condition.lower())
    if dataset not in DATASET_SPECS:
        raise KeyError(f"unknown dataset: {dataset}")
    spec = DATASET_SPECS[dataset]
    if condition not in spec.files:
        raise KeyError(f"unknown condition for {dataset}: {condition}")
    path = Path(data_root) / spec.files[condition] if data_root else DEFAULT_DATA_ROOT / spec.files[condition]
    if not path.is_file():
        raise FileNotFoundError(path)
    header, skiprows = _first_row_mode(path)
    frame = pd.read_csv(path, header=header, skiprows=skiprows, encoding="utf-8-sig")
    if frame.isna().all(axis=1).any():
        frame = frame.loc[~frame.isna().all(axis=1)].reset_index(drop=True)
    if dataset == "wdbc":
        X_raw, y_raw = frame.iloc[:, 2:], frame.iloc[:, 1]
    else:
        X_raw, y_raw = frame.iloc[:, :-1], frame.iloc[:, -1]
    X = X_raw.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    y = _map_labels(dataset, y_raw)
    _validate_loaded(dataset, condition, X, y, validate_classes=validate_classes)
    return X, y


def _validate_loaded(dataset: str, condition: str, X: np.ndarray, y: np.ndarray, *, validate_classes: bool) -> None:
    spec = DATASET_SPECS[dataset]
    if X.shape != (spec.samples, spec.features):
        raise ValueError(
            f"{dataset}/{condition} has shape {X.shape}; expected {(spec.samples, spec.features)}"
        )
    if y.shape != (spec.samples,):
        raise ValueError(f"{dataset}/{condition} has {y.shape[0]} labels; expected {spec.samples}")
    if not np.isfinite(X).all():
        raise ValueError(f"{dataset}/{condition} contains NaN or infinite feature values")
    if set(np.unique(y)) != {-1, 1}:
        raise ValueError(f"{dataset}/{condition} labels are not exactly {{-1, +1}}")
    if validate_classes and condition == "clean":
        positive, negative = int((y == 1).sum()), int((y == -1).sum())
        if (positive, negative) != (spec.positive, spec.negative):
            raise ValueError(
                f"{dataset} class counts are {(positive, negative)}; expected {(spec.positive, spec.negative)}"
            )


def file_hash(dataset: str, condition: str = "clean", *, data_root: str | Path | None = None) -> str:
    condition = CONDITION_ALIASES.get(condition.lower(), condition.lower())
    spec = DATASET_SPECS[dataset]
    path = Path(data_root) / spec.files[condition] if data_root else DEFAULT_DATA_ROOT / spec.files[condition]
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_datasets(*, include_archived_variants: bool = True) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    conditions = ("clean", "mixed", "high_margin", "combined") if include_archived_variants else ("clean",)
    for dataset, spec in DATASET_SPECS.items():
        for condition in conditions:
            X, y = load_dataset(dataset, condition, validate_classes=True)
            report.append({
                "dataset": dataset,
                "condition": condition,
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "positive": int((y == 1).sum()),
                "negative": int((y == -1).sum()),
                "sha256": file_hash(dataset, condition),
                "expected_samples": spec.samples,
            })
    return report
