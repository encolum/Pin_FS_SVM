"""Curated public benchmarks, separate from the original manuscript datasets."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np


DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "dataset"


def _manifest(data_root: str | Path | None) -> tuple[Path, dict]:
    root = Path(data_root) if data_root is not None else DEFAULT_BENCHMARK_ROOT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported benchmark manifest version")
    keys = [(row["dataset"], row["partition"]) for row in manifest["datasets"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate dataset/partition in benchmark manifest")
    return root, manifest


def _verified_path(root: Path, entry: dict) -> Path:
    relative = Path(entry["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("benchmark path escapes data root")
    with path.open("rb") as stream:
        digest = sha256(stream.read()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {entry['path']}")
    return path


def load_benchmark_dataset(
    dataset: str,
    *,
    partition: str,
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an explicit pool/train/test/validation partition with integrity checks.

    Returns float64 features and int64 {-1, +1} labels. Does not scale, split,
    merge partitions, sample rows, select features, or fit a model. Requiring the
    partition prevents accidental use of official test data for training.
    """
    root, manifest = _manifest(data_root)
    entry = next((row for row in manifest["datasets"]
                  if row["dataset"] == dataset and row["partition"] == partition), None)
    if entry is None:
        raise KeyError(f"unknown benchmark partition: {dataset}/{partition}")
    path = _verified_path(root, entry)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"X", "y"}:
            raise ValueError(f"unexpected NPZ members in {path}")
        X, y = archive["X"], archive["y"]
    if X.dtype.kind not in "iuf" or y.dtype.kind not in "iu":
        raise ValueError("benchmark arrays must be numeric with integer labels")
    if str(X.dtype) != entry["X_dtype"] or str(y.dtype) != entry["y_dtype"]:
        raise ValueError(f"dtype mismatch for {dataset}/{partition}")
    if X.shape != (entry["samples"], entry["features"]) or y.shape != (entry["samples"],):
        raise ValueError(f"shape mismatch for {dataset}/{partition}")
    if not np.isfinite(X).all() or set(np.unique(y)) != {-1, 1}:
        raise ValueError(f"invalid values or labels for {dataset}/{partition}")
    if int((y == 1).sum()) != entry["positive"] or int((y == -1).sum()) != entry["negative"]:
        raise ValueError(f"class counts mismatch for {dataset}/{partition}")
    return X.astype(np.float64), y.astype(np.int64)


def audit_benchmark_datasets(*, data_root: str | Path | None = None) -> list[dict]:
    """Validate every committed input and metadata file, without training."""
    root, manifest = _manifest(data_root)
    report = []
    for entry in manifest["datasets"]:
        load_benchmark_dataset(entry["dataset"], partition=entry["partition"], data_root=root)
        report.append({key: entry[key] for key in (
            "dataset", "partition", "samples", "features", "positive", "negative", "sha256"
        )})
    for entry in manifest["metadata"]:
        _verified_path(root, entry)
    return report
