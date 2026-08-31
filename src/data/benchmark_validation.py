"""Measure and validate the original benchmark files without running a solver."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .benchmark_loaders import (
    RawBenchmarkDataset, load_benchmark_dataset, original_inventory,
    verified_source,
)


def _expect(dataset, partition, n, p, X_dtype, y_dtype, counts, *, variant=None, storage="dense"):
    return {
        "dataset": dataset, "partition": partition, "variant": variant,
        "shape": [n, p], "X_dtype": X_dtype, "y_dtype": y_dtype,
        "class_counts": [{"label": label, "count": count} for label, count in counts],
        "storage": storage,
    }


# Expected dimensions/domains are separate from the actual measured report.
# No loader uses these counts to pad, trim, relabel, or manufacture observations.
BENCHMARK_EXPECTATIONS = (
    _expect("basehock", "pool", 1993, 4862, "uint8", "uint8", [(1, 994), (2, 999)]),
    _expect("colon", "pool", 62, 2000, "float64", "float64", [(-1.0, 40), (1.0, 22)], storage="csr"),
    _expect("gina", "pool", 3468, 970, "int64", "object", [("-1", 1763), ("1", 1705)]),
    _expect("hiva", "pool", 4229, 1617, "int64", "object", [("-1", 4080), ("1", 149)]),
    _expect("hill_valley", "train", 606, 100, "float64", "float64", [(0.0, 305), (1.0, 301)], variant="without_noise"),
    _expect("hill_valley", "test", 606, 100, "float64", "float64", [(0.0, 295), (1.0, 311)], variant="without_noise"),
    _expect("madelon", "train", 2000, 500, "float64", "float64", [(-1.0, 1000), (1.0, 1000)]),
    _expect("madelon", "validation", 600, 500, "float64", "float64", [(-1.0, 300), (1.0, 300)]),
)


def source_inventory_fingerprint(manifest: dict) -> str:
    """Hash inventory/provenance only, excluding validation to avoid a self-hash."""
    source = {key: value for key, value in manifest.items() if key != "validation"}
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def describe_benchmark(data: RawBenchmarkDataset) -> dict:
    """Measure loaded values, retaining original label values and their dtype."""
    X, y = data.X, data.y
    is_sparse = sparse.issparse(X)
    values = X
    if is_sparse:
        # Count numerical zeros, not just allocated sparse entries. Work on a
        # copy so even duplicate-entry canonicalization cannot change the input.
        counted = X.tocsr(copy=True)
        counted.sum_duplicates()
        values = counted.data
    size = int(X.shape[0] * X.shape[1])
    nonzero = int(np.count_nonzero(values))
    zero = size - nonzero
    X_info = {
        "shape": list(X.shape), "dtype": str(X.dtype),
        "storage": X.format if is_sparse else "dense",
        "entries": size, "missing_values": int(np.isnan(values).sum()),
        "infinite_values": int(np.isinf(values).sum()),
        "nonzero_values": nonzero, "zero_values": zero,
        "sparsity": zero / size, "density": nonzero / size,
        "stored_sparse_entries": int(X.nnz) if is_sparse else None,
    }
    if y is None:
        y_info = {"available": False, "shape": None, "source_shape": None,
                  "dtype": None, "missing_values": None, "infinite_values": None,
                  "class_counts": None}
    else:
        missing = np.asarray(pd.isna(y))
        if y.dtype.kind in "OU":
            missing |= np.array([isinstance(value, str) and not value.strip() for value in y])
        usable = ~missing
        if y.dtype.kind in "iuf":
            usable &= np.isfinite(y)
        labels, counts = np.unique(y[usable], return_counts=True)
        y_info = {
            "available": True, "shape": list(y.shape), "source_shape": list(data.source_y_shape),
            "dtype": str(y.dtype), "missing_values": int(missing.sum()),
            "infinite_values": int(np.isinf(y).sum()) if y.dtype.kind in "iuf" else 0,
            "class_counts": [{"label": label.item() if isinstance(label, np.generic) else label,
                              "count": int(count)} for label, count in zip(labels, counts)],
        }
    return {
        "dataset": data.dataset, "partition": data.partition, "variant": data.variant,
        "source_format": data.source_format, "dtype_origin": data.dtype_origin,
        "source_files": list(data.source_files), "X": X_info, "y": y_info,
        "supervised_data_available": y is not None, "warnings": list(data.warnings),
    }


def validate_description(record: dict, expected: dict) -> list[str]:
    errors = []
    X, y = record["X"], record["y"]
    for key, observed, target in (
        ("X.shape", X["shape"], expected["shape"]),
        ("X.dtype", X["dtype"], expected["X_dtype"]),
        ("X.storage", X["storage"], expected["storage"]),
        ("y.dtype", y["dtype"], expected["y_dtype"]),
        ("y.available", y["available"], expected["y_dtype"] is not None),
    ):
        if observed != target:
            errors.append(f"{key}: observed {observed!r}, expected {target!r}")
    if y["available"]:
        if y["shape"] != [expected["shape"][0]]:
            errors.append("y.shape does not match the expected row count")
        if y["class_counts"] != expected["class_counts"]:
            errors.append("original class values/counts disagree with expectations")
    for name, array in (("X", X), ("y", y)):
        if array["missing_values"]:
            errors.append(f"{name} contains missing values")
        if array["infinite_values"]:
            errors.append(f"{name} contains infinite values")
    return errors


def audit_benchmark_datasets(*, data_root=None) -> dict:
    """Return a deterministic validation report; this function does not write files."""
    root, inventory = original_inventory(data_root)
    integrity_errors = []
    verified = 0
    for relative in inventory:
        try:
            verified_source(root, inventory, relative)
            verified += 1
        except (ValueError, OSError) as exc:
            integrity_errors.append(str(exc))
    records = []
    for expected in BENCHMARK_EXPECTATIONS:
        identity = {key: expected[key] for key in ("dataset", "partition", "variant")}
        try:
            data = load_benchmark_dataset(**identity, data_root=root)
            record = describe_benchmark(data)
            errors = validate_description(record, expected)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            record = {**identity, "X": None, "y": None}
            errors = [f"load/inspection failed: {exc}"]
        record.update(expected=expected, errors=errors, status="failed" if errors else "passed")
        records.append(record)
    failed = sum(row["status"] == "failed" for row in records)
    return {
        "schema_version": 1, "scope": "original_benchmark_validation",
        "source_inventory_sha256": source_inventory_fingerprint(
            json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        ),
        "transformations": [], "hardness_benchmark_started": False,
        "sparsity_definition": "Fraction of X entries exactly zero, including implicit sparse zeros; "
                               "missing and infinite entries are not counted as zeros.",
        "integrity": {"original_files": len(inventory), "verified_files": verified,
                      "errors": integrity_errors},
        "summary": {"dataset_groups": 6, "partitions": len(records),
                    "passed": len(records) - failed, "failed": failed},
        "status": "failed" if failed or integrity_errors else "passed",
        "partitions": records,
        "notes": [
            "Validation preserves original labels; solver label adaptation is not implemented here.",
            "Text files have no stored array dtype; parsed_float64 identifies the reader's numeric dtype.",
            "BASEHOCK Y is exposed as a 1-D vector; its original (n, 1) shape is recorded.",
            "Only Hill-Valley without_noise and labeled Madelon train/validation are retained.",
            "No train/validation merging or new noise generation is performed by this audit.",
            "Passing this audit does not resolve scientific configuration/author-confirmation gates.",
        ],
    }


def update_dataset_validation(report: dict, *, data_root=None) -> Path:
    """Explicitly refresh only manifest.json's validation block, never input data."""
    root, inventory = original_inventory(data_root)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("scope") != "original_benchmark_validation"
            or report.get("source_inventory_sha256") != source_inventory_fingerprint(manifest)):
        raise ValueError("report does not match the current source inventory")
    # Recheck sources before recording validation; do not install a stale pass.
    for relative in inventory:
        verified_source(root, inventory, relative)
    manifest["validation"] = report
    serialized = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    path.write_text(serialized, encoding="utf-8")
    return path


def write_validation_manifest(report: dict, output: str | Path, *, data_root=None, overwrite=False) -> Path:
    """Save the separate report, never overwrite original inputs or their manifest."""
    root, inventory = original_inventory(data_root)
    path = Path(output).resolve()
    protected = {(root / relative).resolve() for relative in inventory}
    protected.add((root / "manifest.json").resolve())
    aliases_original = path.exists() and any(
        source.exists() and path.samefile(source) for source in protected
    )
    if path in protected or aliases_original or path.suffix != ".json":
        raise ValueError("validation output must be a separate JSON file, not an original input")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists; use --overwrite explicitly: {path}")
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # x mode prevents accidental overwrite when the destination appears mid-run.
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(serialized)
    return path
