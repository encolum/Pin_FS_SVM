"""Read-only solver-facing views; no scaling, noise, model fitting, or disk writes.

"Solver-ready" here means binary labels and explicit partition/storage contracts,
not experiment readiness. Train-only preprocessing and solver integration belong
to subsequent milestones. Original loaders retain their native behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal

import numpy as np
from scipy import sparse

from .benchmark_loaders import load_benchmark_dataset, original_inventory
from .benchmark_registry import (
    DEFAULT_REGISTRY_PATH, SPLIT_PARTITIONS, read_benchmark_registry,
    validate_partition_policy,
)
from .benchmark_validation import (
    describe_benchmark, source_inventory_fingerprint, validate_description,
)


@dataclass(frozen=True)
class SolverReadyPartition:
    """An official holdout kept separate from the training matrix."""

    X: np.ndarray | sparse.csr_matrix
    y: np.ndarray
    sample_ids: np.ndarray
    source_partition: str


@dataclass(frozen=True)
class SolverReadyBenchmark:
    dataset: str
    X: np.ndarray | sparse.csr_matrix
    y: np.ndarray
    sample_ids: np.ndarray
    feature_names: tuple[str, ...] | None
    source_partitions: tuple[str, ...]
    source_files: tuple[dict, ...]
    label_mapping: dict[str, int]
    storage: Literal["dense", "csr"]
    partition_policy: str
    preprocessing_policy: str
    warnings: tuple[str, ...]
    metadata: dict
    holdout: SolverReadyPartition | None = None


def _validated_manifest(data_root):
    root, _ = original_inventory(data_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validation = manifest.get("validation", {})
    fingerprint = source_inventory_fingerprint(manifest)
    if (validation.get("scope") != "original_benchmark_validation"
            or validation.get("status") != "passed"
            or validation.get("source_inventory_sha256") != fingerprint):
        raise ValueError("manifest needs a passing raw validation matching its current source inventory")
    return root, validation, fingerprint


def _verify_partition(raw, validation, entry):
    matches = [row for row in validation.get("partitions", [])
               if (row.get("dataset"), row.get("partition"), row.get("variant"))
               == (raw.dataset, raw.partition, raw.variant)]
    if len(matches) != 1:
        raise ValueError(f"{raw.dataset}/{raw.partition}: expected one manifest validation record")
    saved = matches[0]
    if saved.get("status") != "passed" or saved.get("errors"):
        raise ValueError(f"{raw.dataset}/{raw.partition}: manifest partition did not pass validation")
    if saved.get("source_files") != list(raw.source_files):
        raise ValueError(f"{raw.dataset}/{raw.partition}: manifest validation has stale source hashes")
    try:
        expected = {
            "shape": saved["X"]["shape"], "X_dtype": saved["X"]["dtype"],
            "storage": saved["X"]["storage"], "y_dtype": saved["y"]["dtype"],
            "class_counts": saved["y"]["class_counts"],
        }
        observed = describe_benchmark(raw)
        errors = validate_description(observed, expected)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{raw.dataset}/{raw.partition}: invalid manifest validation record") from exc
    if raw.X.shape[1] != entry["expected_features"]:
        errors.append("feature count differs from registry expected_features")
    if errors:
        raise ValueError(f"{raw.dataset}/{raw.partition}: " + "; ".join(errors))
    return observed


def _map_labels(y, mapping):
    """Map exact native values into a new int64 array; never infer label order."""
    if y is None or y.ndim != 1:
        raise ValueError("solver-ready labels must be a one-dimensional labeled vector")
    output = np.empty(len(y), dtype=np.int64)
    for index, value in enumerate(y):
        native = value.item() if isinstance(value, np.generic) else value
        if (isinstance(native, bool) or not isinstance(native, (int, float, str))
                or (isinstance(native, (int, float)) and not np.isfinite(native))):
            raise ValueError(f"invalid native label at row {index}: {native!r}")
        if native not in mapping:
            raise ValueError(f"unknown native label at row {index}: {native!r}")
        output[index] = mapping[native]
    if set(output.tolist()) != {-1, 1}:
        raise ValueError("solver-ready partition must contain both -1 and +1 classes")
    return output


def _matrix_view(X, storage):
    if storage == "csr":
        return sparse.csr_matrix(X, copy=True)
    if sparse.issparse(X):
        raise ValueError("implicit densification is forbidden; use a CSR registry entry")
    return X.copy()


def _matrix_metadata(X):
    is_sparse = sparse.issparse(X)
    nnz = int(X.count_nonzero()) if is_sparse else int(np.count_nonzero(X))
    return {
        "shape": list(X.shape), "dtype": str(X.dtype),
        "storage": "csr" if is_sparse else "dense", "nnz": nnz,
        "density": nnz / int(X.shape[0] * X.shape[1]),
        "matrix_bytes": int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
        if is_sparse else int(X.nbytes),
        "estimated_dense_bytes": int(X.shape[0] * X.shape[1] * np.dtype(np.float64).itemsize),
        "densified": False,
    }


def _load_solver_ready(dataset, *, data_root, partition_policy, registry, registry_hash):
    if dataset not in registry:
        raise ValueError(f"unknown retained benchmark: {dataset}")
    validate_partition_policy(dataset, partition_policy)
    entry = registry[dataset]
    root, validation, fingerprint = _validated_manifest(data_root)
    partitions = SPLIT_PARTITIONS.get(dataset, ("pool",))
    raw_parts = [load_benchmark_dataset(dataset, partition=part, data_root=root) for part in partitions]
    observed = [_verify_partition(raw, validation, entry) for raw in raw_parts]
    parts = [SolverReadyPartition(
        X=_matrix_view(raw.X, entry["storage"]),
        y=_map_labels(raw.y, entry["label_mapping"]),
        sample_ids=np.array([f"{dataset}:{raw.partition}:{row}" for row in range(raw.X.shape[0])]),
        source_partition=raw.partition,
    ) for raw in raw_parts]
    holdout = parts[1] if partition_policy == "official_holdout" else None
    if len(parts) > 1 and partition_policy == "merge_labeled":
        X = (sparse.vstack([part.X for part in parts], format="csr") if entry["storage"] == "csr"
             else np.concatenate([part.X for part in parts], axis=0))
        y = np.concatenate([part.y for part in parts])
        sample_ids = np.concatenate([part.sample_ids for part in parts])
    else:
        X, y, sample_ids = parts[0].X, parts[0].y, parts[0].sample_ids
    warnings = ["Preprocessing policy is declared only; no scaler has been fitted or applied."]
    if dataset == "colon":
        warnings.append("Colon is normalized upstream; not raw input for a train-only-normalization claim.")
    if dataset in {"gina", "hiva"}:
        warnings.append("Original OpenML export is pooled; original split indices are unavailable.")
    if len(parts) > 1 and partition_policy == "merge_labeled":
        warnings.append("Official labeled partitions were explicitly merged in memory; use new outer splits for evaluation.")
    conversions = []
    for raw in raw_parts:
        if entry["storage"] == "csr" and not sparse.issparse(raw.X):
            conversions.append({"partition": raw.partition, "operation": "dense_to_csr"})
    if conversions:
        warnings.append("Native dense features explicitly converted to CSR in memory; source bytes/values unchanged.")
    source_files = tuple(source for raw in raw_parts for source in raw.source_files)
    metadata = {
        "source_inventory_sha256": fingerprint, "registry_sha256": registry_hash,
        "registry_partition_policy": entry["source_partition_policy"],
        "partition_policy_overridden": partition_policy != entry["source_partition_policy"],
        "preprocessing_applied": False, "feature_values_changed": False,
        "raw_files_modified": False, "storage_conversions": conversions,
        "sample_id_scheme": "dataset:source_partition:zero_based_row_index",
        "matrix_role": "train" if holdout is not None else "pool",
        "source_partitions": [{"partition": raw.partition, "samples": raw.X.shape[0],
                               "X_dtype": str(raw.X.dtype), "y_dtype": str(raw.y.dtype),
                               "storage": row["X"]["storage"], "warnings": list(raw.warnings)}
                              for raw, row in zip(raw_parts, observed)],
        "input_sparse_bytes": sum(int(raw.X.data.nbytes + raw.X.indices.nbytes + raw.X.indptr.nbytes)
                                  for raw in raw_parts if sparse.isspmatrix_csr(raw.X)),
        **_matrix_metadata(X),
        "holdout": None if holdout is None else _matrix_metadata(holdout.X),
    }
    return SolverReadyBenchmark(
        dataset=dataset, X=X, y=y, sample_ids=sample_ids, feature_names=None,
        source_partitions=partitions, source_files=source_files,
        label_mapping={str(key): value for key, value in entry["label_mapping"].items()},
        storage=entry["storage"], partition_policy=partition_policy,
        preprocessing_policy=entry["preprocessing"], warnings=tuple(warnings),
        metadata=metadata, holdout=holdout,
    )


def load_solver_ready_benchmark(dataset: str, *, data_root=None, partition_policy: str,
                                registry_path=DEFAULT_REGISTRY_PATH) -> SolverReadyBenchmark:
    """Verify originals and expose an explicitly chosen, unscaled benchmark view.

    For official_holdout, X/y/sample_ids contain training rows only; holdout has
    the official test (Hill-Valley) or validation (Madelon) rows. No holdout exists
    for already-pooled sources. merge_labeled preserves source order then row order.
    """
    registry, registry_hash = read_benchmark_registry(registry_path)
    return _load_solver_ready(dataset, data_root=data_root, partition_policy=partition_policy,
                              registry=registry, registry_hash=registry_hash)


def _partition_summary(part):
    return {"samples": int(part.X.shape[0]), "features": int(part.X.shape[1]),
            "positive": int(np.count_nonzero(part.y == 1)),
            "negative": int(np.count_nonzero(part.y == -1)),
            "X_dtype": str(part.X.dtype), "y_dtype": str(part.y.dtype),
            "missing_values": 0, "infinite_values": 0}


def audit_solver_ready_benchmarks(*, data_root=None, registry_path=DEFAULT_REGISTRY_PATH) -> list[dict]:
    """Validate all six declared views without a solver or experiment dependency."""
    registry, registry_hash = read_benchmark_registry(registry_path)
    records = []
    for dataset, entry in registry.items():
        identity = {"dataset": dataset, "partition_policy": entry["source_partition_policy"],
                    "preprocessing_policy": entry["preprocessing"],
                    "label_mapping": {str(key): value for key, value in entry["label_mapping"].items()},
                    "registry_sha256": registry_hash}
        try:
            data = _load_solver_ready(dataset, data_root=data_root,
                                      partition_policy=entry["source_partition_policy"],
                                      registry=registry, registry_hash=registry_hash)
            records.append({
                **identity, **_partition_summary(data), "status": "passed", "errors": [],
                "storage": data.storage, "density": data.metadata["density"],
                "source_partitions": list(data.source_partitions),
                "source_hashes": {source["path"]: source["sha256"] for source in data.source_files},
                "warnings": list(data.warnings), "metadata": data.metadata,
                "holdout": None if data.holdout is None else {
                    "source_partition": data.holdout.source_partition, **_partition_summary(data.holdout)},
            })
        except (ValueError, OSError, KeyError, TypeError) as exc:
            records.append({**identity, "status": "failed", "errors": [str(exc)]})
    return records
