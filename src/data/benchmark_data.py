"""Validate benchmark metadata and build solver-ready dataset views.

Raw parsing lives in data_loader.py. This module validates the declared dataset
contract and maps verified native data into unscaled solver-facing structures.
It never fits preprocessing, applies corruption, runs a solver, or rewrites
source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import sparse
import yaml

from src.utils.matrices import matrix_metadata

from .data_loader import (
    BENCHMARK_LOADERS,
    RawBenchmarkDataset,
    load_benchmark_dataset,
    original_inventory,
    verified_source,
)


# Solver-facing registry.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "benchmark_registry.yaml"
SPLIT_PARTITIONS = {"hill_valley": ("train", "test"), "madelon": ("train", "validation")}
PARTITION_POLICIES = {"pool", "merge_labeled", "official_holdout"}
PREPROCESSING_POLICIES = {"standard", "standard_sparse", "max_abs", "none", "passthrough_upstream_normalized"}


class _RegistryLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            if key in result:
                raise ValueError(f"duplicate registry key: {key!r}")
            result[key] = loader.construct_object(value_node, deep=True)
        except TypeError as exc:
            raise ValueError("registry keys must be scalar values") from exc
    return result


_RegistryLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def validate_partition_policy(dataset: str, policy: str) -> None:
    if not isinstance(policy, str) or policy not in PARTITION_POLICIES:
        raise ValueError(f"{dataset}: explicit partition policy must be one of {sorted(PARTITION_POLICIES)}")
    if dataset in SPLIT_PARTITIONS and policy == "pool":
        raise ValueError(f"{dataset}: choose merge_labeled or official_holdout explicitly")
    if dataset not in SPLIT_PARTITIONS and policy == "official_holdout":
        raise ValueError(f"{dataset}: official_holdout unavailable; no official split indices supplied")


def read_benchmark_registry(path=DEFAULT_REGISTRY_PATH) -> tuple[dict, str]:
    """Return a validated registry and the hash of exactly the bytes parsed."""
    content = Path(path).read_bytes()
    try:
        registry = yaml.load(content, Loader=_RegistryLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid benchmark registry YAML: {exc}") from exc
    if not isinstance(registry, dict) or set(registry) != set(BENCHMARK_LOADERS):
        raise ValueError("benchmark registry must define exactly the six retained benchmarks")
    fields = {"loader", "source_partition_policy", "label_mapping", "storage", "preprocessing", "expected_features"}
    for name, entry in registry.items():
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"{name}: registry entry requires exactly {sorted(fields)}")
        if entry["loader"] != name:
            raise ValueError(f"{name}: loader must name the same retained benchmark")
        validate_partition_policy(name, entry["source_partition_policy"])
        if entry["storage"] not in ("dense", "csr"):
            raise ValueError(f"{name}: storage must be dense or csr")
        if entry["preprocessing"] not in tuple(PREPROCESSING_POLICIES):
            raise ValueError(f"{name}: unknown preprocessing policy")
        if entry["storage"] == "csr" and entry["preprocessing"] == "standard":
            raise ValueError(f"{name}: centered standard scaling is not sparse-safe")
        if name == "colon" and entry["preprocessing"] != "passthrough_upstream_normalized":
            raise ValueError("colon: preserve the upstream-normalized preprocessing declaration")
        if type(entry["expected_features"]) is not int or entry["expected_features"] <= 0:
            raise ValueError(f"{name}: expected_features must be a positive integer")
        mapping = entry["label_mapping"]
        if (not isinstance(mapping, dict) or len(mapping) != 2
                or any(type(value) is not int or value not in (-1, 1) for value in mapping.values())
                or set(mapping.values()) != {-1, 1}):
            raise ValueError(f"{name}: label_mapping must explicitly map two native labels to -1/+1")
        key_type = str if name in {"gina", "hiva"} else int
        if any(type(key) is not key_type for key in mapping):
            raise ValueError(f"{name}: native mapping keys must be {key_type.__name__}")
    return registry, sha256(content).hexdigest()


# Original-file validation.
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


def _source_inventory_fingerprint(manifest: dict) -> str:
    """Hash inventory/provenance only, excluding validation to avoid a self-hash."""
    source = {key: value for key, value in manifest.items() if key != "validation"}
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _describe_benchmark(data: RawBenchmarkDataset) -> dict:
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
        if y.dtype.kind in "OU":
            missing = np.array([
                value is None
                or (isinstance(value, str) and not value.strip())
                or (isinstance(value, (float, np.floating)) and np.isnan(value))
                for value in y
            ])
        else:
            missing = np.isnan(y)
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


def _validate_description(record: dict, expected: dict) -> list[str]:
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
            record = _describe_benchmark(data)
            errors = _validate_description(record, expected)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            record = {**identity, "X": None, "y": None}
            errors = [f"load/inspection failed: {exc}"]
        record.update(expected=expected, errors=errors, status="failed" if errors else "passed")
        records.append(record)
    failed = sum(row["status"] == "failed" for row in records)
    return {
        "schema_version": 1, "scope": "original_benchmark_validation",
        "source_inventory_sha256": _source_inventory_fingerprint(
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


# Solver-ready views.
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
    fingerprint = _source_inventory_fingerprint(manifest)
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
        observed = _describe_benchmark(raw)
        errors = _validate_description(observed, expected)
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
        **matrix_metadata(X), "densified": False,
        "holdout": None if holdout is None else {**matrix_metadata(holdout.X), "densified": False},
    }
    return SolverReadyBenchmark(
        dataset=dataset, X=X, y=y, sample_ids=sample_ids,
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
