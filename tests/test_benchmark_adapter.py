from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy import sparse
import yaml

from src.data import benchmark_adapter as adapter
from src.data.benchmark_loaders import DEFAULT_BENCHMARK_ROOT, load_benchmark_dataset, sha256_file
from src.data.benchmark_registry import DEFAULT_REGISTRY_PATH, read_benchmark_registry


EXPECTED = {
    "hill_valley": (1212, 100, 600, 612, "dense"),
    "madelon": (2600, 500, 1300, 1300, "dense"),
    "gina": (3468, 970, 1763, 1705, "dense"),
    "hiva": (4229, 1617, 4080, 149, "dense"),
    "colon": (62, 2000, 40, 22, "csr"),
    "basehock": (1993, 4862, 994, 999, "csr"),
}


@pytest.fixture(scope="module")
def views():
    paths = [path for path in DEFAULT_BENCHMARK_ROOT.rglob("*") if path.is_file()]
    before = {path: sha256_file(path) for path in paths}
    registry, _ = read_benchmark_registry()
    loaded = {name: adapter.load_solver_ready_benchmark(
        name, partition_policy=entry["source_partition_policy"])
        for name, entry in registry.items()}
    yield loaded
    assert {path: sha256_file(path) for path in paths} == before


@pytest.mark.parametrize("name", EXPECTED)
def test_actual_solver_views(name, views):
    data = views[name]
    n, p, negative, positive, storage = EXPECTED[name]
    assert data.X.shape == (n, p)
    assert data.y.shape == data.sample_ids.shape == (n,)
    assert data.y.dtype == np.int64
    assert np.count_nonzero(data.y == -1) == negative
    assert np.count_nonzero(data.y == 1) == positive
    assert set(data.y) == {-1, 1}
    assert data.storage == storage
    assert sparse.isspmatrix_csr(data.X) == (storage == "csr")
    assert len(set(data.sample_ids)) == n
    assert data.holdout is None
    assert data.metadata["preprocessing_applied"] is False
    assert data.metadata["feature_values_changed"] is False
    assert data.metadata["densified"] is False
    assert data.metadata["estimated_dense_bytes"] == n * p * 8
    assert len(data.metadata["source_inventory_sha256"]) == 64
    assert len(data.metadata["registry_sha256"]) == 64
    assert all(sha256_file(DEFAULT_BENCHMARK_ROOT / row["path"]) == row["sha256"]
               for row in data.source_files)


@pytest.mark.parametrize("name", EXPECTED)
def test_native_features_unchanged_and_exact_mapping(name, views):
    data = views[name]
    registry, _ = read_benchmark_registry()
    offset = 0
    for partition in data.source_partitions:
        raw = load_benchmark_dataset(name, partition=partition)
        stop = offset + raw.X.shape[0]
        if sparse.issparse(data.X):
            assert (data.X[offset:stop] != sparse.csr_matrix(raw.X)).nnz == 0
        else:
            np.testing.assert_array_equal(data.X[offset:stop], raw.X)
        assert data.X.dtype == raw.X.dtype
        np.testing.assert_array_equal(data.y[offset:stop],
                                      [registry[name]["label_mapping"][value] for value in raw.y])
        assert data.sample_ids[offset] == f"{name}:{partition}:0"
        assert data.sample_ids[stop - 1] == f"{name}:{partition}:{raw.X.shape[0] - 1}"
        offset = stop


@pytest.mark.parametrize("name,holdout_name,n_train,n_holdout", [
    ("hill_valley", "test", 606, 606), ("madelon", "validation", 2000, 600),
])
def test_official_holdout_never_joins_training(name, holdout_name, n_train, n_holdout, views):
    data = adapter.load_solver_ready_benchmark(name, partition_policy="official_holdout")
    repeated = adapter.load_solver_ready_benchmark(name, partition_policy="official_holdout")
    assert data.X.shape[0] == n_train
    assert data.holdout.X.shape[0] == n_holdout
    assert data.holdout.source_partition == holdout_name
    assert set(data.sample_ids).isdisjoint(data.holdout.sample_ids)
    np.testing.assert_array_equal(data.X, views[name].X[:n_train])
    np.testing.assert_array_equal(data.holdout.X, views[name].X[n_train:])
    np.testing.assert_array_equal(data.y, views[name].y[:n_train])
    np.testing.assert_array_equal(data.holdout.y, views[name].y[n_train:])
    np.testing.assert_array_equal(data.sample_ids, repeated.sample_ids)
    np.testing.assert_array_equal(data.holdout.sample_ids, repeated.holdout.sample_ids)
    assert data.metadata["matrix_role"] == "train"
    assert data.metadata["partition_policy_overridden"] is True
    assert data.source_partitions == ("train", holdout_name)


def test_sparse_views_never_densify(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected densification")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    for name in ("basehock", "colon"):
        data = adapter.load_solver_ready_benchmark(name, partition_policy="pool")
        assert sparse.isspmatrix_csr(data.X)
    assert data.metadata["input_sparse_bytes"] > 0


def test_basehock_storage_and_colon_provenance_are_explicit(views):
    assert views["basehock"].metadata["storage_conversions"] == [
        {"partition": "pool", "operation": "dense_to_csr"}]
    assert views["basehock"].metadata["source_partitions"][0]["storage"] == "dense"
    assert views["colon"].preprocessing_policy == "passthrough_upstream_normalized"
    assert any("upstream" in warning for warning in views["colon"].warnings)


@pytest.mark.parametrize("name,policy", [
    ("hill_valley", "pool"), ("madelon", "pool"), ("gina", "official_holdout"),
    ("hiva", "official_holdout"), ("colon", "official_holdout"), ("basehock", "official_holdout"),
    ("madelon", None), ("madelon", "auto"), ("unknown", "pool"),
])
def test_invalid_partition_requests_fail_closed(name, policy):
    with pytest.raises(ValueError):
        adapter.load_solver_ready_benchmark(name, partition_policy=policy)


def test_partition_policy_cannot_be_omitted():
    with pytest.raises(TypeError, match="partition_policy"):
        adapter.load_solver_ready_benchmark("hill_valley")


def test_pooled_merge_policy_does_not_invent_rows(views):
    data = adapter.load_solver_ready_benchmark("colon", partition_policy="merge_labeled")
    assert data.source_partitions == ("pool",)
    assert data.holdout is None
    np.testing.assert_array_equal(data.sample_ids, views["colon"].sample_ids)
    assert (data.X != views["colon"].X).nnz == 0


@pytest.mark.parametrize("labels", [
    [0, 2], [0, np.nan], [0, np.inf], [False, True], ["0", "1"], [None, 1], [0, 0],
])
def test_unknown_missing_or_degenerate_labels_rejected(labels):
    with pytest.raises(ValueError):
        adapter._map_labels(np.array(labels), {0: -1, 1: 1})


def test_label_mapping_does_not_modify_native_object_array():
    labels = np.array(["1", "-1", "1"], dtype=object)
    original = labels.copy()
    mapped = adapter._map_labels(labels, {"-1": -1, "1": 1})
    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(mapped, [1, -1, 1])
    assert labels.dtype == object and mapped.dtype == np.int64


def _write_registry(tmp_path, name, field, value):
    registry, _ = read_benchmark_registry()
    registry[name][field] = value
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry))
    return path


@pytest.mark.parametrize("field,value", [
    ("label_mapping", {0: -1, 1: -1}), ("label_mapping", {"0": -1, "1": 1}),
    ("label_mapping", {False: -1, True: 1}), ("label_mapping", {0: False, 1: True}),
    ("loader", "colon"), ("storage", "coo"), ("preprocessing", "auto"),
    ("source_partition_policy", "pool"), ("expected_features", True),
    ("expected_features", 0), ("extra_field", 123),
])
def test_invalid_registry_entries_rejected(tmp_path, field, value):
    path = _write_registry(tmp_path, "hill_valley", field, value)
    with pytest.raises(ValueError):
        read_benchmark_registry(path)


def test_duplicate_registry_key_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(DEFAULT_REGISTRY_PATH.read_text() + "\nhill_valley: {}\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_benchmark_registry(path)


def test_registry_feature_expectation_checked_against_actual_data(tmp_path):
    path = _write_registry(tmp_path, "hill_valley", "expected_features", 101)
    with pytest.raises(ValueError, match="feature count"):
        adapter.load_solver_ready_benchmark("hill_valley", partition_policy="merge_labeled", registry_path=path)


def test_sparse_input_cannot_be_densified_via_registry(tmp_path):
    path = _write_registry(tmp_path, "colon", "storage", "dense")
    with pytest.raises(ValueError, match="densification"):
        adapter.load_solver_ready_benchmark("colon", partition_policy="pool", registry_path=path)


def test_sparse_centering_and_colon_preprocessing_overrides_rejected(tmp_path):
    for name in ("basehock", "colon"):
        path = _write_registry(tmp_path, name, "preprocessing", "standard")
        with pytest.raises(ValueError, match="sparse-safe"):
            read_benchmark_registry(path)


@pytest.fixture
def hill_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    shutil.copytree(DEFAULT_BENCHMARK_ROOT / "hill_valley", root / "hill_valley")
    shutil.copyfile(DEFAULT_BENCHMARK_ROOT / "manifest.json", root / "manifest.json")
    return root


@pytest.mark.parametrize("change", ["shape", "source_hash", "fingerprint", "duplicate", "missing", "failed"])
def test_manifest_validation_is_authoritative(hill_root, change):
    path = hill_root / "manifest.json"
    manifest = json.loads(path.read_text())
    validation = manifest["validation"]
    row = next(row for row in validation["partitions"]
               if row["dataset"] == "hill_valley" and row["partition"] == "train")
    if change == "shape":
        row["X"]["shape"][0] += 1
    elif change == "source_hash":
        row["source_files"][0]["sha256"] = "0" * 64
    elif change == "fingerprint":
        validation["source_inventory_sha256"] = "0" * 64
    elif change == "duplicate":
        validation["partitions"].append(deepcopy(row))
    elif change == "missing":
        validation["partitions"].remove(row)
    else:
        row["status"] = "failed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        adapter.load_solver_ready_benchmark("hill_valley", data_root=hill_root, partition_policy="merge_labeled")


def test_corrupt_original_is_not_repaired_or_loaded(hill_root):
    path = hill_root / "hill_valley" / "train.data"
    path.write_bytes(path.read_bytes() + b"\n")
    digest = sha256_file(path)
    with pytest.raises(ValueError, match="checksum"):
        adapter.load_solver_ready_benchmark("hill_valley", data_root=hill_root, partition_policy="merge_labeled")
    assert sha256_file(path) == digest


def test_nonfinite_features_rejected_before_adapter(monkeypatch):
    load = adapter.load_benchmark_dataset
    def corrupted(*args, **kwargs):
        raw = load(*args, **kwargs)
        X = raw.X.copy()
        X[0, 0] = np.inf
        return replace(raw, X=X)
    monkeypatch.setattr(adapter, "load_benchmark_dataset", corrupted)
    with pytest.raises(ValueError, match="infinite"):
        adapter.load_solver_ready_benchmark("hill_valley", partition_policy="merge_labeled")


def test_default_registry_is_independent_of_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = adapter.load_solver_ready_benchmark("hill_valley", partition_policy="merge_labeled")
    assert data.X.shape == (1212, 100)


def test_audit_reports_all_six_without_solver():
    records = adapter.audit_solver_ready_benchmarks()
    assert {row["dataset"] for row in records} == set(EXPECTED)
    for row in records:
        assert row["status"] == "passed" and row["errors"] == []
        n, p, negative, positive, storage = EXPECTED[row["dataset"]]
        assert (row["samples"], row["features"], row["negative"], row["positive"], row["storage"]) == (
            n, p, negative, positive, storage)
        assert row["missing_values"] == row["infinite_values"] == 0
        assert row["source_hashes"]
    json.dumps(records, allow_nan=False)


def test_audit_records_per_dataset_failure_without_hiding_other_results(hill_root):
    records = adapter.audit_solver_ready_benchmarks(data_root=hill_root)
    assert len(records) == 6
    assert next(row for row in records if row["dataset"] == "hill_valley")["status"] == "passed"
    assert sum(row["status"] == "failed" for row in records) == 5
