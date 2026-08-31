from dataclasses import replace
from io import BytesIO
import json
import os
import pickle
import zipfile

import numpy as np
import pytest
from scipy import sparse
from scipy.io import loadmat
from sklearn.datasets import load_svmlight_file

from src.data import (
    BENCHMARK_LOADERS, RawBenchmarkDataset, load_benchmark_dataset,
    load_basehock, load_colon, load_gina, load_hiva, load_hill_valley, load_madelon,
)
from src.data._npz_labels import read_original_npz_labels
from src.data.benchmark_loaders import DEFAULT_BENCHMARK_ROOT, sha256_file
from src.data.benchmark_validation import (
    BENCHMARK_EXPECTATIONS, audit_benchmark_datasets, describe_benchmark,
    source_inventory_fingerprint, update_dataset_validation,
    validate_description, write_validation_manifest,
)


@pytest.fixture(scope="module")
def audit():
    original = json.loads((DEFAULT_BENCHMARK_ROOT / "manifest.json").read_text())
    before = {row["path"]: sha256_file(DEFAULT_BENCHMARK_ROOT / row["path"])
              for row in original["files"]}
    report = audit_benchmark_datasets()
    after = {path: sha256_file(DEFAULT_BENCHMARK_ROOT / path) for path in before}
    assert before == after
    assert before == {row["path"]: row["sha256"] for row in original["files"]}
    return report


def test_six_loaders_separate_from_manuscript():
    from src.data.loaders import DATASET_SPECS

    assert set(BENCHMARK_LOADERS) == {"basehock", "colon", "gina", "hiva", "hill_valley", "madelon"}
    assert set(DATASET_SPECS) == {"diabetes", "cleveland", "wdbc", "ionosphere", "sonar", "colon"}


@pytest.mark.parametrize("index", range(8))
def test_actual_partition_validation(index, audit):
    row = audit["partitions"][index]
    expected = BENCHMARK_EXPECTATIONS[index]
    assert row["status"] == "passed"
    assert row["errors"] == []
    assert row["X"]["shape"] == expected["shape"]
    assert row["X"]["dtype"] == expected["X_dtype"]
    assert row["y"]["dtype"] == expected["y_dtype"]
    assert row["X"]["missing_values"] == row["X"]["infinite_values"] == 0
    assert row["X"]["zero_values"] + row["X"]["nonzero_values"] == row["X"]["entries"]
    assert 0 <= row["X"]["sparsity"] <= 1
    assert row["y"]["available"] and row["supervised_data_available"]
    assert row["y"]["class_counts"] == expected["class_counts"]
    assert row["y"]["missing_values"] == row["y"]["infinite_values"] == 0


def test_saved_manifest_matches_fresh_measurements(audit):
    saved = json.loads((DEFAULT_BENCHMARK_ROOT / "manifest.json").read_text())
    assert saved["validation"] == audit
    assert source_inventory_fingerprint(saved) == audit["source_inventory_sha256"]
    assert audit["status"] == "passed"
    assert audit["integrity"] == {"original_files": 10, "verified_files": 10, "errors": []}
    assert audit["transformations"] == []
    assert audit["hardness_benchmark_started"] is False


def test_basehock_keeps_native_labels_dtype_and_values():
    actual = load_basehock()
    raw = loadmat(DEFAULT_BENCHMARK_ROOT / "BASEHOCK.mat")
    np.testing.assert_array_equal(actual.X, raw["X"])
    np.testing.assert_array_equal(actual.y, raw["Y"].reshape(-1))
    assert actual.source_y_shape == (1993, 1)
    assert actual.X.dtype == actual.y.dtype == np.uint8
    assert set(actual.y) == {1, 2}


def test_colon_retains_libsvm_values_and_sparse_storage():
    data = load_colon()
    X, y = load_svmlight_file(str(DEFAULT_BENCHMARK_ROOT / "colon-cancer.bz2"), zero_based=False)
    assert sparse.isspmatrix_csr(data.X)
    assert (data.X != X).nnz == 0
    np.testing.assert_array_equal(data.y, y)


@pytest.mark.parametrize("loader,name", [(load_gina, "gina"), (load_hiva, "hiva")])
def test_npz_loaders_keep_features_and_object_string_labels(loader, name, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pickle execution is forbidden")

    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)
    data = loader()
    with np.load(DEFAULT_BENCHMARK_ROOT / f"{name}.npz", allow_pickle=False) as raw:
        np.testing.assert_array_equal(data.X, raw["X"])
    assert data.X.dtype == np.int64
    assert data.y.dtype == object
    assert set(data.y) == {"-1", "1"}


@pytest.mark.parametrize("partition", ["train", "test"])
def test_hill_valley_keeps_clean_original_rows_columns_and_labels(partition):
    data = load_hill_valley(partition=partition)
    raw = np.loadtxt(DEFAULT_BENCHMARK_ROOT / "hill_valley" / f"{partition}.data",
                     delimiter=",", skiprows=1)
    np.testing.assert_array_equal(data.X, raw[:, :-1])
    np.testing.assert_array_equal(data.y, raw[:, -1])
    assert set(data.y) == {0, 1}
    assert data.variant == "without_noise"


@pytest.mark.parametrize("partition,suffix", [("train", "train"), ("validation", "valid")])
def test_madelon_keeps_official_labeled_partitions_unmerged(partition, suffix):
    data = load_madelon(partition=partition)
    raw = np.loadtxt(DEFAULT_BENCHMARK_ROOT / "madelon" / f"{suffix}.data")
    np.testing.assert_array_equal(data.X, raw)
    np.testing.assert_array_equal(data.y, np.loadtxt(DEFAULT_BENCHMARK_ROOT / "madelon" / f"{suffix}.labels"))


@pytest.mark.parametrize("dataset,kwargs", [
    ("unknown", {}), ("madelon", {}), ("hill_valley", {}),
    ("gina", {"partition": "test"}), ("hiva", {"variant": "with_noise"}),
    ("hill_valley", {"variant": "clean", "partition": "train"}),
    ("hill_valley", {"variant": "with_noise", "partition": "train"}),
    ("madelon", {"partition": "test"}),
    ("madelon", {"partition": "pool"}),
])
def test_dispatch_never_infers_or_combines_splits(dataset, kwargs):
    with pytest.raises((KeyError, ValueError)):
        load_benchmark_dataset(dataset, **kwargs)


def _raw(X, y):
    return RawBenchmarkDataset("fixture", "pool", None, X, y, (), y.shape if y is not None else None,
                               "test", "stored")


def test_missing_and_infinite_values_are_measured_and_rejected():
    X = np.array([[0, np.nan, np.inf], [0, 2, 0]])
    row = describe_benchmark(_raw(X, np.array([1.0, np.nan])))
    assert row["X"]["missing_values"] == 1
    assert row["X"]["infinite_values"] == 1
    assert row["X"]["zero_values"] == 3
    assert row["X"]["sparsity"] == 0.5
    assert row["y"]["missing_values"] == 1
    expected = {"shape": [2, 3], "X_dtype": "float64", "y_dtype": "float64", "storage": "dense",
                "class_counts": [{"label": 1.0, "count": 2}]}
    errors = validate_description(row, expected)
    assert "X contains missing values" in errors
    assert "X contains infinite values" in errors
    assert "y contains missing values" in errors
    assert "original class values/counts disagree with expectations" in errors


def test_sparse_sparsity_uses_numerical_not_stored_entries_without_mutation():
    X = sparse.csr_matrix((np.array([0.0, 2.0, 3.0]), np.array([0, 1, 1]), np.array([0, 3, 3])),
                          shape=(2, 3))
    before_data, before_indices = X.data.copy(), X.indices.copy()
    row = describe_benchmark(_raw(X, np.array([-1, 1])))
    assert row["X"]["stored_sparse_entries"] == 3
    assert row["X"]["nonzero_values"] == 1
    assert row["X"]["zero_values"] == 5
    assert row["X"]["sparsity"] == pytest.approx(5 / 6)
    np.testing.assert_array_equal(X.data, before_data)
    np.testing.assert_array_equal(X.indices, before_indices)


def test_infinite_labels_reported_separately_from_valid_classes_as_strict_json():
    row = describe_benchmark(_raw(np.ones((2, 2)), np.array([-1.0, np.inf])))
    assert row["y"]["infinite_values"] == 1
    assert row["y"]["class_counts"] == [{"label": -1.0, "count": 1}]
    json.dumps(row, allow_nan=False)


def _npz_export(path, labels):
    stream = BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "shape": (len(labels),), "fortran_order": False, "descr": "|O",
    })
    stream.write(pickle.dumps(np.array(labels, dtype=object), protocol=4))
    features = BytesIO()
    np.lib.format.write_array(features, np.zeros((len(labels), 2)))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("X.npy", features.getvalue())
        archive.writestr("y.npy", stream.getvalue())


def test_literal_labels_reader_keeps_order_across_pickle_batches(tmp_path):
    path = tmp_path / "labels.npz"
    labels = ["-1", "1", "1", "-1"] * 700
    _npz_export(path, labels)
    actual = read_original_npz_labels(path, len(labels))
    np.testing.assert_array_equal(actual, np.array(labels, dtype=object))
    assert actual.dtype == object


class _UnexpectedCallable:
    def __reduce__(self):
        return print, ("PICKLE_MUST_NOT_EXECUTE",)


@pytest.mark.parametrize("labels", [["-1", "unexpected"], ["-1", _UnexpectedCallable()]])
def test_nonliteral_label_payload_rejected_without_execution(tmp_path, capsys, labels):
    path = tmp_path / "labels.npz"
    _npz_export(path, labels)
    with pytest.raises(ValueError):
        read_original_npz_labels(path, len(labels))
    assert "PICKLE_MUST_NOT_EXECUTE" not in capsys.readouterr().out


def test_hash_mismatch_rejected_before_data_parsing(tmp_path, monkeypatch):
    from src.data import benchmark_loaders

    path = tmp_path / "BASEHOCK.mat"
    path.write_bytes(b"not a MAT file")
    manifest = {"schema_version": 3, "representation": "original_uploads", "transformation": "none",
                "files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": "0" * 64}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(benchmark_loaders, "loadmat", lambda *args: pytest.fail("must verify first"))
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        load_basehock(data_root=tmp_path)


def _fixture_originals(root):
    source = root / "gina.npz"
    source.write_bytes(b"fixture original, not a real NPZ")
    manifest = {"schema_version": 3, "representation": "original_uploads", "transformation": "none",
                "files": [{"path": source.name, "bytes": source.stat().st_size,
                           "sha256": sha256_file(source)}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return source


def test_writer_never_overwrites_original_manifest_even_when_requested(tmp_path, audit):
    source = _fixture_originals(tmp_path)
    before = sha256_file(source)
    with pytest.raises(ValueError, match="separate JSON"):
        write_validation_manifest(audit, tmp_path / "manifest.json", data_root=tmp_path, overwrite=True)
    with pytest.raises(ValueError, match="separate JSON"):
        write_validation_manifest(audit, source, data_root=tmp_path, overwrite=True)
    assert sha256_file(source) == before


def test_report_writer_requires_explicit_overwrite(tmp_path, audit):
    path = tmp_path / "validation.json"
    write_validation_manifest(audit, path)
    with pytest.raises(FileExistsError):
        write_validation_manifest(audit, path)
    write_validation_manifest(audit, path, overwrite=True)
    assert json.loads(path.read_text()) == audit
    before = path.read_bytes()
    with pytest.raises(ValueError):
        write_validation_manifest({"invalid": np.nan}, path, overwrite=True)
    assert path.read_bytes() == before


def test_report_writer_rejects_hardlink_to_original(tmp_path, audit):
    alias = tmp_path / "alias.json"
    source = _fixture_originals(tmp_path)
    before = sha256_file(source)
    os.link(source, alias)
    with pytest.raises(ValueError, match="separate JSON"):
        write_validation_manifest(audit, alias, data_root=tmp_path, overwrite=True)
    assert sha256_file(source) == before


def test_embedded_update_changes_only_validation_block(tmp_path):
    source = _fixture_originals(tmp_path)
    path = tmp_path / "manifest.json"
    before = json.loads(path.read_text())
    source_hash = sha256_file(source)
    report = {"scope": "original_benchmark_validation",
              "source_inventory_sha256": source_inventory_fingerprint(before)}
    update_dataset_validation(report, data_root=tmp_path)
    after = json.loads(path.read_text())
    assert after.pop("validation") == report
    assert after == before
    assert sha256_file(source) == source_hash
    assert source_inventory_fingerprint(json.loads(path.read_text())) == report["source_inventory_sha256"]


def test_embedded_update_refuses_stale_inventory_or_changed_input(tmp_path):
    source = _fixture_originals(tmp_path)
    path = tmp_path / "manifest.json"
    before = path.read_bytes()
    report = {"scope": "original_benchmark_validation", "source_inventory_sha256": "stale"}
    with pytest.raises(ValueError, match="current source inventory"):
        update_dataset_validation(report, data_root=tmp_path)
    report["source_inventory_sha256"] = source_inventory_fingerprint(json.loads(path.read_text()))
    source.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        update_dataset_validation(report, data_root=tmp_path)
    assert path.read_bytes() == before


def test_audit_records_bad_shape_instead_of_claiming_success(monkeypatch):
    from src.data import benchmark_validation

    raw_loader = benchmark_validation.load_benchmark_dataset

    def bad_basehock(**kwargs):
        data = raw_loader(**kwargs)
        return replace(data, X=data.X[:-1]) if data.dataset == "basehock" else data

    monkeypatch.setattr(benchmark_validation, "load_benchmark_dataset", bad_basehock)
    report = benchmark_validation.audit_benchmark_datasets()
    assert report["status"] == "failed"
    assert report["summary"]["failed"] == 1
    assert any("X.shape" in error for error in report["partitions"][0]["errors"])


def test_cli_inspection_is_read_only_and_never_runs_hardness(monkeypatch, capsys):
    import main
    from src.experiments import verapin

    def forbidden(*args, **kwargs):
        pytest.fail("data validation must not train or write a report by default")

    monkeypatch.setattr(main, "run_experiment", forbidden)
    monkeypatch.setattr(verapin, "run_hardness_benchmark", forbidden)
    from src.data import benchmark_validation
    monkeypatch.setattr(benchmark_validation, "write_validation_manifest", forbidden)
    monkeypatch.setattr(benchmark_validation, "update_dataset_validation", forbidden)
    assert main.main(["validate-datasets"]) == 0
    assert "'partitions': 8" in capsys.readouterr().out
